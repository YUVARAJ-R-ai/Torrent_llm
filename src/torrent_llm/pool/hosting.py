"""Loading, serving and unloading the stages a device has been asked to hold.

A device may hold several stages (if it has the memory) and the set changes while
it runs: stages come up when the pool needs another copy, and go when the owner
wants the device back. Each held stage is a real gRPC shard server on its own
port, exactly the one a standalone ``torrent-shard`` would run.

Loading a model takes seconds, so it happens on a worker thread and the stage is
advertised as ``loading`` meanwhile. Nothing is routed to it until it says
``ready``.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import grpc
import torch

from torrent_llm.codec import Codec, RawCodec
from torrent_llm.pool.members import FAILED, LOADING, READY, HostedStage
from torrent_llm.shard import ShardRuntime, ShardSpec
from torrent_llm.transport import serve

logger = logging.getLogger(__name__)

#: How long a failed load is remembered, so the pool tries another device
#: instead of asking this one to fail the same way again at once.
FAILURE_COOLDOWN_S = 20.0

Loader = Callable[[ShardSpec], ShardRuntime]


class LoadCounter(grpc.ServerInterceptor):
    """Counts the forward passes a shard is running right now and has run.

    The pool routes by how busy each copy is, and only the device itself can say
    that accurately: it sees every requester, where a requester sees only its own.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.served = 0

    def intercept_service(self, continuation, handler_call_details):
        handler = continuation(handler_call_details)
        if handler is None or not handler_call_details.method.endswith("/Forward"):
            return handler
        if handler.unary_unary is None:
            return handler
        inner = handler.unary_unary

        def counted(request, context):
            with self._lock:
                self.active += 1
            try:
                return inner(request, context)
            finally:
                with self._lock:
                    self.active -= 1
                    self.served += 1

        return grpc.unary_unary_rpc_method_handler(
            counted,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )


@dataclass
class _Held:
    stage: int
    spec: ShardSpec
    status: str = LOADING
    port: int = 0
    runtime: ShardRuntime | None = None
    server: grpc.Server | None = None
    counter: LoadCounter = field(default_factory=LoadCounter)
    failed_at: float = 0.0
    error: str = ""
    #: Set when the stage is asked to go, so a load still in progress gives up
    #: instead of coming up after it was wanted gone.
    cancelled: bool = False


class ShardHost:
    """The stages one device is holding, and the means to change them.

    Args:
        specs: The layer range of every stage in the pool.
        loader: Builds the runtime for one stage. Swapped for a tiny model in tests.
        model_id: Reported to requesters, which check it matches.
        bind: Address the shard servers listen on.
        codec: Codec the servers use on the wire.
    """

    def __init__(
        self,
        specs: list[ShardSpec],
        loader: Loader,
        *,
        model_id: str,
        bind: str = "127.0.0.1",
        codec: Codec | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._specs = specs
        self._loader = loader
        self._model_id = model_id
        self._bind = bind
        self._codec = codec or RawCodec()
        self._clock = clock
        self._held: dict[int, _Held] = {}
        self._lock = threading.RLock()

    # --- what is held --------------------------------------------------------

    def hosting(self) -> tuple[HostedStage, ...]:
        """What to advertise: every stage, how far along it is, and how busy."""
        with self._lock:
            return tuple(
                HostedStage(
                    stage=h.stage,
                    port=h.port,
                    status=h.status,
                    inflight=h.counter.active,
                    served=h.counter.served,
                )
                for h in sorted(self._held.values(), key=lambda h: h.stage)
            )

    def stages(self) -> set[int]:
        """Stages that are loading or ready (not failed)."""
        with self._lock:
            return {s for s, h in self._held.items() if h.status != FAILED}

    def inflight(self, stage: int) -> int:
        with self._lock:
            held = self._held.get(stage)
            return held.counter.active if held else 0

    def total_inflight(self) -> int:
        with self._lock:
            return sum(h.counter.active for h in self._held.values())

    def error(self, stage: int) -> str:
        with self._lock:
            held = self._held.get(stage)
            return held.error if held else ""

    def device(self) -> str:
        """The torch device of whatever is loaded, or ``""`` if nothing is."""
        with self._lock:
            for held in self._held.values():
                if held.runtime is not None:
                    return str(held.runtime.device)
        return ""

    # --- changing it -----------------------------------------------------------

    def start(self, stage: int) -> None:
        """Begin loading ``stage`` in the background. A no-op if already held."""
        with self._lock:
            known = self._held.get(stage)
            if known is not None and known.status != FAILED:
                return
            held = _Held(stage=stage, spec=self._specs[stage])
            self._held[stage] = held
        threading.Thread(
            target=self._load, args=(held,), name=f"load-stage-{stage}", daemon=True
        ).start()

    def stop(self, stage: int) -> None:
        """Stop serving ``stage`` and free what it used."""
        with self._lock:
            held = self._held.pop(stage, None)
        if held is None:
            return
        held.cancelled = True
        if held.server is not None:
            # A short grace lets a pass already running finish rather than be
            # cut off; the caller has normally waited for it to drain already.
            held.server.stop(grace=1.0).wait(2.0)
        held.runtime = None
        held.server = None
        _release_memory()
        logger.info("stopped stage %d", stage)

    def forget_old_failures(self) -> None:
        """Clear failures that have cooled down, so the stage may be tried here again."""
        now = self._clock()
        with self._lock:
            for stage in [
                s
                for s, h in self._held.items()
                if h.status == FAILED and now - h.failed_at > FAILURE_COOLDOWN_S
            ]:
                del self._held[stage]

    def close(self) -> None:
        for stage in list(self._held):
            self.stop(stage)

    # --- the load itself -------------------------------------------------------

    def _load(self, held: _Held) -> None:
        try:
            runtime = self._loader(held.spec)
            if held.cancelled:
                return
            server, port = serve(
                runtime,
                self._codec,
                host=self._bind,
                port=0,
                model_id=self._model_id,
                interceptors=[held.counter],
            )
        except Exception as exc:  # noqa: BLE001 - a failed load must not kill the device
            logger.exception("loading stage %d failed", held.stage)
            with self._lock:
                held.status = FAILED
                held.error = str(exc)
                held.failed_at = self._clock()
            return
        with self._lock:
            if held.cancelled:
                server.stop(grace=0)
                return
            held.runtime = runtime
            held.server = server
            held.port = port
            held.status = READY
        logger.info("stage %d ready on port %d (%s)", held.stage, port, runtime.device)


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
