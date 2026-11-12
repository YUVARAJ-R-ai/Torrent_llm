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
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import grpc
import torch

from torrent_llm.codec import Codec, RawCodec
from torrent_llm.codec.profiles import CodecSet
from torrent_llm.pool.members import FAILED, LOADING, READY, HostedStage
from torrent_llm.shard import ShardRuntime, ShardSpec
from torrent_llm.transport import activation_pb2 as pb
from torrent_llm.transport import serve
from torrent_llm.transport.auth import TokenInterceptor
from torrent_llm.transport.server import ShardService

logger = logging.getLogger(__name__)

#: How long a failed load is remembered, so the pool tries another device
#: instead of asking this one to fail the same way again at once.
FAILURE_COOLDOWN_S = 20.0

Loader = Callable[[ShardSpec], ShardRuntime]

#: How a device's stages are reached: a gRPC server per stage (the original pool and
#: the single-chain harness), or method calls over the device's encrypted overlay.
GRPC, OVERLAY = "grpc", "overlay"


class NotHosted(LookupError):
    """The stage asked for is not ready on this device (not held, loading or failed)."""


class Busy(RuntimeError):
    """The stage is at its limit and refused a new question before doing any work."""


class LoadCounter(grpc.ServerInterceptor):
    """Counts the forward passes a shard is running right now and has run.

    The pool routes by how busy each copy is, and only the device itself can say
    that accurately: it sees every requester, where a requester sees only its own.
    """

    def __init__(self, limit: int | None = None) -> None:
        #: Most forward passes allowed at once before new questions are refused.
        #: ``None`` means no limit.
        self.limit = limit
        self._lock = threading.Lock()
        self.rejected = 0
        self.active = 0
        self.served = 0
        #: Nanoseconds spent inside forward passes, all callers together.
        self.busy_ns = 0

    def admit(self, request) -> bool:
        """Whether ``request`` may start now. Only a new question can be refused."""
        if self.limit is None or not _starts_a_question(request):
            return True
        with self._lock:
            if self.active >= self.limit:
                self.rejected += 1
                return False
            return True

    @contextmanager
    def track(self) -> Iterator[None]:
        """Count one forward pass while it runs."""
        with self._lock:
            self.active += 1
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed = time.perf_counter_ns() - started
            with self._lock:
                self.active -= 1
                self.served += 1
                self.busy_ns += elapsed

    def intercept_service(self, continuation, handler_call_details):
        handler = continuation(handler_call_details)
        if handler is None or not handler_call_details.method.endswith("/Forward"):
            return handler
        if handler.unary_unary is None:
            return handler
        inner = handler.unary_unary

        def counted(request, context):
            if not self.admit(request):
                # Refused before any work, so nothing needs undoing: the requester
                # simply asks another copy.
                context.abort(
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                    "this device is at its limit right now",
                )
            with self.track():
                return inner(request, context)

        return grpc.unary_unary_rpc_method_handler(
            counted,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )


def _starts_a_question(request) -> bool:
    """Whether this call begins a new question, as opposed to continuing one.

    Only the start may be refused. A question already under way has spent effort on
    every stage it passed through, and cutting it off to save a little here would
    throw all of that away and make the overload worse.
    """
    if not request.use_cache:
        return True
    return request.HasField("expected_cache_position") and request.expected_cache_position == 0


@dataclass
class _Held:
    stage: int
    spec: ShardSpec
    status: str = LOADING
    port: int = 0
    runtime: ShardRuntime | None = None
    server: grpc.Server | None = None
    service: ShardService | None = None
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
        token: str | None = None,
        port_base: int = 0,
        max_requests: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        transport: str = GRPC,
        codecs: CodecSet | None = None,
    ) -> None:
        if transport not in (GRPC, OVERLAY):
            raise ValueError(f"transport must be {GRPC!r} or {OVERLAY!r}")
        self.transport = transport
        self._codecs = codecs
        self._max_requests = max_requests
        #: First port for shard servers: stage ``s`` listens on ``port_base + s``. Zero
        #: lets the system choose, which suits several devices on one machine; fixed
        #: ports let a real device open exactly the range its firewall needs.
        self._port_base = port_base
        self._token = token
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
                    busy_ms=h.counter.busy_ns // 1_000_000,
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
            held = _Held(
                stage=stage,
                spec=self._specs[stage],
                counter=LoadCounter(self._max_requests),
            )
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
        held.service = None
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

    def forward(self, stage: int, request: pb.ForwardRequest) -> pb.ForwardReply:
        """Run one forward pass of ``stage`` (overlay transport).

        Raises:
            NotHosted: the stage is not ready here.
            Busy: at the limit; refused before any work.
            CacheSessionLost, UnknownProfile, ValueError: from the shard itself.
        """
        with self._lock:
            held = self._held.get(stage)
            service = held.service if held is not None and held.status == READY else None
        if held is None or service is None:
            raise NotHosted(f"stage {stage} is not ready on this device")
        if not held.counter.admit(request):
            raise Busy("this device is at its limit right now")
        with held.counter.track():
            return service._forward(request)

    def _load(self, held: _Held) -> None:
        if self.transport == OVERLAY:
            self._load_overlay(held)
            return
        try:
            runtime = self._loader(held.spec)
            if held.cancelled:
                return
            server, port = serve(
                runtime,
                self._codec,
                host=self._bind,
                port=self._port_base + held.stage if self._port_base else 0,
                model_id=self._model_id,
                interceptors=[
                    *([TokenInterceptor(self._token)] if self._token else []),
                    held.counter,
                ],
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

    def _load_overlay(self, held: _Held) -> None:
        try:
            runtime = self._loader(held.spec)
            service = ShardService(
                runtime, self._codec, model_id=self._model_id, codecs=self._codecs
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
                return
            held.runtime = runtime
            held.service = service
            held.status = READY
        logger.info("stage %d ready (%s)", held.stage, runtime.device)


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
