"""Running a request across the pool, and surviving a device that disappears.

A request needs one copy of every stage. This runner picks them (see
``router.py``), sends the prompt through, and then decodes one token at a time.
If a device stops answering, it does not give up and it does not start over:

1. The copies on the new route have none of this request's attention cache, so
   the text produced so far is sent through again as a fresh prompt. That costs
   one extra prefill, the same as a longer prompt would.
2. Decoding carries on from the token it was about to produce.

The caller sees one answer, with a record of what happened on the way. The only
visible effect of losing a device is a short pause.

What it deliberately does not do: it does not try to be clever about *which*
requests to retry. A pass that fails is always retried, because the pass is a pure
function of the tokens so far, so repeating it is safe.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Collection
from dataclasses import dataclass, field

import torch

from torrent_llm.link import LinkProfile
from torrent_llm.pool.config import PoolConfig
from torrent_llm.pool.members import MemberTable
from torrent_llm.pool.router import NoReplica, Replica, pick_route
from torrent_llm.runner import ChainRunner, HopError, NodeConfig, TopologyConfig
from torrent_llm.sampling import GREEDY, Sampling

logger = logging.getLogger(__name__)

#: Smoothing for the measured round trip to each copy. Low enough to follow a
#: link that gets worse, high enough that one slow pass does not swing routing.
_RTT_SMOOTHING = 0.3


class PoolUnavailable(RuntimeError):
    """The pool could not finish the request, and says what was missing."""

    def __init__(self, message: str, *, stage: int | None = None) -> None:
        super().__init__(message)
        self.stage = stage


@dataclass
class HopInfo:
    """One stage of one pass: where it ran and what it cost."""

    stage: int
    member: str
    address: str
    sent_bytes: int
    received_bytes: int
    compute_ms: float
    transport_ms: float
    wall_ms: float


@dataclass
class PassInfo:
    """One trip of one position batch through every stage."""

    index: int
    step: int
    phase: str
    #: True when this pass re-sent the text so far to a new route after a failure.
    recovery: bool
    seq_len: int
    route: list[tuple[int, str]]
    hops: list[HopInfo]
    token_id: int | None = None


@dataclass
class PoolEvent:
    """Something that happened to the request on its way."""

    kind: str  # "failover" | "waiting" | "recovered"
    at_pass: int  # how many passes had completed when it happened
    step: int
    stage: int | None = None
    member: str | None = None
    detail: str = ""
    #: Seconds since the request began.
    t: float = 0.0


@dataclass
class PoolResult:
    ids: torch.Tensor
    passes: list[PassInfo] = field(default_factory=list)
    events: list[PoolEvent] = field(default_factory=list)

    @property
    def failovers(self) -> int:
        return sum(1 for e in self.events if e.kind == "failover")


class PoolRunner:
    """Runs requests across whatever the table says is alive.

    One instance serves many concurrent requests: everything per-request lives on
    the stack of :meth:`generate`. What is shared is what *should* be: how many
    requests this runner currently has on each copy, and how long each copy has
    been taking to answer.

    Args:
        me: This device's own name. Stages it holds itself are reached over
            loopback, so the simulated link is not applied to them.
        patience: How long to wait for a missing stage to reappear before giving up.
        max_failovers: How many times one request may switch route.
    """

    def __init__(
        self,
        config: PoolConfig,
        table: MemberTable,
        *,
        me: str | None = None,
        patience: float = 10.0,
        max_failovers: int = 6,
        poll_s: float = 0.25,
    ) -> None:
        self.config = config
        self.table = table
        self.me = me
        self.patience = patience
        self.max_failovers = max_failovers
        self._poll_s = poll_s
        self._lock = threading.Lock()
        self._load: dict[str, int] = {}
        self.rtt_ms: dict[str, float] = {}

    # --- routing -----------------------------------------------------------

    def route(self, *, avoid: Collection[str] = ()) -> list[Replica]:
        """One copy of every stage, or :class:`NoReplica` if one is missing."""
        members = self.table.alive()
        with self._lock:
            load, rtt = dict(self._load), dict(self.rtt_ms)
        try:
            return pick_route(self.config, members, exclude=avoid, local_load=load, rtt_ms=rtt)
        except NoReplica:
            if not avoid:
                raise
        # Everything left is something this request already found unhappy. A
        # second chance beats giving up while a stage has a copy at all.
        return pick_route(self.config, members, local_load=load, rtt_ms=rtt)

    def _await_route(self, avoid: Collection[str], *, on_wait) -> list[Replica]:
        """Route, waiting up to ``patience`` for a missing stage to come back."""
        deadline = time.monotonic() + self.patience
        waited = False
        while True:
            try:
                return self.route(avoid=avoid)
            except NoReplica as exc:
                if time.monotonic() >= deadline:
                    raise PoolUnavailable(
                        f"{exc}, and none came back within {self.patience:.0f}s", stage=exc.stage
                    ) from exc
                if not waited:
                    on_wait(exc.stage)
                    waited = True
                time.sleep(self._poll_s)

    def _topology(self, route: list[Replica]) -> TopologyConfig:
        def link_of(r: Replica) -> LinkProfile | None:
            return None if r.member == self.me else LinkProfile.from_dict(r.link)

        return TopologyConfig(
            model_id=self.config.model_id,
            num_layers=self.config.num_layers,
            dtype=self.config.dtype,
            nodes=[NodeConfig(address=r.address, link=link_of(r)) for r in route],
        )

    def _charge(self, route: list[Replica], delta: int) -> None:
        with self._lock:
            for r in route:
                self._load[r.address] = max(0, self._load.get(r.address, 0) + delta)

    def _observe(self, hops: list[HopInfo]) -> None:
        with self._lock:
            for hop in hops:
                old = self.rtt_ms.get(hop.address)
                self.rtt_ms[hop.address] = (
                    hop.transport_ms
                    if old is None
                    else old + _RTT_SMOOTHING * (hop.transport_ms - old)
                )

    # --- generating ------------------------------------------------------------

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 16,
        stop_token_ids: Collection[int] = (),
        sampling: Sampling = GREEDY,
    ) -> PoolResult:
        """Decode, switching to other copies if a device is lost on the way.

        Raises:
            PoolUnavailable: a stage had no copy for longer than ``patience``, or
                the request had to switch route more than ``max_failovers`` times.
        """
        started = time.monotonic()
        result = PoolResult(ids=input_ids)
        events = result.events

        def note(kind: str, step: int, **kw) -> None:
            events.append(
                PoolEvent(
                    kind=kind,
                    at_pass=len(result.passes),
                    step=step,
                    t=time.monotonic() - started,
                    **kw,
                )
            )

        ids = input_ids
        avoid: set[str] = set()
        route = self._await_route(avoid, on_wait=lambda s: note("waiting", 0, stage=s))
        self._charge(route, +1)
        runner = ChainRunner(self._topology(route))

        request_id = uuid.uuid4().hex
        sent = ids
        cache_position = 0
        recovery = False
        step = 0
        try:
            while step < max_new_tokens:
                is_last_step = step == max_new_tokens - 1
                phase = "prefill" if cache_position == 0 else "decode"
                try:
                    chain = runner.forward(
                        sent,
                        phase=phase,
                        logits_keep_last=1,
                        request_id=request_id,
                        use_cache=True,
                        end_of_request=is_last_step,
                        expected_cache_position=cache_position,
                        sampling=sampling.for_step(step),
                    )
                except HopError as exc:
                    failed = route[exc.hop]
                    logger.warning("%s: switching route (%s)", failed.member, exc)
                    note(
                        "failover",
                        step,
                        stage=failed.stage,
                        member=failed.member,
                        detail=str(exc.code.name if exc.code else exc),
                    )
                    if result.failovers > self.max_failovers:
                        raise PoolUnavailable(
                            f"gave up after {self.max_failovers} route changes", stage=failed.stage
                        ) from exc
                    if exc.is_node_loss:
                        self.table.mark_unreachable(failed.member)
                    avoid.add(failed.member)
                    self._charge(route, -1)
                    runner.close()
                    route = self._await_route(
                        avoid, on_wait=lambda s, at=step: note("waiting", at, stage=s)
                    )
                    self._charge(route, +1)
                    runner = ChainRunner(self._topology(route))
                    # The new copies have none of this request's cache: begin
                    # again from the text so far, as a prompt.
                    request_id = uuid.uuid4().hex
                    sent = ids
                    cache_position = 0
                    recovery = True
                    continue

                assert chain.token_ids is not None
                hops = [
                    HopInfo(
                        stage=route[i].stage,
                        member=route[i].member,
                        address=route[i].address,
                        sent_bytes=h.sent_bytes,
                        received_bytes=h.received_bytes,
                        compute_ms=h.compute_ns / 1e6,
                        transport_ms=h.transport_ns / 1e6,
                        wall_ms=h.wall_ns / 1e6,
                    )
                    for i, h in enumerate(chain.hops)
                ]
                self._observe(hops)
                token = chain.token_ids.to(ids.device)
                result.passes.append(
                    PassInfo(
                        index=len(result.passes),
                        step=step,
                        phase=phase,
                        recovery=recovery,
                        seq_len=int(sent.shape[1]),
                        route=[(r.stage, r.member) for r in route],
                        hops=hops,
                        token_id=int(token.item()),
                    )
                )
                if recovery:
                    note("recovered", step, detail=" -> ".join(r.member for r in route))
                    recovery = False

                cache_position += int(sent.shape[1])
                if int(token.item()) in stop_token_ids:
                    break
                ids = torch.cat([ids, token], dim=1)
                sent = token
                step += 1
        finally:
            self._charge(route, -1)
            runner.close()
        result.ids = ids
        return result
