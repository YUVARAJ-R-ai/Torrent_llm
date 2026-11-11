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
from collections.abc import Callable, Collection
from dataclasses import dataclass, field

import torch

from torrent_llm.codec import Codec
from torrent_llm.codec.profiles import CodecSet
from torrent_llm.link import LinkProfile
from torrent_llm.net.node import Node
from torrent_llm.pool.codec_policy import CodecChooser, LinkEstimator
from torrent_llm.pool.config import PoolConfig
from torrent_llm.pool.members import MemberTable
from torrent_llm.pool.overlay import OverlaySender
from torrent_llm.pool.router import NoReplica, Replica, pick_route
from torrent_llm.pool.spotcheck import SpotChecker
from torrent_llm.pool.trust import TrustStore, Verifier
from torrent_llm.runner import ChainRunner, HopError, NodeConfig, TopologyConfig
from torrent_llm.sampling import GREEDY, Sampling
from torrent_llm.transport import ShardClient

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
    #: What ``sent_bytes`` would have been with no compression.
    uncompressed_bytes: int
    received_bytes: int
    compute_ms: float
    transport_ms: float
    wall_ms: float
    #: Codec (profile id over the overlay) this hop was sent with.
    codec: str = ""


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

    kind: str  # "failover" | "waiting" | "recovered" | "busy"
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
        max_busy_retries: int = 30,
        poll_s: float = 0.25,
        token: str | None = None,
        node: Node | None = None,
        codecs: CodecSet | None = None,
        trust: TrustStore | None = None,
        verify_rate: float = 0.0,
    ) -> None:
        #: Reputation and bans; banned devices are never routed through.
        self.trust = trust
        #: When set, hops travel over this overlay node instead of gRPC (``overlay.py``).
        self.node = node
        self.codecs = codecs or (config.codec_set() if node is not None else None)
        #: Link speed per peer, measured from real hops, and the codec choice made from it.
        self.links = LinkEstimator()
        self.chooser = (
            CodecChooser(config.policy(), self.codecs, self.links)
            if self.codecs is not None
            else None
        )
        self.spot = (
            SpotChecker(
                trust,
                Verifier(verify_rate),
                me=me,
                members=self.table.alive,
                chain=self._chain,
            )
            if trust is not None and verify_rate > 0
            else None
        )
        self.max_busy_retries = max_busy_retries
        self._token = token
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
        if self.trust is not None:
            members = [m for m in members if not self.trust.is_banned(m.name)]
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
                    raise PoolUnavailable(self._why_unavailable(exc), stage=exc.stage) from exc
                if not waited:
                    on_wait(exc.stage)
                    waited = True
                time.sleep(self._poll_s)

    def _why_unavailable(self, exc: NoReplica) -> str:
        """A message that blames the right thing.

        A device that has found no one else cannot route anything, and "stage 1 has no
        copy" would send the person looking at the wrong problem: the fault is that this
        device is not connected to the pool at all.
        """
        others = [m for m in self.table.members() if m.name != self.table.me]
        if not others:
            return (
                "this device has not found any other device in the pool. Check that the "
                "--join address is right and reachable (or that broadcast discovery works on "
                "this network), and run: torrent-peer --check"
            )
        return f"{exc}, and none came back within {self.patience:.0f}s"

    def _topology(self, route: list[Replica]) -> TopologyConfig:
        def link_of(r: Replica) -> LinkProfile | None:
            return None if r.member == self.me else LinkProfile.from_dict(r.link)

        return TopologyConfig(
            model_id=self.config.model_id,
            num_layers=self.config.num_layers,
            dtype=self.config.dtype,
            codec=self.config.codec,
            codec_args=self.config.codec_args,
            nodes=[NodeConfig(address=r.address, link=link_of(r)) for r in route],
        )

    def _chain(self, route: list[Replica]) -> ChainRunner:
        """A chain over ``route``: gRPC, or the overlay when this runner has a node."""
        topology = self._topology(route)
        if self.node is None:
            return ChainRunner(topology, token=self._token)
        node, codecs, chooser = self.node, self.codecs, self.chooser
        if codecs is None or chooser is None:
            raise RuntimeError("an overlay runner needs codec profiles")
        default = codecs.codec(self.config.policy()["fixed"])

        def client(hop: int, cfg: NodeConfig) -> ShardClient:
            replica = route[hop]

            def choose(tensor: torch.Tensor) -> Codec:
                seq_len = int(tensor.shape[-2]) if tensor.dim() >= 2 else 1
                raw = tensor.numel() * tensor.element_size()
                return codecs.codec(chooser.choose(replica.member, raw, seq_len))

            return ShardClient(
                replica.address,
                default,
                link=cfg.link,
                sender=OverlaySender(node, replica.member, replica.stage),
                codecs=codecs,
                choose=choose,
            )

        return ChainRunner(topology, client_factory=client)

    def _connected(self, hop: HopInfo) -> bool:
        return self.node is not None and self.node.connection(hop.member) is not None

    def _charge(self, route: list[Replica], delta: int) -> None:
        with self._lock:
            for r in route:
                self._load[r.address] = max(0, self._load.get(r.address, 0) + delta)

    def _observe(self, hops: list[HopInfo]) -> None:
        for hop in hops:
            if hop.member != self.me:
                rtt = self.node.connection(hop.member).rtt_ms if self._connected(hop) else None
                self.links.observe(hop.member, hop.sent_bytes, hop.transport_ms, rtt)
            if self.chooser is not None and hop.codec in (self.codecs or ()):
                self.chooser.learn(hop.codec, hop.sent_bytes, hop.uncompressed_bytes)
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
        on_token: Callable[[int], None] | None = None,
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
        check_at = self.spot.plan(max_new_tokens) if self.spot is not None else None
        avoid: set[str] = set()
        route = self._await_route(avoid, on_wait=lambda s: note("waiting", 0, stage=s))
        self._charge(route, +1)
        runner = self._chain(route)

        request_id = uuid.uuid4().hex
        sent = ids
        cache_position = 0
        recovery = False
        busy_rejections = 0
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
                    if exc.is_busy:
                        # Not a failure: the device is full and said so before doing any
                        # work. Ask another copy, and back off if they are all full.
                        busy_rejections += 1
                        note(
                            "busy",
                            step,
                            stage=failed.stage,
                            member=failed.member,
                            detail="at its limit",
                        )
                        if busy_rejections > self.max_busy_retries:
                            raise PoolUnavailable(
                                f"every device holding stage {failed.stage + 1} is busy right "
                                "now. Try again in a moment.",
                                stage=failed.stage,
                            ) from exc
                        avoid.add(failed.member)
                        self._charge(route, -1)
                        runner.close()
                        time.sleep(min(0.5, 0.05 * busy_rejections))
                        route = self._await_route(
                            avoid, on_wait=lambda s, at=step: note("waiting", at, stage=s)
                        )
                        self._charge(route, +1)
                        runner = self._chain(route)
                        request_id = uuid.uuid4().hex
                        sent = ids
                        cache_position = 0
                        continue
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
                    runner = self._chain(route)
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
                        uncompressed_bytes=h.sent_uncompressed_bytes,
                        received_bytes=h.received_bytes,
                        compute_ms=h.compute_ns / 1e6,
                        transport_ms=h.transport_ns / 1e6,
                        wall_ms=h.wall_ns / 1e6,
                        codec=h.codec,
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
                if self.spot is not None and check_at == len(result.passes) - 1:
                    # The same text, on the same route but one stage, in the background.
                    self.spot.submit(route, ids, int(token.item()), sampling.for_step(step))

                cache_position += int(sent.shape[1])
                if int(token.item()) in stop_token_ids:
                    break
                if on_token is not None:
                    # Streaming to a person: a slow or failing reader must not end the answer.
                    try:
                        on_token(int(token.item()))
                    except Exception:  # noqa: BLE001
                        logger.exception("token callback failed")
                ids = torch.cat([ids, token], dim=1)
                sent = token
                step += 1
        finally:
            self._charge(route, -1)
            runner.close()
        result.ids = ids
        return result
