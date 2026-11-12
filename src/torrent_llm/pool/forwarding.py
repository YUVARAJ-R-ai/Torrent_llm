"""Peer-to-peer forwarding: hidden states go from stage to stage, not via the asker (#37).

With requester routing, the asking device sends every hop and gets every reply back,
so each activation crosses two links per stage. Here the asker sends hop 0 with the
whole route in its header; each stage runs its layers and hands the result straight to
the next stage; the last stage picks the token and delivers it to the asker. Each
link is crossed once per step, and only a token comes home::

    asker --tokens--> stage 0 --hidden--> stage 1 --hidden--> stage 2 --token--> asker

Every hop is acknowledged as soon as it arrives (``shard.relay`` returns before the
work is done), so a stage that cannot reach the next one, or fails, can tell the asker
which hop broke (an error delivery). A stage that goes silent after acknowledging is
found by the asker when the step's deadline passes: it pings the route in order and
names the first device that does not answer.

The bytes and checks are the same as requester routing: the same ``ForwardRequest``,
the same KV-cache sessions and position checks (#28), the same sampling on the last
stage (#33). A reply is forwarded exactly as it came out of the shard, so a lossy
codec is applied once per hop, never twice.

Trust: a stage accepts hop ``i`` only from the device that holds hop ``i - 1`` on the
route (or from the asker, for hop 0), and only when the route says hop ``i`` is its
own. The asker accepts a delivery only from a device on that step's route.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
import weakref
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch

from torrent_llm.codec import Codec
from torrent_llm.codec.profiles import CodecSet, UnknownProfile
from torrent_llm.net.identity import is_peer_id
from torrent_llm.net.node import Node
from torrent_llm.net.rpc import CallContext, RpcError
from torrent_llm.pool.hosting import Busy, NotHosted, ShardHost
from torrent_llm.pool.overlay import (
    BASE_TIMEOUT_S,
    PER_TOKEN_S,
    WORST_BANDWIDTH_BPS,
    OverlayRpcError,
    hop_timeout,
)
from torrent_llm.runner import ChainResult, HopError
from torrent_llm.sampling import Sampling
from torrent_llm.transport import HopResult, ShardClient
from torrent_llm.transport import activation_pb2 as pb
from torrent_llm.transport.convert import header_from_proto
from torrent_llm.transport.server import CacheSessionLost

logger = logging.getLogger(__name__)

RELAY = "shard.relay"
DELIVER = "shard.deliver"

#: Longest route a stage will forward along; a pool has far fewer stages than this.
MAX_ROUTE = 64
#: Steps a stage will hold queued at once before refusing more as busy.
MAX_PENDING = 32
#: Widest hidden state a step deadline allows for (float32 values per position).
DEADLINE_HIDDEN = 8192
#: How long the asker waits for each route member to answer a ping, once a step is late.
PROBE_TIMEOUT_S = 2.0
#: How long the last stage tries to hand the token to the asker.
DELIVER_TIMEOUT_S = 10.0

_TRACE_KEYS = {"peer", "stage", "compute_ns", "link_ns", "in_bytes", "in_raw", "in_numel", "codec"}


class RouteError(ValueError):
    """A relay header that does not describe a usable route."""


@dataclass(frozen=True)
class Hop:
    peer: str
    stage: int


def parse_route(raw: object) -> list[Hop]:
    """Validate a route from the wire: ``[[peer_id, stage], ...]``."""
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_ROUTE:
        raise RouteError(f"route must be a list of 1..{MAX_ROUTE} hops")
    hops = []
    for item in raw:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not is_peer_id(item[0])
            or not isinstance(item[1], int)
            or isinstance(item[1], bool)
            or item[1] < 0
        ):
            raise RouteError("each hop must be [peer id, stage >= 0]")
        hops.append(Hop(item[0], item[1]))
    return hops


def _error_code(exc: BaseException) -> str:
    if isinstance(exc, NotHosted):
        return "not_found"
    if isinstance(exc, Busy):
        return "busy"
    if isinstance(exc, CacheSessionLost):
        return "cache_lost"
    if isinstance(exc, (UnknownProfile, ValueError)):
        return "bad_request"
    if isinstance(exc, RpcError):
        return exc.code
    return "internal"


def _sizes(header: pb.Header) -> tuple[int, int]:
    """``(elements, uncompressed bytes)`` of the tensor a header describes."""
    numel = header_from_proto(header).logical_numel
    dtype = getattr(torch, header.dtype, None)
    if not isinstance(dtype, torch.dtype):
        return numel, 0
    return numel, numel * torch.empty((), dtype=dtype).element_size()


# --- the stage side ---------------------------------------------------------------


class Forwarder:
    """Serves ``shard.relay``: run this device's stage, pass the result on.

    Work runs on its own small pool, so the acknowledgement goes back at once and a
    slow stage never holds up the connection's other calls.
    """

    def __init__(self, node: Node, host: ShardHost, *, workers: int = 4) -> None:
        self.node = node
        self.host = host
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="forward")
        self._pending = 0
        self._lock = threading.Lock()
        self._closed = False
        node.register(RELAY, self._on_relay, blocking=True)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self.node.unregister(RELAY)
        except Exception:  # noqa: BLE001 - the node may already be closed
            pass
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _on_relay(self, ctx: CallContext, header: dict, body: bytes) -> tuple[dict, bytes]:
        try:
            route = parse_route(header.get("route"))
        except RouteError as exc:
            raise RpcError("bad_request", str(exc)) from exc
        hop, rid, seq = header.get("hop"), header.get("rid"), header.get("seq")
        reply_to, trace = header.get("reply_to"), header.get("trace")
        if not isinstance(hop, int) or isinstance(hop, bool) or not 0 <= hop < len(route):
            raise RpcError("bad_request", "hop is not on the route")
        if not isinstance(rid, str) or not 0 < len(rid) <= 64:
            raise RpcError("bad_request", "rid must be a short string")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
            raise RpcError("bad_request", "seq must be a non-negative integer")
        if not isinstance(reply_to, str) or not is_peer_id(reply_to):
            raise RpcError("bad_request", "reply_to must be a peer id")
        if not isinstance(trace, list) or len(trace) != hop:
            raise RpcError("bad_request", "trace must hold one entry per earlier hop")
        if route[hop].peer != self.node.peer_id:
            raise RpcError("denied", "this hop of the route is not this device")
        sender = reply_to if hop == 0 else route[hop - 1].peer
        if ctx.peer_id != sender:
            raise RpcError("denied", "only the previous hop may send this one")
        request = pb.ForwardRequest()
        try:
            request.ParseFromString(body)
        except Exception as exc:  # noqa: BLE001 - untrusted bytes
            raise RpcError("bad_request", "not a forward request") from exc
        received_ns = time.time_ns()
        with self._lock:
            if self._closed:
                raise RpcError("unavailable", "this device is leaving")
            if self._pending >= MAX_PENDING:
                raise RpcError("busy", "this device is at its limit right now")
            self._pending += 1
        step = _Step(route, hop, rid, seq, reply_to, trace, request, received_ns)
        try:
            self._pool.submit(self._run, step)
        except RuntimeError as exc:  # shut down between the check and the submit
            with self._lock:
                self._pending -= 1
            raise RpcError("unavailable", "this device is leaving") from exc
        return {}, b""

    def _run(self, step: _Step) -> None:
        try:
            self._advance(step)
        except Exception:  # noqa: BLE001 - a worker must never die silently
            logger.exception("forwarding step %s/%d failed", step.rid[:8], step.seq)
        finally:
            with self._lock:
                self._pending -= 1

    def _advance(self, step: _Step) -> None:
        here = step.route[step.hop]
        request = step.request
        sent = request.sent_unix_ns
        entry = {
            "peer": here.peer,
            "stage": here.stage,
            "compute_ns": 0,
            # Clocks of two devices: an estimate, clamped so skew never goes negative.
            "link_ns": max(0, step.received_ns - sent) if sent else 0,
            "in_bytes": len(request.payload),
            "in_raw": 0,
            "in_numel": 0,
            "codec": request.header.codec,
        }
        try:
            entry["in_numel"], entry["in_raw"] = _sizes(request.header)
        except (ValueError, TypeError):  # a malformed header; the shard refuses it below
            pass
        trace = [*step.trace, entry]
        started = time.perf_counter_ns()
        try:
            reply = self.host.forward(here.stage, request)
        except Exception as exc:  # noqa: BLE001 - every failure is reported to the asker
            self._fail(step, step.hop, _error_code(exc), str(exc), trace)
            return
        entry["compute_ns"] = int(reply.compute_ns) or time.perf_counter_ns() - started

        last = step.hop == len(step.route) - 1
        if last:
            if not reply.is_final:
                self._fail(step, step.hop, "internal", "the route ended before the LM head", trace)
                return
            self._deliver(step, {"ok": True, "trace": trace}, reply.SerializeToString())
            return
        if reply.is_final:
            self._fail(
                step, step.hop, "internal", "this stage is final but the route goes on", trace
            )
            return

        onward = pb.ForwardRequest()
        onward.CopyFrom(request)  # keeps cache, sampling and position fields as sent
        onward.header.CopyFrom(reply.header)
        onward.payload = reply.payload
        onward.kind = pb.PAYLOAD_KIND_ACTIVATION
        onward.sent_unix_ns = time.time_ns()
        nxt = step.route[step.hop + 1]
        header = {
            "rid": step.rid,
            "seq": step.seq,
            "hop": step.hop + 1,
            "route": [[h.peer, h.stage] for h in step.route],
            "reply_to": step.reply_to,
            "trace": trace,
        }
        try:
            self.node.call(
                nxt.peer, RELAY, header, onward.SerializeToString(), timeout=hop_timeout(onward)
            )
        except RpcError as exc:
            self._fail(step, step.hop + 1, exc.code, exc.message, trace)
        except RuntimeError:
            logger.debug("node closed before step %d could be forwarded", step.seq)

    def _fail(self, step: _Step, hop: int, code: str, message: str, trace: list) -> None:
        self._deliver(
            step,
            {
                "ok": False,
                "failed_hop": hop,
                "code": code,
                "message": message[:300],
                "trace": trace,
            },
            b"",
        )

    def _deliver(self, step: _Step, fields: dict, body: bytes) -> None:
        header = {"rid": step.rid, "seq": step.seq, **fields}
        try:
            self.node.call(step.reply_to, DELIVER, header, body, timeout=DELIVER_TIMEOUT_S)
        except RpcError as exc:
            # The asker will notice the missing answer at its deadline.
            logger.warning("could not deliver step %d to the asker: %s", step.seq, exc)
        except RuntimeError:
            logger.debug("node closed before step %d could be delivered", step.seq)


@dataclass
class _Step:
    route: list[Hop]
    hop: int
    rid: str
    seq: int
    reply_to: str
    trace: list
    request: pb.ForwardRequest
    received_ns: int


# --- the asker side ---------------------------------------------------------------


@dataclass
class Delivery:
    ok: bool
    trace: list
    body: bytes = b""
    failed_hop: int = -1
    code: str = ""
    message: str = ""


class _Waiter:
    def __init__(self, peers: set[str]) -> None:
        self.peers = peers
        self.done = threading.Event()
        self.delivery: Delivery | None = None


class Deliveries:
    """Where answers to this device's relayed steps arrive (``shard.deliver``)."""

    def __init__(self, node: Node) -> None:
        self.node = node
        self._waiting: dict[tuple[str, int], _Waiter] = {}
        self._lock = threading.Lock()
        node.register(DELIVER, self._on_deliver, blocking=True)

    def expect(self, rid: str, seq: int, peers: set[str]) -> _Waiter:
        waiter = _Waiter(peers)
        with self._lock:
            self._waiting[(rid, seq)] = waiter
        return waiter

    def forget(self, rid: str, seq: int) -> None:
        with self._lock:
            self._waiting.pop((rid, seq), None)

    def _on_deliver(self, ctx: CallContext, header: dict, body: bytes) -> tuple[dict, bytes]:
        key = (header.get("rid"), header.get("seq"))
        with self._lock:
            waiter = self._waiting.get(key)  # type: ignore[arg-type]
        if waiter is None:
            raise RpcError("not_found", "no step is waiting for this")
        if ctx.peer_id not in waiter.peers:
            raise RpcError("denied", "not a device on this step's route")
        ok, trace = header.get("ok"), header.get("trace")
        if not isinstance(ok, bool) or not isinstance(trace, list):
            raise RpcError("bad_request", "a delivery needs ok and trace")
        failed_hop = header.get("failed_hop", -1)
        if not ok and (not isinstance(failed_hop, int) or isinstance(failed_hop, bool)):
            raise RpcError("bad_request", "failed_hop must be an integer")
        if waiter.done.is_set():
            return {}, b""  # first answer wins
        waiter.delivery = Delivery(
            ok=ok,
            trace=trace,
            body=body,
            failed_hop=failed_hop if not ok else -1,
            code=str(header.get("code", ""))[:32],
            message=str(header.get("message", ""))[:300],
        )
        waiter.done.set()
        return {}, b""


_DELIVERIES: weakref.WeakKeyDictionary[Node, Deliveries] = weakref.WeakKeyDictionary()
_DELIVERIES_LOCK = threading.Lock()


def deliveries_for(node: Node) -> Deliveries:
    """The one :class:`Deliveries` of ``node``, registered on first use."""
    with _DELIVERIES_LOCK:
        found = _DELIVERIES.get(node)
        if found is None:
            found = _DELIVERIES[node] = Deliveries(node)
        return found


class _HopFailed(Exception):
    def __init__(self, hop: int, error: RpcError) -> None:
        super().__init__(str(error))
        self.hop = hop
        self.error = error


class _RelaySender:
    """A ``ShardClient`` sender that relays along the route and waits for the token."""

    _seqs = itertools.count()

    def __init__(
        self, node: Node, route: Sequence[Hop], deliveries: Deliveries, deadline_s: float | None
    ) -> None:
        self.node = node
        self.route = list(route)
        self.deliveries = deliveries
        self.deadline_s = deadline_s
        self.trace: list = []
        self.first_ns = 0

    def step_deadline(self, request: pb.ForwardRequest) -> float:
        if self.deadline_s is not None:
            return self.deadline_s
        tokens = max(1, int(request.header.shape[1]) if len(request.header.shape) >= 2 else 1)
        per_hop = (
            BASE_TIMEOUT_S
            + PER_TOKEN_S * tokens
            + tokens * DEADLINE_HIDDEN * 4 / WORST_BANDWIDTH_BPS
        )
        return per_hop * len(self.route)

    def __call__(self, request: pb.ForwardRequest) -> pb.ForwardReply:
        rid = request.header.request_id
        seq = next(self._seqs)
        peers = {h.peer for h in self.route}
        waiter = self.deliveries.expect(rid, seq, peers)
        header = {
            "rid": rid,
            "seq": seq,
            "hop": 0,
            "route": [[h.peer, h.stage] for h in self.route],
            "reply_to": self.node.peer_id,
            "trace": [],
        }
        try:
            started = time.perf_counter_ns()
            try:
                self.node.call(
                    self.route[0].peer,
                    RELAY,
                    header,
                    request.SerializeToString(),
                    timeout=hop_timeout(request),
                )
            except RpcError as exc:
                raise _HopFailed(0, exc) from exc
            self.first_ns = time.perf_counter_ns() - started
            if not waiter.done.wait(self.step_deadline(request)):
                raise self._blame_silence()
            delivery = waiter.delivery
            assert delivery is not None
        finally:
            self.deliveries.forget(rid, seq)
        self.trace = _clean_trace(delivery.trace, len(self.route))
        if not delivery.ok:
            hop = delivery.failed_hop if 0 <= delivery.failed_hop < len(self.route) else 0
            raise _HopFailed(hop, RpcError(delivery.code or "internal", delivery.message))
        reply = pb.ForwardReply()
        try:
            reply.ParseFromString(delivery.body)
        except Exception as exc:  # noqa: BLE001 - protobuf raises DecodeError and others
            raise _HopFailed(
                len(self.route) - 1, RpcError("internal", f"bad delivery: {exc}")
            ) from exc
        return reply

    def _blame_silence(self) -> _HopFailed:
        """The step is late: name the first device on the route that does not answer."""
        for i, hop in enumerate(self.route):
            if hop.peer == self.node.peer_id:
                continue
            try:
                self.node.call(hop.peer, "node.ping", {}, timeout=PROBE_TIMEOUT_S)
            except RpcError:
                return _HopFailed(
                    i,
                    RpcError(
                        "unavailable", f"{hop.peer[:8]} (stage {hop.stage}) stopped answering"
                    ),
                )
        last = self.route[-1]
        return _HopFailed(
            len(self.route) - 1,
            RpcError("deadline", f"no answer from the route; last stage {last.peer[:8]}"),
        )


def _clean_trace(raw: list, hops: int) -> list[dict]:
    """Keep only well-formed trace entries; the trace is the route's word, not ours."""
    out = []
    for item in raw[:hops]:
        if not isinstance(item, dict) or set(item) != _TRACE_KEYS:
            break
        if not isinstance(item["peer"], str) or not isinstance(item["codec"], str):
            break
        ints = ("stage", "compute_ns", "link_ns", "in_bytes", "in_raw", "in_numel")
        if not all(isinstance(item[k], int) and item[k] >= 0 for k in ints):
            break
        out.append(item)
    return out


class RelayChain:
    """A chain over ``route`` that forwards stage to stage; same API as ``ChainRunner``.

    Args:
        route: ``(peer id, stage, address)`` for each stage, in order. ``address`` is
            what failures name, as with requester routing.
        deadline_s: Fixed per-step deadline; ``None`` sizes one from what is sent.
    """

    def __init__(
        self,
        node: Node,
        route: Sequence[tuple[str, int, str]],
        codec: Codec,
        *,
        codecs: CodecSet | None = None,
        deadline_s: float | None = None,
    ) -> None:
        if not route:
            raise ValueError("a route needs at least one stage")
        self.addresses = [address for _, _, address in route]
        self.sender = _RelaySender(
            node, [Hop(p, s) for p, s, _ in route], deliveries_for(node), deadline_s
        )
        self.client = ShardClient(self.addresses[0], codec, sender=self.sender, codecs=codecs)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        phase: str = "prefill",
        logits_keep_last: int = 0,
        request_id: str | None = None,
        use_cache: bool = False,
        end_of_request: bool = False,
        expected_cache_position: int | None = None,
        sampling: Sampling | None = None,
    ) -> ChainResult:
        """One step along the route; see :meth:`ChainRunner.forward`. ``phase`` is
        accepted for the same signature; nothing here profiles by it."""
        del phase
        try:
            result = self.client.forward(
                input_ids,
                request_id=request_id,
                hop=0,
                is_token_ids=True,
                position_ids=position_ids,
                logits_keep_last=logits_keep_last,
                use_cache=use_cache,
                end_of_request=end_of_request,
                expected_cache_position=expected_cache_position,
                sampling=sampling,
            )
        except _HopFailed as exc:
            raise HopError(
                exc.hop, self.addresses[exc.hop], OverlayRpcError(exc.error)
            ) from exc.error
        if not result.is_final:
            raise RuntimeError("the route ended without reaching the LM head")
        hops = self._hops(result)
        rid = result.reply_message.header.request_id if result.reply_message else ""
        if sampling is not None:
            return ChainResult(logits=None, token_ids=result.tensor, hops=hops, request_id=rid)
        return ChainResult(logits=result.tensor, hops=hops, request_id=rid)

    def _hops(self, final: HopResult) -> list[HopResult]:
        """Per-stage measurements from the route's trace (hop 0 timed here)."""
        trace = self.sender.trace
        out = []
        for i, entry in enumerate(trace):
            link_ns = self.sender.first_ns if i == 0 else entry["link_ns"]
            last = i == len(trace) - 1
            out.append(
                HopResult(
                    tensor=final.tensor if last else torch.empty(0),
                    is_final=last,
                    hop=i,
                    sent_bytes=entry["in_bytes"],
                    received_bytes=(final.received_bytes if last else trace[i + 1]["in_bytes"]),
                    sent_numel=entry["in_numel"],
                    sent_uncompressed_bytes=entry["in_raw"],
                    wall_ns=entry["compute_ns"] + link_ns,
                    compute_ns=entry["compute_ns"],
                    codec=str(entry["codec"]),
                )
            )
        return out

    def close(self) -> None:
        self.client.close()
