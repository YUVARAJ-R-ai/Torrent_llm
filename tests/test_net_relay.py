"""Reaching a device nobody can connect to, through a relay."""

from __future__ import annotations

import asyncio
import os
import threading
import time

import pytest

from torrent_llm.net.identity import Identity
from torrent_llm.net.node import Node, NodeConfig
from torrent_llm.net.relay import (
    CIRCUIT_CLOSE,
    CIRCUIT_DATA,
    WINDOW_BYTES,
    Relay,
    RelayLimits,
)
from torrent_llm.net.rpc import RpcError

# Long liveness timeouts: every close in these tests must come from the relay itself.
LIVENESS = {"ping_interval": 5.0, "dead_after": 30.0}


def _node(*, reachable: bool = True) -> Node:
    node = Node(
        NodeConfig(
            identity=Identity.generate(),
            listen_host="127.0.0.1",
            advertise_local=reachable,
            **LIVENESS,
        )
    )
    node.register("echo", lambda ctx, h, body: (h, body))
    node.start()
    return node


def _wait(condition, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


class Net:
    """A relay R, targets that reach the world only through it, and a dialer D."""

    def __init__(self, limits: RelayLimits | None = None, targets: int = 1) -> None:
        self.nodes: list[Node] = []
        self.relays: list[Relay] = []
        self.r = self.add(serve=True, limits=limits)
        self.d = self.add()
        self.targets = [self.add(reachable=False) for _ in range(targets)]
        for t in self.targets:
            t.connect(f"tcp://127.0.0.1:{self.r.listen_port}")
            assert self.relay_of(t).reserve_on([self.r.peer_id]) == [self.r.peer_id]
            self.d.add_record(t.record())
        self.d.add_record(self.r.record())
        self.t = self.targets[0]

    def add(self, *, reachable: bool = True, serve: bool = False, limits=None) -> Node:
        node = _node(reachable=reachable)
        relay = Relay(node, serve=serve, limits=limits)
        relay.attach()
        self.nodes.append(node)
        self.relays.append(relay)
        return node

    def relay_of(self, node: Node) -> Relay:
        return self.relays[self.nodes.index(node)]

    def close(self) -> None:
        for relay in self.relays:
            relay.detach()
        for node in self.nodes:
            node.close()


@pytest.fixture
def net():
    made: list[Net] = []

    def make(**kw) -> Net:
        made.append(Net(**kw))
        return made[-1]

    yield make
    for n in made:
        n.close()


def test_a_target_with_no_address_is_reached_through_the_relay(net):
    n = net()
    assert list(n.t.record().addresses) == [f"relay://{n.r.peer_id}"]
    assert n.d.call(n.t.peer_id, "echo", {"x": 1}, b"hi", timeout=10) == ({"x": 1}, b"hi")
    info = next(c for c in n.d.connections() if c.peer_id == n.t.peer_id)
    assert info.via_relay == n.r.peer_id
    stats = n.relay_of(n.r).stats()
    assert stats["circuits"] == 1 and stats["reservations"] == 1
    assert stats["bytes_relayed"] > 0


def test_a_large_body_crosses_intact_and_the_target_can_call_back(net):
    n = net()
    blob = os.urandom(12 * 1024 * 1024)  # three windows' worth: acks must flow
    _, body = n.d.call(n.t.peer_id, "echo", {}, blob, timeout=60)
    assert body == blob
    assert n.t.call(n.d.peer_id, "echo", {"back": True}, timeout=10)[0] == {"back": True}


def test_the_relay_never_sees_plaintext(net):
    n = net()
    secret = b"the-plaintext-secret-" * 1000
    seen: list[bytes] = []
    relay = n.relay_of(n.r)
    real_forward = relay._forward

    async def spying(circuit, side, kind, header, body):
        seen.append(body)
        await real_forward(circuit, side, kind, header, body)

    relay._forward = spying
    n.d.call(n.t.peer_id, "echo", {}, secret, timeout=10)
    assert seen and not any(b"the-plaintext-secret-" in chunk for chunk in seen)


def test_refusals_name_the_problem(net):
    n = net()
    n.relay_of(n.r).set_serving(False)
    with pytest.raises(RpcError) as caught:
        n.d.call(n.t.peer_id, "echo", timeout=5)
    assert caught.value.code == "unavailable"
    assert n.t.peer_id[:8] in str(caught.value)

    m = net()
    stranger = m.add(reachable=False)  # no reservation anywhere
    m.d.add_record(stranger.record())
    with pytest.raises(RpcError):
        m.d.call(stranger.peer_id, "echo", timeout=5)


def test_losing_the_relay_fails_calls_fast_and_drops_the_reservation(net):
    n = net()
    n.d.call(n.t.peer_id, "echo", timeout=10)
    started = threading.Event()
    n.t.register("slow", lambda ctx, h, b: (started.set(), time.sleep(5), ({}, b""))[2])
    result: dict = {}

    def call() -> None:
        try:
            n.d.call(n.t.peer_id, "slow", timeout=20)
        except RpcError as exc:
            result["error"] = exc

    caller = threading.Thread(target=call)
    caller.start()
    assert started.wait(5)
    t0 = time.monotonic()
    n.r.close()
    caller.join(10)
    assert result["error"].code == "unavailable"
    assert time.monotonic() - t0 < 3
    t_relay = n.relay_of(n.t)
    assert _wait(lambda: t_relay.reservations() == [], 2)
    assert _wait(lambda: list(n.t.record().addresses) == [], 2)


def test_a_closing_target_closes_the_dialer_side_at_once(net):
    n = net()
    n.d.call(n.t.peer_id, "echo", timeout=10)
    n.t.close()
    # Liveness would take 30 s; the relay's CIRCUIT_CLOSE must arrive in about one RTT.
    assert _wait(lambda: not n.d.is_connected(n.t.peer_id), 1.5)
    assert _wait(lambda: n.relay_of(n.r).stats()["circuits"] == 0, 1.5)


def test_per_peer_and_total_limits_hold_under_concurrency(net):
    n = net(limits=RelayLimits(max_circuits=2, max_circuits_per_peer=5), targets=4)
    dialer_relay = n.relay_of(n.d)

    async def connect_all():
        conn = n.d.connection(n.r.peer_id) or None
        if conn is None:
            await n.d.acall(n.r.peer_id, "node.ping", {}, timeout=5)
            conn = n.d.connection(n.r.peer_id)
        calls = [conn.call("relay.connect", {"target": t.peer_id}, timeout=5) for t in n.targets]
        return await asyncio.gather(*calls, return_exceptions=True)

    results = n.d.run(connect_all(), timeout=20)
    ok = [r for r in results if not isinstance(r, BaseException)]
    refused = [r for r in results if isinstance(r, RpcError)]
    assert len(ok) == 2
    assert len(refused) == 2 and all(r.code == "busy" for r in refused)
    assert dialer_relay is not None

    m = net(limits=RelayLimits(max_circuits_per_peer=1), targets=2)
    m.d.call(m.targets[0].peer_id, "echo", timeout=10)
    with pytest.raises(RpcError):
        m.d.call(m.targets[1].peer_id, "echo", timeout=5)


def test_an_idle_circuit_is_closed_on_both_sides(net):
    n = net(limits=RelayLimits(idle_timeout_s=0.3))
    relay = n.relay_of(n.r)
    n.d.call(n.t.peer_id, "echo", timeout=10)
    # The relayed connection's own keepalive is slow (5 s), so the circuit goes idle.
    assert _wait(lambda: relay.stats()["circuits"] == 0, 3)
    assert _wait(lambda: not n.d.is_connected(n.t.peer_id), 2)


def test_two_targets_on_one_relay_are_both_reachable(net):
    n = net(targets=2)
    for t in n.targets:
        assert n.d.call(t.peer_id, "echo", {"to": t.peer_id}, timeout=10)[0] == {"to": t.peer_id}
    assert n.relay_of(n.r).stats()["circuits"] == 2


def test_two_relays_dialing_through_each_other_never_mix_up_circuits(net):
    """R1 and R2 share one connection and each relays over it: ids must not collide."""
    n = net()
    r1, r2 = n.r, n.add(serve=True)
    t2 = n.add(reachable=False)
    t2.connect(f"tcp://127.0.0.1:{r2.listen_port}")
    assert n.relay_of(t2).reserve_on([r2.peer_id]) == [r2.peer_id]
    r1.connect(f"tcp://127.0.0.1:{r2.listen_port}")
    r2.add_record(n.t.record())
    r2.add_record(r1.record())
    r1.add_record(t2.record())
    # R2 reaches T (behind R1) and R1 reaches T2 (behind R2), over the same R1-R2 link.
    assert r2.call(n.t.peer_id, "echo", {"a": 1}, timeout=10)[0] == {"a": 1}
    assert r1.call(t2.peer_id, "echo", {"b": 2}, timeout=10)[0] == {"b": 2}
    for _ in range(3):
        assert r2.call(n.t.peer_id, "echo", {"a": 1}, timeout=10)[0] == {"a": 1}
        assert r1.call(t2.peer_id, "echo", {"b": 2}, timeout=10)[0] == {"b": 2}


def test_incoming_circuits_are_accepted_only_from_our_relays(net):
    n = net()
    with pytest.raises(RpcError) as caught:
        n.d.call(n.t.peer_id, "relay.incoming", {"from": n.d.peer_id, "circuit": 1}, timeout=10)
    assert caught.value.code == "denied"


def test_the_relay_closes_a_circuit_whose_sender_ignores_the_window(net):
    """A raw dialer floods a circuit to a target that never reads: the RELAY closes it,
    and both ends get CIRCUIT_CLOSE from the relay."""
    n = net()
    # A target whose end never reads or acknowledges: a bare node that reserved on R and
    # accepts circuits without creating a stream.
    sink = _node(reachable=False)
    n.nodes.append(sink)
    closes: dict[str, list] = {"sink": [], "dialer": []}

    async def sink_frames(conn, kind, ident, header, body):
        if kind == CIRCUIT_CLOSE:
            closes["sink"].append(header)

    sink.set_extension_handler(sink_frames)
    sink.register("relay.incoming", lambda ctx, h, b: ({}, b""))
    sink.connect(f"tcp://127.0.0.1:{n.r.listen_port}")
    sink.call(n.r.peer_id, "relay.reserve", {}, timeout=5)

    raw = _node()
    n.nodes.append(raw)

    async def raw_frames(conn, kind, ident, header, body):
        if kind == CIRCUIT_CLOSE:
            closes["dialer"].append(header)

    raw.set_extension_handler(raw_frames)
    raw.connect(f"tcp://127.0.0.1:{n.r.listen_port}")
    header, _ = raw.call(n.r.peer_id, "relay.connect", {"target": sink.peer_id}, timeout=5)
    cid = header["circuit"]

    async def flood():
        conn = raw.connection(n.r.peer_id)
        chunk = b"x" * (256 * 1024)
        for _ in range(WINDOW_BYTES // len(chunk) + 2):
            await conn.send_extension(CIRCUIT_DATA, cid, {}, chunk)

    raw.run(flood(), timeout=20)
    assert _wait(lambda: closes["sink"] and closes["dialer"], 3)
    assert n.relay_of(n.r).stats()["circuits"] == 0
    assert "flow control" in closes["sink"][0]["reason"]


def test_reservations_are_renewed(net):
    n = net()
    t_relay = n.relay_of(n.t)
    renewals: list[str] = []
    real = n.relay_of(n.r)._on_reserve

    async def counting(ctx, header, body):
        renewals.append(ctx.peer_id)
        return await real(ctx, header, body)

    n.r.unregister("relay.reserve")
    n.r.register("relay.reserve", counting, blocking=False)
    t_relay._renew_s = 0.1
    n.t.run(_kick_renewal(t_relay), timeout=5)
    assert _wait(lambda: len(renewals) >= 2, 3)
    assert t_relay.reservations() == [n.r.peer_id]


async def _kick_renewal(relay: Relay) -> None:
    """Restart the renewal loop so a shortened interval takes effect."""
    for task in list(relay._tasks):
        if task.get_coro().__name__ == "_renew":
            task.cancel()
    await asyncio.sleep(0)
    relay._renewing = False
    await relay._reserve([])


def test_attach_after_start_and_detach_from_another_thread_are_safe():
    node = _node()
    relay = Relay(node, serve=True)
    relay.attach()
    errors: list[BaseException] = []

    def detach() -> None:
        try:
            relay.detach()
            relay.detach()  # idempotent
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    t = threading.Thread(target=detach)
    t.start()
    t.join(5)
    node.close()
    assert not errors
