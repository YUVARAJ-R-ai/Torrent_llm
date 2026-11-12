"""Tests for AutoNat reachability checks and dialback service."""

from __future__ import annotations

import time

import pytest

from torrent_llm.net.autonat import (
    MAX_DIALBACK_ADDRS,
    PRIVATE,
    PUBLIC,
    UNKNOWN,
    AutoNat,
)
from torrent_llm.net.identity import Identity
from torrent_llm.net.node import Node, NodeConfig
from torrent_llm.net.rpc import RpcError


def _make_node(identity: Identity | None = None, **kwargs) -> Node:
    opts = {
        "listen_host": "127.0.0.1",
        "ping_interval": 0.2,
        "dead_after": 0.8,
    }
    opts.update(kwargs)
    cfg = NodeConfig(
        identity=identity or Identity.generate(),
        **opts,
    )
    return Node(cfg)


def test_autonat_reachability_confirmed_public(monkeypatch):
    """Test 1: A connects to B and C; A's address confirmed, status public, record attr public."""
    monkeypatch.setattr("torrent_llm.net.autonat.allow_private_candidates", True)
    a = _make_node()
    b = _make_node()
    c = _make_node()
    a_nat = AutoNat(a)
    b_nat = AutoNat(b)
    c_nat = AutoNat(c)
    a_nat.attach()
    b_nat.attach()
    c_nat.attach()
    try:
        a.start()
        b.start()
        c.start()
        a.connect(f"tcp://127.0.0.1:{b.listen_port}")
        a.connect(f"tcp://127.0.0.1:{c.listen_port}")

        reachable = a_nat.check([b.peer_id, c.peer_id])
        expected_addr = f"tcp://127.0.0.1:{a.listen_port}"
        assert reachable == [expected_addr]
        assert a_nat.status == PUBLIC
        assert a_nat.reachable == [expected_addr]
        assert a.record().attrs.get("public") is True
        assert expected_addr in a.record().addresses
    finally:
        a.close()
        b.close()
        c.close()


def test_autonat_behind_nat_private(monkeypatch):
    """Test 2: A behind NAT: stopping listener results in status private and attr public False."""
    monkeypatch.setattr("torrent_llm.net.autonat.allow_private_candidates", True)

    a = _make_node()
    b = _make_node()
    c = _make_node()
    a_nat = AutoNat(a)
    b_nat = AutoNat(b)
    c_nat = AutoNat(c)
    a_nat.attach()
    b_nat.attach()
    c_nat.attach()
    try:
        a.start()
        b.start()
        c.start()
        a.connect(f"tcp://127.0.0.1:{b.listen_port}")
        a.connect(f"tcp://127.0.0.1:{c.listen_port}")

        # Initially confirm reachability when public
        initial_reachable = a_nat.check([b.peer_id, c.peer_id])
        expected_addr = f"tcp://127.0.0.1:{a.listen_port}"
        assert initial_reachable == [expected_addr]
        assert a_nat.status == PUBLIC
        assert expected_addr in a.record().addresses

        # Stop A's listener from accepting inbound connections
        assert a._server is not None
        a._server.close()
        a.run(a._server.wait_closed())

        reachable = a_nat.check([b.peer_id, c.peer_id])
        assert reachable == []
        assert a_nat.status == PRIVATE
        assert a_nat.reachable == []
        assert a.record().attrs.get("public") is False
        assert expected_addr not in getattr(a, "_advertised", [])
    finally:
        a.close()
        b.close()
        c.close()


def test_helper_refuses_different_ip():
    """Test 3: helper refuses to dial an address whose IP differs from the caller's."""
    b = _make_node()
    c = _make_node()
    c_nat = AutoNat(c)
    c_nat.attach()
    try:
        b.start()
        c.start()
        b.connect(f"tcp://127.0.0.1:{c.listen_port}")

        t0 = time.perf_counter()
        res_h, res_b = b.call(
            c.peer_id,
            "autonat.dialback",
            {"addrs": ["tcp://10.9.8.7:1"]},
            timeout=2.0,
        )
        elapsed = time.perf_counter() - t0

        assert res_h.get("reachable") == []
        assert res_b == b""
        assert elapsed < 0.5
    finally:
        b.close()
        c.close()


def test_no_helpers_answer_unknown(monkeypatch):
    """Test 4: no helpers answer (helpers are unknown peer ids) -> unknown."""
    monkeypatch.setattr("torrent_llm.net.autonat.allow_private_candidates", True)
    a = _make_node()
    a_nat = AutoNat(a)
    a_nat.attach()
    try:
        a.start()
        unknown_1 = Identity.generate().peer_id
        unknown_2 = Identity.generate().peer_id

        reachable = a_nat.check([unknown_1, unknown_2])
        assert reachable == []
        assert a_nat.status == UNKNOWN
        assert a_nat.reachable == []
        assert "public" not in a.record().attrs
    finally:
        a.close()


def test_loopback_candidates_dropped_by_default(monkeypatch):
    """Test 5: by default (allow_private_candidates=False) loopback candidates are dropped."""
    import torrent_llm.net.autonat as autonat_mod

    assert not autonat_mod.allow_private_candidates
    a = _make_node()
    a_nat = AutoNat(a)
    try:
        a.start()
        # Default behavior: loopback and private addresses are dropped
        cands = a_nat.candidates(
            extra=("tcp://127.0.0.1:9000", "tcp://10.0.0.1:9000", "tcp://192.168.1.1:9000")
        )
        assert cands == []

        # Public candidates are kept
        cands_pub = a_nat.candidates(extra=("tcp://93.184.216.34:9000",))
        assert cands_pub == ["tcp://93.184.216.34:9000"]

        # When hook is True, loopback candidates are kept
        monkeypatch.setattr("torrent_llm.net.autonat.allow_private_candidates", True)
        cands_priv = a_nat.candidates(extra=("tcp://127.0.0.1:9000",))
        assert cands_priv == ["tcp://127.0.0.1:9000"]
    finally:
        a.close()


def test_autonat_serve_false_does_not_attach():
    """AutoNat with serve=False does not register autonat.dialback."""
    a = _make_node()
    a_nat = AutoNat(a, serve=False)
    a_nat.attach()
    try:
        a.start()
        with pytest.raises(RpcError) as exc_info:
            a.call(a.peer_id, "autonat.dialback", {"addrs": []}, timeout=1.0)
        assert exc_info.value.code == "not_found"
    finally:
        a.close()


def test_autonat_caps_dialback_addresses(monkeypatch):
    """Helper dials at most MAX_DIALBACK_ADDRS addresses and rejects malformed inputs."""
    b = _make_node()
    c = _make_node()
    c_nat = AutoNat(c)
    c_nat.attach()
    try:
        b.start()
        c.start()
        b.connect(f"tcp://127.0.0.1:{c.listen_port}")

        dials: list[tuple[str, int]] = []

        async def mock_open_conn(host, port, *args, **kwargs):
            dials.append((host, port))
            raise OSError("mock connection failure")

        monkeypatch.setattr("torrent_llm.net.autonat.asyncio.open_connection", mock_open_conn)

        # Send MAX_DIALBACK_ADDRS + 3 distinct ports
        addrs = [f"tcp://127.0.0.1:{12000 + i}" for i in range(MAX_DIALBACK_ADDRS + 3)]
        res_h, _ = b.call(
            c.peer_id,
            "autonat.dialback",
            {"addrs": addrs},
            timeout=2.0,
        )
        assert res_h.get("reachable") == []
        assert len(dials) == MAX_DIALBACK_ADDRS

        # Malformed input: addrs is not a list
        dials.clear()
        with pytest.raises(RpcError) as caught:
            b.call(c.peer_id, "autonat.dialback", {"addrs": "x"}, timeout=2.0)
        assert caught.value.code == "bad_request"
        assert len(dials) == 0

        # Malformed list items: non-str, relay://, unparsable garbage
        dials.clear()
        res_items, _ = b.call(
            c.peer_id,
            "autonat.dialback",
            {"addrs": [1, None, "relay://abc", "garbage"]},
            timeout=2.0,
        )
        assert res_items.get("reachable") == []
        assert len(dials) == 0
    finally:
        b.close()
        c.close()


def test_helper_refuses_dns_and_non_ip(monkeypatch):
    """Helper refuses to dial DNS hostnames and rejects non-IP caller remotes."""
    b = _make_node()
    c = _make_node()
    c_nat = AutoNat(c)
    c_nat.attach()
    try:
        b.start()
        c.start()
        b.connect(f"tcp://127.0.0.1:{c.listen_port}")

        dials: list[tuple[str, int]] = []

        async def mock_open_conn(host, port, *args, **kwargs):
            dials.append((host, port))
            raise OSError("should not dial")

        monkeypatch.setattr("torrent_llm.net.autonat.asyncio.open_connection", mock_open_conn)

        # Candidate with hostname instead of IP must not be dialed
        res_h, _ = b.call(
            c.peer_id,
            "autonat.dialback",
            {"addrs": ["tcp://localhost:22", "tcp://example.org:80"]},
            timeout=2.0,
        )
        assert res_h.get("reachable") == []
        assert len(dials) == 0
    finally:
        b.close()
        c.close()


def test_helper_refuses_relayed_connection(monkeypatch):
    """Helper refuses to dial back when the caller connection is via relay."""
    b = _make_node()
    c = _make_node()
    c_nat = AutoNat(c)
    c_nat.attach()
    try:
        b.start()
        c.start()
        b.connect(f"tcp://127.0.0.1:{c.listen_port}")

        from torrent_llm.net.node import ConnectionInfo

        orig_connections = c.connections

        def mock_connections() -> list[ConnectionInfo]:
            conns = orig_connections()
            return [
                ConnectionInfo(
                    peer_id=ci.peer_id,
                    remote=ci.remote,
                    inbound=ci.inbound,
                    rtt_ms=ci.rtt_ms,
                    bytes_sent=ci.bytes_sent,
                    bytes_received=ci.bytes_received,
                    via_relay="relay_peer_id",
                )
                for ci in conns
            ]

        monkeypatch.setattr(c, "connections", mock_connections)

        with pytest.raises(RpcError) as caught:
            b.call(
                c.peer_id,
                "autonat.dialback",
                {"addrs": [f"tcp://127.0.0.1:{b.listen_port}"]},
                timeout=2.0,
            )
        assert caught.value.code == "denied"
    finally:
        b.close()
        c.close()


def test_helper_inflight_per_caller_ip(monkeypatch):
    """Helper refuses concurrent dialbacks for the same caller IP."""
    import asyncio
    import threading

    b = _make_node()
    c = _make_node()
    c_nat = AutoNat(c)
    c_nat.attach()
    try:
        b.start()
        c.start()
        b.connect(f"tcp://127.0.0.1:{c.listen_port}")

        dial_entered = threading.Event()
        release_dial = threading.Event()

        async def slow_open(host, port, *args, **kwargs):
            dial_entered.set()
            while not release_dial.is_set():
                await asyncio.sleep(0.01)
            raise OSError("cancelled")

        monkeypatch.setattr("torrent_llm.net.autonat.asyncio.open_connection", slow_open)

        async def _run_concurrent():
            t1 = asyncio.create_task(
                b.acall(
                    c.peer_id,
                    "autonat.dialback",
                    {"addrs": [f"tcp://127.0.0.1:{b.listen_port}"]},
                    timeout=2.0,
                )
            )
            while not dial_entered.is_set():
                await asyncio.sleep(0.01)

            # Second call while first is still in flight on c
            t2 = asyncio.create_task(
                b.acall(
                    c.peer_id,
                    "autonat.dialback",
                    {"addrs": [f"tcp://127.0.0.1:{b.listen_port}"]},
                    timeout=2.0,
                )
            )
            try:
                await t2
                second = None
            except RpcError as exc:
                second = exc.code
            release_dial.set()
            await t1
            return second

        # Refused as busy, not answered with [] (which would mean "you are private").
        assert b.run(_run_concurrent(), timeout=4.0) == "busy"
    finally:
        b.close()
        c.close()


def test_acheck_async_via_asyncio_run(monkeypatch):
    """Test acheck running asynchronously on the node loop."""
    monkeypatch.setattr("torrent_llm.net.autonat.allow_private_candidates", True)
    a = _make_node()
    b = _make_node()
    a_nat = AutoNat(a)
    b_nat = AutoNat(b)
    a_nat.attach()
    b_nat.attach()
    try:
        a.start()
        b.start()
        a.connect(f"tcp://127.0.0.1:{b.listen_port}")

        async def _test():
            return await a_nat.acheck([b.peer_id])

        reachable = a.run(_test(), timeout=5.0)
        assert reachable == [f"tcp://127.0.0.1:{a.listen_port}"]
        assert a_nat.status == PUBLIC
    finally:
        a.close()
        b.close()


def test_a_busy_helper_does_not_make_the_caller_private(monkeypatch):
    """Helpers that refuse (busy, relayed) are not answers: the status stays unknown."""
    b = _make_node()
    c = _make_node()
    nat = AutoNat(b)

    async def refusing(peer_id, method, *args, **kwargs):
        raise RpcError("busy", "a check for this address is already running")

    try:
        b.start()
        c.start()
        monkeypatch.setattr(b, "acall", refusing)
        assert nat.check([c.peer_id], extra=("tcp://203.0.113.5:4000",)) == []
        assert nat.status == UNKNOWN
    finally:
        b.close()
        c.close()


def test_no_candidates_means_unknown_without_asking(monkeypatch):
    b = _make_node()
    nat = AutoNat(b)
    asked: list[str] = []

    async def record(peer_id, method, *args, **kwargs):
        asked.append(method)
        return {"reachable": []}, b""

    try:
        b.start()
        monkeypatch.setattr(b, "acall", record)
        monkeypatch.setattr(nat, "candidates", lambda extra=(): [])
        assert nat.check(["x" * 32]) == []
        assert nat.status == UNKNOWN
        assert asked == []
    finally:
        b.close()


def test_simultaneous_dial_backs_are_capped(monkeypatch):
    """No more than MAX_CONCURRENT_DIALS sockets are opened at once by one helper."""
    import asyncio

    import torrent_llm.net.autonat as autonat_mod

    monkeypatch.setattr(autonat_mod, "MAX_CONCURRENT_DIALS", 2)
    b = _make_node()
    c = _make_node()
    AutoNat(c).attach()
    live = [0]
    peak = [0]

    async def slow_open(host, port, *args, **kwargs):
        live[0] += 1
        peak[0] = max(peak[0], live[0])
        await asyncio.sleep(0.2)
        live[0] -= 1
        raise OSError("nobody home")

    try:
        b.start()
        c.start()
        b.connect(f"tcp://127.0.0.1:{c.listen_port}")
        monkeypatch.setattr("torrent_llm.net.autonat.asyncio.open_connection", slow_open)
        addrs = [f"tcp://127.0.0.1:{40000 + i}" for i in range(4)]
        header, _ = b.call(c.peer_id, "autonat.dialback", {"addrs": addrs}, timeout=5.0)
        assert header == {"reachable": []}
        assert peak[0] == 2
    finally:
        b.close()
        c.close()
