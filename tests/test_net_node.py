"""Tests for the Node class in torrent_llm.net.node."""

from __future__ import annotations

import asyncio
import dataclasses
import socket
import threading
import time

import pytest

from torrent_llm.net.channel import HandshakeError
from torrent_llm.net.identity import Identity
from torrent_llm.net.node import Node, NodeConfig
from torrent_llm.net.ratelimit import Limit
from torrent_llm.net.records import make_record
from torrent_llm.net.rpc import RpcError
from torrent_llm.net.stream import TcpStream


def _make_node(identity: Identity | None = None, **kwargs) -> Node:
    """Create a Node instance with default test configuration."""
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


def test_two_nodes_connect_bidirectional_call_hello():
    """Verify connection, bi-directional RPC calls, hello exchange and observed address."""
    a = _make_node()
    b = _make_node()
    try:
        a.start()
        b.start()

        def echo_b(ctx, header, body):
            """Echo back header and body on node B."""
            return header, body

        def echo_a(ctx, header, body):
            """Echo back with custom header on node A."""
            return {"echo_from": "a"}, body

        b.register("echo", echo_b)
        a.register("echo_a", echo_a)

        peer_b = a.connect(f"tcp://127.0.0.1:{b.listen_port}")
        assert peer_b == b.peer_id

        res_h, res_b = a.call(b.peer_id, "echo", {"msg": "hi"}, b"payload", timeout=2.0)
        assert res_h == {"msg": "hi"}
        assert res_b == b"payload"

        res_h2, res_b2 = b.call(a.peer_id, "echo_a", {}, b"from_b", timeout=2.0)
        assert res_h2 == {"echo_from": "a"}
        assert res_b2 == b"from_b"

        rec_b_on_a = a.record_of(b.peer_id)
        assert rec_b_on_a is not None and rec_b_on_a.verify()

        rec_a_on_b = b.record_of(a.peer_id)
        assert rec_a_on_b is not None and rec_a_on_b.verify()

        observed = a.observed_addresses()
        assert any(addr.startswith("127.0.0.1:") for addr in observed)
    finally:
        a.close()
        b.close()


def test_self_call_in_process():
    """Verify calling self executes in-process without network connection."""
    a = _make_node()
    try:
        a.start()

        def echo(ctx, header, body):
            """Echo back received header and body."""
            return header, body

        a.register("echo", echo)

        res_h, res_b = a.call(a.peer_id, "echo", {"key": "val"}, b"self_data", timeout=2.0)
        assert res_h == {"key": "val"}
        assert res_b == b"self_data"
        assert len(a.connections()) == 0

        with pytest.raises(RpcError) as exc_info:
            a.call(a.peer_id, "nonexistent", timeout=2.0)
        assert exc_info.value.code == "not_found"
    finally:
        a.close()


def test_dial_by_record_without_prior_connect():
    """Verify dialing an unconnected peer using its verified PeerRecord."""
    a = _make_node()
    b = _make_node()
    try:
        a.start()
        b.start()

        def echo(ctx, h, b_data):
            """Echo back."""
            return h, b_data

        b.register("echo", echo)
        assert a.add_record(b.record())

        res_h, res_b = a.call(b.peer_id, "echo", {"k": 1}, b"data", timeout=2.0)
        assert res_h == {"k": 1}
        assert res_b == b"data"
        assert a.is_connected(b.peer_id)
    finally:
        a.close()
        b.close()


def test_concurrent_calls_single_flight_connection():
    """Verify 10 concurrent threads calling an unconnected peer create exactly 1 connection."""
    a = _make_node()
    b = _make_node()
    try:
        a.start()
        b.start()

        def slow_echo(ctx, h, b_data):
            """Slow echo handler."""
            time.sleep(0.05)
            return h, b_data

        b.register("slow_echo", slow_echo)
        assert a.add_record(b.record())

        results = []
        errors = []

        def worker(idx: int) -> None:
            """Call slow_echo from a worker thread."""
            try:
                hdr, _ = a.call(b.peer_id, "slow_echo", {"idx": idx}, timeout=4.0)
                results.append(hdr["idx"])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 10
        assert len(a.connections()) == 1
        assert len(b.connections()) == 1
    finally:
        a.close()
        b.close()


def test_unreachable_record_fails_fast():
    """Verify dialing an unreachable address fails with unavailable within connect_timeout + 1s."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        closed_port = s.getsockname()[1]

    a = _make_node(connect_timeout=0.5)
    try:
        a.start()
        ident_c = Identity.generate()
        rec_c = make_record(ident_c, [f"tcp://127.0.0.1:{closed_port}"], seq=1)
        assert a.add_record(rec_c)

        t0 = time.monotonic()
        with pytest.raises(RpcError) as exc_info:
            a.call(ident_c.peer_id, "echo", timeout=3.0)
        elapsed = time.monotonic() - t0

        assert exc_info.value.code == "unavailable"
        assert elapsed < a.config.connect_timeout + 1.0
    finally:
        a.close()


def test_close_during_inflight_call():
    """Verify B.close() during an in-flight call fails on A with unavailable within ~2s."""
    a = _make_node()
    b = _make_node()
    try:
        a.start()
        b.start()

        started_event = threading.Event()

        def slow_handler(ctx, h, b_data):
            """Slow handler signalling when started."""
            started_event.set()
            time.sleep(5.0)
            return h, b_data

        b.register("slow", slow_handler)
        a.connect(f"tcp://127.0.0.1:{b.listen_port}")

        def closer():
            """Close node B once slow call is in-flight."""
            started_event.wait(timeout=2.0)
            time.sleep(0.05)
            b.close()

        threading.Thread(target=closer, daemon=True).start()

        t0 = time.monotonic()
        with pytest.raises(RpcError) as exc_info:
            a.call(b.peer_id, "slow", timeout=5.0)
        elapsed = time.monotonic() - t0

        assert exc_info.value.code == "unavailable"
        assert elapsed < 2.0
    finally:
        a.close()
        b.close()


def test_silent_peer_fails_within_dead_after():
    """Verify silent peer fails in-flight call with unavailable within dead_after + 1s."""
    a = _make_node(dead_after=0.8, ping_interval=0.2)
    b = _make_node(dead_after=0.8, ping_interval=0.2)
    try:
        a.start()
        b.start()

        started_event = threading.Event()

        def slow_handler(ctx, h, b_data):
            """Slow handler signalling when started."""
            started_event.set()
            time.sleep(5.0)
            return h, b_data

        b.register("slow", slow_handler)
        a.connect(f"tcp://127.0.0.1:{b.listen_port}")

        def silence():
            """Silence node B connection reader and keepalive tasks."""
            started_event.wait(timeout=2.0)
            time.sleep(0.05)
            conn = b.connection(a.peer_id)
            if conn is not None:
                for task in (conn._reader_task, conn._keepalive_task):
                    if task is not None:
                        b.loop.call_soon_threadsafe(task.cancel)

        threading.Thread(target=silence, daemon=True).start()

        t0 = time.monotonic()
        with pytest.raises(RpcError) as exc_info:
            a.call(b.peer_id, "slow", timeout=5.0)
        elapsed = time.monotonic() - t0

        assert exc_info.value.code == "unavailable"
        assert elapsed < a.config.dead_after + 1.0
    finally:
        a.close()
        b.close()


def test_different_psk_refused():
    """Verify nodes with different PSKs fail to connect and leave no connections."""
    a = _make_node(psk=b"\x01" * 32)
    b = _make_node(psk=b"\x02" * 32)
    try:
        a.start()
        b.start()

        with pytest.raises(ConnectionError):
            a.connect(f"tcp://127.0.0.1:{b.listen_port}")

        time.sleep(0.1)
        assert len(a.connections()) == 0
        assert len(b.connections()) == 0
    finally:
        a.close()
        b.close()


def test_min_pow_bits_enforced():
    """Verify min_pow_bits refuses zero-work identities and accepts adequate PoW."""
    b = _make_node(min_pow_bits=8)
    a_zero = _make_node(identity=Identity.generate(0))
    a_pow = _make_node(identity=Identity.generate(8))
    try:
        b.start()
        a_zero.start()
        a_pow.start()

        with pytest.raises(ConnectionError):
            a_zero.connect(f"tcp://127.0.0.1:{b.listen_port}")

        peer_id = a_pow.connect(f"tcp://127.0.0.1:{b.listen_port}")
        assert peer_id == b.peer_id
    finally:
        b.close()
        a_zero.close()
        a_pow.close()


def test_gatekeeper_refusal():
    """Verify a gatekeeper on B refusing A causes connect or call to fail."""
    a = _make_node()
    b = _make_node()
    try:

        def gatekeeper(peer_id: str) -> str | None:
            """Refuse connections from node A."""
            if peer_id == a.peer_id:
                return "banned"
            return None

        b.add_gatekeeper(gatekeeper)
        b.register("echo", lambda ctx, h, b_data: (h, b_data))

        a.start()
        b.start()

        with pytest.raises(ConnectionError):
            a.connect(f"tcp://127.0.0.1:{b.listen_port}")

        assert not a.is_connected(b.peer_id)
        assert not b.is_connected(a.peer_id)

        a.add_record(b.record())
        with pytest.raises(RpcError) as exc_info:
            a.call(b.peer_id, "echo", timeout=2.0)
        assert exc_info.value.code == "unavailable"
    finally:
        a.close()
        b.close()


def test_rate_limiting():
    """Verify rate limit denies third call when burst=2."""
    b = _make_node(limits={"test": Limit(rate=1.0, burst=2.0)})
    a = _make_node()
    try:
        b.register("test.x", lambda ctx, h, b_data: (h, b_data))
        a.start()
        b.start()

        a.connect(f"tcp://127.0.0.1:{b.listen_port}")

        res1, _ = a.call(b.peer_id, "test.x", timeout=1.0)
        assert res1 == {}
        res2, _ = a.call(b.peer_id, "test.x", timeout=1.0)
        assert res2 == {}

        with pytest.raises(RpcError) as exc_info:
            a.call(b.peer_id, "test.x", timeout=1.0)
        assert exc_info.value.code == "busy"
    finally:
        a.close()
        b.close()


def test_thread_execution_and_deadlock_prevention():
    """Verify blocking handlers run on thread pool and async handlers on loop thread."""
    b = _make_node()
    a = _make_node()
    try:
        blocking_threads = []
        async_threads = []
        deadlock_raised = []

        def blocking_fn(ctx, h, b_data):
            """Handler running in thread pool."""
            blocking_threads.append(threading.current_thread().name)
            return h, b_data

        async def async_fn(ctx, h, b_data):
            """Handler running on event loop thread."""
            async_threads.append(threading.current_thread().name)
            try:
                b.call(b.peer_id, "test_b", timeout=1.0)
            except RuntimeError as exc:
                if "would deadlock" in str(exc):
                    deadlock_raised.append(True)
            return h, b_data

        b.register("test_b", blocking_fn, blocking=True)
        b.register("test_a", async_fn, blocking=False)

        a.start()
        b.start()

        a.connect(f"tcp://127.0.0.1:{b.listen_port}")
        a.call(b.peer_id, "test_b", timeout=2.0)
        assert blocking_threads
        assert blocking_threads[0].startswith("tllm-handler")

        a.call(b.peer_id, "test_a", timeout=2.0)
        assert async_threads
        assert async_threads[0].startswith("tllm-node")
        assert deadlock_raised == [True]
    finally:
        a.close()
        b.close()


def test_register_validation():
    """Verify registering a duplicate method or invalid method name raises ValueError."""
    a = _make_node()
    a.register("my_method", lambda ctx, h, b: (h, b))

    with pytest.raises(ValueError):
        a.register("my_method", lambda ctx, h, b: (h, b))

    with pytest.raises(ValueError):
        a.register("Invalid-Method-Name!", lambda ctx, h, b: (h, b))


def test_record_seq_and_add_record_validation():
    """Verify record updates seq on set_attr and add_record validates signature and seq."""
    a = _make_node()
    r1 = a.record()
    assert r1.verify()

    a.set_attr("device", "cuda")
    r2 = a.record()
    assert r2.verify()
    assert r2.seq > r1.seq
    assert r2.attrs["device"] == "cuda"

    other = _make_node()
    rec_old = other.record()
    other.set_attr("foo", "bar")
    rec_new = other.record()

    assert a.add_record(rec_new)
    assert not a.add_record(rec_old)

    tampered = dataclasses.replace(rec_new, attrs={"evil": True})
    assert not a.add_record(tampered)


def test_close_idempotent_and_port_collision():
    """Verify close() is idempotent and starting on a used port raises OSError."""
    a = _make_node()
    a.start()
    a.close()
    a.close()

    b = _make_node()
    b.start()
    try:
        c = _make_node(listen_port=b.listen_port)
        with pytest.raises(OSError):
            c.start()
    finally:
        b.close()


def test_empty_addresses_when_advertise_local_false():
    """Verify a node with advertise_local=False and no advertise has an empty address list."""
    a = _make_node(advertise_local=False, advertise=())
    assert a.addresses() == []


def test_cancel_leader_single_flight_dial_silent_port():
    """Cancelling the leader of a single-flight dial still releases the follower when the
    shared dial fails (here: the handshake to a silent port times out), well before the
    follower's own budget."""
    a = _make_node(handshake_timeout=0.5, connect_timeout=0.5)
    a.start()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    port = srv.getsockname()[1]

    target_id = Identity.generate()
    rec = make_record(target_id, addresses=[f"tcp://127.0.0.1:{port}"], seq=1)
    a.add_record(rec)

    async def _run() -> None:
        """Run leader and follower coroutines concurrently."""
        loop = asyncio.get_running_loop()
        t1 = loop.create_task(a.acall(target_id.peer_id, "echo", timeout=5.0))
        t2 = loop.create_task(a.acall(target_id.peer_id, "echo", timeout=10.0))

        await asyncio.sleep(0.05)
        t1.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t1

        t0 = loop.time()
        with pytest.raises(RpcError) as exc_info:
            await t2
        assert exc_info.value.code == "unavailable"
        assert loop.time() - t0 < 5.0  # the dial's failure, not the 10 s budget

    try:
        a.run(_run())
    finally:
        srv.close()
        a.close()


def test_call_short_timeout_silent_listener():
    """A call whose budget runs out while dialing fails "unavailable", quickly."""
    a = _make_node()
    a.start()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    port = srv.getsockname()[1]

    target_id = Identity.generate()
    rec = make_record(target_id, addresses=[f"tcp://127.0.0.1:{port}"], seq=1)
    a.add_record(rec)

    try:
        t0 = time.monotonic()
        with pytest.raises(RpcError) as exc_info:
            a.call(target_id.peer_id, "echo", timeout=0.5)
        elapsed = time.monotonic() - t0
        assert exc_info.value.code == "unavailable"
        assert 0.4 <= elapsed < 1.2
    finally:
        srv.close()
        a.close()


def test_connect_expect_peer_mismatch_raises_handshake_error():
    """Verify connect with an expect_peer mismatch raises HandshakeError."""
    a = _make_node()
    b = _make_node()
    a.start()
    b.start()
    try:
        wrong_peer = Identity.generate().peer_id
        with pytest.raises(HandshakeError):
            a.connect(f"tcp://127.0.0.1:{b.listen_port}", expect_peer=wrong_peer)
        assert not a.is_connected(b.peer_id)
        assert not b.is_connected(a.peer_id)
    finally:
        a.close()
        b.close()


def test_max_connections_refuses_inbound_and_outbound():
    """Verify max_connections=1 refuses a second inbound peer and prevents outbound dials."""
    b = _make_node(max_connections=1)
    a = _make_node()
    c = _make_node()
    b.start()
    a.start()
    c.start()
    try:
        peer_b = a.connect(f"tcp://127.0.0.1:{b.listen_port}")
        assert peer_b == b.peer_id
        assert b.is_connected(a.peer_id)

        with pytest.raises(ConnectionError):
            c.connect(f"tcp://127.0.0.1:{b.listen_port}")

        b.add_record(c.record())
        with pytest.raises(RpcError) as exc_info:
            b.call(c.peer_id, "echo", timeout=2.0)
        assert exc_info.value.code == "unavailable"
        assert "too many connections" in exc_info.value.message
    finally:
        a.close()
        b.close()
        c.close()


def test_relay_aconnect_validation_and_dialer_order():
    """Verify relay aconnect validation, dialer order, and None-skipping."""
    a = _make_node()
    b = _make_node()
    a.start()
    b.start()
    try:
        relay_id = Identity.generate().peer_id
        with pytest.raises(ValueError):
            a.run(a.aconnect(f"relay://{relay_id}", expect_peer=None))

        call_order: list[str] = []

        async def dialer_none(node, peer_id, addr, timeout):
            """First dialer returns None."""
            call_order.append("dialer_none")
            return None

        async def dialer_stream(node, peer_id, addr, timeout):
            """Second dialer connects to b over loopback TCP and returns TcpStream."""
            call_order.append("dialer_stream")
            reader, writer = await asyncio.open_connection("127.0.0.1", b.listen_port)
            return TcpStream(reader, writer)

        a.add_dialer(dialer_none)
        a.add_dialer(dialer_stream)

        relay_addr = f"relay://{relay_id}"
        peer_id = a.connect(relay_addr, expect_peer=b.peer_id)
        assert peer_id == b.peer_id
        assert call_order == ["dialer_none", "dialer_stream"]
        assert a.is_connected(b.peer_id)
    finally:
        a.close()
        b.close()


def test_on_connect_and_on_disconnect_lifecycle():
    """Verify on_connect and on_disconnect fire exactly once per peer connection lifecycle."""
    a = _make_node()
    b = _make_node()

    a_connects: list[str] = []
    a_disconnects: list[str] = []
    b_connects: list[str] = []
    b_disconnects: list[str] = []

    a.on_connect(a_connects.append)
    a.on_disconnect(a_disconnects.append)
    b.on_connect(b_connects.append)
    b.on_disconnect(b_disconnects.append)

    a.start()
    b.start()
    try:
        a.connect(f"tcp://127.0.0.1:{b.listen_port}")

        time.sleep(0.1)
        assert a_connects == [b.peer_id]
        assert b_connects == [a.peer_id]
        assert a_disconnects == []
        assert b_disconnects == []

        b.close()
        time.sleep(0.2)

        assert a_disconnects == [b.peer_id]
        assert b_disconnects == [a.peer_id]
    finally:
        a.close()
        b.close()


def test_extension_handler_and_set_twice():
    """Verify extension handler receives frames and cannot be set twice."""
    a = _make_node()
    b = _make_node()

    received_frames: list[tuple[str, int, int, dict, bytes]] = []

    async def ext_handler(conn, kind, ident, header, body):
        """Record received extension frames."""
        received_frames.append((conn.peer_id, kind, ident, header, body))

    b.set_extension_handler(ext_handler)

    with pytest.raises(ValueError):
        b.set_extension_handler(ext_handler)

    a.start()
    b.start()
    try:
        a.connect(f"tcp://127.0.0.1:{b.listen_port}")
        conn_a = a.connection(b.peer_id)
        assert conn_a is not None

        a.run(conn_a.send_extension(20, 123, {"ext_key": "ext_val"}, b"payload_bytes"))

        time.sleep(0.1)
        assert len(received_frames) == 1
        peer_id, kind, ident, header, body = received_frames[0]
        assert peer_id == a.peer_id
        assert kind == 20
        assert ident == 123
        assert header == {"ext_key": "ext_val"}
        assert body == b"payload_bytes"
    finally:
        a.close()
        b.close()


def test_connection_pruning_max_two_per_peer():
    """Verify opening > 2 connections to the same peer closes the oldest connection."""
    a = _make_node()
    b = _make_node()
    a.start()
    b.start()
    try:
        addr = f"tcp://127.0.0.1:{b.listen_port}"
        a.connect(addr)
        conn1_a = a.connection(b.peer_id)
        assert conn1_a is not None

        a.connect(addr)
        conn2_a = a.connection(b.peer_id)
        assert conn2_a is not None
        assert conn2_a is not conn1_a

        a.connect(addr)
        conn3_a = a.connection(b.peer_id)
        assert conn3_a is not None
        assert conn3_a is not conn2_a

        time.sleep(0.1)

        assert conn1_a.closed
        assert not conn2_a.closed
        assert not conn3_a.closed

        with a._conn_lock:
            assert len(a._connections[b.peer_id]) == 2
    finally:
        a.close()
        b.close()


def test_a_refused_dial_fires_neither_connect_nor_disconnect():
    """Connect/disconnect events stay balanced when the other side refuses us."""
    a = _make_node()
    b = _make_node()
    events: list[str] = []
    a.on_connect(lambda pid: events.append("connect"))
    a.on_disconnect(lambda pid: events.append("disconnect"))
    b.add_gatekeeper(lambda pid: "banned" if pid == a.peer_id else None)
    a.start()
    b.start()
    try:
        with pytest.raises(ConnectionError):
            a.connect(f"tcp://127.0.0.1:{b.listen_port}")
        time.sleep(0.3)
        assert events == []
    finally:
        a.close()
        b.close()


def test_a_hello_that_fails_with_a_plain_exception_still_connects(monkeypatch):
    """Anything going wrong while building the hello is logged; the connection stays up."""
    a = _make_node()
    b = _make_node()
    b.register("echo", lambda ctx, h, body: (h, body))
    a.start()
    b.start()
    try:
        real_record = a.record
        calls = {"n": 0}

        def flaky_record():
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("record could not be built")
            return real_record()

        monkeypatch.setattr(a, "record", flaky_record)
        assert a.connect(f"tcp://127.0.0.1:{b.listen_port}") == b.peer_id
        assert a.call(b.peer_id, "echo", {"x": 1}, timeout=2.0)[0] == {"x": 1}
    finally:
        a.close()
        b.close()


def test_close_closes_the_event_loop():
    a = _make_node()
    a.start()
    loop = a.loop
    a.close()
    assert loop.is_closed()


def test_a_third_connection_never_drops_one_with_a_call_in_flight():
    """While A waits on a call over its connection to B, B dialling A again (as a
    reachability check does) must not close the connection carrying that call."""
    a = _make_node()
    b = _make_node()
    started = threading.Event()

    def slow(ctx, header, body):
        started.set()
        time.sleep(1.0)
        return {"ok": True}, b""

    b.register("slow", slow)
    a.start()
    b.start()
    try:
        a.connect(f"tcp://127.0.0.1:{b.listen_port}")
        result: dict = {}
        caller = threading.Thread(
            target=lambda: result.setdefault("r", a.call(b.peer_id, "slow", timeout=5.0))
        )
        caller.start()
        assert started.wait(3)
        for _ in range(3):  # B dials A directly, again and again
            b.connect(f"tcp://127.0.0.1:{a.listen_port}")
        caller.join(5)
        assert result["r"][0] == {"ok": True}
    finally:
        a.close()
        b.close()
