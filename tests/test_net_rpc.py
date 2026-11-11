"""Tests for RPC Connection over SecureChannel."""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from torrent_llm.net.channel import (
    SecureChannel,
    accept,
    initiate,
)
from torrent_llm.net.identity import Identity
from torrent_llm.net.rpc import (
    CallContext,
    Connection,
    RpcError,
    encode_frame,
)
from torrent_llm.net.stream import TcpStream


def run(coro):
    return asyncio.run(coro)


async def _pair(
    a: Identity,
    b: Identity,
) -> tuple[SecureChannel, SecureChannel]:
    """Connect a (initiator) to b (responder) over real loopback TCP."""
    accepted: asyncio.Future[SecureChannel] = asyncio.get_running_loop().create_future()

    async def on_conn(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            accepted.set_result(await accept(TcpStream(reader, writer), b))
        except Exception as exc:  # noqa: BLE001
            accepted.set_exception(exc)

    server = await asyncio.start_server(on_conn, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        init = asyncio.ensure_future(initiate(TcpStream(reader, writer), a))
        results = await asyncio.gather(init, accepted, return_exceptions=True)
    finally:
        server.close()
    for r in results:
        if isinstance(r, BaseException):
            for other in results:
                if isinstance(other, SecureChannel):
                    await other.close()
            raise r
    return results[0], results[1]


async def _rpc_pair(
    handlers_a=None,
    handlers_b=None,
    *,
    ping_interval_a: float = 2.0,
    dead_after_a: float = 6.0,
    max_inflight_a: int = 32,
    on_close_a=None,
    on_extension_a=None,
    ping_interval_b: float = 2.0,
    dead_after_b: float = 6.0,
    max_inflight_b: int = 32,
    on_close_b=None,
    on_extension_b=None,
) -> tuple[Connection, Connection]:
    ch_a, ch_b = await _pair(Identity.generate(), Identity.generate())
    conn_a = Connection(
        ch_a,
        handlers_a or (lambda m: None),
        ping_interval=ping_interval_a,
        dead_after=dead_after_a,
        max_inflight=max_inflight_a,
        on_close=on_close_a,
        on_extension=on_extension_a,
    )
    conn_b = Connection(
        ch_b,
        handlers_b or (lambda m: None),
        ping_interval=ping_interval_b,
        dead_after=dead_after_b,
        max_inflight=max_inflight_b,
        on_close=on_close_b,
        on_extension=on_extension_b,
    )
    await conn_a.start()
    await conn_b.start()
    return conn_a, conn_b


# 1. echo call returns header+body; 100 concurrent calls; calls in both directions
def test_echo_and_concurrent_calls():
    async def _test():
        async def echo_b(ctx: CallContext, header: dict, body: bytes) -> tuple[dict, bytes]:
            return {"echo": header.get("k", "")}, body

        async def echo_a(ctx: CallContext, header: dict, body: bytes) -> tuple[dict, bytes]:
            return {"from_a": True}, body.upper()

        conn_a, conn_b = await _rpc_pair(
            handlers_a=lambda m: echo_a if m == "echo" else None,
            handlers_b=lambda m: echo_b if m == "echo" else None,
            max_inflight_a=128,
            max_inflight_b=128,
        )
        try:
            # Single call
            hdr, body = await conn_a.call("echo", {"k": "v1"}, b"hello", timeout=5.0)
            assert hdr == {"echo": "v1"}
            assert body == b"hello"

            # 100 concurrent calls A -> B
            async def single_call(i: int):
                h, b = await conn_a.call(
                    "echo", {"k": str(i)}, f"payload-{i}".encode(), timeout=5.0
                )
                assert h == {"echo": str(i)}
                assert b == f"payload-{i}".encode()

            await asyncio.gather(*[single_call(i) for i in range(100)])

            # Calls in both directions at once
            async def call_b_to_a(i: int):
                h, b = await conn_b.call("echo", {}, f"ping-{i}".encode(), timeout=5.0)
                assert h == {"from_a": True}
                assert b == f"PING-{i}".encode()

            await asyncio.gather(
                *[single_call(i) for i in range(50)],
                *[call_b_to_a(i) for i in range(50)],
            )
        finally:
            await conn_a.close()
            await conn_b.close()

    run(_test())


# 2. handler raising RpcError / ValueError
def test_handler_errors_and_internal_masking():
    async def _test():
        async def handler(ctx: CallContext, header: dict, body: bytes) -> tuple[dict, bytes]:
            if ctx.method == "busy":
                raise RpcError("busy", "full", header={"stage": 1})
            if ctx.method == "explode":
                raise ValueError("secret database password leaked 1234")
            return {}, b""

        def lookup(m: str):
            if m in ("busy", "explode"):
                return handler
            return None

        conn_a, conn_b = await _rpc_pair(handlers_b=lookup)
        try:
            with pytest.raises(RpcError) as exc_busy:
                await conn_a.call("busy", timeout=5.0)
            assert exc_busy.value.code == "busy"
            assert exc_busy.value.message == "full"
            assert exc_busy.value.header.get("stage") == 1

            with pytest.raises(RpcError) as exc_exp:
                await conn_a.call("explode", timeout=5.0)
            assert exc_exp.value.code == "internal"
            assert "Traceback" not in exc_exp.value.message
            assert "ValueError" not in exc_exp.value.message
            assert "1234" not in exc_exp.value.message
        finally:
            await conn_a.close()
            await conn_b.close()

    run(_test())


# 3. unknown method, bad name, reserved headers
def test_unknown_method_bad_request_and_reserved_headers():
    async def _test():
        conn_a, conn_b = await _rpc_pair(handlers_b=lambda m: None)
        try:
            # Unknown method -> not_found
            with pytest.raises(RpcError) as exc:
                await conn_a.call("unknown_method", timeout=5.0)
            assert exc.value.code == "not_found"

            # Method "Bad Name" -> bad_request (bypass client checks via channel.send)
            reply_fut = asyncio.get_running_loop().create_future()
            conn_a._pending[9999] = reply_fut
            raw_req = encode_frame(1, 9999, {"m": "Bad Name", "d": 5000}, b"test")
            await conn_a._channel.send(raw_req)

            with pytest.raises(RpcError) as exc_bad:
                await asyncio.wait_for(reply_fut, timeout=5.0)
            assert exc_bad.value.code == "bad_request"

            # Header containing "m" or "d" in call() -> ValueError
            with pytest.raises(ValueError):
                await conn_a.call("method", {"m": "val"}, timeout=5.0)
            with pytest.raises(ValueError):
                await conn_a.call("method", {"d": 1000}, timeout=5.0)
        finally:
            await conn_a.close()
            await conn_b.close()

    run(_test())


# 4. caller timeout -> RpcError deadline within timeout+0.5s and remote handler cancelled
def test_caller_timeout_and_handler_cancellation():
    async def _test():
        cancelled_event = asyncio.Event()

        async def slow_handler(ctx: CallContext, header: dict, body: bytes) -> tuple[dict, bytes]:
            try:
                await asyncio.sleep(10.0)
            finally:
                cancelled_event.set()
            return {}, b""

        conn_a, conn_b = await _rpc_pair(handlers_b=lambda m: slow_handler if m == "slow" else None)
        try:
            t0 = time.monotonic()
            with pytest.raises(RpcError) as exc:
                await conn_a.call("slow", timeout=0.2)
            elapsed = time.monotonic() - t0
            assert exc.value.code == "deadline"
            assert elapsed < 0.7  # timeout + 0.5s

            # Handler should be cancelled
            await asyncio.wait_for(cancelled_event.wait(), timeout=1.0)
            assert cancelled_event.is_set()
        finally:
            await conn_a.close()
            await conn_b.close()

    run(_test())


# 5. handler slower than its own deadline -> caller gets deadline
def test_handler_slower_than_own_deadline():
    async def _test():
        async def slow_handler(ctx: CallContext, header: dict, body: bytes) -> tuple[dict, bytes]:
            await asyncio.sleep(5.0)
            return {}, b""

        conn_a, conn_b = await _rpc_pair(handlers_b=lambda m: slow_handler if m == "slow" else None)
        try:
            t0 = time.monotonic()
            with pytest.raises(RpcError) as exc:
                await conn_a.call("slow", timeout=0.3)
            elapsed = time.monotonic() - t0
            assert exc.value.code == "deadline"
            assert elapsed < 1.5
        finally:
            await conn_a.close()
            await conn_b.close()

    run(_test())


# 6. max_inflight=2 with 3 concurrent slow calls -> exactly one gets busy
def test_max_inflight_busy():
    async def _test():
        started_event = asyncio.Event()
        release_event = asyncio.Event()
        started_count = 0

        async def slow(ctx: CallContext, header: dict, body: bytes) -> tuple[dict, bytes]:
            nonlocal started_count
            started_count += 1
            if started_count >= 2:
                started_event.set()
            await release_event.wait()
            return {}, b""

        conn_a, conn_b = await _rpc_pair(
            handlers_b=lambda m: slow if m == "slow" else None,
            max_inflight_b=2,
        )
        try:
            # Launch 3 calls concurrently
            task1 = asyncio.create_task(conn_a.call("slow", timeout=5.0))
            task2 = asyncio.create_task(conn_a.call("slow", timeout=5.0))
            await asyncio.wait_for(started_event.wait(), timeout=2.0)

            # 3rd call should get busy immediately
            with pytest.raises(RpcError) as exc:
                await conn_a.call("slow", timeout=5.0)
            assert exc.value.code == "busy"

            release_event.set()
            await asyncio.gather(task1, task2)
        finally:
            await conn_a.close()
            await conn_b.close()

    run(_test())


# 7. SILENT PEER: remote stops reading -> fails pending call in < 2s, on_close called once
def test_silent_peer_fails_pending_with_unavailable():
    async def _test():
        on_close_calls = []

        def on_close(conn):
            on_close_calls.append(conn)

        conn_a, conn_b = await _rpc_pair(
            ping_interval_a=0.2,
            dead_after_a=0.6,
            on_close_a=on_close,
        )
        try:
            # Cancel remote reader and keepalive tasks so it stops reading/answering
            assert conn_b._reader_task is not None
            assert conn_b._keepalive_task is not None
            conn_b._reader_task.cancel()
            conn_b._keepalive_task.cancel()

            t0 = time.monotonic()
            with pytest.raises(RpcError) as exc:
                await conn_a.call("echo", timeout=10.0)
            elapsed = time.monotonic() - t0

            assert exc.value.code == "unavailable"
            assert elapsed < 2.0
            await conn_a.close()
            assert conn_a.closed
            assert len(on_close_calls) == 1
        finally:
            await conn_a.close()
            await conn_b.close()

    run(_test())


# 8. remote close -> pending calls fail unavailable; call after close -> unavailable at once
def test_remote_close_and_call_after_close():
    async def _test():
        started_event = asyncio.Event()

        async def slow(ctx: CallContext, header: dict, body: bytes) -> tuple[dict, bytes]:
            started_event.set()
            await asyncio.sleep(10.0)
            return {}, b""

        conn_a, conn_b = await _rpc_pair(handlers_b=lambda m: slow if m == "slow" else None)
        try:
            task = asyncio.create_task(conn_a.call("slow", timeout=10.0))
            await asyncio.wait_for(started_event.wait(), timeout=2.0)

            # Remote closes
            await conn_b.close()

            with pytest.raises(RpcError) as exc:
                await task
            assert exc.value.code == "unavailable"

            # Call after close raises unavailable at once
            with pytest.raises(RpcError) as exc2:
                await conn_a.call("slow", timeout=5.0)
            assert exc2.value.code == "unavailable"
        finally:
            await conn_a.close()
            await conn_b.close()

    run(_test())


# 9. frame with unknown type or garbage closes connection
def test_protocol_violation_unknown_type_and_garbage():
    async def _test():
        conn_a, conn_b = await _rpc_pair()
        try:
            # Unknown type 99
            raw = encode_frame(99, 1, {}, b"")
            await conn_a._channel.send(raw)

            # Wait for conn_b to close due to protocol violation
            for _ in range(50):
                if conn_b.closed:
                    break
                await asyncio.sleep(0.05)
            assert conn_b.closed
        finally:
            await conn_a.close()
            await conn_b.close()

        # Garbage frame
        conn_a2, conn_b2 = await _rpc_pair()
        try:
            await conn_a2._channel.send(b"\x00\x01\x02\x03\x04")
            for _ in range(50):
                if conn_b2.closed:
                    break
                await asyncio.sleep(0.05)
            assert conn_b2.closed
        finally:
            await conn_a2.close()
            await conn_b2.close()

    run(_test())


# 10. extension frames reach on_extension; send_extension with type 5 or 64 raises ValueError
def test_extension_frames():
    async def _test():
        received_ext = []

        async def on_ext(kind: int, ident: int, header: dict, body: bytes) -> None:
            received_ext.append((kind, ident, header, body))

        conn_a, conn_b = await _rpc_pair(on_extension_b=on_ext)
        try:
            await conn_a.send_extension(16, 42, {"circuit": 1}, b"hello-relay")
            for _ in range(50):
                if received_ext:
                    break
                await asyncio.sleep(0.02)
            assert received_ext == [(16, 42, {"circuit": 1}, b"hello-relay")]

            with pytest.raises(ValueError):
                await conn_a.send_extension(5, 1, {}, b"")
            with pytest.raises(ValueError):
                await conn_a.send_extension(64, 1, {}, b"")
        finally:
            await conn_a.close()
            await conn_b.close()

    run(_test())


# 11. rtt_ms becomes positive float after short ping_interval
def test_rtt_ms_positive_float():
    async def _test():
        conn_a, conn_b = await _rpc_pair(ping_interval_a=0.05, ping_interval_b=0.05)
        try:
            for _ in range(50):
                if conn_a.rtt_ms is not None and conn_a.rtt_ms > 0:
                    break
                await asyncio.sleep(0.05)
            assert conn_a.rtt_ms is not None
            assert conn_a.rtt_ms > 0.0
        finally:
            await conn_a.close()
            await conn_b.close()

    run(_test())


# 12. large body (4 MiB) round-trips intact
def test_large_body_4mib_roundtrip():
    async def _test():
        async def echo(ctx: CallContext, header: dict, body: bytes) -> tuple[dict, bytes]:
            return header, body

        conn_a, conn_b = await _rpc_pair(handlers_b=lambda m: echo if m == "echo" else None)
        try:
            payload = os.urandom(4 * 1024 * 1024)
            hdr, res_body = await conn_a.call("echo", {"size": 4}, payload, timeout=10.0)
            assert hdr == {"size": 4}
            assert res_body == payload
        finally:
            await conn_a.close()
            await conn_b.close()

    run(_test())


# 13. NO DEADLOCK: 3+3 concurrent 8 MiB calls both directions with ping_interval=0.01 / 0.05
def test_no_deadlock_concurrent_8mib_both_directions():
    async def _test():
        async def echo(ctx: CallContext, header: dict, body: bytes) -> tuple[dict, bytes]:
            return {"side": header.get("side")}, body

        conn_a, conn_b = await _rpc_pair(
            handlers_a=lambda m: echo if m == "echo" else None,
            handlers_b=lambda m: echo if m == "echo" else None,
            ping_interval_a=0.01,
            ping_interval_b=0.01,
        )
        try:
            payload = b"x" * (8 * 1024 * 1024)

            async def call_a(idx: int):
                h, b = await conn_a.call("echo", {"side": f"a{idx}"}, payload, timeout=15.0)
                assert h == {"side": f"a{idx}"}
                assert len(b) == len(payload)

            async def call_b(idx: int):
                h, b = await conn_b.call("echo", {"side": f"b{idx}"}, payload, timeout=15.0)
                assert h == {"side": f"b{idx}"}
                assert len(b) == len(payload)

            t0 = time.monotonic()
            await asyncio.gather(
                call_a(1),
                call_a(2),
                call_a(3),
                call_b(1),
                call_b(2),
                call_b(3),
            )
            elapsed = time.monotonic() - t0
            assert elapsed < 20.0

            # Assert close returns in < 2s (Required fix 5(c))
            t_close = time.monotonic()
            await conn_a.close()
            await conn_b.close()
            assert time.monotonic() - t_close < 2.0
        finally:
            await conn_a.close()
            await conn_b.close()

    run(_test())


# 14. BACKPRESSURE + SILENT PEER: 32 MiB body to peer that stopped reading
def test_backpressure_and_silent_peer_large_body():
    async def _test():
        on_close_calls = []

        def on_close(c: Connection) -> None:
            on_close_calls.append(c)

        conn_a, conn_b = await _rpc_pair(
            dead_after_a=0.6,
            ping_interval_a=0.2,
            on_close_a=on_close,
        )
        try:
            # Remote stops reading without closing socket
            assert conn_b._reader_task is not None
            assert conn_b._keepalive_task is not None
            conn_b._reader_task.cancel()
            conn_b._keepalive_task.cancel()

            payload = b"Z" * (32 * 1024 * 1024)
            t0 = time.monotonic()
            with pytest.raises(RpcError) as exc:
                await conn_a.call("echo", body=payload, timeout=10.0)
            elapsed = time.monotonic() - t0

            # Must fail unavailable or deadline within < 2s (or < 3s)
            assert exc.value.code in ("unavailable", "deadline")
            assert elapsed < 3.0
            # close() returns in < 2s and invokes on_close
            t_close = time.monotonic()
            await conn_a.close()
            assert time.monotonic() - t_close < 2.0
            assert conn_a.closed
            assert len(on_close_calls) == 1
        finally:
            await conn_a.close()
            await conn_b.close()

    run(_test())


# 15. call() with timeout 0.5 stuck behind 64 MiB frame -> RpcError("deadline") in ~1s
def test_call_timeout_behind_large_frame():
    async def _test():
        conn_a, conn_b = await _rpc_pair()
        try:
            # Cancel B's reader so sends block on TCP backpressure
            assert conn_b._reader_task is not None
            conn_b._reader_task.cancel()

            # Enqueue a 64 MiB bulk frame
            bulk_payload = b"M" * (64 * 1024 * 1024)
            bulk_task = asyncio.create_task(conn_a.call("echo", body=bulk_payload, timeout=20.0))

            # Let writer start sending the 64 MiB frame and get stuck
            await asyncio.sleep(0.05)

            # Call fast method with timeout 0.5
            t0 = time.monotonic()
            with pytest.raises(RpcError) as exc:
                await conn_a.call("fast", timeout=0.5)
            elapsed = time.monotonic() - t0

            assert exc.value.code == "deadline"
            assert elapsed < 1.5

            bulk_task.cancel()
        finally:
            await conn_a.close()
            await conn_b.close()

    run(_test())


# 16. duplicate REQ id while first handler runs -> connection closes
def test_duplicate_req_id_closes_connection():
    async def _test():
        started_event = asyncio.Event()

        async def slow(ctx: CallContext, header: dict, body: bytes) -> tuple[dict, bytes]:
            started_event.set()
            await asyncio.sleep(5.0)
            return {}, b""

        conn_a, conn_b = await _rpc_pair(handlers_b=lambda m: slow if m == "slow" else None)
        try:
            # Send REQ with id 555
            req1 = encode_frame(1, 555, {"m": "slow", "d": 5000}, b"")
            await conn_a._channel.send(req1)
            await asyncio.wait_for(started_event.wait(), timeout=2.0)

            # Send duplicate REQ with id 555
            req2 = encode_frame(1, 555, {"m": "slow", "d": 5000}, b"")
            await conn_a._channel.send(req2)

            for _ in range(50):
                if conn_b.closed:
                    break
                await asyncio.sleep(0.05)
            assert conn_b.closed
        finally:
            await conn_a.close()
            await conn_b.close()

    run(_test())


# 17. validation of ping_interval, dead_after, max_inflight, timeout
def test_parameter_validation():
    async def _test():
        ch_a, ch_b = await _pair(Identity.generate(), Identity.generate())
        try:
            with pytest.raises(ValueError):
                Connection(ch_a, lambda m: None, ping_interval=0)
            with pytest.raises(ValueError):
                Connection(ch_a, lambda m: None, ping_interval=-1.0)
            with pytest.raises(ValueError):
                Connection(ch_a, lambda m: None, ping_interval=float("nan"))
            with pytest.raises(ValueError):
                Connection(ch_a, lambda m: None, dead_after=0)
            with pytest.raises(ValueError):
                Connection(ch_a, lambda m: None, dead_after=float("nan"))
            with pytest.raises(ValueError):
                Connection(ch_a, lambda m: None, max_inflight=0)
            with pytest.raises(ValueError):
                Connection(ch_a, lambda m: None, max_inflight=True)  # bool rejected

            conn = Connection(ch_a, lambda m: None)
            with pytest.raises(ValueError):
                await conn.call("m", timeout=0)
            with pytest.raises(ValueError):
                await conn.call("m", timeout=-1.0)
            with pytest.raises(ValueError):
                await conn.call("m", timeout=float("inf"))
            with pytest.raises(ValueError):
                await conn.call("m", timeout=float("nan"))
        finally:
            await ch_a.close()
            await ch_b.close()

    run(_test())


# 18. Too large detection in call and reply
def test_too_large_payload():
    async def _test():
        async def reply_huge(ctx: CallContext, header: dict, body: bytes) -> tuple[dict, bytes]:
            return {}, b"x" * (128 * 1024 * 1024)

        conn_a, conn_b = await _rpc_pair(handlers_b=lambda m: reply_huge if m == "huge" else None)
        try:
            # Client call too large
            too_large_body = b"y" * (128 * 1024 * 1024)
            with pytest.raises(RpcError) as exc:
                await conn_a.call("echo", body=too_large_body, timeout=5.0)
            assert exc.value.code == "too_large"

            # Reply too large
            with pytest.raises(RpcError) as exc2:
                await conn_a.call("huge", timeout=5.0)
            assert exc2.value.code == "too_large"
        finally:
            await conn_a.close()
            await conn_b.close()

    run(_test())


# --- regressions from review (escalated to the manager) --------------------------


def test_extension_frames_arrive_in_order():
    async def go():
        seen = []

        async def on_ext(kind, ident, header, body):
            await asyncio.sleep(0.001 * (10 - ident))  # later frames would overtake if racing
            seen.append(ident)

        a, b = await _rpc_pair(on_extension_b=on_ext)
        for i in range(10):
            await a.send_extension(16, i, {}, b"x")
        for _ in range(100):
            if len(seen) == 10:
                break
            await asyncio.sleep(0.02)
        assert seen == list(range(10))
        await a.close()
        await b.close()

    asyncio.run(go())


def test_a_raising_extension_handler_does_not_kill_the_reader():
    async def go():
        async def on_ext(kind, ident, header, body):
            raise ValueError("boom")

        async def echo(ctx, header, body):
            return {"ok": True}, body

        a, b = await _rpc_pair(handlers_b=lambda m: echo, on_extension_b=on_ext)
        await a.send_extension(16, 1, {}, b"")
        assert (await a.call("echo", {}, b"hi", timeout=2))[1] == b"hi"
        await a.close()
        await b.close()

    asyncio.run(go())


def test_reply_errors_reach_the_caller_instead_of_hanging():
    async def go():
        async def huge(ctx, header, body):
            return {"x": "y" * 70_000}, b""

        async def bad_rpc_header(ctx, header, body):
            raise RpcError("busy", "full", header={"o": object()})

        table = {"huge": huge, "bad": bad_rpc_header}
        a, b = await _rpc_pair(handlers_b=table.get)
        with pytest.raises(RpcError) as caught:
            await a.call("huge", timeout=2)
        assert caught.value.code == "too_large"
        with pytest.raises(RpcError) as caught:
            await a.call("bad", timeout=2)
        assert caught.value.code == "busy" and caught.value.message == "full"
        await a.close()
        await b.close()

    asyncio.run(go())


def test_a_raising_handler_lookup_answers_internal_and_keeps_the_connection():
    async def go():
        def lookup(method):
            raise KeyError(method)

        a, b = await _rpc_pair(handlers_b=lookup)
        with pytest.raises(RpcError) as caught:
            await a.call("anything", timeout=2)
        assert caught.value.code == "internal"
        assert not b.closed
        await a.close()
        await b.close()

    asyncio.run(go())


def test_on_close_runs_once_even_if_close_is_cancelled():
    async def go():
        calls = []
        a, b = await _rpc_pair(on_close_a=lambda c: calls.append(c))

        async def slow_close():
            await asyncio.sleep(5)

        a._channel.close = slow_close  # a channel that will not close politely
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(a.close(), 0.2)
        assert calls == [a]
        await a.close()  # idempotent
        assert calls == [a]
        await b.close()

    asyncio.run(go())


def test_byte_counters_move():
    async def go():
        async def echo(ctx, header, body):
            return {}, body

        a, b = await _rpc_pair(handlers_b=lambda m: echo)
        await a.call("echo", {}, b"z" * 1000, timeout=2)
        assert a.bytes_sent >= 1000 and a.bytes_received >= 1000
        assert b.bytes_sent >= 1000 and b.bytes_received >= 1000
        await a.close()
        await b.close()

    asyncio.run(go())
