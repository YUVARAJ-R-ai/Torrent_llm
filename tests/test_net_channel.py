"""Identity and secure channel: who you talk to, and that nobody else can read or forge it."""

from __future__ import annotations

import asyncio
import os

import pytest

from torrent_llm.net.channel import (
    ChannelError,
    HandshakeError,
    SecureChannel,
    accept,
    initiate,
)
from torrent_llm.net.identity import (
    Identity,
    is_peer_id,
    leading_zero_bits,
    mine_pow,
    peer_id_of,
    pow_ok,
    verify_signature,
)
from torrent_llm.net.stream import FrameTooLarge, TcpStream, read_frame, write_frame


def run(coro):
    return asyncio.run(coro)


async def _pair(
    a: Identity,
    b: Identity,
    *,
    psk_a=None,
    psk_b=None,
    expect=None,
    pow_a=0,
    pow_b=0,
) -> tuple[SecureChannel, SecureChannel]:
    """Connect a (initiator) to b (responder) over real loopback TCP."""
    accepted: asyncio.Future = asyncio.get_running_loop().create_future()

    async def on_conn(reader, writer):
        try:
            accepted.set_result(
                await accept(TcpStream(reader, writer), b, psk=psk_b, min_pow_bits=pow_b)
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the test below
            accepted.set_exception(exc)

    server = await asyncio.start_server(on_conn, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        init = asyncio.ensure_future(
            initiate(TcpStream(reader, writer), a, psk=psk_a, expect_peer=expect, min_pow_bits=pow_a)
        )
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


# --- identity -----------------------------------------------------------------


def test_peer_id_is_stable_and_shaped():
    ident = Identity.generate()
    assert ident.peer_id == peer_id_of(ident.public_key)
    assert is_peer_id(ident.peer_id)
    assert not is_peer_id("short")
    assert not is_peer_id(ident.peer_id.upper())
    with pytest.raises(ValueError):
        peer_id_of(b"x" * 31)


def test_signatures_verify_only_for_the_signer_and_the_data():
    a, b = Identity.generate(), Identity.generate()
    sig = a.sign(b"hello")
    assert verify_signature(a.public_key, sig, b"hello")
    assert not verify_signature(a.public_key, sig, b"hellO")
    assert not verify_signature(b.public_key, sig, b"hello")
    assert not verify_signature(a.public_key, sig[:-1], b"hello")


def test_proof_of_work_mines_and_checks():
    key = Identity.generate().public_key
    nonce = mine_pow(key, 12)
    assert pow_ok(key, nonce, 12)
    assert pow_ok(key, 12345, 0)
    assert not pow_ok(key, -1, 4)
    assert leading_zero_bits(b"\x00\x0f") == 12
    assert leading_zero_bits(b"\x00\x00") == 16


def test_identity_round_trips_and_keeps_its_id_when_work_is_raised(tmp_path):
    path = tmp_path / "sub" / "identity.json"
    first = Identity.load_or_create(path, pow_bits=0)
    again = Identity.load_or_create(path, pow_bits=0)
    assert again.peer_id == first.peer_id
    upgraded = Identity.load_or_create(path, pow_bits=10)
    assert upgraded.peer_id == first.peer_id
    assert pow_ok(upgraded.public_key, upgraded.pow_nonce, 10)
    assert Identity.load(path).pow_bits == 10
    if os.name == "posix":
        assert (path.stat().st_mode & 0o777) == 0o600


def test_a_damaged_identity_file_is_refused(tmp_path):
    path = tmp_path / "identity.json"
    path.write_text("{not json")
    with pytest.raises(ValueError, match="damaged"):
        Identity.load(path)
    good = Identity.generate(pow_bits=8)
    path.write_text(good.to_json().replace(f'"pow_nonce": {good.pow_nonce}', '"pow_nonce": 1'))
    if not pow_ok(good.public_key, 1, 8):
        with pytest.raises(ValueError, match="proof of work"):
            Identity.load(path)


# --- channel ------------------------------------------------------------------


def test_both_sides_learn_each_others_id_and_exchange_messages():
    async def go():
        a, b = Identity.generate(), Identity.generate()
        ca, cb = await _pair(a, b)
        assert ca.peer_id == b.peer_id and cb.peer_id == a.peer_id
        await ca.send(b"ping")
        await cb.send(b"pong" * 1000)
        assert await cb.recv() == b"ping"
        assert await ca.recv() == b"pong" * 1000
        for i in range(50):  # counters advance in step
            await ca.send(str(i).encode())
        assert [await cb.recv() for _ in range(50)] == [str(i).encode() for i in range(50)]
        await ca.close()
        await cb.close()

    run(go())


def test_a_large_frame_arrives_with_progress_reports():
    async def go():
        ca, cb = await _pair(Identity.generate(), Identity.generate())
        blob = os.urandom(3 * 1024 * 1024)
        ticks = []
        send = asyncio.ensure_future(ca.send(blob))
        got = await cb.recv(on_progress=lambda: ticks.append(1))
        await send
        assert got == blob
        assert len(ticks) > 10
        await ca.close()
        await cb.close()

    run(go())


def test_matching_swarm_secret_connects_and_a_wrong_one_fails_on_both_sides():
    async def go():
        secret = os.urandom(32)
        ca, cb = await _pair(Identity.generate(), Identity.generate(), psk_a=secret, psk_b=secret)
        await ca.close()
        await cb.close()
        with pytest.raises(HandshakeError, match="different swarm"):
            await _pair(Identity.generate(), Identity.generate(), psk_a=os.urandom(32), psk_b=secret)
        with pytest.raises(HandshakeError):
            await _pair(Identity.generate(), Identity.generate(), psk_a=None, psk_b=secret)

    run(go())


def test_pinning_the_expected_peer_refuses_an_impostor():
    async def go():
        a, b, someone_else = Identity.generate(), Identity.generate(), Identity.generate()
        ca, cb = await _pair(a, b, expect=b.peer_id)
        await ca.close()
        await cb.close()
        with pytest.raises(HandshakeError, match="expected peer"):
            await _pair(a, b, expect=someone_else.peer_id)

    run(go())


def test_proof_of_work_is_enforced_both_ways():
    async def go():
        cheap, worked = Identity.generate(), Identity.generate(pow_bits=10)
        ca, cb = await _pair(worked, Identity.generate(pow_bits=10), pow_a=10, pow_b=10)
        await ca.close()
        await cb.close()
        with pytest.raises(HandshakeError, match="proof of work"):
            await _pair(cheap, worked, pow_b=10)  # responder refuses the cheap initiator
        with pytest.raises(HandshakeError, match="proof of work"):
            await _pair(worked, cheap, pow_a=10)  # initiator refuses the cheap responder

    run(go())


def test_connecting_to_yourself_is_refused():
    async def go():
        me = Identity.generate()
        with pytest.raises(HandshakeError, match="itself"):
            await _pair(me, me)

    run(go())


def test_a_tampered_frame_fails_authentication():
    async def go():
        ca, cb = await _pair(Identity.generate(), Identity.generate())
        # Seal one frame properly, then flip a bit before it goes out.
        stream = ca.stream
        original_write = stream.write

        async def corrupting(data: bytes) -> None:
            data = bytearray(data)
            data[-1] ^= 0x01
            await original_write(bytes(data))

        stream.write = corrupting  # type: ignore[method-assign]
        await ca.send(b"secret")
        with pytest.raises(ChannelError, match="authentication"):
            await cb.recv()
        await ca.close()
        await cb.close()

    run(go())


def test_a_stranger_speaking_another_protocol_is_refused():
    async def go():
        result: asyncio.Future = asyncio.get_running_loop().create_future()

        async def on_conn(reader, writer):
            try:
                await accept(TcpStream(reader, writer), Identity.generate(), timeout=2)
            except Exception as exc:  # noqa: BLE001
                result.set_result(exc)

        server = await asyncio.start_server(on_conn, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        await write_frame(TcpStream(reader, writer), b"GET / HTTP/1.1\r\n\r\nxxxxxxxxxxxxxxxxxxx")
        exc = await asyncio.wait_for(result, 5)
        server.close()
        writer.close()
        assert isinstance(exc, HandshakeError)

    run(go())


def test_frames_over_the_limit_are_refused_before_reading_them():
    async def go():
        server_got: asyncio.Future = asyncio.get_running_loop().create_future()

        async def on_conn(reader, writer):
            try:
                await read_frame(TcpStream(reader, writer), max_bytes=1024)
            except Exception as exc:  # noqa: BLE001
                server_got.set_result(exc)

        server = await asyncio.start_server(on_conn, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write((10**9).to_bytes(4, "big"))
        await writer.drain()
        exc = await asyncio.wait_for(server_got, 5)
        server.close()
        writer.close()
        assert isinstance(exc, FrameTooLarge)

    run(go())


def test_a_silent_responder_times_out_the_handshake():
    async def go():
        async def on_conn(reader, writer):
            await asyncio.sleep(5)

        server = await asyncio.start_server(on_conn, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        with pytest.raises(HandshakeError, match="timed out"):
            await initiate(TcpStream(reader, writer), Identity.generate(), timeout=0.5)
        server.close()

    run(go())
