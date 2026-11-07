"""The byte streams the secure channel runs on.

The channel does not care how bytes travel: over a TCP connection, or through a
relay that forwards them between two other connections (``relay.py``). Both
implement :class:`ByteStream`, and everything above (channel, RPC) is written
against it.

``read_exactly`` reports progress while a long read is under way. A 50 MB hidden
state on a slow link takes many seconds to arrive, and the RPC layer must be able
to tell "still receiving" from "silent" (``rpc.py`` declares a connection dead only
when no bytes at all have arrived for a while).
"""

from __future__ import annotations

import asyncio
import struct
from collections.abc import Callable
from typing import Protocol

#: Largest single frame accepted. A 4096-token prompt of an 8B model's hidden states
#: in float32 is 64 MiB; anything far above that is an attack, not a tensor.
MAX_FRAME_BYTES = 128 * 1024 * 1024
#: Bytes read per step of a long read, so progress is reported often.
CHUNK_BYTES = 64 * 1024

_LENGTH = struct.Struct(">I")


class StreamClosed(ConnectionError):
    """The other end went away (cleanly or not)."""


class FrameTooLarge(ConnectionError):
    """The other end announced a frame bigger than we accept."""


class ByteStream(Protocol):
    """An ordered, reliable byte stream to one other endpoint."""

    @property
    def remote(self) -> str:
        """Where the bytes come from, for logs and observed-address reports."""
        ...

    async def read_exactly(self, n: int, *, on_progress: Callable[[], None] | None = None) -> bytes:
        """Exactly ``n`` bytes, or :class:`StreamClosed`."""
        ...

    async def write(self, data: bytes) -> None: ...

    async def close(self) -> None: ...


class TcpStream:
    """A :class:`ByteStream` over an asyncio TCP connection."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._reader = reader
        self._writer = writer
        peer = writer.get_extra_info("peername")
        self._remote = f"{peer[0]}:{peer[1]}" if isinstance(peer, tuple) else str(peer)
        self._write_lock = asyncio.Lock()

    @property
    def remote(self) -> str:
        return self._remote

    @property
    def local(self) -> str:
        sock = self._writer.get_extra_info("sockname")
        return f"{sock[0]}:{sock[1]}" if isinstance(sock, tuple) else str(sock)

    async def read_exactly(self, n: int, *, on_progress: Callable[[], None] | None = None) -> bytes:
        if n == 0:
            return b""
        parts: list[bytes] = []
        left = n
        try:
            while left:
                chunk = await self._reader.read(min(left, CHUNK_BYTES))
                if not chunk:
                    raise StreamClosed(f"{self._remote} closed the connection")
                parts.append(chunk)
                left -= len(chunk)
                if on_progress is not None:
                    on_progress()
        except (ConnectionError, OSError) as exc:
            if isinstance(exc, StreamClosed):
                raise
            raise StreamClosed(f"{self._remote}: {exc}") from exc
        return b"".join(parts)

    async def write(self, data: bytes) -> None:
        async with self._write_lock:
            try:
                self._writer.write(data)
                await self._writer.drain()
            except (ConnectionError, OSError) as exc:
                raise StreamClosed(f"{self._remote}: {exc}") from exc

    async def close(self) -> None:
        if self._writer.is_closing():
            return
        self._writer.close()
        try:
            await self._writer.wait_closed()
        except (ConnectionError, OSError):
            pass


async def write_frame(stream: ByteStream, data: bytes) -> None:
    """Send one length-prefixed frame."""
    if len(data) > MAX_FRAME_BYTES:
        raise FrameTooLarge(f"frame of {len(data)} bytes is over {MAX_FRAME_BYTES}")
    await stream.write(_LENGTH.pack(len(data)) + data)


async def read_frame(
    stream: ByteStream,
    *,
    max_bytes: int = MAX_FRAME_BYTES,
    on_progress: Callable[[], None] | None = None,
) -> bytes:
    """Receive one length-prefixed frame, refusing anything over ``max_bytes``."""
    (length,) = _LENGTH.unpack(await stream.read_exactly(_LENGTH.size, on_progress=on_progress))
    if length > max_bytes:
        raise FrameTooLarge(f"{stream.remote} announced a {length}-byte frame (limit {max_bytes})")
    return await stream.read_exactly(length, on_progress=on_progress)
