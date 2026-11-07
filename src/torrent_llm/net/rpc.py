"""Request/response calls over one secure channel, with liveness.

One :class:`Connection` wraps one :class:`~torrent_llm.net.channel.SecureChannel`
and lets either side call methods on the other, many at once (multiplexed by a
per-direction call id). It is the layer that makes a silent peer cost seconds,
not minutes: each side pings when idle, and a connection that has received no
bytes at all for ``dead_after`` seconds is closed, failing every pending call at
once with ``unavailable``. Bytes still arriving inside a large frame count as
life, so a big hidden state on a slow link is not mistaken for silence.

Frame plaintext (inside the channel's encryption)::

    u8 type | u32 id | u32 header_length | header (UTF-8 JSON object) | body (bytes)

=======  ====  ===========================================================
type     code  meaning
=======  ====  ===========================================================
REQ      1     call; header has ``"m"`` (method) and ``"d"`` (deadline, ms)
RES      2     reply to REQ ``id``; header and body are the handler's result
ERR      3     failure of REQ ``id``; header has ``"code"`` and ``"msg"``
CANCEL   4     the caller gave up on REQ ``id``; no reply is sent
PING     5     ``id`` is a nonce; answered at once with PONG of the same id
PONG     6
16-63          extension frames, passed to ``on_extension`` (relay circuits)
=======  ====  ===========================================================

The method name and the remaining header keys belong to the application; this
layer only reserves ``m`` and ``d`` in REQ headers.
"""

from __future__ import annotations

import json
import re
import struct
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from torrent_llm.net.channel import SecureChannel

REQ, RES, ERR, CANCEL, PING, PONG = 1, 2, 3, 4, 5, 6
EXTENSION_MIN, EXTENSION_MAX = 16, 63

#: Error codes a caller can receive. Anything else from the wire becomes ``internal``.
CODES = frozenset(
    {
        "unavailable",  # connection lost or peer unreachable: try another peer
        "deadline",  # the call took longer than its deadline
        "busy",  # the peer refused before doing any work: try another copy
        "not_found",  # no such method
        "bad_request",  # the request was malformed
        "denied",  # not allowed (banned, rate-limited, wrong swarm)
        "internal",  # the handler failed
        "cache_lost",  # a shard lost this request's cache: re-prefill
        "cancelled",  # the caller cancelled
        "too_large",  # the message was over the frame limit
    }
)
#: Longest header accepted, in bytes. Headers are small JSON; bodies carry tensors.
MAX_HEADER_BYTES = 64 * 1024
_METHOD = re.compile(r"^[a-z][a-z0-9_.]{0,63}$")
_PREFIX = struct.Struct(">BII")


class RpcError(Exception):
    """A call failed. ``code`` is one of :data:`CODES`; ``peer`` is who failed, if known."""

    def __init__(
        self,
        code: str,
        message: str = "",
        *,
        peer: str | None = None,
        header: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = code if code in CODES else "internal"
        self.message = message
        self.peer = peer
        #: Extra fields the failing handler attached (e.g. which hop of a relay failed).
        self.header = dict(header or {})
        super().__init__(f"{self.code}: {message}" if message else self.code)


@dataclass
class CallContext:
    """What a handler knows about the call it is serving."""

    peer_id: str
    method: str
    #: Absolute deadline on the event loop's clock (``loop.time()``).
    deadline: float
    #: The connection the call came in on; ``None`` for an in-process self-call.
    connection: Connection | None = None
    #: The caller's address as this side sees it ("ip:port"), if known.
    remote: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


#: A handler gets the request header (without ``m``/``d``) and body, and returns
#: the reply header and body, or raises :class:`RpcError`.
Handler = Callable[[CallContext, dict, bytes], Awaitable[tuple[dict, bytes]]]
#: Looks up the handler for a method name; ``None`` means not found.
HandlerLookup = Callable[[str], Handler | None]
#: Receives extension frames: (type, id, header, body).
ExtensionHandler = Callable[[int, int, dict, bytes], Awaitable[None]]


def encode_frame(kind: int, ident: int, header: Mapping[str, Any], body: bytes = b"") -> bytes:
    """Serialise one frame (plaintext; the channel encrypts it)."""
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(raw) > MAX_HEADER_BYTES:
        raise RpcError("too_large", f"header of {len(raw)} bytes is over {MAX_HEADER_BYTES}")
    return _PREFIX.pack(kind, ident & 0xFFFFFFFF, len(raw)) + raw + body


def decode_frame(data: bytes) -> tuple[int, int, dict, bytes]:
    """Inverse of :func:`encode_frame`. Raises ``ValueError`` on anything malformed."""
    if len(data) < _PREFIX.size:
        raise ValueError("frame shorter than its prefix")
    kind, ident, length = _PREFIX.unpack_from(data)
    if length > MAX_HEADER_BYTES or _PREFIX.size + length > len(data):
        raise ValueError("frame header length is out of range")
    try:
        header = json.loads(data[_PREFIX.size : _PREFIX.size + length].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"frame header is not JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError("frame header is not a JSON object")
    return kind, ident, header, bytes(data[_PREFIX.size + length :])


def valid_method(name: object) -> bool:
    return isinstance(name, str) and bool(_METHOD.match(name))


class Connection:
    """Calls in both directions over one secure channel.

    Args:
        channel: An established channel.
        handlers: Finds the handler for an incoming method.
        ping_interval: Send a PING when nothing has been sent for this long.
        dead_after: Close the connection when no bytes have arrived for this long.
        max_inflight: Incoming calls served at once; more get ``busy`` at once.
        on_close: Called once, with this connection, after it closes for any reason.
        on_extension: Receives frames of types 16-63; without it they are ignored.

    Behaviour the tests pin down:

    * ``call`` returns the handler's ``(header, body)``; a handler raising
      :class:`RpcError` reaches the caller as the same ``code``/``message``/``header``;
      any other exception reaches it as ``internal`` without the traceback.
    * ``call`` with ``timeout`` raises ``RpcError("deadline")`` after ``timeout``
      seconds and sends CANCEL; the callee cancels the handler task. A handler still
      running at its own deadline is cancelled and answered with ``deadline``.
    * Unknown method → ``not_found``; invalid method name or header → ``bad_request``.
    * More than ``max_inflight`` concurrent incoming calls → ``busy`` immediately.
    * A malformed frame or a REQ/RES protocol violation closes the connection.
    * Losing the channel (closed, error, or silent for ``dead_after``) fails every
      pending call with ``unavailable`` within one check period, and cancels every
      running handler.
    * ``rtt_ms`` is the last PING round trip.
    * After ``close`` returns, ``call`` raises ``unavailable`` at once.
    """

    def __init__(
        self,
        channel: SecureChannel,
        handlers: HandlerLookup,
        *,
        ping_interval: float = 2.0,
        dead_after: float = 6.0,
        max_inflight: int = 32,
        on_close: Callable[[Connection], None] | None = None,
        on_extension: ExtensionHandler | None = None,
    ) -> None:
        raise NotImplementedError

    @property
    def peer_id(self) -> str:
        raise NotImplementedError

    @property
    def remote(self) -> str:
        raise NotImplementedError

    @property
    def closed(self) -> bool:
        raise NotImplementedError

    @property
    def rtt_ms(self) -> float | None:
        raise NotImplementedError

    @property
    def bytes_sent(self) -> int:
        raise NotImplementedError

    @property
    def bytes_received(self) -> int:
        raise NotImplementedError

    async def start(self) -> None:
        """Start the reader and keepalive tasks. Returns at once."""
        raise NotImplementedError

    async def call(
        self,
        method: str,
        header: Mapping[str, Any] | None = None,
        body: bytes = b"",
        *,
        timeout: float,
    ) -> tuple[dict, bytes]:
        raise NotImplementedError

    async def send_extension(
        self, kind: int, ident: int, header: Mapping[str, Any], body: bytes
    ) -> None:
        """Send an extension frame (type 16-63)."""
        raise NotImplementedError

    async def close(self, reason: str = "closed") -> None:
        raise NotImplementedError
