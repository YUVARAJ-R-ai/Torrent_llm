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

import asyncio
import itertools
import json
import logging
import math
import re
import struct
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from torrent_llm.net.channel import SecureChannel
from torrent_llm.net.stream import MAX_FRAME_BYTES

logger = logging.getLogger(__name__)

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


def _consume(future: asyncio.Future[Any]) -> None:
    """Mark a future's outcome as seen (its waiter may have given up already)."""
    if not future.cancelled():
        future.exception()


def valid_method(name: object) -> bool:
    return isinstance(name, str) and bool(_METHOD.match(name))


class _QueueItem:
    __slots__ = ("cancelled", "frame", "ident", "kind", "sent_fut", "started")

    def __init__(
        self,
        ident: int,
        kind: int,
        frame: bytes,
        sent_fut: asyncio.Future[None] | None = None,
    ) -> None:
        self.ident = ident
        self.kind = kind
        self.frame = frame
        self.sent_fut = sent_fut
        self.started = False
        self.cancelled = False


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
        if (
            not isinstance(ping_interval, (int, float))
            or not math.isfinite(ping_interval)
            or ping_interval <= 0
        ):
            raise ValueError(f"ping_interval must be finite and > 0, got {ping_interval!r}")
        if (
            not isinstance(dead_after, (int, float))
            or not math.isfinite(dead_after)
            or dead_after <= 0
        ):
            raise ValueError(f"dead_after must be finite and > 0, got {dead_after!r}")
        if not isinstance(max_inflight, int) or isinstance(max_inflight, bool) or max_inflight < 1:
            raise ValueError(f"max_inflight must be an integer >= 1, got {max_inflight!r}")

        self._channel = channel
        self._handlers = handlers
        self._ping_interval = float(ping_interval)
        self._dead_after = float(dead_after)
        self._max_inflight = max_inflight
        self._on_close = on_close
        self._on_extension = on_extension

        self._closed = False
        self._close_reason = ""
        self._close_event = asyncio.Event()
        self._on_close_called = False
        self._started = False

        self._rtt_ms: float | None = None
        self._bytes_sent = 0
        self._bytes_received = 0
        self._max_payload = MAX_FRAME_BYTES - 16

        self._last_rx = 0.0
        self._last_tx = 0.0

        self._call_counter = itertools.count(1)
        self._ping_counter = itertools.count(1)
        self._ping_times: dict[int, float] = {}

        self._pending: dict[int, asyncio.Future[tuple[dict, bytes]]] = {}
        self._running_handlers: dict[int, asyncio.Task[None]] = {}
        self._handler_tasks: set[asyncio.Task[Any]] = set()

        self._control_queue: deque[_QueueItem] = deque()
        self._bulk_queue: deque[_QueueItem] = deque()
        self._writer_event = asyncio.Event()
        self._current_item: _QueueItem | None = None

        #: Extension frames (relay circuit data) must be handled in arrival order, without
        #: blocking the reader on a slow handler: one consumer task, a bounded queue. A full
        #: queue makes the reader wait, which pushes back on the sender through TCP.
        self._extensions: asyncio.Queue[tuple[int, int, dict, bytes]] = asyncio.Queue(maxsize=256)
        self._extension_task: asyncio.Task[None] | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._writer_task: asyncio.Task[None] | None = None
        self._keepalive_task: asyncio.Task[None] | None = None

    @property
    def peer_id(self) -> str:
        return self._channel.peer_id

    @property
    def remote(self) -> str:
        return self._channel.remote

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def pending_calls(self) -> int:
        """Calls in flight on this connection: sent and awaited, or being served."""
        return len(self._pending) + len(self._running_handlers)

    @property
    def rtt_ms(self) -> float | None:
        return self._rtt_ms

    @property
    def bytes_sent(self) -> int:
        return self._bytes_sent

    @property
    def bytes_received(self) -> int:
        return self._bytes_received

    def _enqueue_control(self, frame: bytes) -> None:
        if self._closed:
            return
        item = _QueueItem(ident=0, kind=0, frame=frame)
        self._control_queue.append(item)
        self._writer_event.set()

    def _enqueue_item(self, item: _QueueItem) -> None:
        if self._closed:
            if item.sent_fut and not item.sent_fut.done():
                item.sent_fut.set_exception(
                    RpcError("unavailable", "connection is closed", peer=self.peer_id)
                )
            return
        if len(item.frame) < 65536 or item.kind in (PING, PONG, ERR, CANCEL):
            self._control_queue.append(item)
        else:
            self._bulk_queue.append(item)
        self._writer_event.set()

    def _send_error(
        self, ident: int, code: str, msg: str, extra: Mapping[str, Any] | None = None
    ) -> None:
        """Queue an ERR for ``ident``. Extra header fields that cannot be sent (not JSON,
        too large) are dropped rather than leaving the caller waiting for nothing."""
        try:
            frame = encode_frame(ERR, ident, {**dict(extra or {}), "code": code, "msg": msg})
        except (TypeError, ValueError, RpcError):
            frame = encode_frame(ERR, ident, {"code": code, "msg": msg[:1000]})
        self._enqueue_control(frame)

    def _spawn(self, coro: Awaitable[Any]) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro)
        self._handler_tasks.add(task)
        task.add_done_callback(self._handler_tasks.discard)
        return task

    async def start(self) -> None:
        """Start the reader and keepalive tasks. Returns at once."""
        if self._started or self._closed:
            return
        self._started = True
        loop = asyncio.get_running_loop()
        now = loop.time()
        self._last_rx = now
        self._last_tx = now
        self._writer_task = loop.create_task(
            self._writer_loop(), name=f"rpc-writer-{self.peer_id[:8]}"
        )
        self._reader_task = loop.create_task(
            self._reader_loop(), name=f"rpc-reader-{self.peer_id[:8]}"
        )
        self._keepalive_task = loop.create_task(
            self._keepalive_loop(), name=f"rpc-keepalive-{self.peer_id[:8]}"
        )
        if self._on_extension is not None:
            self._extension_task = loop.create_task(
                self._extension_loop(), name=f"rpc-ext-{self.peer_id[:8]}"
            )

    async def _writer_loop(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            while not self._closed:
                item: _QueueItem | None = None
                if self._control_queue:
                    item = self._control_queue.popleft()
                elif self._bulk_queue:
                    item = self._bulk_queue.popleft()
                else:
                    self._writer_event.clear()
                    if not self._control_queue and not self._bulk_queue:
                        try:
                            await self._writer_event.wait()
                        except asyncio.CancelledError:
                            break
                    continue

                if item.cancelled:
                    if item.sent_fut and not item.sent_fut.done():
                        item.sent_fut.set_result(None)
                    continue

                item.started = True
                self._current_item = item
                try:
                    await self._channel.send(item.frame)
                    self._bytes_sent += len(item.frame)
                    self._last_tx = loop.time()
                    if item.sent_fut and not item.sent_fut.done():
                        item.sent_fut.set_result(None)
                except Exception as exc:
                    if item.sent_fut and not item.sent_fut.done():
                        item.sent_fut.set_exception(exc)
                    if not self._closed:
                        logger.debug("channel send failed: %s", exc)
                        self._spawn(self.close(f"send failed: {exc}"))
                    break
                finally:
                    self._current_item = None
        except asyncio.CancelledError:
            if self._current_item is not None:
                if self._current_item.sent_fut and not self._current_item.sent_fut.done():
                    self._current_item.sent_fut.set_exception(
                        RpcError(
                            "unavailable",
                            self._close_reason or "closed",
                            peer=self.peer_id,
                        )
                    )

    async def _reader_loop(self) -> None:
        loop = asyncio.get_running_loop()

        def on_progress() -> None:
            self._last_rx = loop.time()

        try:
            while not self._closed:
                try:
                    frame_bytes = await self._channel.recv(on_progress=on_progress)
                except Exception as exc:
                    if not self._closed:
                        logger.debug("channel recv failed: %s", exc)
                        await self.close(f"channel recv failed: {exc}")
                    break

                self._last_rx = loop.time()
                self._bytes_received += len(frame_bytes)

                try:
                    kind, ident, header, body = decode_frame(frame_bytes)
                except ValueError as exc:
                    logger.warning(
                        "protocol violation: malformed frame from %s: %s", self.peer_id, exc
                    )
                    await self.close("protocol violation")
                    break

                if kind == REQ:
                    if ident in self._running_handlers:
                        logger.warning(
                            "protocol violation: duplicate REQ id %d from %s", ident, self.peer_id
                        )
                        await self.close("protocol violation")
                        break

                    m = header.get("m")
                    d = header.get("d")
                    if (
                        not valid_method(m)
                        or not isinstance(d, int)
                        or isinstance(d, bool)
                        or d <= 0
                    ):
                        self._enqueue_control(
                            encode_frame(
                                ERR,
                                ident,
                                {"code": "bad_request", "msg": "invalid method or deadline"},
                            )
                        )
                        continue

                    try:
                        handler = self._handlers(m)
                    except Exception:
                        logger.exception("handler lookup for %r failed", m)
                        self._send_error(ident, "internal", "handler lookup failed")
                        continue
                    if handler is None:
                        self._enqueue_control(
                            encode_frame(
                                ERR,
                                ident,
                                {"code": "not_found", "msg": f"method '{m}' not found"},
                            )
                        )
                        continue

                    if len(self._running_handlers) >= self._max_inflight:
                        self._enqueue_control(
                            encode_frame(ERR, ident, {"code": "busy", "msg": "server busy"})
                        )
                        continue

                    task = self._spawn(self._run_handler(ident, m, d, header, body, handler))
                    self._running_handlers[ident] = task

                elif kind == RES:
                    fut = self._pending.pop(ident, None)
                    if fut is not None and not fut.done():
                        fut.set_result((header, body))

                elif kind == ERR:
                    fut = self._pending.pop(ident, None)
                    if fut is not None and not fut.done():
                        code = header.get("code", "internal")
                        msg = header.get("msg", "")
                        extra = {k: v for k, v in header.items() if k not in ("code", "msg")}
                        fut.set_exception(RpcError(code, msg, peer=self.peer_id, header=extra))

                elif kind == CANCEL:
                    task = self._running_handlers.get(ident)
                    if task is not None and not task.done():
                        task.cancel()

                elif kind == PING:
                    self._enqueue_control(encode_frame(PONG, ident, {}))

                elif kind == PONG:
                    sent_at = self._ping_times.pop(ident, None)
                    if sent_at is not None:
                        self._rtt_ms = (loop.time() - sent_at) * 1000.0

                elif EXTENSION_MIN <= kind <= EXTENSION_MAX:
                    if self._on_extension is not None:
                        await self._extensions.put((kind, ident, header, body))

                else:
                    logger.warning(
                        "protocol violation: unknown frame kind %d from %s", kind, self.peer_id
                    )
                    await self.close("protocol violation")
                    break
        except asyncio.CancelledError:
            pass

    async def _run_handler(
        self,
        ident: int,
        method: str,
        d_ms: int,
        header: dict,
        body: bytes,
        handler: Handler,
    ) -> None:
        loop = asyncio.get_running_loop()
        ctx = CallContext(
            peer_id=self.peer_id,
            method=method,
            deadline=loop.time() + (d_ms / 1000.0),
            connection=self,
            remote=self.remote,
        )
        user_header = {k: v for k, v in header.items() if k not in ("m", "d")}
        try:
            remaining = max(0.0, ctx.deadline - loop.time())
            res_header, res_body = await asyncio.wait_for(
                handler(ctx, user_header, body),
                timeout=remaining,
            )
        except asyncio.CancelledError:
            return
        except TimeoutError:
            self._send_error(ident, "deadline", "deadline exceeded")
            return
        except RpcError as exc:
            self._send_error(ident, exc.code, exc.message, exc.header)
            return
        except Exception:
            logger.exception("handler for '%s' failed", method)
            self._send_error(ident, "internal", "handler failed")
            return
        finally:
            self._running_handlers.pop(ident, None)

        try:
            res_frame = encode_frame(RES, ident, res_header, res_body)
        except RpcError as exc:  # header over MAX_HEADER_BYTES
            self._send_error(ident, exc.code, "reply header is too large")
            return
        except (TypeError, ValueError):
            logger.exception("handler reply header for '%s' is not JSON serializable", method)
            self._send_error(ident, "internal", "invalid reply header")
            return

        if len(res_frame) > self._max_payload:
            self._send_error(ident, "too_large", "reply exceeds frame limit")
            return

        item = _QueueItem(ident=ident, kind=RES, frame=res_frame)
        self._enqueue_item(item)

    async def _extension_loop(self) -> None:
        """Deliver extension frames one at a time, in arrival order."""
        while True:
            kind, ident, header, body = await self._extensions.get()
            try:
                if self._on_extension is not None:
                    await self._on_extension(kind, ident, header, body)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("on_extension callback failed")

    async def _keepalive_loop(self) -> None:
        loop = asyncio.get_running_loop()
        wake_interval = min(self._ping_interval, self._dead_after) / 4.0
        try:
            while not self._closed:
                await asyncio.sleep(wake_interval)
                now = loop.time()

                if now - self._last_rx > self._dead_after:
                    logger.info("peer %s silent for > %fs; closing", self.peer_id, self._dead_after)
                    await self.close(f"silent for {self._dead_after}s")
                    break

                cutoff = now - self._dead_after
                self._ping_times = {n: t for n, t in self._ping_times.items() if t >= cutoff}

                if now - self._last_tx >= self._ping_interval:
                    nonce = next(self._ping_counter)
                    if nonce >= 0xFFFFFFFF:
                        self._ping_counter = itertools.count(1)
                    self._ping_times[nonce] = now
                    self._enqueue_control(encode_frame(PING, nonce, {}))
                    self._last_tx = now
        except asyncio.CancelledError:
            pass

    async def call(
        self,
        method: str,
        header: Mapping[str, Any] | None = None,
        body: bytes = b"",
        *,
        timeout: float,
    ) -> tuple[dict, bytes]:
        if self._closed:
            raise RpcError("unavailable", "connection is closed", peer=self.peer_id)

        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(f"timeout must be finite and > 0, got {timeout!r}")

        if not valid_method(method):
            raise ValueError(f"invalid method name: {method!r}")

        if header is not None:
            if "m" in header or "d" in header:
                raise ValueError("header must not contain reserved keys 'm' or 'd'")

        d_ms = max(1, int(timeout * 1000))
        req_header = {**dict(header or {}), "m": method, "d": d_ms}

        try:
            req_id = next(self._call_counter)
            if req_id >= 0xFFFFFFFF:
                self._call_counter = itertools.count(1)
            frame = encode_frame(REQ, req_id, req_header, body)
        except TypeError as exc:
            raise ValueError(f"header is not JSON serializable: {exc}") from exc

        if len(frame) > self._max_payload:
            raise RpcError("too_large", f"request frame of {len(frame)} bytes exceeds limit")

        if not self._started and not self._closed:
            await self.start()

        loop = asyncio.get_running_loop()
        reply_fut: asyncio.Future[tuple[dict, bytes]] = loop.create_future()
        reply_fut.add_done_callback(_consume)  # the caller may be gone when close() fails it
        sent_fut: asyncio.Future[None] = loop.create_future()
        sent_fut.add_done_callback(_consume)
        item = _QueueItem(ident=req_id, kind=REQ, frame=frame, sent_fut=sent_fut)

        self._pending[req_id] = reply_fut
        self._enqueue_item(item)

        deadline = loop.time() + timeout
        try:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError
            await asyncio.wait_for(asyncio.shield(sent_fut), timeout=remaining)

            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError
            return await asyncio.wait_for(reply_fut, timeout=remaining)

        except TimeoutError:
            self._pending.pop(req_id, None)
            if not item.started:
                item.cancelled = True
            else:
                self._enqueue_control(encode_frame(CANCEL, req_id, {}))
            raise RpcError(
                "deadline", f"call '{method}' timed out after {timeout}s", peer=self.peer_id
            ) from None
        except asyncio.CancelledError:
            self._pending.pop(req_id, None)
            if not item.started:
                item.cancelled = True
            else:
                self._enqueue_control(encode_frame(CANCEL, req_id, {}))
            raise
        except RpcError:
            self._pending.pop(req_id, None)
            raise
        except Exception as exc:
            self._pending.pop(req_id, None)
            raise RpcError("unavailable", str(exc), peer=self.peer_id) from exc

    async def send_extension(
        self, kind: int, ident: int, header: Mapping[str, Any], body: bytes
    ) -> None:
        """Send an extension frame (type 16-63)."""
        if (
            not isinstance(kind, int)
            or isinstance(kind, bool)
            or not (EXTENSION_MIN <= kind <= EXTENSION_MAX)
        ):
            raise ValueError(
                f"extension kind must be in {EXTENSION_MIN}..{EXTENSION_MAX}, got {kind!r}"
            )
        if self._closed:
            raise RpcError("unavailable", "connection is closed", peer=self.peer_id)

        try:
            frame = encode_frame(kind, ident, header, body)
        except TypeError as exc:
            raise ValueError(f"header is not JSON serializable: {exc}") from exc

        if len(frame) > self._max_payload:
            raise RpcError("too_large", f"extension frame of {len(frame)} bytes exceeds limit")

        if not self._started and not self._closed:
            await self.start()

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        fut.add_done_callback(_consume)
        item = _QueueItem(ident=ident, kind=kind, frame=frame, sent_fut=fut)
        self._enqueue_item(item)
        await fut

    async def wait_closed(self) -> None:
        """Return once this connection is fully closed, for whatever reason."""
        await self._close_event.wait()

    async def close(self, reason: str = "closed") -> None:
        if self._closed:
            await self._close_event.wait()
            return
        self._closed = True
        self._close_reason = reason
        self._writer_event.set()

        try:
            pending = list(self._pending.values())
            self._pending.clear()
            for fut in pending:
                if not fut.done():
                    fut.set_exception(RpcError("unavailable", reason, peer=self.peer_id))

            if self._current_item is not None:
                if self._current_item.sent_fut and not self._current_item.sent_fut.done():
                    self._current_item.sent_fut.set_exception(
                        RpcError("unavailable", reason, peer=self.peer_id)
                    )

            for q in (self._control_queue, self._bulk_queue):
                while q:
                    item = q.popleft()
                    if item.sent_fut and not item.sent_fut.done():
                        item.sent_fut.set_exception(
                            RpcError("unavailable", reason, peer=self.peer_id)
                        )

            current_task = asyncio.current_task()
            for task in (
                self._keepalive_task,
                self._reader_task,
                self._writer_task,
                self._extension_task,
            ):
                if task is not None and task is not current_task and not task.done():
                    task.cancel()

            for task in list(self._handler_tasks):
                if task is not current_task and not task.done():
                    task.cancel()
            self._handler_tasks.clear()

            try:
                await asyncio.wait_for(self._channel.close(), timeout=2.0)
            except Exception as exc:
                logger.debug("channel.close timed out or failed: %s; aborting transport", exc)
                self._abort_transport()
        finally:
            # Runs even when the task calling close() is cancelled mid-way.
            if self._on_close is not None and not self._on_close_called:
                self._on_close_called = True
                try:
                    self._on_close(self)
                except Exception:
                    logger.exception("on_close callback raised")
            self._close_event.set()

    def _abort_transport(self) -> None:
        """Drop the TCP connection at once when a polite close did not finish."""
        stream = getattr(self._channel, "stream", None)
        writer = getattr(stream, "_writer", None)
        transport = getattr(writer, "transport", None)
        if transport is None:
            return
        try:
            transport.abort()
        except Exception as exc:  # noqa: BLE001 - nothing more can be done here
            logger.debug("aborting the transport failed: %s", exc)
