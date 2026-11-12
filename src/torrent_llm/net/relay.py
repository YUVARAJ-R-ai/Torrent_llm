"""Reaching a device nobody can connect to, through one that everybody can.

A laptop on a phone hotspot or behind carrier-grade NAT can make outgoing
connections but accepts none. It keeps one connection open to a reachable peer
(a *relay*) and advertises ``relay://<relay peer id>``. Someone who wants to reach
it connects to the relay and asks for a *circuit*; the relay pipes bytes between
the two connections. The two ends then run the normal secure handshake *through*
the circuit, so the relay forwards ciphertext it cannot read or alter undetected.

Wire (all over existing secure connections to the relay):

=====================  ==========================================================
``relay.reserve``      unreachable peer -> relay: ``{} -> {"ttl": seconds}``.
                       Keep my connection; route circuits for me. Renewed every
                       ``ttl / 2``. Valid while that connection lives.
``relay.connect``      dialer -> relay: ``{"target": peer id} -> {"circuit": id}``.
``relay.incoming``     relay -> target: ``{"from": peer id, "circuit": id} -> {}``.
                       The target answers once its end of the circuit exists.
CIRCUIT_DATA (16)      extension frame, ``id`` = circuit id on this connection,
                       body = bytes of the end-to-end stream.
CIRCUIT_CLOSE (17)     extension frame, header ``{"reason": str}``.
CIRCUIT_ACK (18)       extension frame, header ``{"n": bytes consumed}`` (flow control).
=====================  ==========================================================

Circuit ids are chosen by the relay, separately on each of its two connections. Both
ends of a connection may relay over it, so on each connection the side with the
smaller peer id uses odd ids and the other even ones: ids never collide, and are never
reused while the connection lives.

Flow control: a sender may have at most ``WINDOW_BYTES`` unacknowledged bytes in
flight on a circuit; the receiver acknowledges as its reader consumes, and the relay
forwards acknowledgements like data. So a relay never buffers more than one window
per direction per circuit, and a slow reader slows the writer instead of filling
anyone's memory. A sender that ignores the window gets its circuit closed, by the
relay as well as by the receiving end.

Limits a relay enforces (``RelayLimits``): reservations, circuits in total and per
peer, bytes per circuit, idle time per circuit. A refused request gets
``RpcError("busy")`` (limits) or ``RpcError("not_found")`` (no reservation for target)
or ``RpcError("denied")`` (relaying switched off).
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from torrent_llm.net.identity import is_peer_id
from torrent_llm.net.node import Node
from torrent_llm.net.records import Address, RelayAddress
from torrent_llm.net.rpc import CallContext, Connection, RpcError
from torrent_llm.net.stream import ByteStream, StreamClosed

logger = logging.getLogger(__name__)

CIRCUIT_DATA, CIRCUIT_CLOSE, CIRCUIT_ACK = 16, 17, 18
WINDOW_BYTES = 4 * 1024 * 1024
#: Bytes per CIRCUIT_DATA frame.
CHUNK_BYTES = 256 * 1024
RESERVATION_TTL_S = 600

#: A reader acknowledges at least this often (and whenever its buffer runs dry).
_ACK_EVERY = 64 * 1024
#: How often the relay looks for idle circuits.
_SWEEP_S = 5.0
#: How long the target has to accept a circuit.
_INCOMING_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class RelayLimits:
    max_reservations: int = 128
    max_circuits: int = 256
    max_circuits_per_peer: int = 8
    max_bytes_per_circuit: int = 8 * 1024 * 1024 * 1024
    idle_timeout_s: float = 300.0


def _valid_id(value: object) -> bool:
    return type(value) is int and 0 < value < 2**53


class CircuitStream:
    """One end of a circuit, as a :class:`~torrent_llm.net.stream.ByteStream`.

    ``read_exactly`` waits for data (reporting progress per chunk received), sends
    CIRCUIT_ACK as it consumes, and raises ``StreamClosed`` once the circuit is closed
    and drained. ``write`` splits into CHUNK_BYTES frames and waits while the window
    is full. ``close`` sends CIRCUIT_CLOSE once; closing the underlying connection
    closes every circuit on it.
    """

    def __init__(
        self,
        connection: Connection,
        circuit_id: int,
        remote: str,
        on_closed: Callable[[CircuitStream], None],
    ) -> None:
        self._conn = connection
        self._id = circuit_id
        self._remote = remote
        self._on_closed = on_closed
        self._buffer = bytearray()
        self._arrived = asyncio.Event()
        self._progress: Callable[[], None] | None = None
        #: Bytes we have sent that the other end has not acknowledged yet.
        self._unacked_out = 0
        #: Bytes received but not yet acknowledged (bounded by the other side's window).
        self._unacked_in = 0
        self._consumed = 0
        self._window = asyncio.Condition()
        self._write_lock = asyncio.Lock()
        self._closed = False
        self.close_reason = ""

    @property
    def remote(self) -> str:
        return self._remote

    @property
    def closed(self) -> bool:
        return self._closed

    async def read_exactly(self, n: int, *, on_progress: Callable[[], None] | None = None) -> bytes:
        # Bytes are taken (and acknowledged) as they arrive, so a read larger than the
        # window still completes; if the read is abandoned they go back for the next one.
        got = bytearray()
        try:
            while len(got) < n:
                if self._buffer:
                    take = min(n - len(got), len(self._buffer))
                    got += self._buffer[:take]
                    del self._buffer[:take]
                    self._consumed += take
                    await self._maybe_ack()
                    continue
                if self._closed:
                    raise StreamClosed(f"circuit closed: {self.close_reason or 'closed'}")
                self._progress = on_progress
                self._arrived.clear()
                try:
                    await self._arrived.wait()
                finally:
                    self._progress = None
        except BaseException:
            self._buffer[:0] = got
            raise
        return bytes(got)

    async def _maybe_ack(self) -> None:
        if self._consumed < _ACK_EVERY and self._buffer:
            return
        consumed, self._consumed = self._consumed, 0
        self._unacked_in -= consumed
        if consumed and not self._closed:
            try:
                await self._conn.send_extension(CIRCUIT_ACK, self._id, {"n": consumed}, b"")
            except (ConnectionError, RpcError) as exc:
                self._shut(f"connection lost: {exc}")

    async def write(self, data: bytes) -> None:
        async with self._write_lock:
            for start in range(0, len(data), CHUNK_BYTES):
                chunk = data[start : start + CHUNK_BYTES]
                async with self._window:
                    await self._window.wait_for(functools.partial(self._can_send, len(chunk)))
                    if self._closed:
                        raise StreamClosed(f"circuit closed: {self.close_reason or 'closed'}")
                    self._unacked_out += len(chunk)
                try:
                    await self._conn.send_extension(CIRCUIT_DATA, self._id, {}, chunk)
                except (ConnectionError, RpcError) as exc:
                    self._shut(f"connection lost: {exc}")
                    raise StreamClosed(str(exc)) from exc

    def _can_send(self, n: int) -> bool:
        return self._closed or self._unacked_out + n <= WINDOW_BYTES

    async def close(self) -> None:
        if self._closed:
            return
        self._shut("closed")
        try:
            await self._conn.send_extension(CIRCUIT_CLOSE, self._id, {"reason": "closed"}, b"")
        except (ConnectionError, RpcError) as exc:
            logger.debug("circuit %d: close not sent: %s", self._id, exc)

    # --- driven by the relay's frame handler (on the loop) ------------------------------

    def _on_data(self, body: bytes) -> bool:
        """Take one DATA frame; ``False`` if the sender broke the window (circuit closed)."""
        if self._closed:
            return True
        if self._unacked_in + len(body) > WINDOW_BYTES:
            self._shut("the other end ignored flow control")
            return False
        self._buffer += body
        self._unacked_in += len(body)
        self._arrived.set()
        if self._progress is not None:
            try:
                self._progress()
            except Exception:  # noqa: BLE001 - a progress callback must not break the stream
                logger.exception("progress callback failed")
        return True

    async def _on_ack(self, n: object) -> bool:
        if type(n) is not int or not 0 < n <= self._unacked_out:
            self._shut("bad acknowledgement")
            return False
        async with self._window:
            self._unacked_out -= n
            self._window.notify_all()
        return True

    def _shut(self, reason: str) -> None:
        """Close locally (no frame sent): wake every reader and writer, forget the circuit."""
        if self._closed:
            return
        self._closed = True
        self.close_reason = reason
        self._arrived.set()
        asyncio.get_running_loop().create_task(self._wake_writers())
        self._on_closed(self)

    async def _wake_writers(self) -> None:
        async with self._window:
            self._window.notify_all()


@dataclass(eq=False)
class _Circuit:
    """A circuit this node relays: dialer leg ``d`` and target leg ``t``."""

    dialer: str
    target: str
    conn: dict[str, Connection]
    cid: dict[str, int]
    #: Bytes sent from that side and not yet acknowledged by the other.
    unacked: dict[str, int] = field(default_factory=lambda: {"d": 0, "t": 0})
    bytes: int = 0
    last_active: float = field(default_factory=time.monotonic)


_OTHER = {"d": "t", "t": "d"}


class Relay:
    """Relaying for one node: the service (if ``serve``), the client side (reserving on
    relays and accepting circuits through them), and the dialer for ``relay://``.

    ``attach()`` registers ``relay.reserve``/``relay.connect`` (only when ``serve``) and
    ``relay.incoming``, installs the node's extension handler, and adds the dialer.
    ``serve`` also sets the node's record attr ``"relay": True``.

    ``reserve_on(relay_peer_ids)`` reserves on each (skipping failures), keeps the
    reservations renewed and re-made after a relay connection drops (trying the next
    candidate given to ``set_candidates``), and keeps
    ``node.set_relay_addresses([f"relay://{id}" ...])`` equal to the live reservations.

    Every public method may be called from any thread; all circuit state lives on the
    node's event loop.
    """

    def __init__(
        self, node: Node, *, serve: bool = False, limits: RelayLimits | None = None
    ) -> None:
        self.node = node
        self.limits = limits or RelayLimits()
        self._serve = serve
        self._service_registered = False
        self._register_lock = threading.Lock()
        self._attached = False
        self._detached = False
        #: Our ends of circuits: (connection, id) -> stream.
        self._streams: dict[tuple[Connection, int], CircuitStream] = {}
        #: Circuits we relay: (connection, id) -> (circuit, which leg).
        self._legs: dict[tuple[Connection, int], tuple[_Circuit, str]] = {}
        self._circuits: set[_Circuit] = set()
        self._next_id: dict[Connection, int] = {}
        self._watched: dict[Connection, asyncio.Task] = {}
        self._tasks: set[asyncio.Task] = set()
        #: Service side: peers reserved on us, and the connection they reserved on.
        self._served: dict[str, Connection] = {}
        #: Client side: relays we hold a reservation on.
        self._mine: dict[str, Connection] = {}
        self._want = 2
        self._candidates: list[str] = []
        self._bytes_relayed = 0
        self._sweeping = False
        self._renewing = False
        self._renew_s = RESERVATION_TTL_S / 2
        self._sweep_s = _SWEEP_S

    # --- lifecycle --------------------------------------------------------------------

    def attach(self) -> None:
        if self._attached:
            return
        self._attached = True
        self.node.register("relay.incoming", self._on_incoming, blocking=False)
        self.node.set_extension_handler(self._on_frame)
        self.node.add_dialer(self.dial)
        self.set_serving(self._serve)

    def detach(self) -> None:
        """Stop renewing, close every circuit, cancel tasks. Idempotent."""
        if self._detached or not self._attached:
            self._detached = True
            return
        self._detached = True
        self._set_service(False)
        try:
            self.node.unregister("relay.incoming")
        except Exception as exc:  # noqa: BLE001 - the node may be closing already
            logger.debug("unregistering relay.incoming: %s", exc)
        self._on_loop(self._shutdown())

    async def _shutdown(self) -> None:
        for stream in list(self._streams.values()):
            await stream.close()
        for circuit in list(self._circuits):
            await self._close_circuit(circuit, "relay shutting down")
        self._mine.clear()
        self._publish()
        current = asyncio.current_task()
        for task in [*self._tasks, *self._watched.values()]:
            if task is not current:
                task.cancel()

    @property
    def serving(self) -> bool:
        return self._serve

    def set_serving(self, serve: bool) -> None:
        was = self._serve
        self._serve = serve
        if serve or was:
            self.node.set_attr("relay", serve)
        if self._attached and not self._detached:
            self._set_service(serve)

    def _set_service(self, on: bool) -> None:
        with self._register_lock:
            if on and not self._service_registered:
                self.node.register("relay.reserve", self._on_reserve, blocking=False)
                self.node.register("relay.connect", self._on_connect, blocking=False)
                self._service_registered = True
            elif not on and self._service_registered:
                for method in ("relay.reserve", "relay.connect"):
                    self.node.unregister(method)
                self._service_registered = False

    # --- thread plumbing ----------------------------------------------------------------

    def _on_loop(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Run ``coro`` on the node's loop, from whatever thread this is."""
        try:
            loop = self.node.loop
        except RuntimeError:  # the node has not started (or is gone): nothing to do
            coro.close()
            return
        if not loop.is_running():
            coro.close()
            return
        try:
            on_loop = asyncio.get_running_loop() is loop
        except RuntimeError:
            on_loop = False
        if on_loop:
            self._spawn(coro)
        else:
            asyncio.run_coroutine_threadsafe(coro, loop)

    def _spawn(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
        """A tracked task on the current (node) loop; its failure is logged."""
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("relay task failed", exc_info=task.exception())

    def _watch(self, conn: Connection) -> None:
        """Clean up everything on ``conn`` the moment it closes (one watcher each)."""
        if conn not in self._watched:
            self._watched[conn] = self._spawn(self._follow(conn))

    async def _follow(self, conn: Connection) -> None:
        await conn.wait_closed()
        self._watched.pop(conn, None)
        await self._purge(conn)

    async def _purge(self, conn: Connection) -> None:
        for key, stream in list(self._streams.items()):
            if key[0] is conn:
                stream._shut("connection closed")
        for circuit in {c for (c_conn, _), (c, _) in self._legs.items() if c_conn is conn}:
            await self._close_circuit(circuit, "connection closed")
        for peer, served_conn in list(self._served.items()):
            if served_conn is conn:
                del self._served[peer]
        lost = [relay for relay, mine in self._mine.items() if mine is conn]
        for relay in lost:
            del self._mine[relay]
        self._next_id.pop(conn, None)
        if lost:
            self._publish()
            await self._refill(exclude=set(lost))

    def _new_id(self, conn: Connection) -> int:
        """The next circuit id this node allocates on ``conn`` (its parity, never reused)."""
        start = 1 if self.node.peer_id < conn.peer_id else 2
        cid = self._next_id.get(conn, start)
        self._next_id[conn] = cid + 2
        return cid

    def _their_parity(self, conn: Connection, cid: int) -> bool:
        """Whether ``cid`` has the parity the other side of ``conn`` allocates with."""
        theirs_odd = conn.peer_id < self.node.peer_id
        return (cid % 2 == 1) == theirs_odd

    # --- frames ---------------------------------------------------------------------

    async def _on_frame(
        self, conn: Connection, kind: int, ident: int, header: dict, body: bytes
    ) -> None:
        key = (conn, ident)
        stream = self._streams.get(key)
        if stream is not None:
            if kind == CIRCUIT_DATA:
                if not stream._on_data(body):
                    await self._send_close(conn, ident, "flow control ignored")
            elif kind == CIRCUIT_ACK:
                if not await stream._on_ack(header.get("n")):
                    await self._send_close(conn, ident, "bad acknowledgement")
            elif kind == CIRCUIT_CLOSE:
                reason = header.get("reason")
                stream._shut(reason[:100] if isinstance(reason, str) else "closed by peer")
            return
        leg = self._legs.get(key)
        if leg is not None:
            await self._forward(leg[0], leg[1], kind, header, body)
            return
        logger.debug("frame %d for unknown circuit %d from %s", kind, ident, conn.peer_id[:8])

    async def _forward(
        self, circuit: _Circuit, side: str, kind: int, header: dict, body: bytes
    ) -> None:
        other = _OTHER[side]
        if kind == CIRCUIT_CLOSE:
            await self._close_circuit(circuit, "closed by the other end", skip=side)
            return
        if kind == CIRCUIT_DATA:
            circuit.unacked[side] += len(body)
            circuit.bytes += len(body)
            if circuit.unacked[side] > WINDOW_BYTES:
                await self._close_circuit(circuit, "flow control ignored")
                return
            if circuit.bytes > self.limits.max_bytes_per_circuit:
                await self._close_circuit(circuit, "circuit byte limit reached")
                return
            frame_header: dict = {}
            self._bytes_relayed += len(body)
        elif kind == CIRCUIT_ACK:
            n = header.get("n")
            # An ACK from this side releases data the other side sent.
            if type(n) is not int or not 0 < n <= circuit.unacked[other]:
                await self._close_circuit(circuit, "bad acknowledgement")
                return
            circuit.unacked[other] -= n
            frame_header, body = {"n": n}, b""
        else:
            return
        circuit.last_active = time.monotonic()
        try:
            await circuit.conn[other].send_extension(kind, circuit.cid[other], frame_header, body)
        except (ConnectionError, RpcError) as exc:
            await self._close_circuit(circuit, f"forwarding failed: {exc}")

    async def _send_close(self, conn: Connection, cid: int, reason: str) -> None:
        try:
            await conn.send_extension(CIRCUIT_CLOSE, cid, {"reason": reason}, b"")
        except (ConnectionError, RpcError) as exc:
            logger.debug("circuit %d: close not sent: %s", cid, exc)

    async def _close_circuit(self, circuit: _Circuit, reason: str, *, skip: str = "") -> None:
        """Forget a relayed circuit and tell both legs (except ``skip``) once."""
        if circuit not in self._circuits:
            return
        self._circuits.discard(circuit)
        for side in ("d", "t"):
            self._legs.pop((circuit.conn[side], circuit.cid[side]), None)
        for side in ("d", "t"):
            if side != skip and not circuit.conn[side].closed:
                await self._send_close(circuit.conn[side], circuit.cid[side], reason)

    def _forget_stream(self, stream: CircuitStream) -> None:
        self._streams.pop((stream._conn, stream._id), None)

    # --- the service ----------------------------------------------------------------------

    def _ensure_sweep(self) -> None:
        if not self._sweeping:
            self._sweeping = True
            self._spawn(self._sweep())

    async def _sweep(self) -> None:
        try:
            while True:
                await asyncio.sleep(min(self._sweep_s, max(0.05, self.limits.idle_timeout_s / 2)))
                cutoff = time.monotonic() - self.limits.idle_timeout_s
                for circuit in [c for c in self._circuits if c.last_active < cutoff]:
                    await self._close_circuit(circuit, "idle")
        finally:
            self._sweeping = False

    async def _on_reserve(self, ctx: CallContext, header: dict, body: bytes) -> tuple[dict, bytes]:
        if not self._serve:
            raise RpcError("denied", "this device does not relay")
        conn = ctx.connection
        if conn is None:
            raise RpcError("bad_request", "reserve over a connection")
        if ctx.peer_id not in self._served and len(self._served) >= self.limits.max_reservations:
            raise RpcError("busy", "no room for more reservations")
        self._served[ctx.peer_id] = conn
        self._watch(conn)
        self._ensure_sweep()
        return {"ttl": RESERVATION_TTL_S}, b""

    async def _on_connect(self, ctx: CallContext, header: dict, body: bytes) -> tuple[dict, bytes]:
        if not self._serve:
            raise RpcError("denied", "this device does not relay")
        conn_d = ctx.connection
        target = header.get("target")
        if conn_d is None or not is_peer_id(target):
            raise RpcError("bad_request", "target must be a peer id")
        conn_t = self._served.get(target)
        if conn_t is None or conn_t.closed:
            raise RpcError("not_found", f"{target[:8]} has no reservation here")
        # Counted from live state, and checked and taken with no await in between, so
        # concurrent connects can never pass the limits together.
        if len(self._circuits) >= self.limits.max_circuits:
            raise RpcError("busy", "this relay is full")
        if sum(1 for c in self._circuits if c.dialer == ctx.peer_id) >= (
            self.limits.max_circuits_per_peer
        ):
            raise RpcError("busy", "too many circuits from you")
        circuit = _Circuit(
            dialer=ctx.peer_id,
            target=target,
            conn={"d": conn_d, "t": conn_t},
            cid={"d": self._new_id(conn_d), "t": self._new_id(conn_t)},
        )
        self._circuits.add(circuit)
        for side in ("d", "t"):
            self._legs[(circuit.conn[side], circuit.cid[side])] = (circuit, side)
        self._watch(conn_d)
        self._watch(conn_t)
        self._ensure_sweep()
        try:
            await conn_t.call(
                "relay.incoming",
                {"from": ctx.peer_id, "circuit": circuit.cid["t"]},
                timeout=_INCOMING_TIMEOUT_S,
            )
        except BaseException as exc:
            await self._close_circuit(circuit, "the target did not accept")
            if isinstance(exc, RpcError):
                raise RpcError("unavailable", f"{target[:8]} did not accept: {exc}") from exc
            raise
        if circuit not in self._circuits:  # a leg closed while we waited
            raise RpcError("unavailable", "the circuit closed while it was being set up")
        return {"circuit": circuit.cid["d"]}, b""

    def stats(self) -> dict[str, int]:
        """``{"reservations", "circuits", "bytes_relayed"}`` for the service side."""
        return {
            "reservations": len(self._served),
            "circuits": len(self._circuits),
            "bytes_relayed": self._bytes_relayed,
        }

    # --- being reached through a relay -------------------------------------------------

    async def _on_incoming(self, ctx: CallContext, header: dict, body: bytes) -> tuple[dict, bytes]:
        conn = ctx.connection
        if conn is None or self._mine.get(ctx.peer_id) is not conn:
            raise RpcError("denied", "this device holds no reservation with you")
        cid, dialer = header.get("circuit"), header.get("from")
        if not _valid_id(cid) or not is_peer_id(dialer) or not self._their_parity(conn, cid):
            raise RpcError("bad_request", "bad circuit")
        if (conn, cid) in self._streams or (conn, cid) in self._legs:
            raise RpcError("bad_request", "circuit id in use")
        stream = CircuitStream(conn, cid, f"relay:{ctx.peer_id}/{dialer}", self._forget_stream)
        self._streams[(conn, cid)] = stream
        self._watch(conn)
        self._spawn(self._accept(stream, ctx.peer_id))
        return {}, b""

    async def _accept(self, stream: CircuitStream, relay: str) -> None:
        try:
            await self.node.accept_stream(stream, via_relay=relay)
        except Exception as exc:  # noqa: BLE001 - a failed handshake ends only this circuit
            logger.info("circuit through %s not accepted: %s", relay[:8], exc)
            await stream.close()

    async def dial(
        self, node: Node, target: str, address: Address, timeout: float
    ) -> ByteStream | None:
        """The node dialer: ``None`` for non-relay addresses."""
        if not isinstance(address, RelayAddress) or address.relay == node.peer_id:
            return None
        relay = address.relay
        await node.acall(relay, "node.ping", {}, timeout=timeout)  # dials the relay if needed
        conn = node.connection(relay)
        if conn is None:
            raise ConnectionError(f"no connection to relay {relay[:8]}")
        header, _ = await conn.call("relay.connect", {"target": target}, timeout=timeout)
        cid = header.get("circuit")
        if not _valid_id(cid) or not self._their_parity(conn, cid):
            raise ConnectionError(f"relay {relay[:8]} gave a bad circuit id")
        if (conn, cid) in self._streams or (conn, cid) in self._legs:
            raise ConnectionError(f"relay {relay[:8]} reused a circuit id")
        stream = CircuitStream(conn, cid, f"relay:{relay}/{target}", self._forget_stream)
        self._streams[(conn, cid)] = stream
        self._watch(conn)
        return stream

    # --- holding reservations --------------------------------------------------------------

    def reserve_on(self, relay_peer_ids: list[str], *, want: int = 2) -> list[str]:
        """Synchronous; returns the relays now holding a reservation (at most ``want``)."""
        self._want = max(1, want)
        for relay in relay_peer_ids:
            if relay not in self._candidates:
                self._candidates.append(relay)
        try:
            return self.node.run(self._reserve(list(relay_peer_ids)), timeout=30.0)
        except (RuntimeError, TimeoutError) as exc:
            logger.info("reserving on relays failed: %s", exc)
            return self.reservations()

    def set_candidates(self, relay_peer_ids: list[str]) -> None:
        self._candidates = [r for r in relay_peer_ids if r != self.node.peer_id]

    def reservations(self) -> list[str]:
        return list(self._mine)

    async def _reserve(self, relays: list[str], exclude: set[str] = frozenset()) -> list[str]:
        for relay in relays:
            if len(self._mine) >= self._want:
                break
            if relay in self._mine or relay in exclude or relay == self.node.peer_id:
                continue
            try:
                await self.node.acall(relay, "relay.reserve", {}, timeout=10.0)
            except RpcError as exc:
                logger.info("relay %s refused a reservation: %s", relay[:8], exc)
                continue
            conn = self.node.connection(relay)
            if conn is None:
                continue
            self._mine[relay] = conn
            self._watch(conn)
        self._publish()
        if self._mine and not self._renewing:
            self._renewing = True
            self._spawn(self._renew())
        return list(self._mine)

    async def _refill(self, exclude: set[str]) -> None:
        if len(self._mine) < self._want and not self._detached:
            await self._reserve(list(self._candidates), exclude)

    async def _renew(self) -> None:
        try:
            while self._mine:
                await asyncio.sleep(self._renew_s)
                for relay, conn in list(self._mine.items()):
                    try:
                        await conn.call("relay.reserve", {}, timeout=10.0)
                    except (RpcError, ConnectionError) as exc:
                        logger.info("reservation on %s lapsed: %s", relay[:8], exc)
                        self._mine.pop(relay, None)
                self._publish()
                await self._refill(exclude=set())
        finally:
            self._renewing = False

    def _publish(self) -> None:
        self.node.set_relay_addresses([f"relay://{relay}" for relay in self._mine])
