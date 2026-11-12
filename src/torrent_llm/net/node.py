"""One device's presence on the overlay: its identity, its connections, its methods.

A :class:`Node` listens for peers, dials them, runs the secure handshake on every
connection, and lets the rest of the program call methods on any peer by peer id
without caring how that peer is reached (directly, through its LAN address, its
public address, or a relay). It runs an asyncio event loop on a background thread
and offers a synchronous facade, because the pool around it (daemon, runner,
torch) is thread-based.

How a call to ``peer_id`` is carried, in order:

1. ``peer_id`` is this node: the handler runs in-process. A device never reaches its
   own stages through the network (a home router or a cloud network often refuses a
   machine connecting to its own public address).
2. A live connection to ``peer_id`` exists: use the newest one.
3. Otherwise dial it from its :class:`~torrent_llm.net.records.PeerRecord`: every
   ``tcp://`` address at once (first handshake that proves the right peer id wins,
   the others are cancelled), then each ``relay://`` address in turn through the
   registered dialers. Concurrent calls to the same unconnected peer share one dial.

Every new connection, in either direction, passes the gatekeepers (bans) and starts
with ``node.hello``: each side sends its signed record and learns the address the
other side sees it at (the start of NAT detection, ``autonat.py``).

Built-in methods:

==============  ==============================================================
``node.ping``   ``{} -> {"t": <unix time>}``
``node.hello``  ``{"record": <record dict>} -> {"record": <ours>, "observed": "ip:port"}``
==============  ==============================================================
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import threading
import time
from collections.abc import Awaitable, Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, TypeVar

from torrent_llm.net.channel import (
    SecureChannel,
    accept,
    initiate,
)
from torrent_llm.net.identity import Identity
from torrent_llm.net.ratelimit import Limit, RateLimiter
from torrent_llm.net.records import (
    MAX_ADDRESSES,
    Address,
    PeerRecord,
    RelayAddress,
    TcpAddress,
    make_record,
    parse_address,
)
from torrent_llm.net.rpc import CallContext, Connection, Handler, RpcError, valid_method
from torrent_llm.net.stream import ByteStream, TcpStream
from torrent_llm.pool.net import lan_address, tailscale_address

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _consume_task(task: asyncio.Task[Any]) -> None:
    """Consume any unhandled exception on task to avoid logging."""
    if not task.cancelled():
        task.exception()


#: Receives an extension frame: (connection, type, id, header, body).
NodeExtensionHandler = Callable[[Connection, int, int, dict, bytes], Awaitable[None]]

#: A synchronous handler, run on the node's thread pool (torch work belongs here).
SyncHandler = Callable[[CallContext, dict, bytes], tuple[dict, bytes]]
#: Opens a byte stream towards ``target`` (a peer id) through an address this dialer
#: understands, or returns ``None`` if it does not handle that kind of address.
#: Raises ``ConnectionError`` on failure. Arguments: (node, target, address, timeout).
Dialer = Callable[["Node", str, Address, float], Awaitable[ByteStream | None]]
#: Returns a reason to refuse a peer (e.g. "banned"), or ``None`` to let it in.
Gatekeeper = Callable[[str], str | None]

#: Default per-peer limits, by method prefix (the part before the first dot).
DEFAULT_LIMITS: dict[str, Limit] = {
    "node": Limit(rate=20.0, burst=40.0),
    "dht": Limit(rate=50.0, burst=100.0),
    "relay": Limit(rate=10.0, burst=20.0),
    "pool": Limit(rate=20.0, burst=40.0),
    "shard": Limit(rate=200.0, burst=400.0),
    "blob": Limit(rate=50.0, burst=100.0),
    "trust": Limit(rate=20.0, burst=40.0),
    # Each check makes this node dial out (up to 4 sockets): a few a minute is plenty.
    "autonat": Limit(rate=0.2, burst=3.0),
    #: Any prefix not listed above, so a new service is never unlimited by accident.
    "*": Limit(rate=20.0, burst=40.0),
}


@dataclass
class NodeConfig:
    identity: Identity
    #: The swarm secret; ``None`` for the public swarm.
    psk: bytes | None = None
    swarm_id: str = "public"
    #: Proof of work demanded of every peer (the public swarm uses PUBLIC_POW_BITS).
    min_pow_bits: int = 0
    listen_host: str = "0.0.0.0"
    #: 0 picks a free port.
    listen_port: int = 0
    #: Addresses to advertise besides the local interface addresses (a UPnP mapping,
    #: a confirmed public address, a DNS name).
    advertise: tuple[str, ...] = ()
    #: Advertise ``tcp://<ip>:<port>`` for every local non-loopback IPv4 address.
    #: Off in tests that simulate a device behind NAT.
    advertise_local: bool = True
    connect_timeout: float = 5.0
    handshake_timeout: float = 10.0
    ping_interval: float = 2.0
    dead_after: float = 6.0
    max_connections: int = 256
    max_inflight_per_connection: int = 32
    handler_threads: int = 8
    #: Per-peer rate limits by method prefix; ``None`` means DEFAULT_LIMITS.
    limits: dict[str, Limit] | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConnectionInfo:
    peer_id: str
    remote: str
    inbound: bool
    rtt_ms: float | None
    bytes_sent: int
    bytes_received: int
    #: Peer id of the relay this connection goes through, if any.
    via_relay: str | None = None


@dataclass
class _ConnEntry:
    """Internal container tracking a connection and its metadata."""

    connection: Connection
    inbound: bool
    via_relay: str | None = None


class Node:
    """See the module docstring. Thread-safe unless a method says otherwise.

    Behaviour the tests pin down:

    * ``start`` binds the listener (``OSError`` if the port is taken) and returns once
      it accepts connections; ``close`` is idempotent and stops every task and thread.
    * ``connect(address)`` returns the peer id proven by the handshake;
      ``expect_peer`` refuses anyone else.
    * ``call`` to self runs the handler in-process; to a peer it dials as described.
      Unknown method -> ``RpcError("not_found")``; unreachable -> ``RpcError("unavailable")``
      naming the peer, within ``connect_timeout`` + a little.
    * A peer that dies or goes silent fails in-flight calls with ``unavailable``
      within ``dead_after`` + one check period.
    * A refused gatekeeper closes the connection right after the handshake, both
      directions; a peer past its rate limit gets ``RpcError("denied")``.
    * Blocking handlers run on the thread pool, never on the event-loop thread;
      ``call`` from the event-loop thread raises ``RuntimeError`` (it would deadlock).
    * Peers in another swarm (different psk) or lacking proof of work never connect.
    * After ``node.hello`` both sides hold each other's verified record, and
      ``observed_addresses()`` reports how peers see this node.
    """

    def __init__(self, config: NodeConfig) -> None:
        """Initialise a Node with the given configuration."""
        self._config = config
        self._identity = config.identity
        self._peer_id = self._identity.peer_id

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._server: asyncio.Server | None = None
        self._listen_port: int = config.listen_port

        self._started = False
        self._closed = False
        self._lock = threading.Lock()

        limits = config.limits if config.limits is not None else DEFAULT_LIMITS
        self._limited = frozenset(limits)
        self._rate_limiter = RateLimiter(limits)

        self._handlers: dict[str, tuple[SyncHandler | Handler, bool]] = {}
        self._gatekeepers: list[Gatekeeper] = []
        self._dialers: list[Dialer] = []
        self._on_connect_cbs: list[Callable[[str], None]] = []
        self._on_disconnect_cbs: list[Callable[[str], None]] = []
        self._extension_handler: NodeExtensionHandler | None = None

        self._conn_lock = threading.Lock()
        self._connections: dict[str, list[_ConnEntry]] = {}
        #: Peers whose on_connect fired; on_disconnect fires only for these, once.
        self._announced: set[str] = set()
        self._dials_in_flight: dict[str, asyncio.Task[Connection]] = {}
        self._background_tasks: set[asyncio.Task[Any]] = set()

        self._records_lock = threading.Lock()
        self._records: dict[str, PeerRecord] = {}

        self._observed_lock = threading.Lock()
        self._peer_observed: dict[str, str] = {}
        self._observed_counts: dict[str, int] = {}

        self._record_lock = threading.Lock()
        self._advertised: list[str] = []
        self._relay_addresses: list[str] = []
        self._attrs: dict[str, Any] = dict(config.attrs)
        self._cached_record: PeerRecord | None = None
        self._last_addresses: list[str] = []
        self._record_seq: int = 0
        self._record_dirty: bool = True

    def _spawn(self, coro: Awaitable[Any]) -> asyncio.Task[Any]:
        """Spawn a background task and track it until completion."""
        task = self.loop.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        task.add_done_callback(_consume_task)
        return task

    # --- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Start the background event loop, thread pool and listener socket."""
        with self._lock:
            if self._started:
                return
            if self._closed:
                raise RuntimeError("node is closed")

            loop = asyncio.new_event_loop()
            self._loop = loop
            ready = threading.Event()

            def _run_loop() -> None:
                """Run the event loop in the daemon thread."""
                asyncio.set_event_loop(loop)
                loop.call_soon(ready.set)
                loop.run_forever()
                loop.close()  # only close() stops the loop

            thread_name = f"tllm-node-{self._peer_id[:8]}"
            thread = threading.Thread(target=_run_loop, name=thread_name, daemon=True)
            self._thread = thread
            thread.start()
            ready.wait()

            self._executor = ThreadPoolExecutor(
                max_workers=self._config.handler_threads,
                thread_name_prefix="tllm-handler",
            )

            fut = asyncio.run_coroutine_threadsafe(self._bind_server(), loop)
            try:
                self._server = fut.result()
            except Exception as exc:
                # close() takes self._lock, so it must run after this block releases it.
                self._started = True
                bind_error: Exception | None = exc
            else:
                bind_error = None
                sockets = self._server.sockets
                if sockets:
                    sockname = sockets[0].getsockname()
                    if isinstance(sockname, tuple):
                        self._listen_port = sockname[1]
                self._started = True

        if bind_error is not None:
            self.close()
            raise bind_error

    async def _bind_server(self) -> asyncio.Server:
        """Bind and start the listening TCP server."""
        return await asyncio.start_server(
            self._handle_inbound_connection,
            host=self._config.listen_host,
            port=self._config.listen_port,
        )

    async def _handle_inbound_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Accept an incoming TCP connection, perform handshake, and adopt it."""
        stream = TcpStream(reader, writer)
        try:
            ch = await accept(
                stream,
                self._identity,
                psk=self._config.psk,
                min_pow_bits=self._config.min_pow_bits,
                timeout=self._config.handshake_timeout,
            )
        except Exception as exc:
            logger.debug("inbound handshake failed from %s: %s", stream.remote, exc)
            await stream.close()
            return

        try:
            await self._adopt_channel(ch, inbound=True)
        except ConnectionRefusedError as exc:
            logger.debug("inbound connection refused from %s: %s", ch.peer_id, exc)
        except Exception as exc:
            logger.debug("failed to adopt inbound channel from %s: %s", ch.peer_id, exc)
            await ch.close()

    def close(self) -> None:
        """Close every connection, the server, cancel pending tasks, and stop threads."""
        with self._lock:
            if self._closed:
                return
            self._closed = True

        if self._loop is None or not self._loop.is_running():
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
            return

        is_loop_thread = threading.current_thread() == self._thread

        async def _async_close() -> None:
            """Asynchronously shut down listener, connections, and loop tasks."""
            if self._server is not None:
                self._server.close()

            with self._conn_lock:
                all_conns = [
                    entry.connection for entries in self._connections.values() for entry in entries
                ]
                self._connections.clear()

            if all_conns:
                await asyncio.gather(
                    *(conn.close(reason="node closing") for conn in all_conns),
                    return_exceptions=True,
                )

            if self._server is not None:
                await self._server.wait_closed()

            for t in list(self._background_tasks):
                t.cancel()
            self._background_tasks.clear()

            current = asyncio.current_task()
            tasks = [t for t in asyncio.all_tasks(self._loop) if t is not current and not t.done()]
            for t in tasks:
                t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        if not is_loop_thread:
            fut = asyncio.run_coroutine_threadsafe(_async_close(), self._loop)
            try:
                fut.result(timeout=5.0)
            except Exception as exc:
                logger.debug("error closing node asynchronously: %s", exc)

            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=5.0)
            if not self._loop.is_running() and not self._loop.is_closed():
                self._loop.close()
        else:
            # On the loop thread: stop once closing has finished, not before it starts.
            task = self._loop.create_task(_async_close())
            task.add_done_callback(lambda _t: self._loop.stop())

        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> Node:
        """Enter context manager, starting the node."""
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        """Exit context manager, closing the node."""
        self.close()

    @property
    def peer_id(self) -> str:
        """This node's unique peer id."""
        return self._peer_id

    @property
    def identity(self) -> Identity:
        """This node's cryptographic identity."""
        return self._identity

    @property
    def config(self) -> NodeConfig:
        """Configuration options given to this node."""
        return self._config

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """The node's background asyncio event loop."""
        if self._loop is None:
            raise RuntimeError("node not started")
        return self._loop

    @property
    def listen_port(self) -> int:
        """The TCP port this node is listening on."""
        return self._listen_port

    def run(self, coro: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
        """Run ``coro`` on the node's loop from another thread and return its result."""
        if self._loop is None or not self._loop.is_running():
            coro.close()
            raise RuntimeError("node is not running")
        if self._thread is not None and threading.current_thread() == self._thread:
            coro.close()
            raise RuntimeError("would deadlock")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout)
        except TimeoutError:
            fut.cancel()
            raise

    # --- methods -------------------------------------------------------------

    def register(
        self, method: str, handler: SyncHandler | Handler, *, blocking: bool = True
    ) -> None:
        """Serve ``method``. ``blocking=True``: a plain function run on the thread pool;
        ``False``: an ``async def`` run on the loop. ``ValueError`` for a bad name or a
        name registered twice."""
        if not valid_method(method):
            raise ValueError(f"invalid method name: {method!r}")
        if method in self._handlers or method in ("node.ping", "node.hello"):
            raise ValueError(f"method already registered: {method!r}")
        self._handlers[method] = (handler, blocking)

    def unregister(self, method: str) -> None:
        """Stop serving ``method``."""
        self._handlers.pop(method, None)

    def call(
        self,
        peer_id: str,
        method: str,
        header: dict | None = None,
        body: bytes = b"",
        *,
        timeout: float,
    ) -> tuple[dict, bytes]:
        """Perform an RPC call to ``peer_id`` synchronously with a deadline."""
        return self.run(
            self.acall(peer_id, method, header, body, timeout=timeout),
            timeout=timeout + 1.0,
        )

    async def acall(
        self,
        peer_id: str,
        method: str,
        header: dict | None = None,
        body: bytes = b"",
        *,
        timeout: float,
    ) -> tuple[dict, bytes]:
        """Perform an RPC call to ``peer_id`` asynchronously with a deadline."""
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise ValueError(f"timeout must be finite and > 0, got {timeout!r}")

        loop = self._loop or asyncio.get_running_loop()

        if peer_id == self._peer_id:
            entry = self._get_handler(method)
            if entry is None:
                raise RpcError("not_found", f"method '{method}' not found", peer=self._peer_id)
            fn, blocking = entry
            ctx = CallContext(
                peer_id=self._peer_id,
                method=method,
                deadline=loop.time() + timeout,
                connection=None,
                remote="self",
            )
            req_header = dict(header or {})

            async def _run_self() -> tuple[dict, bytes]:
                """Run self handler locally."""
                if blocking:
                    return await loop.run_in_executor(self._executor, fn, ctx, req_header, body)
                else:
                    return await fn(ctx, req_header, body)

            try:
                return await asyncio.wait_for(_run_self(), timeout=timeout)
            except TimeoutError:
                raise RpcError(
                    "deadline",
                    f"call '{method}' timed out after {timeout}s",
                    peer=self._peer_id,
                ) from None
            except RpcError:
                raise
            except Exception as exc:
                logger.exception("self handler for '%s' failed", method)
                raise RpcError("internal", str(exc), peer=self._peer_id) from exc

        conn = self.connection(peer_id)
        if conn is not None:
            return await conn.call(method, header, body, timeout=timeout)

        dial_task = self._dials_in_flight.get(peer_id)
        if dial_task is None or dial_task.done():
            dial_task = loop.create_task(
                self._dial_peer(peer_id),
                name=f"dial-{peer_id[:8]}",
            )
            dial_task.add_done_callback(_consume_task)
            self._dials_in_flight[peer_id] = dial_task

            def _cleanup(t: asyncio.Task[Any], pid: str = peer_id) -> None:
                """Clean up finished dial task from in-flight registry."""
                if self._dials_in_flight.get(pid) is t:
                    self._dials_in_flight.pop(pid, None)

            dial_task.add_done_callback(_cleanup)

        t0 = loop.time()
        try:
            conn = await asyncio.wait_for(asyncio.shield(dial_task), timeout=timeout)
        except TimeoutError:
            raise RpcError(
                "unavailable",
                f"could not reach {peer_id[:8]}: no connection within {timeout}s",
                peer=peer_id,
            ) from None
        except RpcError:
            raise
        except Exception as exc:
            raise RpcError(
                "unavailable",
                f"could not reach {peer_id[:8]}: {exc}",
                peer=peer_id,
            ) from exc

        elapsed = loop.time() - t0
        remaining = max(0.0, timeout - elapsed)
        if remaining <= 0:
            raise RpcError("deadline", f"call '{method}' timed out after {timeout}s", peer=peer_id)

        return await conn.call(method, header, body, timeout=remaining)

    def _get_handler(self, method: str) -> tuple[SyncHandler | Handler, bool] | None:
        """Return the handler and blocking flag for method if known."""
        if method in self._handlers:
            return self._handlers[method]
        if method == "node.ping":
            return (self._builtin_ping, False)
        if method == "node.hello":
            return (self._builtin_hello, False)
        return None

    def _lookup_handler(self, method: str) -> Handler | None:
        """Find and wrap the RPC handler for method, applying gatekeeper and limits."""
        entry = self._get_handler(method)
        if entry is None:
            return None
        fn, blocking = entry

        async def _wrapped(ctx: CallContext, header: dict, body: bytes) -> tuple[dict, bytes]:
            """Wrapped handler checking gatekeepers and rate limits."""
            for gk in list(self._gatekeepers):
                refusal = gk(ctx.peer_id)
                if refusal is not None:
                    raise RpcError("denied", str(refusal), peer=ctx.peer_id)

            kind = method.split(".", 1)[0]
            if kind not in self._limited:
                kind = "*"
            if not self._rate_limiter.allow(ctx.peer_id, kind):
                raise RpcError("denied", "rate limited", peer=ctx.peer_id)

            if blocking:
                return await self.loop.run_in_executor(self._executor, fn, ctx, header, body)
            else:
                return await fn(ctx, header, body)

        return _wrapped

    async def _builtin_ping(
        self, ctx: CallContext, header: dict, body: bytes
    ) -> tuple[dict, bytes]:
        """Built-in handler for node.ping returning current unix timestamp."""
        return {"t": time.time()}, b""

    async def _builtin_hello(
        self, ctx: CallContext, header: dict, body: bytes
    ) -> tuple[dict, bytes]:
        """Built-in handler for node.hello exchanging signed peer records."""
        rec_dict = header.get("record")
        if isinstance(rec_dict, dict):
            try:
                rec = PeerRecord.from_dict(rec_dict)
                if rec.peer_id == ctx.peer_id:
                    self.add_record(rec)
            except Exception as exc:
                logger.debug("invalid record in node.hello from %s: %s", ctx.peer_id, exc)

        res_header = {
            "record": self.record().to_dict(),
            "observed": ctx.remote,
        }
        return res_header, b""

    # --- peers ---------------------------------------------------------------

    def connect(
        self, address: str, *, expect_peer: str | None = None, timeout: float | None = None
    ) -> str:
        """Connect to a peer at address synchronously and return its peer id."""
        return self.run(self.aconnect(address, expect_peer=expect_peer, timeout=timeout))

    async def _open_tcp(
        self, addr: TcpAddress, expect_peer: str | None, timeout: float
    ) -> SecureChannel:
        """Open a TCP stream and initiate a SecureChannel."""
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(addr.host, addr.port),
            timeout=timeout,
        )
        stream = TcpStream(reader, writer)
        try:
            return await initiate(
                stream,
                self._identity,
                psk=self._config.psk,
                expect_peer=expect_peer,
                min_pow_bits=self._config.min_pow_bits,
                timeout=self._config.handshake_timeout,
            )
        except BaseException:
            await stream.close()
            raise

    async def _dial_relay(self, addr: RelayAddress, peer_id: str, timeout: float) -> SecureChannel:
        """Dial through registered dialers to open a relay SecureChannel."""
        stream = None
        last_exc: Exception | None = None
        for dialer in list(self._dialers):
            try:
                stream = await dialer(self, peer_id, addr, timeout)
                if stream is not None:
                    break
            except Exception as exc:
                last_exc = exc
        if stream is None:
            if last_exc is not None:
                raise ConnectionError(f"relay dialer failed: {last_exc}") from last_exc
            raise ConnectionError(f"no dialer could handle relay address {addr}")
        try:
            return await initiate(
                stream,
                self._identity,
                psk=self._config.psk,
                expect_peer=peer_id,
                min_pow_bits=self._config.min_pow_bits,
                timeout=self._config.handshake_timeout,
            )
        except BaseException:
            await stream.close()
            raise

    async def aconnect(
        self, address: str, *, expect_peer: str | None = None, timeout: float | None = None
    ) -> str:
        """Connect to a peer at address asynchronously and return its peer id."""
        parsed = parse_address(address)
        conn_timeout = timeout if timeout is not None else self._config.connect_timeout

        if isinstance(parsed, TcpAddress):
            ch = await self._open_tcp(parsed, expect_peer, conn_timeout)
            conn = await self._adopt_channel(ch, inbound=False)
            return conn.peer_id

        elif isinstance(parsed, RelayAddress):
            if expect_peer is None:
                raise ValueError("connecting via relay requires expect_peer")
            ch = await self._dial_relay(parsed, expect_peer, conn_timeout)
            conn = await self._adopt_channel(ch, inbound=False, via_relay=parsed.relay)
            return conn.peer_id
        else:
            raise ValueError(f"unsupported address type: {type(parsed).__name__}")

    async def accept_stream(self, stream: ByteStream, *, via_relay: str | None = None) -> None:
        """Run the responder side of the handshake on ``stream`` and adopt the connection
        (used by the relay for circuits arriving through it)."""
        ch = await accept(
            stream,
            self._identity,
            psk=self._config.psk,
            min_pow_bits=self._config.min_pow_bits,
            timeout=self._config.handshake_timeout,
        )
        await self._adopt_channel(ch, inbound=True, via_relay=via_relay)

    async def _adopt_channel(
        self,
        channel: SecureChannel,
        *,
        inbound: bool,
        via_relay: str | None = None,
    ) -> Connection:
        """Run gatekeepers, register the connection, fire callbacks, and perform hello."""
        with self._conn_lock:
            total = sum(len(entries) for entries in self._connections.values())
            over_limit = total >= self._config.max_connections

        if over_limit:
            await channel.close()
            if inbound:
                raise ConnectionRefusedError("too many connections")
            else:
                raise RpcError("unavailable", "too many connections", peer=channel.peer_id)

        for gk in list(self._gatekeepers):
            refusal = gk(channel.peer_id)
            if refusal is not None:
                await channel.close()
                raise ConnectionRefusedError(f"gatekeeper refused {channel.peer_id}: {refusal}")

        async def _on_extension_wrapper(kind: int, ident: int, header: dict, body: bytes) -> None:
            """Dispatch received extension frames to the node extension handler."""
            if self._extension_handler is not None:
                await self._extension_handler(conn, kind, ident, header, body)

        conn = Connection(
            channel,
            self._lookup_handler,
            ping_interval=self._config.ping_interval,
            dead_after=self._config.dead_after,
            max_inflight=self._config.max_inflight_per_connection,
            on_close=self._on_connection_close,
            on_extension=_on_extension_wrapper,
        )

        await conn.start()

        oldest_to_close: Connection | None = None
        with self._conn_lock:
            total = sum(len(entries) for entries in self._connections.values())
            if total >= self._config.max_connections:
                over_limit_at_reg = True
            else:
                over_limit_at_reg = False
                entry = _ConnEntry(connection=conn, inbound=inbound, via_relay=via_relay)
                peer_conns = self._connections.setdefault(conn.peer_id, [])
                peer_conns.append(entry)
                if len(peer_conns) > 2:
                    # Drop the oldest connection with nothing in flight. If every older
                    # one is busy (a reachability dial-back arriving while we wait on
                    # that very peer), the newcomer is the one to go.
                    idle = [e for e in peer_conns[:-1] if e.connection.pending_calls == 0]
                    victim = idle[0] if idle else peer_conns[-1]
                    peer_conns.remove(victim)
                    oldest_to_close = victim.connection

        if oldest_to_close is conn:
            # Every older connection to this peer is busy: keep them, use one of them.
            await conn.close("superseded")
            if inbound:
                return conn
            kept = self.connection(conn.peer_id)
            if kept is None:
                raise RpcError("unavailable", "connection superseded", peer=conn.peer_id)
            return kept
        if oldest_to_close is not None:
            self._spawn(oldest_to_close.close("superseded"))

        if over_limit_at_reg:
            await conn.close("too many connections")
            if inbound:
                raise ConnectionRefusedError("too many connections")
            else:
                raise RpcError("unavailable", "too many connections", peer=channel.peer_id)

        if not inbound:
            try:
                res_header, _ = await conn.call(
                    "node.hello",
                    {"record": self.record().to_dict()},
                    timeout=5.0,
                )
                if isinstance(res_header.get("record"), dict):
                    try:
                        reply_rec = PeerRecord.from_dict(res_header["record"])
                        if reply_rec.peer_id == conn.peer_id:
                            self.add_record(reply_rec)
                    except Exception as exc:
                        logger.debug("invalid record in hello reply: %s", exc)
                if isinstance(res_header.get("observed"), str):
                    self._record_observed(conn.peer_id, res_header["observed"])
            except RpcError as exc:
                if conn.closed or exc.code == "unavailable":
                    # The other side hung up straight after the handshake: it refused us
                    # (a ban, too little proof of work, too many connections).
                    await conn.close("refused by peer")
                    raise ConnectionRefusedError(
                        f"{conn.peer_id[:8]} closed the connection right after the handshake "
                        "(it refused this device)"
                    ) from exc
                logger.debug("hello exchange failed with %s: %s", conn.peer_id, exc)
            except Exception as exc:
                logger.debug("hello exchange error with %s: %s", conn.peer_id, exc)

        with self._conn_lock:
            fresh = not conn.closed and conn.peer_id not in self._announced
            if fresh:
                self._announced.add(conn.peer_id)
        for cb in list(self._on_connect_cbs) if fresh else []:

            def _fire(callback: Callable[[str], None] = cb, pid: str = conn.peer_id) -> None:
                """Invoke an on_connect callback safely."""
                try:
                    callback(pid)
                except Exception:
                    logger.exception("on_connect callback failed")

            self.loop.call_soon(_fire)

        return conn

    def _on_connection_close(self, conn: Connection) -> None:
        """Handle a connection closing, removing it and notifying disconnect listeners."""
        gone = False
        with self._conn_lock:
            entries = self._connections.get(conn.peer_id)
            if entries:
                self._connections[conn.peer_id] = [
                    e for e in entries if e.connection is not conn and not e.connection.closed
                ]
                if not self._connections[conn.peer_id]:
                    del self._connections[conn.peer_id]
                    gone = True
            else:
                self._connections.pop(conn.peer_id, None)
                gone = True
            # A connection refused or dropped before it was announced is not a disconnect.
            disconnected = gone and conn.peer_id in self._announced
            if gone:
                self._announced.discard(conn.peer_id)

        if gone:
            with self._observed_lock:
                prev = self._peer_observed.pop(conn.peer_id, None)
                if prev is not None:
                    self._observed_counts[prev] = self._observed_counts.get(prev, 1) - 1
                    if self._observed_counts[prev] <= 0:
                        self._observed_counts.pop(prev, None)

            for cb in list(self._on_disconnect_cbs) if disconnected else []:

                def _fire(callback: Callable[[str], None] = cb, pid: str = conn.peer_id) -> None:
                    """Invoke an on_disconnect callback safely."""
                    try:
                        callback(pid)
                    except Exception:
                        logger.exception("on_disconnect callback failed")

                if self._loop is not None and self._loop.is_running():
                    self._loop.call_soon(_fire)
                else:
                    _fire()

    async def _dial_peer(self, peer_id: str) -> Connection:
        """Dial a peer using addresses from its verified record."""
        with self._conn_lock:
            total = sum(len(entries) for entries in self._connections.values())
            over_limit = total >= self._config.max_connections

        if over_limit:
            raise RpcError("unavailable", "too many connections", peer=peer_id)

        rec = self.record_of(peer_id)
        if rec is None:
            raise RpcError("unavailable", f"no known address for {peer_id[:8]}", peer=peer_id)

        parsed_addrs = rec.parsed_addresses()
        tcp_addrs = [a for a in parsed_addrs if isinstance(a, TcpAddress)]
        relay_addrs = [a for a in parsed_addrs if isinstance(a, RelayAddress)]

        if not tcp_addrs and not relay_addrs:
            raise RpcError(
                "unavailable",
                f"could not reach {peer_id[:8]}: no valid addresses in record",
                peer=peer_id,
            )

        last_error: Exception | None = None
        channel: SecureChannel | None = None
        via_relay: str | None = None

        if tcp_addrs:
            tasks = [
                asyncio.create_task(
                    self._open_tcp(a, peer_id, self._config.connect_timeout),
                    name=f"dial-tcp-{a}",
                )
                for a in tcp_addrs
            ]
            pending = set(tasks)
            try:
                while pending:
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    for t in done:
                        try:
                            ch = t.result()
                            if channel is None:
                                channel = ch
                            else:
                                self._spawn(ch.close())
                        except Exception as exc:
                            last_error = exc

                    if channel is not None:
                        break
            finally:
                for t in pending:
                    t.cancel()
                if pending:

                    async def _drain_extra(rem: set[asyncio.Task[SecureChannel]]) -> None:
                        """Close channels from cancelled pending tasks."""
                        res = await asyncio.gather(*rem, return_exceptions=True)
                        for r in res:
                            if isinstance(r, SecureChannel):
                                await r.close()

                    self._spawn(_drain_extra(pending))

        if channel is None and relay_addrs:
            for r_addr in relay_addrs:
                try:
                    channel = await self._dial_relay(r_addr, peer_id, self._config.connect_timeout)
                    via_relay = r_addr.relay
                    break
                except Exception as exc:
                    last_error = exc

        if channel is None:
            raise RpcError(
                "unavailable",
                f"could not reach {peer_id[:8]}: {last_error}",
                peer=peer_id,
            )

        try:
            return await self._adopt_channel(channel, inbound=False, via_relay=via_relay)
        except (ConnectionError, OSError) as exc:
            raise RpcError(
                "unavailable",
                f"could not reach {peer_id[:8]}: {exc}",
                peer=peer_id,
            ) from exc

    def disconnect(self, peer_id: str) -> None:
        """Close all live connections to peer_id."""
        with self._conn_lock:
            entries = self._connections.pop(peer_id, [])

        if entries:

            async def _close_entries() -> None:
                """Close removed connection entries."""
                for entry in entries:
                    await entry.connection.close("disconnect requested")

            if self._loop is not None and self._loop.is_running():
                if threading.current_thread() == self._thread:
                    self._loop.create_task(_close_entries())
                else:
                    fut = asyncio.run_coroutine_threadsafe(_close_entries(), self._loop)
                    try:
                        fut.result(timeout=5.0)
                    except Exception as exc:
                        logger.debug("error closing connections on disconnect: %s", exc)

    def is_connected(self, peer_id: str) -> bool:
        """Return True if at least one live connection exists to peer_id."""
        with self._conn_lock:
            entries = self._connections.get(peer_id, [])
            return any(not e.connection.closed for e in entries)

    def connections(self) -> list[ConnectionInfo]:
        """Return metadata for all currently active connections."""
        infos: list[ConnectionInfo] = []
        with self._conn_lock:
            for entries in self._connections.values():
                for entry in entries:
                    if not entry.connection.closed:
                        infos.append(
                            ConnectionInfo(
                                peer_id=entry.connection.peer_id,
                                remote=entry.connection.remote,
                                inbound=entry.inbound,
                                rtt_ms=entry.connection.rtt_ms,
                                bytes_sent=entry.connection.bytes_sent,
                                bytes_received=entry.connection.bytes_received,
                                via_relay=entry.via_relay,
                            )
                        )
        return infos

    def connected_peers(self) -> list[str]:
        """Return a list of all distinct peer ids with at least one active connection."""
        with self._conn_lock:
            return [
                peer_id
                for peer_id, entries in self._connections.items()
                if any(not e.connection.closed for e in entries)
            ]

    # --- records -------------------------------------------------------------

    def addresses(self) -> list[str]:
        """What this node advertises: ``advertise`` + local interface addresses (if
        ``advertise_local``) + ``relay_addresses``, deduplicated, at most 16."""
        candidates: list[str] = []
        candidates.extend(self._config.advertise)
        candidates.extend(self._advertised)

        # Nothing local to advertise until the listener has a real port.
        if self._config.advertise_local and self._listen_port > 0:
            port = self._listen_port
            is_loopback = False
            try:
                ip = ipaddress.ip_address(self._config.listen_host)
                is_loopback = ip.is_loopback
            except ValueError:
                is_loopback = self._config.listen_host in ("localhost", "127.0.0.1", "::1")

            if is_loopback:
                candidates.append(f"tcp://127.0.0.1:{port}")
            else:
                lan = lan_address()
                if lan != "127.0.0.1":
                    candidates.append(f"tcp://{lan}:{port}")
                ts = tailscale_address()
                if ts is not None:
                    candidates.append(f"tcp://{ts}:{port}")

        candidates.extend(self._relay_addresses)

        seen: set[str] = set()
        result: list[str] = []
        for addr in candidates:
            if addr not in seen:
                seen.add(addr)
                result.append(addr)
            if len(result) >= MAX_ADDRESSES:
                break
        return result

    def set_relay_addresses(self, addresses: list[str]) -> None:
        """Update relay addresses advertised by this node."""
        for addr in addresses:
            parse_address(addr)
        with self._record_lock:
            if self._relay_addresses != list(addresses):
                self._relay_addresses = list(addresses)
                self._record_dirty = True

    def set_advertised(self, addresses: list[str]) -> None:
        """Update extra addresses advertised by this node."""
        for addr in addresses:
            parse_address(addr)
        with self._record_lock:
            if self._advertised != list(addresses):
                self._advertised = list(addresses)
                self._record_dirty = True

    def update_advertised(self, change: Callable[[list[str]], list[str]]) -> list[str]:
        """Replace the extra advertised addresses with ``change(current)``, atomically,
        so two services adjusting them at once never lose each other's update."""
        with self._record_lock:
            new = list(change(list(self._advertised)))
            for addr in new:
                parse_address(addr)
            if self._advertised != new:
                self._advertised = new
                self._record_dirty = True
            return new

    def set_attr(self, key: str, value: Any) -> None:
        """Set a metadata attribute in this node's signed record."""
        if not isinstance(key, str) or len(key) > 32:
            raise ValueError(f"invalid attribute key: {key!r}")
        with self._record_lock:
            if self._attrs.get(key) != value:
                self._attrs[key] = value
                self._record_dirty = True

    def record(self) -> PeerRecord:
        """This node's signed record; re-signed with a higher ``seq`` whenever its
        addresses or attrs change."""
        with self._record_lock:
            addrs = self.addresses()
            if self._cached_record is None or self._record_dirty or self._last_addresses != addrs:
                seq = max(self._record_seq + 1, int(time.time() * 1000))
                self._record_seq = seq
                self._cached_record = make_record(
                    self._identity,
                    addrs,
                    seq=seq,
                    attrs=self._attrs,
                )
                self._last_addresses = addrs
                self._record_dirty = False
            return self._cached_record

    def add_record(self, record: PeerRecord) -> bool:
        """Keep ``record`` if it verifies and is newer than what we have."""
        if not record.verify():
            return False
        if record.peer_id == self._peer_id:
            return False

        with self._records_lock:
            existing = self._records.get(record.peer_id)
            if not record.newer_than(existing):
                return False
            self._records[record.peer_id] = record
            while len(self._records) > 10_000:
                oldest_peer = min(
                    self._records.values(), key=lambda r: (r.issued_at, r.seq)
                ).peer_id
                del self._records[oldest_peer]
            return True

    def record_of(self, peer_id: str) -> PeerRecord | None:
        """Return the known verified record for peer_id, or None."""
        if peer_id == self._peer_id:
            return self.record()
        with self._records_lock:
            return self._records.get(peer_id)

    def records(self) -> list[PeerRecord]:
        """Return all stored peer records."""
        with self._records_lock:
            return list(self._records.values())

    def observed_addresses(self) -> dict[str, int]:
        """``"ip:port"`` as peers report seeing this node, with how many peers said so."""
        with self._observed_lock:
            return dict(self._observed_counts)

    def _record_observed(self, peer_id: str, observed: str) -> None:
        """Record the address a peer observed this node at."""
        with self._observed_lock:
            if not isinstance(observed, str) or not observed:
                return
            prev = self._peer_observed.get(peer_id)
            if prev == observed:
                return
            if prev is not None:
                self._observed_counts[prev] = self._observed_counts.get(prev, 1) - 1
                if self._observed_counts[prev] <= 0:
                    self._observed_counts.pop(prev, None)
            self._peer_observed[peer_id] = observed
            self._observed_counts[observed] = self._observed_counts.get(observed, 0) + 1

    # --- extension points ----------------------------------------------------

    def add_dialer(self, dialer: Dialer) -> None:
        """Register a dialer for non-TCP addresses."""
        self._dialers.append(dialer)

    def add_gatekeeper(self, gatekeeper: Gatekeeper) -> None:
        """Register a gatekeeper to inspect and possibly refuse connections."""
        self._gatekeepers.append(gatekeeper)

    def on_connect(self, callback: Callable[[str], None]) -> None:
        """Register a callback invoked when a connection to a peer is established."""
        self._on_connect_cbs.append(callback)

    def on_disconnect(self, callback: Callable[[str], None]) -> None:
        """Register a callback invoked when all connections to a peer are closed."""
        self._on_disconnect_cbs.append(callback)

    def connection(self, peer_id: str) -> Connection | None:
        """The newest live connection to ``peer_id`` (for the relay), or ``None``."""
        with self._conn_lock:
            entries = self._connections.get(peer_id, [])
            for entry in reversed(entries):
                if not entry.connection.closed:
                    return entry.connection
        return None

    def set_extension_handler(self, handler: NodeExtensionHandler) -> None:
        """Receive extension frames (types 16-63) from every connection, with the
        connection they arrived on. Only one handler; setting it twice is ``ValueError``."""
        if self._extension_handler is not None:
            raise ValueError("extension handler already set")
        self._extension_handler = handler
