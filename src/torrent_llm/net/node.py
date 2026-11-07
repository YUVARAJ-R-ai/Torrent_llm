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
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, TypeVar

from torrent_llm.net.identity import Identity
from torrent_llm.net.ratelimit import Limit
from torrent_llm.net.records import Address, PeerRecord
from torrent_llm.net.rpc import CallContext, Connection, Handler
from torrent_llm.net.stream import ByteStream

T = TypeVar("T")

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
        raise NotImplementedError

    # --- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> Node:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def peer_id(self) -> str:
        raise NotImplementedError

    @property
    def identity(self) -> Identity:
        raise NotImplementedError

    @property
    def config(self) -> NodeConfig:
        raise NotImplementedError

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        raise NotImplementedError

    @property
    def listen_port(self) -> int:
        raise NotImplementedError

    def run(self, coro: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
        """Run ``coro`` on the node's loop from another thread and return its result."""
        raise NotImplementedError

    # --- methods -------------------------------------------------------------

    def register(
        self, method: str, handler: SyncHandler | Handler, *, blocking: bool = True
    ) -> None:
        """Serve ``method``. ``blocking=True``: a plain function run on the thread pool;
        ``False``: an ``async def`` run on the loop. ``ValueError`` for a bad name or a
        name registered twice."""
        raise NotImplementedError

    def unregister(self, method: str) -> None:
        raise NotImplementedError

    def call(
        self,
        peer_id: str,
        method: str,
        header: dict | None = None,
        body: bytes = b"",
        *,
        timeout: float,
    ) -> tuple[dict, bytes]:
        raise NotImplementedError

    async def acall(
        self,
        peer_id: str,
        method: str,
        header: dict | None = None,
        body: bytes = b"",
        *,
        timeout: float,
    ) -> tuple[dict, bytes]:
        raise NotImplementedError

    # --- peers ---------------------------------------------------------------

    def connect(
        self, address: str, *, expect_peer: str | None = None, timeout: float | None = None
    ) -> str:
        raise NotImplementedError

    async def aconnect(
        self, address: str, *, expect_peer: str | None = None, timeout: float | None = None
    ) -> str:
        raise NotImplementedError

    async def accept_stream(self, stream: ByteStream, *, via_relay: str | None = None) -> None:
        """Run the responder side of the handshake on ``stream`` and adopt the connection
        (used by the relay for circuits arriving through it)."""
        raise NotImplementedError

    def disconnect(self, peer_id: str) -> None:
        raise NotImplementedError

    def is_connected(self, peer_id: str) -> bool:
        raise NotImplementedError

    def connections(self) -> list[ConnectionInfo]:
        raise NotImplementedError

    def connected_peers(self) -> list[str]:
        raise NotImplementedError

    # --- records -------------------------------------------------------------

    def addresses(self) -> list[str]:
        """What this node advertises: ``advertise`` + local interface addresses (if
        ``advertise_local``) + ``relay_addresses``, deduplicated, at most 16."""
        raise NotImplementedError

    def set_relay_addresses(self, addresses: list[str]) -> None:
        raise NotImplementedError

    def set_advertised(self, addresses: list[str]) -> None:
        raise NotImplementedError

    def set_attr(self, key: str, value: Any) -> None:
        raise NotImplementedError

    def record(self) -> PeerRecord:
        """This node's signed record; re-signed with a higher ``seq`` whenever its
        addresses or attrs change."""
        raise NotImplementedError

    def add_record(self, record: PeerRecord) -> bool:
        """Keep ``record`` if it verifies and is newer than what we have."""
        raise NotImplementedError

    def record_of(self, peer_id: str) -> PeerRecord | None:
        raise NotImplementedError

    def records(self) -> list[PeerRecord]:
        raise NotImplementedError

    def observed_addresses(self) -> dict[str, int]:
        """``"ip:port"`` as peers report seeing this node, with how many peers said so."""
        raise NotImplementedError

    # --- extension points ----------------------------------------------------

    def add_dialer(self, dialer: Dialer) -> None:
        raise NotImplementedError

    def add_gatekeeper(self, gatekeeper: Gatekeeper) -> None:
        raise NotImplementedError

    def on_connect(self, callback: Callable[[str], None]) -> None:
        raise NotImplementedError

    def on_disconnect(self, callback: Callable[[str], None]) -> None:
        raise NotImplementedError

    def connection(self, peer_id: str) -> Connection | None:
        """The newest live connection to ``peer_id`` (for the relay), or ``None``."""
        raise NotImplementedError

    def set_extension_handler(self, handler: NodeExtensionHandler) -> None:
        """Receive extension frames (types 16-63) from every connection, with the
        connection they arrived on. Only one handler; setting it twice is ``ValueError``."""
        raise NotImplementedError
