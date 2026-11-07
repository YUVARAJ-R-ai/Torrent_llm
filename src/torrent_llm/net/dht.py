"""Finding peers with no server: a Kademlia distributed hash table over the overlay.

Every node has a 256-bit key (sha256 of its peer id) and keeps a routing table of
peers in buckets by XOR distance to its own key. To find who is close to some key it
asks the closest peers it knows, who answer with closer ones, ``ALPHA`` at a time,
until nobody closer turns up (an *iterative lookup*). A swarm is a key too
(``topic_key``): its members announce themselves as *providers* of that key on the
``K`` nodes closest to it, and a newcomer looks the key up to find them. So any
node of the swarm can bootstrap a newcomer, and no node is special.

Methods (all bodies empty, all fields in the header):

======================  ================================================================
``dht.find_node``       ``{"key": hex} -> {"peers": [record dict, ...]}`` (<= K closest)
``dht.add_provider``    ``{"key": hex, "record": record dict, "ttl": s} -> {}``
``dht.get_providers``   ``{"key": hex} -> {"providers": [...], "peers": [closer, ...]}``
======================  ================================================================

A provider record must verify and belong to the caller (``record.peer_id ==
ctx.peer_id``): a node can only announce itself, never others. TTL is clamped to
``MAX_TTL_S``. Records received in replies are verified before use and added to the
node's address book (``node.add_record``) so they can be dialled.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from torrent_llm.net.node import Node
from torrent_llm.net.records import PeerRecord

K = 16
ALPHA = 3
KEY_BYTES = 32
MAX_TTL_S = 3600
PROVIDE_TTL_S = 1800
REPROVIDE_S = 600


def node_key(peer_id: str) -> bytes:
    return hashlib.sha256(peer_id.encode("ascii")).digest()


def topic_key(swarm_id: str) -> bytes:
    return hashlib.sha256(b"tllm-swarm" + swarm_id.encode("utf-8")).digest()


def distance(a: bytes, b: bytes) -> int:
    return int.from_bytes(a, "big") ^ int.from_bytes(b, "big")


class RoutingTable:
    """256 buckets of at most ``k`` records, by XOR distance from ``own``.

    * ``add`` puts a new peer at the tail of its bucket, or moves a known one there (and
      replaces its record if newer). A full bucket keeps its old peers unless the
      least-recently-seen one has not been seen for ``stale_s`` (then it is replaced):
      Kademlia's preference for long-lived nodes, which also resists eclipse attacks.
      Never adds ``own``. Returns whether the peer is now in the table.
    * ``closest(key, n)`` returns up to ``n`` records sorted by distance to ``key``.
    """

    def __init__(
        self,
        own: bytes,
        *,
        k: int = K,
        stale_s: float = 900.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        raise NotImplementedError

    def add(self, record: PeerRecord) -> bool:
        raise NotImplementedError

    def remove(self, peer_id: str) -> None:
        raise NotImplementedError

    def seen(self, peer_id: str) -> None:
        raise NotImplementedError

    def closest(
        self, key: bytes, n: int = K, *, exclude: frozenset[str] = frozenset()
    ) -> list[PeerRecord]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class ProviderStore:
    """Provider records by key, expiring. At most ``max_keys`` keys and ``max_per_key``
    providers per key (the oldest expiry goes first)."""

    def __init__(
        self,
        *,
        max_keys: int = 10_000,
        max_per_key: int = 64,
        clock: Callable[[], float] | None = None,
    ) -> None:
        raise NotImplementedError

    def add(self, key: bytes, record: PeerRecord, ttl_s: float) -> None:
        raise NotImplementedError

    def get(self, key: bytes) -> list[PeerRecord]:
        raise NotImplementedError

    def expire(self) -> int:
        raise NotImplementedError


class Dht:
    """The DHT for one node. ``attach()`` registers the three methods and adds every
    connected peer's record to the routing table (``node.on_connect``); peers whose calls
    fail are removed. The async methods run on ``node.loop``; the sync ones wrap them.
    """

    def __init__(
        self, node: Node, *, k: int = K, alpha: int = ALPHA, rpc_timeout: float = 3.0
    ) -> None:
        raise NotImplementedError

    def attach(self) -> None:
        raise NotImplementedError

    @property
    def table(self) -> RoutingTable:
        raise NotImplementedError

    async def afind_node(self, key: bytes) -> list[PeerRecord]:
        raise NotImplementedError

    async def aprovide(self, key: bytes, ttl_s: float = PROVIDE_TTL_S) -> int:
        """Announce this node on the K closest nodes; returns how many accepted."""
        raise NotImplementedError

    async def afind_providers(self, key: bytes, limit: int = 32) -> list[PeerRecord]:
        raise NotImplementedError

    async def abootstrap(self) -> int:
        """Look up our own key to fill the table; returns the table size."""
        raise NotImplementedError

    def find_node(self, key: bytes) -> list[PeerRecord]:
        raise NotImplementedError

    def provide(self, key: bytes, ttl_s: float = PROVIDE_TTL_S) -> int:
        raise NotImplementedError

    def find_providers(self, key: bytes, limit: int = 32) -> list[PeerRecord]:
        raise NotImplementedError

    def bootstrap(self) -> int:
        raise NotImplementedError
