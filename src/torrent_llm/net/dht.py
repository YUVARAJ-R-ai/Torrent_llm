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

import asyncio
import hashlib
import json
import logging
import math
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from torrent_llm.net.node import Node
from torrent_llm.net.records import PeerRecord
from torrent_llm.net.rpc import MAX_HEADER_BYTES, CallContext, RpcError

logger = logging.getLogger(__name__)

K = 16
ALPHA = 3
KEY_BYTES = 32
MAX_TTL_S = 3600
PROVIDE_TTL_S = 1800
REPROVIDE_S = 600

_MAX_KEYS_PER_PROVIDER = 64
_HEADER_BUDGET = MAX_HEADER_BYTES - 4096


def node_key(peer_id: str) -> bytes:
    """Return the 256-bit key for a peer id (sha256 of its ascii bytes)."""
    return hashlib.sha256(peer_id.encode("ascii")).digest()


def topic_key(swarm_id: str) -> bytes:
    """Return the 256-bit topic key for a swarm."""
    return hashlib.sha256(b"tllm-swarm" + swarm_id.encode("utf-8")).digest()


def distance(a: bytes, b: bytes) -> int:
    """Return XOR distance between two equal-length byte sequences as an integer."""
    return int.from_bytes(a, "big") ^ int.from_bytes(b, "big")


def _parse_hex_key(raw: Any) -> bytes:
    """Validate and parse a 64-hex-char key string from an RPC header."""
    if not isinstance(raw, str) or len(raw) != 64:
        raise RpcError("bad_request", "key must be 64 hex characters")
    try:
        return bytes.fromhex(raw)
    except ValueError as exc:
        raise RpcError("bad_request", "key must be 64 hex characters") from exc


def _verified_records(raw: object, cap: int) -> list[PeerRecord]:
    """Parse and verify up to cap records from raw list, ignoring invalid ones."""
    if not isinstance(raw, list) or cap <= 0:
        return []
    records: list[PeerRecord] = []
    for item in raw[:cap]:
        if not isinstance(item, dict):
            continue
        try:
            rec = PeerRecord.from_dict(item)
            if rec.verify():
                records.append(rec)
        except Exception:
            continue
    return records


def _fit_records(records: list[PeerRecord], budget: int) -> tuple[list[dict], int]:
    """Convert records to dicts until total serialized JSON size exceeds budget."""
    fitted: list[dict] = []
    used = 0
    for r in records:
        d = r.to_dict()
        size = len(json.dumps(d, separators=(",", ":")).encode("utf-8")) + 2
        if used + size > budget:
            break
        fitted.append(d)
        used += size
    return fitted, used


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
        self._own = own
        self._k = k
        self._stale_s = stale_s
        self._clock = clock if clock is not None else time.monotonic
        self._buckets: list[OrderedDict[str, tuple[PeerRecord, float]]] = [
            OrderedDict() for _ in range(256)
        ]

    def _bucket(self, peer_id: str) -> OrderedDict[str, tuple[PeerRecord, float]] | None:
        """Return bucket for peer_id, or None if peer is own."""
        pkey = node_key(peer_id)
        dist = distance(self._own, pkey)
        if dist == 0:
            return None
        idx = dist.bit_length() - 1
        return self._buckets[idx]

    def add(self, record: PeerRecord) -> bool:
        """Add record or update known peer. Returns True if peer is now in table."""
        bucket = self._bucket(record.peer_id)
        if bucket is None:
            return False

        now = self._clock()

        if record.peer_id in bucket:
            old_rec, _ = bucket[record.peer_id]
            rec = record if record.newer_than(old_rec) else old_rec
            bucket.move_to_end(record.peer_id)
            bucket[record.peer_id] = (rec, now)
            return True

        if len(bucket) < self._k:
            bucket[record.peer_id] = (record, now)
            return True

        lrs_peer, (_, lrs_seen) = next(iter(bucket.items()))
        if now - lrs_seen >= self._stale_s:
            del bucket[lrs_peer]
            bucket[record.peer_id] = (record, now)
            return True

        return False

    def remove(self, peer_id: str) -> None:
        """Remove a peer from the routing table if present."""
        bucket = self._bucket(peer_id)
        if bucket is not None:
            bucket.pop(peer_id, None)

    def seen(self, peer_id: str) -> None:
        """Mark a peer as recently seen and move to bucket tail if present."""
        bucket = self._bucket(peer_id)
        if bucket is not None and peer_id in bucket:
            rec, _ = bucket[peer_id]
            bucket.move_to_end(peer_id)
            bucket[peer_id] = (rec, self._clock())

    def closest(
        self, key: bytes, n: int = K, *, exclude: frozenset[str] = frozenset()
    ) -> list[PeerRecord]:
        """Return up to n records closest to key, excluding any in exclude."""
        if n <= 0:
            return []
        candidates: list[PeerRecord] = []
        for bucket in self._buckets:
            for pid, (rec, _) in bucket.items():
                if pid not in exclude:
                    candidates.append(rec)
        candidates.sort(key=lambda r: distance(key, node_key(r.peer_id)))
        return candidates[:n]

    def __len__(self) -> int:
        """Return total number of records across all buckets."""
        return sum(len(b) for b in self._buckets)


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
        self._max_keys = max_keys
        self._max_per_key = max_per_key
        self._clock = clock if clock is not None else time.monotonic
        self._store: dict[bytes, dict[str, tuple[PeerRecord, float]]] = {}
        self._provider_keys: dict[str, set[bytes]] = {}

    def _untrack_provider_key(self, peer_id: str, key: bytes) -> None:
        """Remove a key from a provider's tracked set."""
        keys = self._provider_keys.get(peer_id)
        if keys is not None:
            keys.discard(key)
            if not keys:
                self._provider_keys.pop(peer_id, None)

    def keys_for_provider(self, peer_id: str) -> int:
        """Return how many distinct keys peer_id currently provides."""
        return len(self._provider_keys.get(peer_id, ()))

    def provides_key(self, peer_id: str, key: bytes) -> bool:
        """Return True if peer_id is currently tracked as providing key."""
        return key in self._provider_keys.get(peer_id, ())

    def add(self, key: bytes, record: PeerRecord, ttl_s: float) -> None:
        """Store a provider record for key with a TTL in seconds."""
        now = self._clock()
        expires_at = now + ttl_s

        if key in self._store:
            providers = self._store[key]
        else:
            if len(self._store) >= self._max_keys:
                self.expire()
                if len(self._store) >= self._max_keys:
                    # The least-provided key goes first: a real swarm key has many
                    # providers, keys made up by one identity have one each.
                    oldest_key = min(
                        self._store.keys(),
                        key=lambda k: (
                            len(self._store[k]),
                            min((exp for _, exp in self._store[k].values()), default=0.0),
                        ),
                    )
                    for pid in list(self._store[oldest_key].keys()):
                        self._untrack_provider_key(pid, oldest_key)
                    del self._store[oldest_key]
            providers = {}
            self._store[key] = providers

        if record.peer_id in providers:
            providers[record.peer_id] = (record, expires_at)
        else:
            if len(providers) >= self._max_per_key:
                expired = [pid for pid, (_, exp) in providers.items() if exp <= now]
                for pid in expired:
                    del providers[pid]
                    self._untrack_provider_key(pid, key)
            if len(providers) >= self._max_per_key:
                oldest_pid = min(providers.keys(), key=lambda pid: providers[pid][1])
                del providers[oldest_pid]
                self._untrack_provider_key(oldest_pid, key)
            providers[record.peer_id] = (record, expires_at)
            self._provider_keys.setdefault(record.peer_id, set()).add(key)

    def get(self, key: bytes) -> list[PeerRecord]:
        """Return unexpired provider records for key, dropping expired ones."""
        now = self._clock()
        providers = self._store.get(key)
        if providers is None:
            return []
        expired = [pid for pid, (_, exp) in providers.items() if exp <= now]
        for pid in expired:
            del providers[pid]
            self._untrack_provider_key(pid, key)
        if not providers:
            self._store.pop(key, None)
            return []
        return [rec for rec, _ in providers.values()]

    def expire(self) -> int:
        """Sweep all expired provider records and empty keys; return count removed."""
        now = self._clock()
        removed = 0
        empty_keys: list[bytes] = []
        for key, providers in self._store.items():
            expired = [pid for pid, (_, exp) in providers.items() if exp <= now]
            removed += len(expired)
            for pid in expired:
                del providers[pid]
                self._untrack_provider_key(pid, key)
            if not providers:
                empty_keys.append(key)
        for key in empty_keys:
            self._store.pop(key, None)
        return removed


class Dht:
    """The DHT for one node. ``attach()`` registers the three methods and adds every
    connected peer's record to the routing table (``node.on_connect``); peers whose calls
    fail are removed. The async methods run on ``node.loop``; the sync ones wrap them.
    """

    def __init__(
        self, node: Node, *, k: int = K, alpha: int = ALPHA, rpc_timeout: float = 3.0
    ) -> None:
        self._node = node
        self._k = k
        self._alpha = alpha
        self._rpc_timeout = rpc_timeout
        self._table = RoutingTable(node_key(node.peer_id), k=k)
        self._provider_store = ProviderStore()
        self._attached = False

    def attach(self) -> None:
        """Register DHT handlers and listen for newly connected peers."""
        if self._attached:
            return
        self._attached = True

        self._node.register("dht.find_node", self._handle_find_node, blocking=False)
        self._node.register("dht.add_provider", self._handle_add_provider, blocking=False)
        self._node.register("dht.get_providers", self._handle_get_providers, blocking=False)

        def _on_connect(pid: str) -> None:
            rec = self._node.record_of(pid)
            if rec is not None:
                self._table.add(rec)

        self._node.on_connect(_on_connect)

        def _populate() -> None:
            for pid in self._node.connected_peers():
                rec = self._node.record_of(pid)
                if rec is not None:
                    self._table.add(rec)

        try:
            loop = self._node.loop
            if loop.is_running():
                loop.call_soon_threadsafe(_populate)
            else:
                _populate()
        except RuntimeError:  # the node has not started: nothing to schedule on yet
            pass

    @property
    def table(self) -> RoutingTable:
        """The underlying routing table."""
        return self._table

    def _touch_caller(self, ctx: CallContext) -> None:
        """Add caller's known verified record to routing table if available."""
        rec = self._node.record_of(ctx.peer_id)
        if rec is not None:
            self._table.add(rec)

    async def _handle_find_node(
        self, ctx: CallContext, header: dict, body: bytes
    ) -> tuple[dict, bytes]:
        """Handle dht.find_node request returning up to K closest known peers."""
        self._touch_caller(ctx)
        key = _parse_hex_key(header.get("key"))
        peers = self._table.closest(key, self._k)
        fitted_peers, _ = _fit_records(peers, _HEADER_BUDGET)
        return {"peers": fitted_peers}, b""

    async def _handle_add_provider(
        self, ctx: CallContext, header: dict, body: bytes
    ) -> tuple[dict, bytes]:
        """Handle dht.add_provider request storing caller as provider."""
        self._touch_caller(ctx)
        key = _parse_hex_key(header.get("key"))

        raw_rec = header.get("record")
        if not isinstance(raw_rec, dict):
            raise RpcError("denied", "invalid record")
        try:
            record = PeerRecord.from_dict(raw_rec)
        except Exception as exc:
            raise RpcError("denied", "invalid record") from exc

        if not record.verify():
            raise RpcError("denied", "unverified record")
        if record.peer_id != ctx.peer_id:
            raise RpcError("denied", "record does not match caller peer id")

        ttl = header.get("ttl")
        if not isinstance(ttl, (int, float)) or isinstance(ttl, bool) or not math.isfinite(ttl):
            raise RpcError("bad_request", "invalid ttl")
        ttl_clamped = max(1.0, min(float(ttl), float(MAX_TTL_S)))

        store = self._provider_store
        if not store.provides_key(ctx.peer_id, key):
            if store.keys_for_provider(ctx.peer_id) >= _MAX_KEYS_PER_PROVIDER:
                store.expire()  # only at the quota: expired keys must not count
            if store.keys_for_provider(ctx.peer_id) >= _MAX_KEYS_PER_PROVIDER:
                raise RpcError("denied", "too many keys for provider")

        self._node.add_record(record)
        self._table.add(record)
        self._provider_store.add(key, record, ttl_clamped)
        return {}, b""

    async def _handle_get_providers(
        self, ctx: CallContext, header: dict, body: bytes
    ) -> tuple[dict, bytes]:
        """Handle dht.get_providers request returning providers and closer peers."""
        self._touch_caller(ctx)
        key = _parse_hex_key(header.get("key"))
        providers = self._provider_store.get(key)
        fitted_provs, used_provs = _fit_records(providers, _HEADER_BUDGET)
        remaining = max(0, _HEADER_BUDGET - used_provs)
        closer_peers = self._table.closest(key, self._k)
        fitted_peers, _ = _fit_records(closer_peers, remaining)
        return {
            "providers": fitted_provs,
            "peers": fitted_peers,
        }, b""

    async def _iterative_lookup(
        self, key: bytes, *, is_providers: bool = False, limit: int = 32
    ) -> tuple[list[PeerRecord], list[PeerRecord]]:
        """Run iterative Kademlia lookup for key, finding closest nodes and/or providers."""
        if not isinstance(key, (bytes, bytearray)) or len(key) != KEY_BYTES:
            raise ValueError(f"key must be {KEY_BYTES} bytes")
        key_bytes = bytes(key)

        providers_found: dict[str, PeerRecord] = {}
        if is_providers:
            for rec in self._provider_store.get(key_bytes):
                providers_found[rec.peer_id] = rec
            if len(providers_found) >= limit:
                return [], list(providers_found.values())[:limit]

        candidates: dict[str, PeerRecord] = {
            r.peer_id: r
            for r in self._table.closest(key_bytes, self._k)
            if r.peer_id != self._node.peer_id
        }

        queried: set[str] = {self._node.peer_id}
        failed: set[str] = set()

        def dist(r: PeerRecord) -> int:
            return distance(key_bytes, node_key(r.peer_id))

        closest_seen_dist: int | None = (
            min(dist(r) for r in candidates.values()) if candidates else None
        )

        method = "dht.get_providers" if is_providers else "dht.find_node"
        req_header = {"key": key_bytes.hex()}

        while True:
            unqueried = [r for pid, r in candidates.items() if pid not in queried]
            if not unqueried:
                break
            unqueried.sort(key=dist)
            batch = unqueried[: self._alpha]
            for r in batch:
                queried.add(r.peer_id)
                # Only records about to be dialled go into the address book, so a flood
                # of fake records in replies cannot push honest ones out of it.
                self._node.add_record(r)

            async def _query_one(peer: PeerRecord) -> tuple[PeerRecord, dict | None, bool]:
                try:
                    res_h, _ = await self._node.acall(
                        peer.peer_id,
                        method,
                        req_header,
                        timeout=self._rpc_timeout,
                    )
                    return peer, res_h, False
                except RpcError as exc:
                    logger.debug("query to %s failed: %s", peer.peer_id[:8], exc)
                    return peer, None, _peer_seems_gone(exc)
                except Exception:  # a local bug, not the peer's fault: keep it
                    logger.exception("query to %s failed here", peer.peer_id[:8])
                    return peer, None, False

            results = await asyncio.gather(*[_query_one(r) for r in batch])

            for peer, res_header, should_remove in results:
                if res_header is None:
                    if should_remove:
                        failed.add(peer.peer_id)
                        self._table.remove(peer.peer_id)
                        candidates.pop(peer.peer_id, None)
                else:
                    self._table.add(peer)

                    if is_providers:
                        raw_provs = res_header.get("providers")
                        prov_recs = _verified_records(raw_provs, limit - len(providers_found))
                        for p_rec in prov_recs:
                            self._node.add_record(p_rec)
                            if p_rec.peer_id not in providers_found:
                                providers_found[p_rec.peer_id] = p_rec

                    raw_peers = res_header.get("peers")
                    peer_recs = _verified_records(raw_peers, self._k)
                    for r_rec in peer_recs:
                        if r_rec.peer_id != self._node.peer_id and r_rec.peer_id not in failed:
                            if r_rec.peer_id not in candidates:
                                candidates[r_rec.peer_id] = r_rec

            if is_providers and len(providers_found) >= limit:
                break

            if not candidates:
                break

            k_closest = sorted(candidates.values(), key=dist)[: self._k]
            keep_pids = {r.peer_id for r in k_closest} | (set(candidates.keys()) & queried)
            candidates = {pid: candidates[pid] for pid in keep_pids}

            current_closest_dist = min(dist(r) for r in candidates.values())
            if closest_seen_dist is None or current_closest_dist < closest_seen_dist:
                closest_seen_dist = current_closest_dist
            else:
                sorted_candidates = sorted(candidates.values(), key=dist)
                k_closest = sorted_candidates[: self._k]
                if all(r.peer_id in queried for r in k_closest):
                    break

        sorted_final = sorted(candidates.values(), key=dist)
        final_closest = [r for r in sorted_final if r.peer_id != self._node.peer_id][: self._k]
        for r in final_closest:  # at most K, so the caller can dial what it was given
            self._node.add_record(r)
        return final_closest, list(providers_found.values())[:limit]

    async def afind_node(self, key: bytes) -> list[PeerRecord]:
        """Perform iterative lookup finding up to K closest verified records to key."""
        closest_peers, _ = await self._iterative_lookup(key, is_providers=False)
        return closest_peers

    async def aprovide(self, key: bytes, ttl_s: float = PROVIDE_TTL_S) -> int:
        """Announce this node on the K closest nodes; returns how many accepted."""
        if not isinstance(key, (bytes, bytearray)) or len(key) != KEY_BYTES:
            raise ValueError(f"key must be {KEY_BYTES} bytes")
        key_bytes = bytes(key)
        ttl = max(1.0, min(float(ttl_s), float(MAX_TTL_S)))
        our_record = self._node.record()
        self._provider_store.add(key_bytes, our_record, ttl)

        closest_peers = await self.afind_node(key_bytes)
        if not closest_peers:
            return 0

        req_header = {
            "key": key_bytes.hex(),
            "record": our_record.to_dict(),
            "ttl": ttl,
        }

        async def _send(peer: PeerRecord) -> bool:
            try:
                await self._node.acall(
                    peer.peer_id,
                    "dht.add_provider",
                    req_header,
                    timeout=self._rpc_timeout,
                )
                return True
            except RpcError as exc:
                logger.debug("add_provider to %s failed: %s", peer.peer_id[:8], exc)
                if _peer_seems_gone(exc):
                    self._table.remove(peer.peer_id)
                return False
            except Exception:  # a local bug, not the peer's fault: keep it
                logger.exception("add_provider to %s failed here", peer.peer_id[:8])
                return False

        results = await asyncio.gather(*[_send(p) for p in closest_peers])
        return sum(1 for ok in results if ok)

    async def afind_providers(self, key: bytes, limit: int = 32) -> list[PeerRecord]:
        """Perform iterative lookup collecting provider records for key."""
        _, providers = await self._iterative_lookup(key, is_providers=True, limit=limit)
        return providers

    async def abootstrap(self) -> int:
        """Look up our own key to fill the table; returns the table size."""
        own_key = node_key(self._node.peer_id)
        await self.afind_node(own_key)
        return len(self._table)

    def find_node(self, key: bytes) -> list[PeerRecord]:
        """Synchronously look up up to K closest records to key."""
        return self._node.run(self.afind_node(key))

    def provide(self, key: bytes, ttl_s: float = PROVIDE_TTL_S) -> int:
        """Synchronously announce this node on K closest nodes."""
        return self._node.run(self.aprovide(key, ttl_s=ttl_s))

    def find_providers(self, key: bytes, limit: int = 32) -> list[PeerRecord]:
        """Synchronously look up provider records for key."""
        return self._node.run(self.afind_providers(key, limit=limit))

    def bootstrap(self) -> int:
        """Synchronously look up own key to fill routing table; returns size."""
        return self._node.run(self.abootstrap())


__all__ = [
    "ALPHA",
    "K",
    "KEY_BYTES",
    "MAX_TTL_S",
    "PROVIDE_TTL_S",
    "REPROVIDE_S",
    "Dht",
    "ProviderStore",
    "RoutingTable",
    "distance",
    "node_key",
    "topic_key",
]


def _peer_seems_gone(exc: RpcError) -> bool:
    """Whether a failed call means the peer is gone, rather than us being at a limit."""
    if exc.code == "deadline":
        return True
    # Our own node refuses to dial when it is at max_connections; that says nothing
    # about the peer, so it must not cost the peer its place in the table.
    return exc.code == "unavailable" and "too many connections" not in exc.message
