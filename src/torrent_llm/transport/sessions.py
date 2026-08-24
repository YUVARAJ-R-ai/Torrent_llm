"""Server-side KV cache sessions, keyed by request id (issue #24).

gRPC calls are stateless by design -- ``Forward`` has no idea it was ever called
before. A KV cache breaks that on purpose: the whole point is that a decode step
only sends the *new* token because the shard remembers everything before it. That
memory has to live somewhere between calls, and it has to live on the shard that
owns the layers it belongs to, since layer indices are global and a middle
shard's cache entries only exist at that shard's own layer range (see the
``ShardRuntime.forward`` docstring). So each :class:`ShardService` keeps a small
in-memory table: request id -> that request's cache on this shard.

Two things a session table like this always needs, or it turns into a slow leak:
eviction on an explicit "done" signal (the common case) and a TTL sweep as a
backstop for the case a client never sends one (a crash, a dropped connection, a
test that forgets to clean up). Both are here.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from transformers.cache_utils import Cache, DynamicCache


class SessionStore:
    """A small, thread-safe table of request id -> KV cache, with TTL eviction.

    Thread safety matters here specifically because :func:`torrent_llm.transport
    .server.serve` runs the gRPC service on a thread pool: two different requests
    can call ``Forward`` on the same shard concurrently, and each must get its own
    cache without racing on the shared dict. The lock only ever guards dict
    bookkeeping (microseconds); the actual forward pass runs outside it.
    """

    def __init__(
        self, *, ttl_seconds: float = 300.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._sessions: dict[str, tuple[Cache, float]] = {}
        self._lock = threading.Lock()
        self._ttl_seconds = ttl_seconds
        # Injectable so tests can simulate a session going idle without a real
        # five-minute sleep.
        self._clock = clock

    def get_or_create(self, request_id: str) -> Cache:
        """Return this request's cache, creating one on the first call.

        Every call -- not just the first -- touches the session's last-seen
        time, so a slow but still-active generation never gets swept out from
        under itself.
        """
        now = self._clock()
        with self._lock:
            self._evict_expired_locked(now)
            entry = self._sessions.get(request_id)
            if entry is None:
                cache: Cache = DynamicCache()
                self._sessions[request_id] = (cache, now)
                return cache
            cache, _ = entry
            self._sessions[request_id] = (cache, now)
            return cache

    def drop(self, request_id: str) -> None:
        """Remove a session immediately. Safe to call on a session that never existed."""
        with self._lock:
            self._sessions.pop(request_id, None)

    def _evict_expired_locked(self, now: float) -> None:
        """Sweep stale sessions. Caller must already hold ``self._lock``.

        Runs on every ``get_or_create`` rather than on a background timer: it is
        cheap (a dict of live generations is never large) and it means no thread
        needs to be started or torn down alongside the gRPC server.
        """
        expired = [
            rid
            for rid, (_, last_seen) in self._sessions.items()
            if now - last_seen > self._ttl_seconds
        ]
        for rid in expired:
            del self._sessions[rid]

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def __contains__(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._sessions
