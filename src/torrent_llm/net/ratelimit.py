"""Rate limits: how much one peer may ask of this device.

A token bucket per (peer, kind): ``rate`` tokens refill per second up to ``burst``;
each request spends ``cost`` tokens and is refused when the bucket is short. Buckets
for peers not seen for ``idle_s`` are forgotten so the table cannot grow without
bound (at most ``max_keys`` buckets; past that the least recently used is dropped).

Thread-safe: the overlay's event loop and handler threads both consult it.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Limit:
    rate: float  # tokens per second, > 0
    burst: float  # bucket size, >= 1

    def __post_init__(self) -> None:
        if not self.rate > 0:
            raise ValueError(f"rate must be > 0, got {self.rate}")
        if not self.burst >= 1:
            raise ValueError(f"burst must be >= 1, got {self.burst}")


@dataclass
class _Bucket:
    tokens: float
    last_time: float


class RateLimiter:
    """Behaviour (tests pin these down, with an injected clock):

    * a fresh key allows ``burst`` requests of cost 1 at once, then refuses;
    * after ``t`` seconds it has ``min(burst, left + rate * t)`` tokens again;
    * ``cost`` larger than ``burst`` is always refused (and spends nothing);
    * an unknown ``kind`` (not in ``limits``) is always allowed;
    * ``retry_after(key, kind, cost)`` is the seconds until ``allow`` would pass (0 if now);
    * keys idle for ``idle_s`` are dropped by ``prune()`` (also run every 1000 calls);
    * never more than ``max_keys`` buckets: the least recently used is evicted.
    """

    def __init__(
        self,
        limits: dict[str, Limit],
        *,
        idle_s: float = 600.0,
        max_keys: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not max_keys >= 1:
            raise ValueError(f"max_keys must be >= 1, got {max_keys}")
        if not idle_s > 0:
            raise ValueError(f"idle_s must be > 0, got {idle_s}")
        for kind, limit in limits.items():
            if not isinstance(limit, Limit):  # Limit validates rate and burst itself
                raise TypeError(f"limit for {kind!r} must be a Limit, got {type(limit).__name__}")

        self._limits = dict(limits)
        self._idle_s = idle_s
        self._max_keys = max_keys
        self._clock = clock
        self._buckets: OrderedDict[tuple[str, str], _Bucket] = OrderedDict()
        self._lock = threading.Lock()
        self._calls = 0

    def _prune_locked(self, now: float) -> int:
        to_drop = [k for k, b in self._buckets.items() if now - b.last_time >= self._idle_s]
        for k in to_drop:
            del self._buckets[k]
        return len(to_drop)

    def allow(self, key: str, kind: str, cost: float = 1.0) -> bool:
        if not cost >= 0:
            raise ValueError(f"cost must be non-negative and not NaN, got {cost}")
        with self._lock:
            now = self._clock()
            self._calls += 1
            if self._calls % 1000 == 0:
                self._prune_locked(now)

            limit = self._limits.get(kind)
            if limit is None:
                return True

            if cost > limit.burst:
                return False

            bucket_key = (key, kind)
            bucket = self._buckets.get(bucket_key)
            if bucket is None:
                self._buckets[bucket_key] = _Bucket(
                    tokens=limit.burst - cost,
                    last_time=now,
                )
                while len(self._buckets) > self._max_keys:
                    self._buckets.popitem(last=False)
                return True

            elapsed = max(0.0, now - bucket.last_time)
            tokens = min(limit.burst, bucket.tokens + elapsed * limit.rate)
            bucket.last_time = now
            self._buckets.move_to_end(bucket_key)

            if tokens >= cost:
                bucket.tokens = tokens - cost
                return True
            bucket.tokens = tokens
            return False

    def retry_after(self, key: str, kind: str, cost: float = 1.0) -> float:
        if not cost >= 0:
            raise ValueError(f"cost must be non-negative and not NaN, got {cost}")
        with self._lock:
            limit = self._limits.get(kind)
            if limit is None:
                return 0.0

            if cost > limit.burst:
                return float("inf")

            bucket_key = (key, kind)
            bucket = self._buckets.get(bucket_key)
            if bucket is None:
                tokens = limit.burst
            else:
                now = self._clock()
                elapsed = max(0.0, now - bucket.last_time)
                tokens = min(limit.burst, bucket.tokens + elapsed * limit.rate)

            return max(0.0, (cost - tokens) / limit.rate)

    def prune(self) -> int:
        """Drop idle buckets; returns how many were dropped."""
        with self._lock:
            return self._prune_locked(self._clock())

    def __len__(self) -> int:
        with self._lock:
            return len(self._buckets)


__all__ = ["Limit", "RateLimiter"]
