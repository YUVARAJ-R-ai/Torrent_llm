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
        raise NotImplementedError

    def allow(self, key: str, kind: str, cost: float = 1.0) -> bool:
        raise NotImplementedError

    def retry_after(self, key: str, kind: str, cost: float = 1.0) -> float:
        raise NotImplementedError

    def prune(self) -> int:
        """Drop idle buckets; returns how many were dropped."""
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


__all__ = ["Limit", "RateLimiter"]
