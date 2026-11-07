"""Tests for torrent_llm.net.ratelimit."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable

import pytest

from torrent_llm.net.ratelimit import Limit, RateLimiter


def _make_clock(initial: float = 1000.0) -> tuple[list[float], Callable[[], float]]:
    clock = [initial]

    def _tick() -> float:
        return clock[0]

    return clock, _tick


def test_invalid_limit_values_raise_value_error():
    with pytest.raises(ValueError, match="rate"):
        Limit(rate=0.0, burst=5.0)

    with pytest.raises(ValueError, match="rate"):
        Limit(rate=-1.0, burst=5.0)

    with pytest.raises(ValueError, match="rate"):
        Limit(rate=float("nan"), burst=5.0)

    with pytest.raises(ValueError, match="burst"):
        Limit(rate=1.0, burst=0.0)

    with pytest.raises(ValueError, match="burst"):
        Limit(rate=1.0, burst=0.9)

    with pytest.raises(ValueError, match="burst"):
        Limit(rate=1.0, burst=-2.0)

    with pytest.raises(ValueError, match="burst"):
        Limit(rate=1.0, burst=float("nan"))

    with pytest.raises(ValueError, match="max_keys"):
        RateLimiter({"k": Limit(rate=1.0, burst=1.0)}, max_keys=0)

    with pytest.raises(ValueError, match="idle_s"):
        RateLimiter({"k": Limit(rate=1.0, burst=1.0)}, idle_s=0.0)

    with pytest.raises(ValueError, match="idle_s"):
        RateLimiter({"k": Limit(rate=1.0, burst=1.0)}, idle_s=-10.0)

    with pytest.raises(ValueError, match="idle_s"):
        RateLimiter({"k": Limit(rate=1.0, burst=1.0)}, idle_s=float("nan"))


def test_invalid_cost_raises_value_error():
    limiter = RateLimiter({"req": Limit(rate=1.0, burst=5.0)})

    with pytest.raises(ValueError, match="cost"):
        limiter.allow("peer-1", "req", -1.0)

    with pytest.raises(ValueError, match="cost"):
        limiter.allow("peer-1", "req", float("nan"))

    with pytest.raises(ValueError, match="cost"):
        limiter.allow("peer-1", "req", math.nan)

    with pytest.raises(ValueError, match="cost"):
        limiter.retry_after("peer-1", "req", -0.5)

    with pytest.raises(ValueError, match="cost"):
        limiter.retry_after("peer-1", "req", float("nan"))


def test_fresh_key_allows_burst_requests_then_refuses():
    clock, tick = _make_clock(100.0)
    limiter = RateLimiter({"req": Limit(rate=1.0, burst=5.0)}, clock=tick)

    for _ in range(5):
        assert limiter.allow("peer-1", "req", 1.0) is True

    # 6th request at the same clock time is refused
    assert limiter.allow("peer-1", "req", 1.0) is False
    assert limiter.allow("peer-1", "req", 1.0) is False


def test_refill_after_t_seconds():
    clock, tick = _make_clock(100.0)
    limiter = RateLimiter({"req": Limit(rate=2.0, burst=10.0)}, clock=tick)

    # Spend 8 tokens out of 10 -> 2 left
    assert limiter.allow("peer-1", "req", 8.0) is True

    # After 1.5 seconds, 3.0 tokens refilled -> min(10, 2 + 3) = 5.0 tokens
    clock[0] += 1.5
    assert limiter.allow("peer-1", "req", 5.0) is True
    # 0 tokens left now
    assert limiter.allow("peer-1", "req", 1.0) is False

    # After 10 seconds, refilled up to burst (capped at 10.0)
    clock[0] += 10.0
    assert limiter.allow("peer-1", "req", 10.0) is True
    assert limiter.allow("peer-1", "req", 0.5) is False


def test_cost_larger_than_burst_refused_and_spends_nothing():
    clock, tick = _make_clock(100.0)
    limiter = RateLimiter({"req": Limit(rate=1.0, burst=5.0)}, clock=tick)

    # Fresh key with cost > burst
    assert limiter.allow("peer-1", "req", 6.0) is False
    assert limiter.retry_after("peer-1", "req", 6.0) == float("inf")

    # Bucket should be intact with full burst
    assert limiter.allow("peer-1", "req", 5.0) is True

    # Existing bucket with cost > burst
    clock[0] += 1.0  # 1 token refilled
    assert limiter.allow("peer-1", "req", 10.0) is False
    assert limiter.retry_after("peer-1", "req", 10.0) == float("inf")
    # Spent nothing: the 1 refilled token is still there
    assert limiter.allow("peer-1", "req", 1.0) is True


def test_unknown_kind_is_always_allowed():
    clock, tick = _make_clock(100.0)
    limiter = RateLimiter({"known": Limit(rate=1.0, burst=1.0)}, clock=tick)

    assert limiter.allow("peer-1", "unknown", cost=100.0) is True
    assert limiter.allow("peer-1", "unknown", cost=1000.0) is True
    assert limiter.retry_after("peer-1", "unknown", cost=500.0) == 0.0


def test_retry_after_returns_wait_time():
    clock, tick = _make_clock(100.0)
    limiter = RateLimiter({"req": Limit(rate=2.0, burst=6.0)}, clock=tick)

    # Fresh key with cost <= burst
    assert limiter.retry_after("peer-1", "req", 3.0) == 0.0

    # Drain all 6 tokens
    assert limiter.allow("peer-1", "req", 6.0) is True

    # Need 4 tokens at rate 2.0 -> 2.0 seconds
    assert limiter.retry_after("peer-1", "req", 4.0) == 2.0

    # Advance 0.5s -> 1 token refilled, needs 3 tokens -> 1.5 seconds
    clock[0] += 0.5
    assert limiter.retry_after("peer-1", "req", 4.0) == 1.5

    # Advance 1.5s -> 4 tokens available, wait time is 0.0
    clock[0] += 1.5
    assert limiter.retry_after("peer-1", "req", 4.0) == 0.0
    assert limiter.allow("peer-1", "req", 4.0) is True


def test_retry_after_does_not_mutate_bucket_or_prevent_prune():
    clock, tick = _make_clock(100.0)
    limiter = RateLimiter({"req": Limit(rate=1.0, burst=5.0)}, idle_s=20.0, clock=tick)

    assert limiter.allow("peer-1", "req", 5.0) is True
    assert len(limiter) == 1

    # Querying retry_after at t=115
    clock[0] = 115.0
    assert limiter.retry_after("peer-1", "req", 5.0) == 0.0

    # At t=125, peer-1 has had no allow() calls since t=100 (25s >= 20s idle)
    # retry_after must not have refreshed last_time, so prune drops it
    clock[0] = 125.0
    assert limiter.prune() == 1
    assert len(limiter) == 0

    # Querying retry_after on an evicted/absent key does not resurrect a bucket
    assert limiter.retry_after("peer-1", "req", 3.0) == 0.0
    assert len(limiter) == 0


def test_retry_after_does_not_reorder_lru():
    clock, tick = _make_clock(100.0)
    limiter = RateLimiter({"req": Limit(rate=1.0, burst=5.0)}, max_keys=2, clock=tick)

    assert limiter.allow("k1", "req", 5.0) is True
    assert limiter.allow("k2", "req", 5.0) is True

    # k1 is LRU, k2 is MRU. Querying retry_after on k1 should NOT make k1 MRU.
    assert limiter.retry_after("k1", "req", 1.0) == 1.0

    # Adding k3 must evict the true LRU (k1), not k2.
    assert limiter.allow("k3", "req", 1.0) is True
    assert len(limiter) == 2

    # k1 was evicted, so retry_after treats it as a fresh key (wait time 0.0 for cost <= burst)
    assert limiter.retry_after("k1", "req", 5.0) == 0.0
    assert len(limiter) == 2  # still not resurrected

    # Accessing k1 allocates a fresh full bucket
    assert limiter.allow("k1", "req", 5.0) is True


def test_prune_drops_idle_buckets():
    clock, tick = _make_clock(100.0)
    limiter = RateLimiter({"req": Limit(rate=1.0, burst=5.0)}, idle_s=60.0, clock=tick)

    # peer-1 at t=100
    assert limiter.allow("peer-1", "req") is True

    # peer-2 at t=130
    clock[0] = 130.0
    assert limiter.allow("peer-2", "req") is True
    assert len(limiter) == 2

    # At t=165, peer-1 idle for 65s (>=60), peer-2 idle for 35s (<60)
    clock[0] = 165.0
    dropped = limiter.prune()
    assert dropped == 1
    assert len(limiter) == 1

    # At t=195, peer-2 idle for 65s (>=60)
    clock[0] = 195.0
    dropped = limiter.prune()
    assert dropped == 1
    assert len(limiter) == 0


def test_allow_prunes_every_1000_calls():
    clock, tick = _make_clock(100.0)
    limiter = RateLimiter({"req": Limit(rate=1.0, burst=5.0)}, idle_s=10.0, clock=tick)

    # peer-idle at t=100
    assert limiter.allow("peer-idle", "req") is True

    # Advance clock past idle_s
    clock[0] = 120.0

    # 998 calls on active peer (total 999 calls so far)
    for _ in range(998):
        limiter.allow("peer-active", "req", 0.0)

    # peer-idle should still be present before 1000th call
    assert len(limiter) == 2

    # 1000th call triggers prune
    limiter.allow("peer-active", "req", 0.0)
    assert len(limiter) == 1


def test_max_keys_eviction_drops_least_recently_used():
    clock, tick = _make_clock(100.0)
    limiter = RateLimiter(
        {"req": Limit(rate=1.0, burst=5.0)},
        max_keys=2,
        clock=tick,
    )

    # Drain k1 and k2 completely at t=100
    assert limiter.allow("k1", "req", 5.0) is True
    assert limiter.allow("k2", "req", 5.0) is True
    assert len(limiter) == 2

    # Touch k1 with an unsuccessful request (0 tokens left, refuses)
    # k1 is now MRU, k2 is LRU
    assert limiter.allow("k1", "req", 1.0) is False

    # Insert k3 -> causes eviction of LRU (k2)
    assert limiter.allow("k3", "req", 1.0) is True
    assert len(limiter) == 2

    # k1 is still in the cache with 0 tokens left
    assert limiter.allow("k1", "req", 1.0) is False

    # k2 was evicted, so accessing k2 gives it a fresh full bucket (5.0 tokens)
    assert limiter.allow("k2", "req", 5.0) is True


def test_concurrent_allow_never_exceeds_burst():
    burst = 16
    limiter = RateLimiter(
        {"req": Limit(rate=1.0, burst=burst)},
        clock=lambda: 100.0,  # frozen clock
    )

    allowed: list[int] = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        count = 0
        for _ in range(50):
            if limiter.allow("shared-peer", "req", 1.0):
                count += 1
        allowed.append(count)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Sum of all allowed calls across 8 concurrent threads must equal burst
    assert sum(allowed) == burst


def test_bucket_key_is_key_and_kind():
    clock, tick = _make_clock(100.0)
    limiter = RateLimiter(
        {
            "kindA": Limit(rate=1.0, burst=2.0),
            "kindB": Limit(rate=1.0, burst=2.0),
        },
        clock=tick,
    )

    assert limiter.allow("peer-1", "kindA", 2.0) is True
    assert limiter.allow("peer-1", "kindB", 2.0) is True
    assert len(limiter) == 2

    # Both exhausted independently
    assert limiter.allow("peer-1", "kindA", 1.0) is False
    assert limiter.allow("peer-1", "kindB", 1.0) is False


def test_limiter_refuses_non_limit_values_and_prunes_exactly_at_idle_s():
    import pytest as _pytest

    from torrent_llm.net.ratelimit import Limit as _Limit
    from torrent_llm.net.ratelimit import RateLimiter as _RateLimiter

    with _pytest.raises(TypeError):
        _RateLimiter({"x": (1.0, 2.0)})  # type: ignore[dict-item]
    now = [0.0]
    limiter = _RateLimiter({"x": _Limit(rate=1.0, burst=2.0)}, idle_s=10.0, clock=lambda: now[0])
    assert limiter.allow("peer", "x")
    now[0] = 9.999
    assert limiter.prune() == 0
    now[0] = 10.0
    assert limiter.prune() == 1
    assert len(limiter) == 0
