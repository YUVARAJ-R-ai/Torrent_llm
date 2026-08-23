"""SessionStore correctness: the part of KV caching that only exists because
gRPC calls are otherwise stateless.

A fake clock is injected everywhere a TTL is exercised, so these tests run in
milliseconds instead of actually sleeping for five minutes.
"""

from torrent_llm.transport.sessions import SessionStore


class FakeClock:
    """A controllable monotonic clock for testing TTL eviction without sleeping."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_first_call_for_a_request_id_creates_a_cache():
    store = SessionStore()

    cache = store.get_or_create("req-1")

    assert cache is not None
    assert "req-1" in store
    assert len(store) == 1


def test_second_call_for_the_same_request_id_returns_the_same_cache():
    store = SessionStore()

    first = store.get_or_create("req-1")
    second = store.get_or_create("req-1")

    assert first is second
    assert len(store) == 1


def test_different_request_ids_get_independent_caches():
    store = SessionStore()

    a = store.get_or_create("req-a")
    b = store.get_or_create("req-b")

    assert a is not b
    assert len(store) == 2


def test_drop_removes_a_session_immediately():
    store = SessionStore()
    store.get_or_create("req-1")

    store.drop("req-1")

    assert "req-1" not in store
    assert len(store) == 0


def test_dropping_a_session_that_never_existed_is_a_no_op():
    store = SessionStore()

    store.drop("never-existed")  # must not raise

    assert len(store) == 0


def test_dropping_one_session_leaves_others_untouched():
    store = SessionStore()
    store.get_or_create("keep")
    store.get_or_create("drop-me")

    store.drop("drop-me")

    assert "keep" in store
    assert "drop-me" not in store


# --- TTL eviction: the backstop for a client that never sends end_of_request ---


def test_a_session_survives_within_its_ttl():
    clock = FakeClock()
    store = SessionStore(ttl_seconds=10.0, clock=clock)
    store.get_or_create("req-1")

    clock.advance(5.0)
    store.get_or_create("other")  # any call sweeps expired sessions

    assert "req-1" in store


def test_a_session_is_swept_once_its_ttl_has_elapsed():
    clock = FakeClock()
    store = SessionStore(ttl_seconds=10.0, clock=clock)
    store.get_or_create("req-1")

    clock.advance(11.0)
    store.get_or_create("other")  # triggers the sweep

    assert "req-1" not in store
    assert "other" in store


def test_touching_a_session_resets_its_ttl_clock():
    # A slow but still-active generation must not be swept out from under
    # itself just because it has been running for a while.
    clock = FakeClock()
    store = SessionStore(ttl_seconds=10.0, clock=clock)
    store.get_or_create("req-1")

    clock.advance(7.0)
    store.get_or_create("req-1")  # touch: last-seen resets to t=7
    clock.advance(7.0)  # t=14; would be expired if last-seen were still t=0
    store.get_or_create("other")

    assert "req-1" in store


def test_sweep_only_removes_sessions_past_their_own_ttl():
    clock = FakeClock()
    store = SessionStore(ttl_seconds=10.0, clock=clock)
    store.get_or_create("old")

    clock.advance(6.0)
    store.get_or_create("newer")

    clock.advance(6.0)  # old is at t=12 (expired), newer is at t=6 (not yet)
    store.get_or_create("trigger-sweep")

    assert "old" not in store
    assert "newer" in store


# --- concurrency: the reason a lock exists at all ---


def test_concurrent_creation_of_different_sessions_does_not_corrupt_the_table():
    import threading

    store = SessionStore()
    request_ids = [f"req-{i}" for i in range(50)]
    errors: list[Exception] = []

    def worker(rid: str) -> None:
        try:
            store.get_or_create(rid)
        except Exception as exc:  # noqa: BLE001 - captured for the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(rid,)) for rid in request_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(store) == len(request_ids)
