"""A device that is full says so and is routed around; and the pool keeps score."""

from types import SimpleNamespace
from unittest.mock import Mock

import grpc
import httpx
import pytest
from tests.test_pool import Pool, wait_for

from torrent_llm.pool.config import DeviceProfile
from torrent_llm.pool.hosting import LoadCounter


def request(*, cached: bool, position: int | None):
    """A stand-in for a forward request: just the two fields admission looks at."""
    return SimpleNamespace(
        use_cache=cached,
        expected_cache_position=position or 0,
        HasField=lambda name: position is not None,
    )


def intercepted(counter, work):
    """The call the counter would wrap, with ``work`` as what the shard does."""
    handler = SimpleNamespace(unary_unary=work, request_deserializer=None, response_serializer=None)
    details = SimpleNamespace(method="/torrent.ShardService/Forward")
    return counter.intercept_service(lambda _details: handler, details).unary_unary


def context():
    ctx = Mock()
    ctx.abort.side_effect = RuntimeError("aborted")  # as grpc's abort does: it raises
    return ctx


# --- the limit ------------------------------------------------------------------------


def test_a_device_with_no_limit_never_refuses():
    counter = LoadCounter(limit=None)
    call = intercepted(counter, lambda req, ctx: "done")

    assert all(call(request(cached=False, position=None), context()) == "done" for _ in range(20))
    assert counter.rejected == 0


def test_a_full_device_refuses_a_new_question():
    counter = LoadCounter(limit=0)
    ctx = context()
    call = intercepted(counter, lambda req, c: "done")

    with pytest.raises(RuntimeError, match="aborted"):
        call(request(cached=True, position=0), ctx)

    assert ctx.abort.call_args.args[0] == grpc.StatusCode.RESOURCE_EXHAUSTED
    assert counter.rejected == 1
    assert counter.active == 0  # nothing was started, so nothing needs undoing


def test_a_full_device_still_continues_a_question_already_under_way():
    counter = LoadCounter(limit=0)
    call = intercepted(counter, lambda req, c: "done")

    # A decode step (the cache already holds 12 positions) is never refused.
    assert call(request(cached=True, position=12), context()) == "done"
    assert counter.rejected == 0


def test_an_uncached_call_counts_as_a_new_question():
    counter = LoadCounter(limit=0)
    call = intercepted(counter, lambda req, c: "done")

    with pytest.raises(RuntimeError):
        call(request(cached=False, position=None), context())


def test_the_limit_counts_calls_in_flight():
    import threading

    counter = LoadCounter(limit=1)
    inside, release = threading.Event(), threading.Event()

    def slow(req, ctx):
        inside.set()
        release.wait(5)
        return "done"

    call = intercepted(counter, slow)
    first = threading.Thread(target=lambda: call(request(cached=True, position=0), context()))
    first.start()
    assert inside.wait(5)
    try:
        with pytest.raises(RuntimeError):
            call(request(cached=True, position=0), context())  # the device is occupied
    finally:
        release.set()
        first.join(5)

    assert counter.active == 0 and counter.served == 1


def test_a_gpu_is_allowed_more_at_once_than_a_cpu():
    gpu = DeviceProfile(name="g", device="cuda").request_limit("cuda")
    cpu = DeviceProfile(name="c", device="cpu").request_limit("cpu")

    assert gpu > cpu >= 1
    assert DeviceProfile(name="x", max_requests=2).request_limit("cuda") == 2


# --- routing around a full device -------------------------------------------------------


@pytest.fixture
def busy_pool(model_factory, num_layers):
    pool = Pool(model_factory, num_layers)
    yield pool
    pool.close()


def test_questions_avoid_a_device_that_is_full(busy_pool):
    # "full" holds a copy of everything but refuses every new question.
    busy_pool.add("full", memory_mb=2000, speed=8.0, max_requests=0)
    for name in "bcd":
        busy_pool.add(name)
    wait_for(
        lambda: busy_pool.settled([3, 3]) or busy_pool.settled([2, 2]), what="the pool to form"
    )

    for _ in range(6):
        body = busy_pool.ask("b", max_new_tokens=3).json()
        assert "full" not in {member for _, member in body["passes"][-1]["route"]}
        assert body["totals"]["tokens"] == 3


def test_when_every_copy_is_full_the_answer_says_so(model_factory, num_layers):
    pool = Pool(model_factory, num_layers, replicas=1)
    try:
        for name in "ab":
            pool.add(name, memory_mb=2000, max_requests=0)
        wait_for(lambda: pool.settled([2, 2]) or pool.settled([1, 1]), what="the pool to form")
        pool.peers["a"].daemon.runner.max_busy_retries = 3

        response = pool.ask("a", max_new_tokens=2)

        assert response.status_code == 503
        assert "busy right now" in response.json()["detail"]
    finally:
        pool.close()


# --- keeping score ----------------------------------------------------------------------


def test_the_pool_counts_what_each_device_gave_and_asked(model_factory, num_layers):
    pool = Pool(model_factory, num_layers)
    try:
        for name in "abcd":
            pool.add(name)
        wait_for(lambda: pool.settled([2, 2]), what="two copies of each stage")

        body = pool.ask("a", max_new_tokens=12).json()
        pool.ask("a", max_new_tokens=12)
        wait_for(
            lambda: next(m for m in pool.view("b")["members"] if m["name"] == "a")["asked"] == 2,
            what="a's questions to be reported to b",
        )

        members = {m["name"]: m for m in pool.view("b")["members"]}
        assert members["a"]["asked"] == 2
        assert members["a"]["asked_tokens"] >= body["totals"]["tokens"]
        assert members["b"]["asked"] == 0
        gave = sum(h["busy_ms"] for m in members.values() for h in m["hosting"])
        assert gave > 0
    finally:
        pool.close()


def test_a_device_that_only_asks_has_given_nothing(model_factory, num_layers):
    pool = Pool(model_factory, num_layers)
    try:
        for name in "abcd":
            pool.add(name)
        pool.add("asker", memory_mb=0)  # lends nothing, so it holds no stage
        wait_for(lambda: pool.settled([2, 2]), what="two copies of each stage")

        pool.ask("asker", max_new_tokens=6)

        def me():
            members = httpx.get(f"{pool.peers['asker'].url}/pool").json()["members"]
            return next(m for m in members if m["is_self"])

        # The counter is published on the device's next heartbeat.
        wait_for(lambda: me()["asked"] == 1, what="the question to be counted")
        me = me()
        assert sum(h["busy_ms"] for h in me["hosting"]) == 0
    finally:
        pool.close()
