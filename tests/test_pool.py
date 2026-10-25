"""A whole pool of devices, in one process, over real sockets.

Each device is a real ``PeerDaemon`` with a real HTTP server and real gRPC shard
servers; only the model is the tiny random one the rest of the suite uses. They
find each other by gossip, split the model between them, and answer questions.

The correctness anchor is the same as everywhere else in this repo: whatever
route a request takes, and whatever happens to the devices along the way, the
tokens must be exactly what the monolithic model would have produced.
"""

import threading
import time

import httpx
import pytest
import torch

from torrent_llm.link import LinkProfile
from torrent_llm.pool.config import DeviceProfile, PoolConfig
from torrent_llm.pool.members import ACTIVE, AWAY, DRAINING
from torrent_llm.pool.peer import launch
from torrent_llm.shard import ShardRuntime

FAST = {"tick_s": 0.1, "suspect_after": 0.8, "down_after": 1.6, "drain_grace_s": 3.0}


class CharTokenizer:
    """Reversible and trivial: ids stay inside the tiny model's vocabulary."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 256 for c in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(int(i) % 256) for i in ids)


def wait_for(condition, *, timeout=15.0, what="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


class Pool:
    """A handful of devices that find each other, with helpers to poke them."""

    def __init__(self, model_factory, num_layers, *, replicas=2, stages=2):
        self.config = PoolConfig(
            model_id="tiny", num_layers=num_layers, stages=stages, replicas=replicas, shard_mb=1000
        )
        self._model_factory = model_factory
        self.peers: dict[str, object] = {}

    def add(self, name, *, memory_mb=1000, speed=1.0, link=None, join=True):
        spec_loader = lambda spec: ShardRuntime(self._model_factory(), spec)  # noqa: E731
        seeds = [p.url.removeprefix("http://") for p in self.peers.values()][:1] if join else []
        peer = launch(
            self.config,
            DeviceProfile(name=name, memory_mb=memory_mb, speed=speed, link=link, device="cpu"),
            spec_loader,
            CharTokenizer(),
            join=seeds,
            **FAST,
        )
        self.peers[name] = peer
        return peer

    def view(self, name):
        return httpx.get(f"{self.peers[name].url}/pool").json()

    def settled(self, expect_copies):
        """Every live device agrees every stage has ``expect_copies`` ready copies."""
        for peer in self.peers.values():
            if peer.daemon.state == "left":
                continue
            counts = httpx.get(f"{peer.url}/pool").json()["ready_copies"]
            if counts != expect_copies:
                return False
        return True

    def ask(self, name, prompt="hello", **kw):
        body = {"prompt": prompt, "chat": False, **kw}
        return httpx.post(f"{self.peers[name].url}/generate", json=body, timeout=60)

    def close(self):
        for peer in self.peers.values():
            peer.close(graceful=False)


@pytest.fixture
def pool(model_factory, num_layers):
    p = Pool(model_factory, num_layers)
    yield p
    p.close()


def reference(tiny_model, prompt, n):
    ids = torch.tensor([CharTokenizer().encode(prompt)])
    out = ids
    with torch.inference_mode():
        for _ in range(n):
            nxt = tiny_model(input_ids=out).logits[:, -1, :].argmax(-1, keepdim=True)
            out = torch.cat([out, nxt], dim=1)
    return out[0, ids.shape[1] :].tolist()


def four_devices(pool):
    for name in "abcd":
        pool.add(name)
    wait_for(lambda: pool.settled([2, 2]), what="every stage to have two copies")


# --- forming ---------------------------------------------------------------------


def test_devices_find_each_other_and_split_the_model(pool):
    four_devices(pool)

    view = pool.view("a")

    assert {m["name"] for m in view["members"]} == set("abcd")
    assert view["health"] == "healthy"
    # One stage each: four devices, two stages, two copies of each.
    assert sorted(len(m["hosting"]) for m in view["members"]) == [1, 1, 1, 1]


def test_a_device_needs_only_one_address_to_join(pool):
    pool.add("a")
    pool.add("b")
    # c knows only about b, not about a.
    pool.add("c", join=False)
    pool.peers["c"].daemon.join.append(pool.peers["b"].url.removeprefix("http://"))

    wait_for(
        lambda: {m["name"] for m in pool.view("c")["members"]} == {"a", "b", "c"},
        what="c to learn about a through b",
    )


def test_every_device_serves_the_same_view(pool):
    four_devices(pool)

    names = [sorted(m["name"] for m in pool.view(n)["members"]) for n in "abcd"]

    assert all(n == names[0] for n in names)


# --- answering ---------------------------------------------------------------------


def test_any_device_can_answer_and_the_answer_is_exact(pool, tiny_model):
    four_devices(pool)

    expected = reference(tiny_model, "hello", 6)
    for requester in "abcd":
        body = pool.ask(requester, max_new_tokens=6).json()
        ids = [ord(c) for c in body["completion"]]
        assert ids == [t % 256 for t in expected], requester
        assert body["requester"] == requester


def test_the_response_says_which_device_ran_each_stage(pool):
    four_devices(pool)

    body = pool.ask("a", max_new_tokens=3).json()

    first = body["passes"][0]
    assert [stage for stage, _ in first["route"]] == [0, 1]
    assert body["passes"][0]["phase"] == "prefill"
    assert {p["phase"] for p in body["passes"][1:]} == {"decode"}
    assert body["totals"]["failovers"] == 0


def test_concurrent_requests_are_spread_across_the_copies(pool):
    four_devices(pool)
    seen: set[str] = set()
    lock = threading.Lock()

    def worker():
        body = pool.ask("a", max_new_tokens=4).json()
        with lock:
            seen.update(member for _, member in body["passes"][0]["route"])

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Twelve requests across two copies of each stage should touch more than one.
    assert len(seen) > 2


# --- surviving loss ------------------------------------------------------------------


def test_losing_a_device_mid_answer_changes_nothing_but_the_pause(pool, tiny_model):
    link = LinkProfile(latency_ms=25, bandwidth_mbps=100)
    for name in "abcd":
        pool.add(name, link=link)
    wait_for(lambda: pool.settled([2, 2]), what="two copies of each stage")
    n = 24
    expected = [t % 256 for t in reference(tiny_model, "hello", n)]

    box: dict = {}
    worker = threading.Thread(
        target=lambda: box.update(body=pool.ask("a", max_new_tokens=n).json())
    )
    worker.start()

    # Kill whichever device holds stage 1 for this request, once it is underway.
    runner = pool.peers["a"].daemon.runner
    wait_for(lambda: any(v > 0 for v in runner._load.values()), what="the request to start")
    busy = [addr for addr, v in runner._load.items() if v > 0]
    victim = next(
        name
        for name, peer in pool.peers.items()
        for h in peer.daemon.shards.hosting()
        if f"127.0.0.1:{h.port}" in busy and name != "a"
    )
    time.sleep(0.3)
    pool.peers[victim].close(graceful=False)
    worker.join(timeout=60)

    body = box["body"]
    assert [ord(c) for c in body["completion"]] == expected
    assert body["totals"]["failovers"] >= 1
    kinds = [e["kind"] for e in body["events"]]
    assert "failover" in kinds and "recovered" in kinds
    assert any(p["recovery"] for p in body["passes"])


def test_the_pool_replaces_a_lost_copy_when_a_spare_exists(pool):
    for name in "abcde":
        pool.add(name)
    wait_for(lambda: pool.settled([2, 2]), what="two copies of each stage")

    holders = [m["name"] for m in pool.view("a")["members"] if m["hosting"]]
    victim = next(n for n in holders if n != "a")
    pool.peers[victim].close(graceful=False)

    live = [n for n in pool.peers if n != victim]
    wait_for(
        lambda: all(
            httpx.get(f"{pool.peers[n].url}/pool").json()["ready_copies"] == [2, 2] for n in live
        ),
        timeout=20,
        what="the spare to take the lost stage",
    )


def test_with_no_spare_the_pool_runs_degraded_but_still_answers(pool, tiny_model):
    four_devices(pool)
    victim = next(n for n in "bcd")
    pool.peers[victim].close(graceful=False)

    wait_for(
        lambda: min(pool.view("a")["ready_copies"]) == 1 and pool.view("a")["health"] == "degraded",
        what="the pool to notice and degrade",
    )

    body = pool.ask("a", max_new_tokens=5).json()
    assert [ord(c) for c in body["completion"]] == [
        t % 256 for t in reference(tiny_model, "hello", 5)
    ]


def test_a_stage_with_no_copies_waits_then_gives_a_clear_error(pool):
    four_devices(pool)
    pool.peers["a"].daemon.runner.patience = 1.0
    # Take both holders of stage 1 away, leaving only stage 0 copies and the asker.
    stage1 = [
        n
        for n, p in pool.peers.items()
        if n != "a" and any(h.stage == 1 for h in p.daemon.shards.hosting())
    ]
    for n in stage1:
        pool.peers[n].close(graceful=False)
    wait_for(lambda: pool.view("a")["health"] == "down", what="stage 1 to be lost")

    response = pool.ask("a", max_new_tokens=3)

    assert response.status_code == 503
    assert "stage 1" in response.json()["detail"]


def test_a_device_that_leaves_politely_is_gone_at_once(pool):
    four_devices(pool)

    pool.peers["d"].close(graceful=True)

    wait_for(
        lambda: (
            next(m for m in pool.view("a")["members"] if m["name"] == "d")["liveness"] == "down"
        ),
        timeout=5,
        what="d's goodbye to arrive",
    )


# --- the owner wanting their device back ------------------------------------------------


def test_a_device_whose_owner_returns_hands_over_and_goes_quiet(pool, tiny_model):
    for name in "abcde":
        pool.add(name)
    wait_for(lambda: pool.settled([2, 2]), what="two copies of each stage")
    holders = [m["name"] for m in pool.view("a")["members"] if m["hosting"] and m["name"] != "a"]
    busy_device = holders[0]

    state = httpx.post(f"{pool.peers[busy_device].url}/admin/owner", json={"busy": True}).json()
    assert state["state"] == DRAINING

    daemon = pool.peers[busy_device].daemon
    wait_for(
        lambda: daemon.state == AWAY and not daemon.shards.stages(), what="the device to let go"
    )
    # The pool never dropped below two copies: a replacement came up first.
    wait_for(lambda: pool.settled([2, 2]), what="the pool to settle with two copies again")
    assert daemon.shards.hosting() == ()
    # And it still answers correctly meanwhile.
    body = pool.ask("a", max_new_tokens=4).json()
    assert [ord(c) for c in body["completion"]] == [
        t % 256 for t in reference(tiny_model, "hello", 4)
    ]


def test_a_draining_device_is_not_given_new_work(pool):
    for name in "abcde":
        pool.add(name)
    wait_for(lambda: pool.settled([2, 2]), what="two copies of each stage")
    holders = [m["name"] for m in pool.view("a")["members"] if m["hosting"] and m["name"] != "a"]
    busy_device = holders[0]
    httpx.post(f"{pool.peers[busy_device].url}/admin/owner", json={"busy": True})
    wait_for(
        lambda: (
            next(m for m in pool.view("a")["members"] if m["name"] == busy_device)["state"]
            != ACTIVE
        ),
        what="the draining state to spread",
    )

    for _ in range(8):
        body = pool.ask("a", max_new_tokens=2).json()
        used = {member for _, member in body["passes"][0]["route"]}
        assert busy_device not in used


def test_a_device_that_gets_its_owner_back_out_of_the_way_rejoins(pool):
    for name in "abcde":
        pool.add(name)
    wait_for(lambda: pool.settled([2, 2]), what="two copies of each stage")
    holders = [m["name"] for m in pool.view("a")["members"] if m["hosting"] and m["name"] != "a"]
    device = holders[0]
    daemon = pool.peers[device].daemon
    httpx.post(f"{pool.peers[device].url}/admin/owner", json={"busy": True})
    wait_for(lambda: daemon.state == AWAY, what="the device to go away")

    httpx.post(f"{pool.peers[device].url}/admin/owner", json={"busy": False})

    wait_for(lambda: daemon.state == ACTIVE, what="the device to rejoin")
    assert pool.view(device)["self"] == device


# --- protecting the pool ------------------------------------------------------------------


def test_pool_changes_need_the_shared_token(model_factory, num_layers):
    config = PoolConfig(model_id="tiny", num_layers=num_layers, stages=2, replicas=1)
    peer = launch(
        config,
        DeviceProfile(name="solo", memory_mb=3000, device="cpu"),
        lambda spec: ShardRuntime(model_factory(), spec),
        CharTokenizer(),
        token="s3cret",
        **FAST,
    )
    try:
        refused = httpx.post(f"{peer.url}/admin/owner", json={"busy": True})
        gossip = httpx.post(f"{peer.url}/gossip", json={"sender": "x", "members": []})
        allowed = httpx.post(
            f"{peer.url}/admin/owner", json={"busy": False}, headers={"x-pool-token": "s3cret"}
        )
        reading = httpx.get(f"{peer.url}/pool")

        assert refused.status_code == 401
        assert gossip.status_code == 401
        assert allowed.status_code == 200
        assert reading.status_code == 200  # looking is open
    finally:
        peer.close(graceful=False)
