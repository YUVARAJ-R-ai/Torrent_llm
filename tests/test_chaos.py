"""A pool under random failures, with questions flowing the whole time.

The claim the project makes is "no downtime while any copy of each stage is up". The
hand-written tests check the cases someone thought of. This one lets a seeded random
sequence of crashes, polite departures, newcomers and owners reclaiming devices run
while several clients keep asking, and checks two things on every single answer:

* it is **exactly** the answer the whole model gives (a failover must never change a
  token), and
* it arrives (a request must not fail just because a device disappeared).

The chaos is deliberately one failure at a time: a device is only taken away once the
pool has healed from the last one. Losing both copies of a stage at the same instant is
a real outage, tested elsewhere (the request waits, then reports which stage is missing).
"""

import random
import threading
import time

import httpx
import pytest
from tests.test_pool import Pool, reference, wait_for

PROMPTS = ["hello", "the pool", "chaos monkey", "zebra", "ok?"]
NEW_TOKENS = 8


def alive_gateways(pool, dead):
    return [p for n, p in pool.peers.items() if n not in dead]


class Client(threading.Thread):
    """Asks questions through whichever device answers, the way the dashboard does."""

    def __init__(self, pool, expected, dead, stop, seed):
        super().__init__(daemon=True)
        self.pool, self.expected, self.dead, self.stop_flag = pool, expected, dead, stop
        self.rng = random.Random(seed)
        self.ok = 0
        self.wrong: list[str] = []
        self.failed: list[str] = []

    def run(self):
        while not self.stop_flag.is_set():
            prompt = self.rng.choice(PROMPTS)
            candidates = alive_gateways(self.pool, self.dead)
            self.rng.shuffle(candidates)
            for peer in candidates:
                try:
                    response = httpx.post(
                        f"{peer.url}/generate",
                        json={"prompt": prompt, "chat": False, "max_new_tokens": NEW_TOKENS},
                        timeout=40,
                    )
                except httpx.HTTPError:
                    continue  # that device is gone; ask another, as a real client would
                if response.status_code != 200:
                    self.failed.append(
                        f"{peer.daemon.name}: {response.status_code} {response.text[:120]}"
                    )
                    break
                got = [ord(c) for c in response.json()["completion"]]
                if got == self.expected[prompt]:
                    self.ok += 1
                else:
                    self.wrong.append(f"{prompt!r}: {got} != {self.expected[prompt]}")
                break
            time.sleep(0.02)


def healed(pool, dead):
    """Every live device sees at least two ready copies of each stage (spares may add more)."""
    for name, peer in pool.peers.items():
        if name in dead:
            continue
        try:
            copies = httpx.get(f"{peer.url}/pool", timeout=2).json()["ready_copies"]
            if min(copies) < 2:
                return False
        except httpx.HTTPError:
            return False
    return True


def run_chaos(model_factory, num_layers, tiny_model, seed, seconds=14.0):
    rng = random.Random(seed)
    pool = Pool(model_factory, num_layers)
    pool.config = pool.config  # replicas=2, stages=2
    for name in ["a", "b", "c", "d", "e", "f"]:
        pool.add(name, memory_mb=2000 if name in "ab" else 1000, speed=rng.choice([1.0, 2.0, 4.0]))
    wait_for(lambda: healed(pool, set()), timeout=25, what="the pool to form")
    for peer in pool.peers.values():
        peer.daemon.runner.patience = 10.0

    expected = {p: [t % 256 for t in reference(tiny_model, p, NEW_TOKENS)] for p in PROMPTS}
    dead: set[str] = set()
    stop = threading.Event()
    clients = [Client(pool, expected, dead, stop, seed * 10 + i) for i in range(3)]
    for c in clients:
        c.start()

    actions: list[str] = []
    spare_names = iter("ghijklmnop")
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            time.sleep(rng.uniform(0.3, 0.9))
            if not healed(pool, dead):
                continue  # one failure at a time: let the pool recover first
            live = [n for n in pool.peers if n not in dead]
            action = rng.choice(["crash", "leave", "join", "owner", "owner"])
            if action in ("crash", "leave") and len(live) > 4:
                victim = rng.choice(live)
                dead.add(victim)
                pool.peers[victim].close(graceful=(action == "leave"))
                actions.append(f"{action} {victim}")
            elif action == "join":
                name = next(spare_names)
                pool.add(
                    name,
                    memory_mb=1000,
                    speed=rng.choice([1.0, 2.0, 4.0]),
                    seed=pool.peers[rng.choice(live)],  # a device that is actually up
                )
                actions.append(f"join {name}")
            elif action == "owner":
                target = rng.choice(live)
                try:
                    httpx.post(
                        f"{pool.peers[target].url}/admin/owner", json={"busy": True}, timeout=2
                    )
                    actions.append(f"owner takes {target}")
                    time.sleep(1.5)
                    httpx.post(
                        f"{pool.peers[target].url}/admin/owner", json={"busy": False}, timeout=2
                    )
                    actions.append(f"owner leaves {target}")
                except httpx.HTTPError:
                    pass
    finally:
        stop.set()
        for c in clients:
            c.join(timeout=60)
        pool.close()
    return clients, actions


@pytest.mark.parametrize(
    "seed", [1, 2, 3, *([4, 5, 6, 7] if __import__("os").environ.get("CHAOS_MORE") else [])]
)
def test_one_failure_at_a_time_never_fails_a_request_or_changes_an_answer(
    model_factory, num_layers, tiny_model, seed
):
    clients, actions = run_chaos(model_factory, num_layers, tiny_model, seed)

    wrong = [w for c in clients for w in c.wrong]
    failed = [f for c in clients for f in c.failed]
    answered = sum(c.ok for c in clients)
    note = f"seed {seed}: {answered} answered; chaos: {actions}"

    print(note)  # shown with `pytest -rP`: proves the chaos really ran
    assert not wrong, f"{note}\nwrong answers: {wrong[:3]}"
    assert not failed, f"{note}\nfailed requests: {failed[:3]}"
    assert answered >= 10, note
    assert len(actions) >= 2, f"the chaos did nothing ({note})"
