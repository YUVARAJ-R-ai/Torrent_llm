"""Spot-checks: one stage swapped for another copy, a third copy as referee."""

from __future__ import annotations

import time
from types import SimpleNamespace

import torch

from torrent_llm.pool.members import ACTIVE, READY, HostedStage, Member
from torrent_llm.pool.router import Replica
from torrent_llm.pool.spotcheck import SpotChecker
from torrent_llm.pool.trust import TrustStore, Verifier
from torrent_llm.sampling import GREEDY


def device(name: str, stages: tuple[int, ...]) -> Member:
    return Member(
        name=name,
        host="h",
        control_port=1,
        slots=3,
        state=ACTIVE,
        heartbeat=1,
        hosting=tuple(HostedStage(stage=s, port=10 + s, status=READY) for s in stages),
    )


def replica(member: str, stage: int) -> Replica:
    return Replica(
        stage=stage, member=member, host="h", port=10 + stage, speed=1, link=None, load=0
    )


class FakeChain:
    """Answers token 3, unless a liar holds any stage of the route (then 7)."""

    def __init__(self, route, liars):
        self.route, self.liars = route, liars

    def forward(self, ids, **kw):
        bad = any(r.member in self.liars for r in self.route)
        return SimpleNamespace(token_ids=torch.tensor([[7 if bad else 3]]))

    def close(self):
        pass


def checker(members, liars, trust):
    return SpotChecker(
        trust,
        Verifier(1.0),
        me="me",
        members=lambda: members,
        chain=lambda route: FakeChain(route, liars),
    )


def wait_idle(spot: SpotChecker) -> None:
    for _ in range(200):
        if spot._busy.acquire(blocking=False):
            spot._busy.release()
            return
        time.sleep(0.01)
    raise AssertionError("spot-check did not finish")


def test_a_liar_is_caught_by_the_referee_and_the_honest_copies_are_credited():
    members = [device("liar", (1,)), device("b", (1,)), device("c", (1,)), device("me", (0,))]
    trust = TrustStore(None)
    spot = checker(members, {"liar"}, trust)
    route = [replica("me", 0), replica("liar", 1)]
    assert spot.submit(route, torch.tensor([[1, 2]]), token_id=7, sampling=GREEDY)
    wait_idle(spot)
    assert trust.get("liar").failures == 1
    assert trust.get("liar").checks == 1
    assert sum(trust.get(n).failures for n in ("b", "c")) == 0


def test_honest_copies_agree_and_nobody_is_blamed():
    members = [device("a", (1,)), device("b", (1,)), device("me", (0,))]
    trust = TrustStore(None)
    spot = checker(members, set(), trust)
    route = [replica("me", 0), replica("a", 1)]
    assert spot.submit(route, torch.tensor([[1]]), token_id=3, sampling=GREEDY)
    wait_idle(spot)
    assert trust.get("a").checks == 1 and trust.get("a").failures == 0
    assert trust.get("b").checks == 1 and trust.get("b").failures == 0


def test_nothing_to_compare_with_means_no_check():
    members = [device("a", (1,)), device("me", (0,))]
    trust = TrustStore(None)
    spot = checker(members, set(), trust)
    assert not spot.submit([replica("me", 0), replica("a", 1)], torch.tensor([[1]]), 3, GREEDY)
    assert trust.all() == []


def test_a_disagreement_without_a_referee_blames_nobody():
    members = [device("liar", (1,)), device("b", (1,)), device("me", (0,))]
    trust = TrustStore(None)
    spot = checker(members, {"liar"}, trust)
    assert spot.submit([replica("me", 0), replica("liar", 1)], torch.tensor([[1]]), 7, GREEDY)
    wait_idle(spot)
    assert trust.get("liar").failures == 0 and trust.get("b").failures == 0
