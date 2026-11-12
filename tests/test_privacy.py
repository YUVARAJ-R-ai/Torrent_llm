"""Privacy mode: the first and last stage stay on the asking device."""

from __future__ import annotations

import pytest

from torrent_llm.pool.config import PoolConfig
from torrent_llm.pool.members import ACTIVE, READY, HostedStage, Member, MemberTable
from torrent_llm.pool.router import NoReplica, pick_route
from torrent_llm.pool.runner import PoolRunner, PoolUnavailable

CONFIG = PoolConfig(model_id="m", num_layers=12, stages=3, replicas=1, shard_mb=100)


def device(name: str, stages: tuple[int, ...], slots: int = 3) -> Member:
    return Member(
        name=name,
        host="h",
        control_port=1,
        slots=slots,
        state=ACTIVE,
        hosting=tuple(HostedStage(stage=s, port=100 + s, status=READY) for s in stages),
        heartbeat=1,
    )


def test_pins_force_a_stage_onto_one_device():
    members = [device("me", (0, 2)), device("a", (0, 1, 2))]
    for _ in range(20):  # the choice would otherwise be random among equals
        route = pick_route(CONFIG, members, pin={0: "me", 2: "me"})
        assert [r.member for r in route] == ["me", "a", "me"]


def test_a_pin_the_device_cannot_honour_is_no_replica():
    members = [device("me", (1,)), device("a", (0, 1, 2))]
    with pytest.raises(NoReplica) as caught:
        pick_route(CONFIG, members, pin={0: "me"})
    assert caught.value.stage == 0


def test_the_runner_routes_privately_and_explains_when_it_cannot():
    table = MemberTable("me")
    table.set_self(device("me", (0, 2)))
    table.merge([device("a", (0, 1, 2))])
    runner = PoolRunner(CONFIG, table, me="me", privacy=True, patience=0.1)
    assert [r.member for r in runner.route()] == ["me", "a", "me"]

    lonely = MemberTable("me")
    lonely.set_self(device("me", (1,)))
    lonely.merge([device("a", (0, 1, 2))])
    runner = PoolRunner(CONFIG, lonely, me="me", privacy=True, patience=0.1)
    with pytest.raises(PoolUnavailable, match="privacy mode keeps stage 1"):
        runner._await_route(set(), on_wait=lambda stage: None)


def test_without_privacy_any_copy_is_fine():
    table = MemberTable("me")
    table.set_self(device("me", (1,)))
    table.merge([device("a", (0, 1, 2))])
    runner = PoolRunner(CONFIG, table, me="me", patience=0.1)
    assert len(runner.route()) == 3
