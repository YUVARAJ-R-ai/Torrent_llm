"""The parts of the pool that need no network: config, membership, placement, routing."""

import random

import pytest

from torrent_llm.pool.config import DeviceProfile, PoolConfig
from torrent_llm.pool.members import (
    ACTIVE,
    ALIVE,
    AWAY,
    DOWN,
    DRAINING,
    FAILED,
    LEFT,
    READY,
    SUSPECT,
    HostedStage,
    Member,
    MemberTable,
)
from torrent_llm.pool.placement import coverage, wanted_stages
from torrent_llm.pool.router import NoReplica, pick_route, replicas_of

CONFIG = PoolConfig(model_id="tiny", num_layers=6, stages=2, replicas=2, shard_mb=1000)


def member(name, *, slots=1, speed=1.0, state=ACTIVE, hosting=(), **kw) -> Member:
    return Member(
        name=name,
        host="127.0.0.1",
        control_port=9000,
        slots=slots,
        speed=speed,
        state=state,
        hosting=tuple(hosting),
        **kw,
    )


def ready(stage, port=50000, **kw) -> HostedStage:
    return HostedStage(stage=stage, port=port + stage, status=READY, **kw)


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


# --- config ---------------------------------------------------------------


def test_stages_cover_the_whole_model():
    specs = CONFIG.specs()

    assert [(s.start, s.end) for s in specs] == [(0, 3), (3, 6)]


def test_a_pool_cannot_have_more_stages_than_layers():
    with pytest.raises(ValueError, match="stages"):
        PoolConfig(model_id="tiny", num_layers=4, stages=5)


def test_device_slots_follow_memory():
    assert DeviceProfile(name="a", memory_mb=3200).slots(1500) == 2
    assert DeviceProfile(name="b", memory_mb=1400).slots(1500) == 0


def test_a_device_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown keys"):
        DeviceProfile.from_dict({"name": "a", "gpu": "rtx"})


def test_a_device_link_is_parsed():
    profile = DeviceProfile.from_dict(
        {"name": "a", "link": {"latency_ms": 40, "bandwidth_mbps": 50}}
    )

    assert profile.link.latency_ms == 40


# --- membership -----------------------------------------------------------


def test_a_higher_stamp_wins_and_a_lower_one_is_ignored():
    table = MemberTable("me")
    table.merge([member("a", heartbeat=5)])

    assert table.merge([member("a", heartbeat=3)]) == []
    assert table.merge([member("a", heartbeat=6, label="new")]) == ["a"]
    assert table.get("a").label == "new"


def test_a_restarted_device_is_not_mistaken_for_old_news():
    table = MemberTable("me")
    table.merge([member("a", incarnation=1, heartbeat=900)])

    # After a restart the counter starts over, but the incarnation is later.
    assert table.merge([member("a", incarnation=2, heartbeat=1)]) == ["a"]


def test_merging_is_order_independent():
    a, b = MemberTable("x"), MemberTable("y")
    rows = [member("p", heartbeat=1), member("p", heartbeat=3), member("q", heartbeat=2)]

    a.merge(rows)
    b.merge(reversed(rows))

    assert a.digest() == b.digest()


def test_nobody_else_can_overwrite_my_own_entry():
    table = MemberTable("me")
    table.set_self(member("me", heartbeat=1, label="mine"))

    table.merge([member("me", heartbeat=99, label="forged")])

    assert table.get("me").label == "mine"


def test_a_device_is_suspected_then_declared_down_when_it_goes_quiet():
    clock = Clock()
    table = MemberTable("me", suspect_after=2, down_after=5, clock=clock)
    table.merge([member("a", heartbeat=1)])

    assert table.liveness("a") == ALIVE
    clock.now += 3
    assert table.liveness("a") == SUSPECT
    clock.now += 3
    assert table.liveness("a") == DOWN


def test_a_new_heartbeat_brings_a_quiet_device_back():
    clock = Clock()
    table = MemberTable("me", suspect_after=2, down_after=5, clock=clock)
    table.merge([member("a", heartbeat=1)])
    clock.now += 10

    table.merge([member("a", heartbeat=2)])

    assert table.liveness("a") == ALIVE


def test_the_same_heartbeat_again_does_not_count_as_alive():
    clock = Clock()
    table = MemberTable("me", suspect_after=2, down_after=5, clock=clock)
    table.merge([member("a", heartbeat=1)])
    clock.now += 10

    # A stale copy relayed by someone else must not revive it.
    table.merge([member("a", heartbeat=1)])

    assert table.liveness("a") == DOWN


def test_a_device_that_said_goodbye_is_gone_at_once():
    table = MemberTable("me")
    table.merge([member("a", heartbeat=1)])
    table.merge([member("a", heartbeat=2, state=LEFT)])

    assert table.liveness("a") == DOWN


def test_direct_evidence_marks_a_device_down_without_waiting():
    table = MemberTable("me")
    table.merge([member("a", heartbeat=1)])

    table.mark_unreachable("a")

    assert table.liveness("a") == DOWN


def test_a_member_survives_the_wire():
    original = member("a", hosting=[ready(0, inflight=2)], label="Laptop", link={"latency_ms": 5})

    assert Member.from_dict(original.to_dict()) == original


# --- placement ------------------------------------------------------------


def test_an_empty_pool_gets_every_stage_covered_twice():
    devices = [member(n) for n in "abcd"]

    wanted = wanted_stages(CONFIG, devices)

    held = [s for stages in wanted.values() for s in stages]
    assert sorted(held) == [0, 0, 1, 1]
    # No device holds the same stage twice.
    assert all(len(set(stages)) == len(stages) for stages in wanted.values())


def test_every_device_computes_the_same_answer_from_the_same_table():
    devices = [member("c"), member("a"), member("b"), member("d")]

    assert wanted_stages(CONFIG, devices) == wanted_stages(CONFIG, list(reversed(devices)))


def test_a_device_never_gets_more_stages_than_it_has_room_for():
    devices = [member("a", slots=1), member("b", slots=1), member("c", slots=2)]

    wanted = wanted_stages(CONFIG, devices)

    assert all(len(wanted[m.name]) <= m.slots for m in devices)


def test_a_bigger_device_holds_more_of_the_model():
    devices = [member("big", slots=2, speed=4.0), member("s1"), member("s2")]

    wanted = wanted_stages(CONFIG, devices)

    assert wanted["big"] == [0, 1]


def test_a_stage_with_no_copies_is_filled_before_one_that_wants_more():
    # Stage 0 already has two copies; stage 1 has none. The spare goes to stage 1.
    devices = [
        member("a", hosting=[ready(0)]),
        member("b", hosting=[ready(0)]),
        member("spare"),
    ]

    assert wanted_stages(CONFIG, devices)["spare"] == [1]


def test_adding_a_device_never_moves_what_others_already_hold():
    before = [member("a", hosting=[ready(0)]), member("b", hosting=[ready(1)])]
    after = [*before, member("c")]

    wanted = wanted_stages(CONFIG, after)

    assert wanted["a"] == [0]
    assert wanted["b"] == [1]


def test_a_draining_device_stops_counting_so_a_replacement_comes_up_first():
    devices = [
        member("a", hosting=[ready(0)]),
        member("b", hosting=[ready(0)], state=DRAINING),
        member("c", hosting=[ready(1)]),
        member("d", hosting=[ready(1)]),
        member("spare"),
    ]

    wanted = wanted_stages(CONFIG, devices)

    assert wanted["spare"] == [0]  # replaces the copy b is about to give up
    assert wanted["b"] == []  # and b is asked to let go


def test_an_away_device_is_asked_to_hold_nothing():
    devices = [member("a", hosting=[ready(0)], state=AWAY), member("b")]

    assert wanted_stages(CONFIG, devices)["a"] == []


def test_a_failed_load_is_not_counted_as_cover():
    broken = HostedStage(stage=0, status=FAILED)
    devices = [member("a", hosting=[broken]), member("b")]

    wanted = wanted_stages(CONFIG, devices)

    assert 0 not in wanted["a"]  # not asked to retry what just failed on it
    assert wanted["b"] == [0]
    assert wanted["a"] == [1]  # but still useful for the other stage


def test_with_too_few_devices_the_pool_covers_what_it_can():
    wanted = wanted_stages(CONFIG, [member("only")])

    assert len(wanted["only"]) == 1


def settled_pool():
    """Both stages already on two slow devices, as after a lender left and came back."""
    return [
        member("a", speed=1.0, hosting=[ready(0)]),
        member("b", speed=1.0, hosting=[ready(0)]),
        member("c", speed=1.0, hosting=[ready(1)]),
        member("d", speed=1.0, hosting=[ready(1)]),
    ]


def test_a_faster_idle_device_takes_work_from_slower_holders():
    devices = [*settled_pool(), member("gpu", speed=4.0, slots=2)]

    wanted = wanted_stages(CONFIG, devices)

    assert wanted["gpu"] == [0, 1]


def test_a_spare_no_faster_than_the_holders_stays_spare():
    devices = [*settled_pool(), member("same", speed=1.0), member("slower", speed=0.5)]

    wanted = wanted_stages(CONFIG, devices)

    assert wanted["same"] == [] and wanted["slower"] == []


def test_a_spare_only_takes_as_many_extra_copies_as_it_has_room_for():
    devices = [*settled_pool(), member("gpu", speed=4.0, slots=1)]

    assert len(wanted_stages(CONFIG, devices)["gpu"]) == 1


def test_taking_extra_copies_never_makes_anyone_drop_what_they_hold():
    before = settled_pool()
    after = [*before, member("gpu", speed=4.0, slots=2)]

    wanted = wanted_stages(CONFIG, after)

    for m in before:
        assert [h.stage for h in m.hosting] == wanted[m.name]


def test_spare_capacity_can_be_switched_off():
    config = PoolConfig(
        model_id="tiny", num_layers=6, stages=2, replicas=2, shard_mb=1000, use_spare_capacity=False
    )
    devices = [*settled_pool(), member("gpu", speed=4.0, slots=2)]

    assert wanted_stages(config, devices)["gpu"] == []


def test_a_draining_fast_device_is_not_given_extra_copies():
    devices = [*settled_pool(), member("gpu", speed=4.0, slots=2, state=DRAINING)]

    assert wanted_stages(CONFIG, devices)["gpu"] == []


def test_coverage_counts_only_usable_copies():
    devices = [
        member("a", hosting=[ready(0)]),
        member("b", hosting=[HostedStage(stage=0, status="loading")]),
        member("c", hosting=[ready(1)], state=DRAINING),
    ]

    assert coverage(CONFIG, devices) == [1, 0]
    assert coverage(CONFIG, devices, ready_only=False) == [2, 0]


# --- routing --------------------------------------------------------------


def two_stage_pool():
    return [
        member("a0", hosting=[ready(0, 50000)]),
        member("a1", hosting=[ready(0, 50100)]),
        member("b0", hosting=[ready(1, 50200)]),
        member("b1", hosting=[ready(1, 50300)]),
    ]


def test_a_route_has_one_copy_of_every_stage_in_order():
    route = pick_route(CONFIG, two_stage_pool(), rng=random.Random(1))

    assert [r.stage for r in route] == [0, 1]


def test_a_missing_stage_is_named():
    pool = [m for m in two_stage_pool() if m.name != "b0" and m.name != "b1"]

    with pytest.raises(NoReplica) as excinfo:
        pick_route(CONFIG, pool)

    assert excinfo.value.stage == 1


def test_excluded_devices_are_not_used():
    route = pick_route(CONFIG, two_stage_pool(), exclude={"a0", "b0"}, rng=random.Random(2))

    assert [r.member for r in route] == ["a1", "b1"]


def test_draining_and_away_devices_are_never_chosen():
    pool = two_stage_pool()
    pool[0] = member("a0", hosting=[ready(0)], state=DRAINING)
    pool[3] = member("b1", hosting=[ready(1)], state=AWAY)

    for seed in range(10):
        route = pick_route(CONFIG, pool, rng=random.Random(seed))
        assert [r.member for r in route] == ["a1", "b0"]


def test_a_stage_still_loading_is_not_routed_to():
    pool = [
        member("a", hosting=[HostedStage(stage=0, port=1, status="loading")]),
        member("b", hosting=[ready(0)]),
    ]

    assert [r.member for r in replicas_of(0, pool)] == ["b"]


def test_the_less_busy_copy_is_preferred():
    pool = two_stage_pool()
    pool[0] = member("a0", hosting=[ready(0, 50000, inflight=5)])

    for seed in range(10):
        assert pick_route(CONFIG, pool, rng=random.Random(seed))[0].member == "a1"


def test_load_this_requester_has_sent_counts_before_it_is_reported():
    pool = two_stage_pool()

    route = pick_route(CONFIG, pool, local_load={"127.0.0.1:50000": 3}, rng=random.Random(0))

    assert route[0].member == "a1"


def test_a_faster_device_is_preferred_when_load_is_equal():
    pool = two_stage_pool()
    pool[1] = member("a1", speed=4.0, hosting=[ready(0, 50100)])

    assert pick_route(CONFIG, pool, rng=random.Random(0))[0].member == "a1"


def test_a_nearer_device_is_preferred_when_all_else_is_equal():
    pool = two_stage_pool()
    pool[0] = member("a0", hosting=[ready(0, 50000)], link={"latency_ms": 200.0})
    pool[1] = member("a1", hosting=[ready(0, 50100)], link={"latency_ms": 10.0})

    assert pick_route(CONFIG, pool, rng=random.Random(0))[0].member == "a1"


def test_equal_copies_share_the_work():
    pool = two_stage_pool()
    rng = random.Random(7)

    chosen = {pick_route(CONFIG, pool, rng=rng)[0].member for _ in range(40)}

    assert chosen == {"a0", "a1"}
