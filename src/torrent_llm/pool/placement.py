"""Deciding which device holds which stage of the model.

This is a pure function of what the table says. There is no leader and no
vote: every device runs the same function over its own view and acts only on its
own answer ("should *I* be holding stage 1?"). Views that agree give answers that
agree, and views that briefly disagree can at worst load one stage too many,
which is harmless.

Three rules keep it calm:

* **Sticky.** A device that already holds a stage keeps it. Adding a device to
  the pool never makes anyone drop or reload anything.
* **Make before break.** A device that is draining or away stops counting as
  cover for what it holds, so the pool brings up a replacement *before* it
  unloads, not after.
* **Coverage first.** The stage with the fewest copies is always filled next, so
  a stage with none beats a stage that merely wants a third.
"""

from __future__ import annotations

from collections.abc import Sequence

from torrent_llm.pool.config import PoolConfig
from torrent_llm.pool.members import ACTIVE, FAILED, Member


def wanted_stages(config: PoolConfig, members: Sequence[Member]) -> dict[str, list[int]]:
    """For every device, the stages it should be holding.

    Pass only devices that are alive. Devices that are not :data:`ACTIVE` are
    wanted to hold nothing.
    """
    eligible = [m for m in members if m.state == ACTIVE]
    wanted: dict[str, list[int]] = {m.name: [] for m in members}

    # What each device already holds stays held. A failed load does not count:
    # the device is not holding it, whatever it intended.
    for m in eligible:
        wanted[m.name] = sorted(h.stage for h in m.hosting if h.status != FAILED)

    coverage = {stage: 0 for stage in range(config.stages)}
    for m in eligible:
        for stage in wanted[m.name]:
            coverage[stage] = coverage.get(stage, 0) + 1

    def free(m: Member) -> int:
        return m.slots - len(wanted[m.name])

    # Fill the thinnest stage first, until everything has enough copies or the
    # pool has no room left.
    while True:
        thin = sorted(
            (stage for stage, count in coverage.items() if count < config.replicas),
            key=lambda stage: (coverage[stage], stage),
        )
        placed = False
        for stage in thin:
            candidates = [
                m
                for m in eligible
                if free(m) > 0 and stage not in wanted[m.name] and not _failed_here(m, stage)
            ]
            if not candidates:
                continue
            # Fastest first, then the one with the most room, then by name so
            # every device reaches the same answer from the same table.
            best = min(candidates, key=lambda m: (-m.speed, -free(m), m.name))
            wanted[best.name] = sorted([*wanted[best.name], stage])
            coverage[stage] += 1
            placed = True
            break
        if not placed:
            break

    if config.use_spare_capacity:
        _add_faster_copies(eligible, wanted, free)
    return wanted


def _add_faster_copies(eligible, wanted, free) -> None:
    """Put spare room to work.

    Once every stage has the copies it needs, a device that still has room and is
    *faster than the slowest copy of some stage* gains by holding that stage too:
    requests prefer the fastest copy, so the pool gets quicker and the idle device
    stops being idle. A device no faster than the current holders would only
    duplicate what exists, so it stays spare.

    This only ever adds copies. Nothing already held is dropped, so it cannot make
    the pool churn.
    """
    while True:
        for stage in sorted({s for stages in wanted.values() for s in stages}):
            holders = [m for m in eligible if stage in wanted[m.name]]
            slowest = min(m.speed for m in holders)
            candidates = [
                m
                for m in eligible
                if free(m) > 0
                and stage not in wanted[m.name]
                and not _failed_here(m, stage)
                and m.speed > slowest
            ]
            if candidates:
                best = min(candidates, key=lambda m: (-m.speed, -free(m), m.name))
                wanted[best.name] = sorted([*wanted[best.name], stage])
                break
        else:
            return


def _failed_here(member: Member, stage: int) -> bool:
    """A device whose load of ``stage`` just failed is not asked to try again.

    Another device is tried instead. The device clears its own failure after a
    cooldown, so a bad moment (out of memory while a game was running) does not
    rule it out for good.
    """
    return any(h.stage == stage and h.status == FAILED for h in member.hosting)


def coverage(
    config: PoolConfig, members: Sequence[Member], *, ready_only: bool = True
) -> list[int]:
    """How many usable copies of each stage exist among these devices."""
    counts = [0] * config.stages
    for m in members:
        if m.state != ACTIVE:
            continue
        for stage in range(config.stages):
            if m.hosts(stage, ready_only=ready_only):
                counts[stage] += 1
    return counts
