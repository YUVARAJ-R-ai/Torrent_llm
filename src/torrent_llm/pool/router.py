"""Choosing which copy of each stage a request goes through.

With replicas, a request has a choice at every stage, and the choice is what
spreads work across the pool: two people asking at once should land on different
devices when there are different devices to land on, and nobody should be sent
to a device that is busy when an idle one holds the same stage.

The score is deliberately simple. It is the work already queued on a copy,
scaled by how fast that device is, plus how far away it is. The lowest wins, and
ties are broken at random so equal devices share load instead of the first one
in the list taking everything.
"""

from __future__ import annotations

import random
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from torrent_llm.pool.config import PoolConfig
from torrent_llm.pool.members import ACTIVE, READY, Member


class NoReplica(RuntimeError):
    """No usable copy of a stage exists right now."""

    def __init__(self, stage: int) -> None:
        super().__init__(f"no device is holding stage {stage} right now")
        self.stage = stage


@dataclass(frozen=True)
class Replica:
    """One device's copy of one stage, as a request would use it."""

    stage: int
    member: str
    host: str
    port: int
    speed: float
    link: dict[str, float] | None
    #: Work queued on this copy: the larger of what it reports and what this
    #: requester knows it has sent it.
    load: int

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


def replicas_of(
    stage: int,
    members: Sequence[Member],
    *,
    exclude: Collection[str] = (),
    local_load: Mapping[str, int] | None = None,
) -> list[Replica]:
    """Every copy of ``stage`` a new request may use.

    Draining and away devices are left out: they finish what they have and take
    nothing new, which is the whole point of draining.
    """
    local_load = local_load or {}
    found = []
    for m in members:
        if m.state != ACTIVE or m.name in exclude:
            continue
        hosted = m.hosts(stage, ready_only=True)
        if hosted is None or hosted.status != READY:
            continue
        address = f"{m.host}:{hosted.port}"
        found.append(
            Replica(
                stage=stage,
                member=m.name,
                host=m.host,
                port=hosted.port,
                speed=m.speed,
                link=m.link,
                load=max(hosted.inflight, local_load.get(address, 0)),
            )
        )
    return found


def score(replica: Replica, rtt_ms: Mapping[str, float] | None = None) -> float:
    """Lower is better. See the module docstring for what goes into it."""
    rtt = (rtt_ms or {}).get(replica.address)
    if rtt is None and replica.link:
        rtt = replica.link.get("latency_ms", 0.0)
    return (replica.load + 1) / max(replica.speed, 0.1) + (rtt or 0.0) / 100.0


def pick_route(
    config: PoolConfig,
    members: Sequence[Member],
    *,
    exclude: Collection[str] = (),
    local_load: Mapping[str, int] | None = None,
    rtt_ms: Mapping[str, float] | None = None,
    rng: random.Random | None = None,
) -> list[Replica]:
    """One copy of every stage, in pipeline order.

    Raises:
        NoReplica: some stage has no usable copy, naming the first one.
    """
    rng = rng or random
    route = []
    for stage in range(config.stages):
        options = replicas_of(stage, members, exclude=exclude, local_load=local_load)
        if not options:
            raise NoReplica(stage)
        best = min(score(r, rtt_ms) for r in options)
        # Within a hair of the best counts as a tie, so tiny differences in
        # reported load do not pin everyone to one device.
        route.append(rng.choice([r for r in options if score(r, rtt_ms) - best < 0.05]))
    return route
