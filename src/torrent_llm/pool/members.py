"""Who is in the pool, as each device currently believes it.

There is no registry server. Every device keeps its own copy of the table and
swaps it with a few random others every half second (gossip, see ``gossip.py``).
A copy becomes the same as everyone else's within a couple of rounds, and the
pool keeps working while any subset of devices is unreachable.

The rule that makes this safe to merge is **one writer per row**. A device is
the only one that ever changes its own entry, and it stamps every change with a
heartbeat that only goes up. Merging two tables is then "for each device, keep
whichever entry has the higher stamp", which can be done in any order, any
number of times, with the same result.

Whether a device is alive is *not* gossiped. It is each reader's own judgement:
a device is alive while its heartbeat keeps advancing in the entries this reader
receives. A device that is merely quiet looks the same as a dead one for a
moment, which is why there is a "suspect" stage in between.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

#: Device states, chosen by the device's owner.
ACTIVE = "active"  # lending resources and accepting work
DRAINING = "draining"  # finishing what it has, taking nothing new
AWAY = "away"  # the owner is using the device; it hosts nothing
LEFT = "left"  # said goodbye; a tombstone so others stop waiting for it

#: What a hosted stage is doing.
LOADING = "loading"
READY = "ready"
FAILED = "failed"

#: Liveness, as judged by the reader.
ALIVE = "alive"
SUSPECT = "suspect"
DOWN = "down"


@dataclass(frozen=True)
class HostedStage:
    """One stage of the model a device is holding."""

    stage: int
    port: int = 0
    status: str = LOADING
    #: Requests this stage is serving right now.
    inflight: int = 0
    #: Requests it has finished since it came up.
    served: int = 0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> HostedStage:
        return cls(**raw)


@dataclass(frozen=True)
class Member:
    """One device's entry in the table. Only that device ever writes it."""

    name: str
    host: str
    control_port: int
    label: str = ""
    os: str = ""
    hostname: str = ""
    #: The torch device actually in use, after ``auto`` was resolved.
    device: str = "cpu"
    memory_mb: int = 0
    slots: int = 0
    speed: float = 1.0
    link: dict[str, float] | None = None
    state: str = ACTIVE
    hosting: tuple[HostedStage, ...] = ()
    #: Together these order a device's changes: a restart begins a new
    #: incarnation, so its counter starting over is not mistaken for old news.
    incarnation: int = 0
    heartbeat: int = 0

    @property
    def stamp(self) -> tuple[int, int]:
        return (self.incarnation, self.heartbeat)

    @property
    def control_url(self) -> str:
        return f"http://{self.host}:{self.control_port}"

    def hosts(self, stage: int, *, ready_only: bool = False) -> HostedStage | None:
        for hosted in self.hosting:
            if hosted.stage == stage and (not ready_only or hosted.status == READY):
                return hosted
        return None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Member:
        fields = dict(raw)
        fields["hosting"] = tuple(HostedStage.from_dict(h) for h in fields.get("hosting", ()))
        return cls(**fields)


@dataclass
class _Seen:
    member: Member
    #: When this reader last saw the heartbeat advance, by its own clock.
    last_advance: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class MemberTable:
    """This device's view of the pool.

    Args:
        me: The name of the device that owns this view. It is always considered
            alive to itself.
        suspect_after: Seconds without a heartbeat advancing before a device is
            only *suspected* of being gone. Requests already avoid it.
        down_after: Seconds before it is declared gone, which is what makes the
            pool replace whatever it was holding.
        clock: Injected so tests can move time without sleeping.
    """

    def __init__(
        self,
        me: str,
        *,
        suspect_after: float = 2.5,
        down_after: float = 6.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.me = me
        self.suspect_after = suspect_after
        self.down_after = down_after
        self._clock = clock
        self._seen: dict[str, _Seen] = {}
        self._lock = threading.RLock()

    # --- writing -----------------------------------------------------------

    def set_self(self, member: Member) -> None:
        """Replace this device's own entry. The only way an entry is ever changed."""
        if member.name != self.me:
            raise ValueError(f"this table belongs to {self.me!r}, not {member.name!r}")
        with self._lock:
            self._seen[member.name] = _Seen(member=member, last_advance=self._clock())

    def merge(self, incoming: Iterable[Member]) -> list[str]:
        """Fold another device's table into this one.

        Returns the names whose entries actually changed, so a caller can tell
        whether the exchange taught it anything.
        """
        changed: list[str] = []
        now = self._clock()
        with self._lock:
            for member in incoming:
                if member.name == self.me:
                    # Nobody else's account of me beats my own.
                    continue
                known = self._seen.get(member.name)
                if known is None or member.stamp > known.member.stamp:
                    self._seen[member.name] = _Seen(member=member, last_advance=now)
                    changed.append(member.name)
        return changed

    # --- reading -----------------------------------------------------------

    def get(self, name: str) -> Member | None:
        with self._lock:
            seen = self._seen.get(name)
            return seen.member if seen else None

    def members(self) -> list[Member]:
        """Every device ever heard of, in a stable order."""
        with self._lock:
            return sorted((s.member for s in self._seen.values()), key=lambda m: m.name)

    def liveness(self, name: str) -> str:
        """Whether this reader thinks ``name`` is reachable."""
        with self._lock:
            seen = self._seen.get(name)
            if seen is None:
                return DOWN
            if name == self.me:
                return ALIVE
            if seen.member.state == LEFT:
                return DOWN
            quiet = self._clock() - seen.last_advance
            if quiet < self.suspect_after:
                return ALIVE
            return SUSPECT if quiet < self.down_after else DOWN

    def alive(self) -> list[Member]:
        """Devices that are reachable by this reader's judgement."""
        return [m for m in self.members() if self.liveness(m.name) == ALIVE]

    def snapshot(self) -> list[tuple[Member, str]]:
        """Every member with this reader's verdict on it, for display."""
        return [(m, self.liveness(m.name)) for m in self.members()]

    def digest(self) -> list[dict[str, Any]]:
        """The whole table in wire form, for gossip."""
        return [m.to_dict() for m in self.members()]

    def mark_unreachable(self, name: str) -> None:
        """Treat ``name`` as gone right now, on direct evidence.

        A request that just failed against a device knows more than the
        heartbeat timer does, so it should not have to wait for the timer.
        The device is brought back by its own next heartbeat, which is what
        makes a wrong call here harmless.
        """
        with self._lock:
            seen = self._seen.get(name)
            if seen is not None and name != self.me:
                seen.last_advance = self._clock() - self.down_after
