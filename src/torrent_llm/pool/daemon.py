"""One device in the pool.

Every device runs this same program. It has no special role: it holds some
stages of the model, keeps its view of who else is in the pool, and can take a
question from anyone and route it across the pool. Nothing here is a head node,
and the pool keeps going if any single device (including the one a user happens
to be talking to) disappears.

A device does four things on a half-second beat:

1. **Heartbeat.** Publish its own entry: what it is holding, how busy it is.
2. **Gossip.** Swap tables with a couple of others.
3. **Reconcile.** Ask the placement function what it should be holding, and load
   or unload to match.
4. **Notice.** Compare the table with last time and record what changed, for the
   activity feed.

And it serves a small HTTP API for everything else (see ``api.py``).
"""

from __future__ import annotations

import logging
import os
import platform
import random
import socket
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from torrent_llm.pool.config import DeviceProfile, PoolConfig
from torrent_llm.pool.gossip import choose_targets, exchange
from torrent_llm.pool.hosting import Loader, ShardHost
from torrent_llm.pool.members import (
    ACTIVE,
    ALIVE,
    AWAY,
    DRAINING,
    LEFT,
    READY,
    Member,
    MemberTable,
)
from torrent_llm.pool.placement import wanted_stages
from torrent_llm.pool.runner import PoolRunner

logger = logging.getLogger(__name__)

#: The beat of the daemon loop.
TICK_S = 0.5
#: A device that has just started learns about everyone within its first few
#: beats. Reporting each of them as "joined" would read as news when it is only
#: the device catching up, so it stays quiet for this long.
WARMUP_S = 4.0
#: How long a draining device waits for a replacement before letting go anyway.
#: Bounded because the owner asked for the device back and should get it.
DRAIN_GRACE_S = 10.0


@dataclass
class Event:
    """One line of the activity feed."""

    t: float
    kind: str
    text: str
    member: str | None = None


class PeerDaemon:
    """One device: its stages, its view of the pool, and its beat."""

    def __init__(
        self,
        config: PoolConfig,
        profile: DeviceProfile,
        loader: Loader,
        *,
        host: str = "127.0.0.1",
        control_port: int = 0,
        bind: str | None = None,
        join: list[str] | None = None,
        token: str | None = None,
        tick_s: float = TICK_S,
        suspect_after: float = 2.5,
        down_after: float = 6.0,
        drain_grace_s: float = DRAIN_GRACE_S,
        warmup_s: float = WARMUP_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.profile = profile
        self.host = host
        self.bind = bind or host
        self.control_port = control_port
        self.join = list(join or [])
        self.token = token
        self.tick_s = tick_s
        self.drain_grace_s = drain_grace_s
        self.warmup_s = warmup_s

        self.table = MemberTable(
            profile.name, suspect_after=suspect_after, down_after=down_after, clock=clock
        )
        self.shards = ShardHost(
            config.specs(), loader, model_id=config.model_id, bind=self.bind, clock=clock
        )
        self.runner = PoolRunner(config, self.table, me=profile.name)

        self._incarnation = time.time_ns()
        self._heartbeat = 0
        self._state = ACTIVE
        self._drain_started = 0.0
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.events: deque[Event] = deque(maxlen=80)
        self._previous: dict[str, tuple[str, str, tuple]] = {}
        self._clock = clock
        self._started = clock()
        self._rng = random.Random()
        self.publish()

    # --- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Begin the beat in the background."""
        self._thread = threading.Thread(target=self._loop, name="peer-beat", daemon=True)
        self._thread.start()

    def stop(self, *, graceful: bool = True) -> None:
        """Leave the pool. Graceful tells the others so they need not wait to find out."""
        if graceful:
            with self._lock:
                self._state = LEFT
            self.publish()
            self._gossip()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.shards.close()

    def _loop(self) -> None:
        while not self._stop.wait(self.tick_s):
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - one bad beat must not end the device
                logger.exception("beat failed")

    def tick(self) -> None:
        """One beat: heartbeat, gossip, reconcile, notice."""
        self.shards.forget_old_failures()
        self.publish()
        self._gossip()
        self._reconcile()
        self.publish()
        self._notice()

    # --- who I am ------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.profile.name

    @property
    def state(self) -> str:
        return self._state

    def publish(self) -> Member:
        """Write my own entry, with a heartbeat that has moved on."""
        from torrent_llm.shard import resolve_device

        with self._lock:
            self._heartbeat += 1
            link = self.profile.link.describe() if self.profile.link else None
            member = Member(
                name=self.name,
                host=self.host,
                control_port=self.control_port,
                label=self.profile.label or self.name,
                os=platform.system(),
                hostname=socket.gethostname(),
                device=self.shards.device() or resolve_device(self.profile.device),
                memory_mb=self.profile.memory_mb,
                slots=self.profile.slots(self.config.shard_mb),
                speed=self.profile.speed,
                link=link,
                state=self._state,
                hosting=self.shards.hosting(),
                incarnation=self._incarnation,
                heartbeat=self._heartbeat,
            )
            self.table.set_self(member)
            return member

    # --- gossip --------------------------------------------------------------

    def _gossip(self) -> None:
        digest = self.table.digest()
        for url in choose_targets(self.table, self.join, rng=self._rng):
            theirs = exchange(url, self.name, digest, token=self.token)
            if theirs is not None:
                self.table.merge(theirs)

    def receive(self, sender: str, incoming: list[Member]) -> list[dict]:
        """Another device pushed its table; merge it and answer with mine."""
        self.table.merge(incoming)
        return self.table.digest()

    # --- reconcile -----------------------------------------------------------

    def _reconcile(self) -> None:
        with self._lock:
            state = self._state
        if state == LEFT:
            return

        alive = self.table.alive()
        wanted = set(wanted_stages(self.config, alive).get(self.name, []))
        held = self.shards.stages()

        if state == ACTIVE:
            for stage in sorted(wanted - held):
                logger.info("%s: taking on stage %d", self.name, stage)
                self.shards.start(stage)
            return

        # Draining or away: give up what I hold, but only once it is safe to.
        waited = self._clock() - self._drain_started
        for stage in sorted(held):
            if self.shards.inflight(stage) == 0 and self._safe_to_release(stage, alive, waited):
                self.shards.stop(stage)
        if not self.shards.stages():
            with self._lock:
                if self._state == DRAINING:
                    self._state = AWAY

    def _safe_to_release(self, stage: int, alive: list[Member], waited: float) -> bool:
        """Whether letting go of ``stage`` now leaves the pool no worse off than it must be.

        The aim is make before break: do not drop a copy until another one has
        taken its place. Three things end the wait:

        * the pool already has all the copies it wants without me;
        * at least one other copy exists and no free device could add another,
          so waiting would achieve nothing;
        * the grace period ran out. The owner asked for the device back and
          should get it even if that leaves the pool thinner.
        """
        others = [m for m in alive if m.name != self.name and m.state == ACTIVE]
        ready = sum(1 for m in others if m.hosts(stage, ready_only=True))
        if ready >= self.config.replicas:
            return True
        could_replace = any(m.slots - len(m.hosting) > 0 and not m.hosts(stage) for m in others)
        if ready >= 1 and not could_replace:
            return True
        return waited >= self.drain_grace_s

    # --- the owner's controls --------------------------------------------------

    def set_owner_busy(self, busy: bool) -> str:
        """The owner wants the device (or is done with it).

        Busy begins a drain: stop taking new work, let a replacement come up
        elsewhere, then release the memory. Not busy rejoins as a spare, and the
        pool decides what, if anything, to give it.
        """
        with self._lock:
            if busy and self._state == ACTIVE:
                self._state = DRAINING
                self._drain_started = self._clock()
            elif not busy and self._state in (DRAINING, AWAY):
                self._state = ACTIVE
        self.publish()
        return self._state

    # --- noticing --------------------------------------------------------------

    def _record(self, kind: str, text: str, member: str | None = None) -> None:
        self.events.append(Event(t=time.time(), kind=kind, text=text, member=member))

    def _notice(self) -> None:
        """Compare the table with the last beat and say what changed."""
        now: dict[str, tuple[str, str, tuple]] = {}
        for member, liveness in self.table.snapshot():
            stages = tuple(sorted((h.stage, h.status) for h in member.hosting))
            now[member.name] = (liveness, member.state, stages)

        if self._clock() - self._started < self.warmup_s:
            self._previous = now
            return

        for name, (liveness, state, stages) in now.items():
            if name == self.name and name not in self._previous:
                continue
            before = self._previous.get(name)
            if before is None:
                self._record("joined", f"{name} joined the pool", name)
                before = (liveness, state, ())
            if before[0] == ALIVE and liveness != ALIVE:
                verb = "left" if state == LEFT else "went down"
                self._record("down", f"{name} {verb}", name)
            elif before[0] != ALIVE and liveness == ALIVE and name in self._previous:
                self._record("up", f"{name} is back", name)
            if before[1] != state and state != LEFT:
                text = {
                    DRAINING: f"{name} is being used by its owner, handing its work over",
                    AWAY: f"{name} has released its memory",
                    ACTIVE: f"{name} is lending its resources again",
                }.get(state)
                if text:
                    self._record("state", text, name)
            for stage, status in set(stages) - set(before[2]):
                if status == READY:
                    self._record("stage", f"{name} now serves stage {stage + 1}", name)
            gone = {s for s, _ in before[2]} - {s for s, _ in stages}
            for stage in sorted(gone):
                if liveness == ALIVE:
                    self._record("stage", f"{name} stopped serving stage {stage + 1}", name)
        self._previous = now

    # --- reading ---------------------------------------------------------------

    def recent_events(self, since: float = 0.0) -> list[dict]:
        return [
            {"t": e.t, "kind": e.kind, "text": e.text, "member": e.member}
            for e in self.events
            if e.t > since
        ]


def crash() -> None:  # pragma: no cover - ends the process
    """Die at once, without saying goodbye. For simulating a device failing."""
    os._exit(1)
