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

from torrent_llm.codec import get_codec
from torrent_llm.pool.config import DeviceProfile, PoolConfig
from torrent_llm.pool.discovery import DISCOVERY_PORT, Announcement, Discovery
from torrent_llm.pool.gossip import PoolMismatch, choose_targets, exchange
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
from torrent_llm.pool.owner import OwnerMonitor
from torrent_llm.pool.placement import wanted_stages
from torrent_llm.pool.runner import PoolRunner
from torrent_llm.shard import resolve_device

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
        discover: bool = False,
        discovery_port: int = DISCOVERY_PORT,
        shard_port_base: int = 0,
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
            config.specs(),
            loader,
            model_id=config.model_id,
            bind=self.bind,
            token=token,
            codec=get_codec(config.codec, **config.codec_args),
            port_base=shard_port_base,
            max_requests=profile.request_limit(resolve_device(profile.device)),
            clock=clock,
        )
        self.owner_monitor = None
        self.discover = discover
        self.discovery_port = discovery_port
        self.discovery: Discovery | None = None
        self._reported_clashes: set[str] = set()
        self._announced = False
        self._asked = 0
        self._asked_tokens = 0
        self._reported_mismatches: set[str] = set()
        self.runner = PoolRunner(config, self.table, me=profile.name, token=token)

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
        if self.discover:
            self.discovery = Discovery(
                Announcement(self.name, self.host, self.control_port, self.pool_id),
                self._found,
                port=self.discovery_port,
            )
            self.discovery.start()
        if self.profile.watch_owner:
            self.owner_monitor = OwnerMonitor(
                self.set_owner_busy, busy_mb=self.profile.owner_busy_mb
            )
            self.owner_monitor.start()

    def stop(self, *, graceful: bool = True) -> None:
        """Leave the pool. Graceful tells the others so they need not wait to find out."""
        if self.owner_monitor is not None:
            self.owner_monitor.stop()
        if self.discovery is not None:
            self.discovery.stop()
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

    # --- finding others ------------------------------------------------------

    @property
    def pool_id(self) -> str:
        """Names the pool, so two groups on one network never mix."""
        c = self.config
        return f"{c.model_id}/{c.stages}/{c.replicas}/{c.codec_signature}"

    def _found(self, found: Announcement) -> None:
        """Another device announced itself. If it is new to us, introduce ourselves."""
        known = {(m.host, m.control_port) for m in self.table.members()}
        if (found.host, found.port) in known:
            return
        try:
            theirs = exchange(
                f"http://{found.address}",
                self.name,
                self.table.digest(),
                token=self.token,
                pool_id=self.pool_id,
            )
        except PoolMismatch:
            self._report_mismatch(f"http://{found.address}", found.name)
            return
        if theirs is not None:
            self.table.merge(theirs)
            self._record("joined", f"found {found.name} on the network", found.name)

    # --- who I am ------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.profile.name

    @property
    def state(self) -> str:
        return self._state

    def publish(self) -> Member:
        """Write my own entry, with a heartbeat that has moved on."""
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
                asked=self._asked,
                asked_tokens=self._asked_tokens,
                incarnation=self._incarnation,
                heartbeat=self._heartbeat,
            )
            self.table.set_self(member)
            return member

    # --- gossip --------------------------------------------------------------

    def _gossip(self) -> None:
        digest = self.table.digest()
        for url in choose_targets(self.table, self.join, rng=self._rng):
            try:
                theirs = exchange(url, self.name, digest, token=self.token, pool_id=self.pool_id)
            except PoolMismatch:
                self._report_mismatch(url)
                continue
            if theirs is not None:
                self.table.merge(theirs)

    def _report_mismatch(self, url: str, name: str | None = None) -> None:
        """Say once that another device runs different settings, and so is ignored."""
        if url in self._reported_mismatches:
            return
        self._reported_mismatches.add(url)
        self._record(
            "down",
            f"{name or url} is running different pool settings (model, stages or codec), "
            "so it was ignored. Everyone must use the same configs/pool.yaml",
            name,
        )

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

    def record_asked(self, tokens: int) -> None:
        """A question was answered through this device: count what it used."""
        with self._lock:
            self._asked += 1
            self._asked_tokens += tokens

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

        for address in sorted(self.table.name_clashes - self._reported_clashes):
            self._reported_clashes.add(address)
            self._record(
                "down",
                f"another device at {address} is also called {self.name}; "
                "give each device its own --name or they will confuse each other",
                self.name,
            )

        if self._clock() - self._started < self.warmup_s:
            # Learning who is in the pool is not news. Take this view as the baseline
            # and say nothing until the device has had time to settle.
            self._previous = now
            return
        if not self._announced:
            # What was held back is summed up once, so the feed is not empty on a
            # pool that formed a moment ago.
            self._announced = True
            online = sum(1 for _, live in self.table.snapshot() if live == ALIVE)
            self._record(
                "joined",
                f"{self.name} is up, with {online} device{'s' if online != 1 else ''} in the pool",
                self.name,
            )

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
