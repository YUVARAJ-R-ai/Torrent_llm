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

import base64
import dataclasses
import json
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from torrent_llm.net.identity import Identity, peer_id_of, verify_signature

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
    #: Milliseconds this stage has spent computing for requests since it came up.
    #: Summed over a device's stages, it is what the device has given the pool.
    busy_ms: int = 0

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
    #: Questions asked through this device, and the tokens they produced. What the
    #: device has *used*, to set beside what it has given.
    asked: int = 0
    asked_tokens: int = 0
    incarnation: int = 0
    heartbeat: int = 0
    #: Over the overlay a device's ``name`` is its peer id, and its row is signed by
    #: the key behind that id, so nobody can forge or alter another device's entry.
    #: ``pubkey`` and ``sig`` are base64; ``addrs`` are overlay addresses.
    pubkey: str = ""
    addrs: tuple[str, ...] = ()
    sig: str = ""

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
        """Build from wire form. ``ValueError`` on anything malformed (it is untrusted)."""
        if not isinstance(raw, dict):
            raise ValueError("member entry is not an object")
        allowed = {f.name for f in dataclasses.fields(cls)}
        unknown = raw.keys() - allowed
        if unknown:
            raise ValueError(f"member entry has unknown keys: {sorted(unknown)[:5]}")
        fields = dict(raw)
        try:
            hosting = fields.get("hosting", ())
            if not isinstance(hosting, list | tuple) or len(hosting) > 64:
                raise ValueError("bad hosting list")
            fields["hosting"] = tuple(HostedStage.from_dict(h) for h in hosting)
            addrs = fields.get("addrs", ())
            if not isinstance(addrs, list | tuple) or len(addrs) > 16:
                raise ValueError("bad address list")
            fields["addrs"] = tuple(str(a) for a in addrs)
            member = cls(**fields)
        except (TypeError, AttributeError) as exc:
            raise ValueError(f"member entry is malformed: {exc}") from exc
        if not isinstance(member.name, str) or not isinstance(member.heartbeat, int):
            raise ValueError("member entry is malformed")
        return member

    def signed_bytes(self) -> bytes:
        """Canonical bytes covered by ``sig``: every field except ``sig``."""
        data = self.to_dict()
        data.pop("sig")
        return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()

    def signed_by(self, identity: Identity) -> Member:
        """This entry with ``pubkey`` and ``sig`` set by ``identity`` (whose peer id
        must be ``name``)."""
        if identity.peer_id != self.name:
            raise ValueError("a device can only sign its own entry")
        unsigned = dataclasses.replace(
            self, pubkey=base64.b64encode(identity.public_key).decode(), sig=""
        )
        signature = identity.sign(unsigned.signed_bytes())
        return dataclasses.replace(unsigned, sig=base64.b64encode(signature).decode())

    def verify(self) -> bool:
        """True when the entry is signed by the key behind its ``name``."""
        try:
            public = base64.b64decode(self.pubkey, validate=True)
            signature = base64.b64decode(self.sig, validate=True)
            return peer_id_of(public) == self.name and verify_signature(
                public, signature, self.signed_bytes()
            )
        except (ValueError, TypeError):
            return False


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
        require_signatures: bool = False,
        max_members: int = 4096,
    ) -> None:
        self.me = me
        #: Drop entries that are not signed by their own device (the overlay pool).
        self.require_signatures = require_signatures
        self.max_members = max_members
        self.suspect_after = suspect_after
        self.down_after = down_after
        self._clock = clock
        self._seen: dict[str, _Seen] = {}
        self._lock = threading.RLock()
        #: Addresses of other devices that are also calling themselves ``me``.
        self.name_clashes: set[str] = set()

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
                    # Nobody else's account of me beats my own. But if the
                    # account comes from a different place, two devices share a
                    # name, which would make them overwrite each other: say so.
                    mine = self._seen.get(self.me)
                    if mine is not None and (member.host, member.control_port) != (
                        mine.member.host,
                        mine.member.control_port,
                    ):
                        self.name_clashes.add(f"{member.host}:{member.control_port}")
                    continue
                known = self._seen.get(member.name)
                if known is None and len(self._seen) >= self.max_members:
                    continue
                if known is None or member.stamp > known.member.stamp:
                    if self.require_signatures and not member.verify():
                        continue
                    self._seen[member.name] = _Seen(member=member, last_advance=now)
                    changed.append(member.name)
        return changed

    def stamps(self) -> dict[str, list[int]]:
        """``{name: [incarnation, heartbeat]}`` for every entry: the gossip digest."""
        with self._lock:
            return {n: list(s.member.stamp) for n, s in self._seen.items()}

    def newer_than(self, stamps: dict[str, Any], *, limit: int = 512) -> list[Member]:
        """Entries the holder of ``stamps`` lacks or holds an older version of."""
        out: list[Member] = []
        with self._lock:
            for name, seen in self._seen.items():
                theirs = stamps.get(name)
                try:
                    older = theirs is None or tuple(theirs) < seen.member.stamp
                except TypeError:
                    older = True
                if older:
                    out.append(seen.member)
                    if len(out) >= limit:
                        break
        return out

    def forget(self, older_than_s: float) -> list[str]:
        """Drop entries that have been down (or gone) for ``older_than_s`` seconds."""
        now = self._clock()
        with self._lock:
            stale = [
                n
                for n, s in self._seen.items()
                if n != self.me and now - s.last_advance > older_than_s
            ]
            for name in stale:
                del self._seen[name]
        return stale

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
