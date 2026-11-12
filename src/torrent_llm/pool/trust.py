"""Who to trust in a swarm: reputation, bans, and a ledger of give and take.

In a public swarm anyone can run a stage, including someone who returns wrong
numbers on purpose (or a broken machine that does it by accident). A device cannot
check every answer, but it can check some: now and then it recomputes one pass on a
different copy of one stage and compares (``Verifier``). A device whose answers
disagree with the majority twice is banned **locally**: this device stops routing
through it and refuses its connections. Bans are never shared, because a shared
accusation is an easy way to frame an honest device; every device judges for itself,
as BitTorrent clients do.

The ledger (issue #19) records, per peer, the compute this device spent for it and
the compute it spent for this device. When this device is at capacity it serves
the peers with the best give-and-take ratio first, which rewards devices that lend
and costs free-riders only when there is contention.

Everything is local and persisted in ``trust.json`` (``net.paths.trust_path``).
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Reputation:
    peer_id: str
    checks: int = 0
    failures: int = 0
    banned: bool = False
    reason: str = ""
    banned_at: float = 0.0
    #: Milliseconds of compute this device spent serving the peer / the peer spent
    #: serving this device.
    served_ms: float = 0.0
    used_ms: float = 0.0

    @property
    def score(self) -> float:
        """0..1: share of checks passed, with a prior of one pass (new peers score 1.0)."""
        return (self.checks - self.failures + 1) / (self.checks + 1)

    @property
    def ratio(self) -> float:
        """What the peer gave over what it took, with +1 smoothing both sides."""
        return (self.used_ms + 1.0) / (self.served_ms + 1.0)


class TrustStore:
    """Reputation and bans for every peer this device has dealt with.

    Behaviour (tests pin these down):

    * ``record_check(peer, ok)`` counts a check; after ``ban_after`` failures the peer
      is banned with reason ``"returned wrong results"``.
    * ``ban``/``unban`` by hand; ``is_banned``; ``gatekeeper(peer)`` returns
      ``"banned: <reason>"`` for a banned peer, else ``None`` (for ``Node.add_gatekeeper``).
    * ``record_served(peer, ms)`` / ``record_used(peer, ms)`` update the ledger; negative,
      NaN or infinite ms raise ``ValueError``.
    * ``priority(peer)`` = ``reputation.ratio * reputation.score``, 0 for a banned peer.
    * ``get(peer)`` returns a :class:`Reputation` (a fresh one for unknown peers); ``all()``
      every known peer, sorted by peer id.
    * With a ``path``, every change is saved atomically (temp file + replace) at most once
      per ``save_interval_s`` (and on ``flush()``); loading a missing file starts empty, a
      corrupt one is renamed to ``trust.json.corrupt`` and starts empty. At most
      ``max_peers`` entries are kept (unbanned peers with the oldest activity go first;
      bans are never evicted).
    * Thread-safe.
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        ban_after: int = 2,
        max_peers: int = 10_000,
        save_interval_s: float = 2.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Initialize the trust store with storage path and configuration."""
        self._path = Path(path) if path is not None else None
        self._ban_after = ban_after
        self._max_peers = max_peers
        self._save_interval_s = save_interval_s
        self._clock = clock
        self._lock = threading.RLock()
        self._dirty: bool = False
        self._peers: dict[str, tuple[Reputation, float]] = {}
        self._last_save: float = self._clock()

        if self._path is not None:
            self._load()

    def _load(self) -> None:
        """Load reputation and ledger data from disk, handling corruption."""
        if self._path is None or not self._path.exists():
            return
        try:
            raw_text = self._path.read_text(encoding="utf-8")
            data = json.loads(raw_text)
            if not isinstance(data, dict):
                raise ValueError("root is not an object")
            if data.get("version") != 1 or type(data.get("version")) is not int:
                raise ValueError(f"unsupported version: {data.get('version')}")
            raw_peers = data.get("peers")
            if not isinstance(raw_peers, dict):
                raise ValueError("peers is not an object")

            loaded_peers: dict[str, tuple[Reputation, float]] = {}
            for peer_id, entry in raw_peers.items():
                if type(peer_id) is not str or len(peer_id) > 64:
                    continue
                if not isinstance(entry, dict):
                    raise ValueError(f"entry for peer {peer_id} is not an object")

                checks = entry.get("checks")
                failures = entry.get("failures")
                banned = entry.get("banned")
                reason = entry.get("reason")
                banned_at = entry.get("banned_at")
                served_ms = entry.get("served_ms")
                used_ms = entry.get("used_ms")
                last_seen = entry.get("last_seen")

                if type(checks) is not int or checks < 0:
                    raise ValueError(f"invalid checks for {peer_id}")
                if type(failures) is not int or failures < 0:
                    raise ValueError(f"invalid failures for {peer_id}")
                if type(banned) is not bool:
                    raise ValueError(f"invalid banned for {peer_id}")
                if type(reason) is not str:
                    raise ValueError(f"invalid reason for {peer_id}")
                if (
                    type(banned_at) not in (int, float)
                    or type(banned_at) is bool
                    or math.isnan(banned_at)
                    or math.isinf(banned_at)
                ):
                    raise ValueError(f"invalid banned_at for {peer_id}")
                if (
                    type(served_ms) not in (int, float)
                    or type(served_ms) is bool
                    or math.isnan(served_ms)
                    or math.isinf(served_ms)
                    or served_ms < 0
                ):
                    raise ValueError(f"invalid served_ms for {peer_id}")
                if (
                    type(used_ms) not in (int, float)
                    or type(used_ms) is bool
                    or math.isnan(used_ms)
                    or math.isinf(used_ms)
                    or used_ms < 0
                ):
                    raise ValueError(f"invalid used_ms for {peer_id}")
                if (
                    type(last_seen) not in (int, float)
                    or type(last_seen) is bool
                    or math.isnan(last_seen)
                    or math.isinf(last_seen)
                ):
                    raise ValueError(f"invalid last_seen for {peer_id}")

                rep = Reputation(
                    peer_id=peer_id,
                    checks=checks,
                    failures=failures,
                    banned=banned,
                    reason=reason,
                    banned_at=float(banned_at),
                    served_ms=float(served_ms),
                    used_ms=float(used_ms),
                )
                loaded_peers[peer_id] = (rep, float(last_seen))

            self._peers = loaded_peers
            self._evict(keep=peer_id)
        except Exception:
            corrupt_path = self._path.with_name(self._path.name + ".corrupt")
            try:
                os.replace(self._path, corrupt_path)
            except OSError:
                pass
            self._peers = {}

    def _evict(self, keep: str | None = None) -> None:
        """Evict unbanned peers with oldest activity if over capacity, never ``keep``
        (the peer just recorded: its record must not vanish the moment it is written)."""
        while len(self._peers) > self._max_peers:
            unbanned = [
                (peer_id, last_seen)
                for peer_id, (rep, last_seen) in self._peers.items()
                if not rep.banned and peer_id != keep
            ]
            if not unbanned:
                break
            oldest_peer_id = min(unbanned, key=lambda item: (item[1], item[0]))[0]
            del self._peers[oldest_peer_id]

    def _save_to_disk(self) -> None:
        """Write internal state to path atomically."""
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        peers_dict = {
            peer_id: {
                "banned": rep.banned,
                "banned_at": rep.banned_at,
                "checks": rep.checks,
                "failures": rep.failures,
                "last_seen": last_seen,
                "reason": rep.reason,
                "served_ms": rep.served_ms,
                "used_ms": rep.used_ms,
            }
            for peer_id, (rep, last_seen) in self._peers.items()
        }
        payload = {
            "peers": peers_dict,
            "version": 1,
        }
        data = json.dumps(payload, sort_keys=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=".trust-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
            os.replace(tmp, self._path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def _maybe_save(self) -> None:
        """Save if dirty and more than save_interval_s passed since last save."""
        if self._path is None or not self._dirty:
            return
        now = self._clock()
        if (now - self._last_save) >= self._save_interval_s:
            try:
                self._save_to_disk()
                self._dirty = False
                self._last_save = now
            except Exception as exc:
                # Wait a full interval before trying again, rather than retrying (and
                # warning) on every single record under the lock.
                self._last_save = now
                logger.warning("failed to save trust store: %s", exc)

    def get(self, peer_id: str) -> Reputation:
        """Get the reputation of a peer, or a fresh default if unknown."""
        with self._lock:
            if peer_id in self._peers:
                return self._peers[peer_id][0]
            return Reputation(peer_id=peer_id)

    def all(self) -> list[Reputation]:
        """Return reputations for all known peers, sorted by peer id."""
        with self._lock:
            return [self._peers[k][0] for k in sorted(self._peers.keys())]

    def record_check(self, peer_id: str, ok: bool) -> Reputation:
        """Record the result of spot-checking a peer's computation."""
        with self._lock:
            now = self._clock()
            rep, _ = self._peers.get(peer_id, (Reputation(peer_id=peer_id), now))
            new_checks = rep.checks + 1
            new_failures = rep.failures if ok else rep.failures + 1

            if rep.banned:
                updated_rep = replace(rep, checks=new_checks, failures=new_failures)
            elif new_failures >= self._ban_after:
                updated_rep = replace(
                    rep,
                    checks=new_checks,
                    failures=new_failures,
                    banned=True,
                    reason="returned wrong results",
                    banned_at=now,
                )
            else:
                updated_rep = replace(
                    rep,
                    checks=new_checks,
                    failures=new_failures,
                )

            self._peers[peer_id] = (updated_rep, now)
            self._evict(keep=peer_id)
            self._dirty = True
            self._maybe_save()
            return updated_rep

    def record_served(self, peer_id: str, ms: float) -> None:
        """Record compute spent by this device serving the given peer."""
        if (
            type(ms) not in (int, float)
            or type(ms) is bool
            or math.isnan(ms)
            or math.isinf(ms)
            or ms < 0
        ):
            raise ValueError(f"invalid compute ms: {ms!r}")
        with self._lock:
            now = self._clock()
            rep, _ = self._peers.get(peer_id, (Reputation(peer_id=peer_id), now))
            updated_rep = replace(rep, served_ms=rep.served_ms + float(ms))
            self._peers[peer_id] = (updated_rep, now)
            self._evict(keep=peer_id)
            self._dirty = True
            self._maybe_save()

    def record_used(self, peer_id: str, ms: float) -> None:
        """Record compute spent by the peer serving this device."""
        if (
            type(ms) not in (int, float)
            or type(ms) is bool
            or math.isnan(ms)
            or math.isinf(ms)
            or ms < 0
        ):
            raise ValueError(f"invalid compute ms: {ms!r}")
        with self._lock:
            now = self._clock()
            rep, _ = self._peers.get(peer_id, (Reputation(peer_id=peer_id), now))
            updated_rep = replace(rep, used_ms=rep.used_ms + float(ms))
            self._peers[peer_id] = (updated_rep, now)
            self._evict(keep=peer_id)
            self._dirty = True
            self._maybe_save()

    def ban(self, peer_id: str, reason: str) -> None:
        """Ban a peer manually with a specified reason."""
        with self._lock:
            now = self._clock()
            rep, _ = self._peers.get(peer_id, (Reputation(peer_id=peer_id), now))
            updated_rep = replace(
                rep,
                banned=True,
                reason=reason,
                banned_at=now,
            )
            self._peers[peer_id] = (updated_rep, now)
            self._evict(keep=peer_id)
            self._dirty = True
            self._maybe_save()

    def unban(self, peer_id: str) -> None:
        """Unban a peer, resetting failure count to give a second chance."""
        with self._lock:
            if peer_id not in self._peers:
                return
            now = self._clock()
            rep, _ = self._peers[peer_id]
            updated_rep = replace(
                rep,
                banned=False,
                reason="",
                banned_at=0.0,
                failures=0,
            )
            self._peers[peer_id] = (updated_rep, now)
            self._dirty = True
            self._maybe_save()

    def is_banned(self, peer_id: str) -> bool:
        """Check whether a peer is currently banned."""
        with self._lock:
            if peer_id in self._peers:
                return self._peers[peer_id][0].banned
            return False

    def gatekeeper(self, peer_id: str) -> str | None:
        """Return rejection reason for banned peers, or None if accepted."""
        with self._lock:
            if peer_id in self._peers and self._peers[peer_id][0].banned:
                return f"banned: {self._peers[peer_id][0].reason}"
            return None

    def priority(self, peer_id: str) -> float:
        """Calculate peer priority based on give/take ratio and verification score."""
        with self._lock:
            rep = self._peers[peer_id][0] if peer_id in self._peers else Reputation(peer_id=peer_id)
            if rep.banned:
                return 0.0
            return rep.ratio * rep.score

    def flush(self) -> None:
        """Save immediately if the store has unsaved changes."""
        with self._lock:
            if self._path is None or not self._dirty:
                return
            self._save_to_disk()
            self._dirty = False
            self._last_save = self._clock()


def outputs_agree(a: torch.Tensor, b: torch.Tensor, *, tolerance: float = 0.02) -> bool:
    """Whether two devices' outputs for the same input are the same answer.

    Integer tensors (chosen token ids) must be equal. Floating tensors (hidden states)
    differ slightly between hardware and kernels, so they agree when the shapes match,
    neither has NaN/inf, the relative error ``||a-b|| / max(||a||, 1e-6)`` is at most
    ``tolerance`` and the cosine similarity is at least ``1 - tolerance / 10``. A shape or
    dtype-kind mismatch is disagreement. Compared in float32 on the CPU.
    """
    if a.shape != b.shape:
        return False
    if a.dtype.is_complex or b.dtype.is_complex:
        # Hidden states and token ids are never complex: treat it as a broken answer.
        return False

    is_int_a = (not a.dtype.is_floating_point) and (not a.dtype.is_complex)
    is_int_b = (not b.dtype.is_floating_point) and (not b.dtype.is_complex)
    is_float_a = a.dtype.is_floating_point
    is_float_b = b.dtype.is_floating_point

    if is_int_a != is_int_b or is_float_a != is_float_b:
        return False

    if a.numel() == 0:
        return True

    if is_int_a:
        return bool(torch.equal(a.detach().cpu(), b.detach().cpu()))

    a_f = a.detach().to(device="cpu", dtype=torch.float32)
    b_f = b.detach().to(device="cpu", dtype=torch.float32)

    if not torch.isfinite(a_f).all() or not torch.isfinite(b_f).all():
        return False

    norm_diff = torch.linalg.norm(a_f - b_f).item()
    norm_a = torch.linalg.norm(a_f).item()
    norm_b = torch.linalg.norm(b_f).item()

    rel_err = norm_diff / max(norm_a, 1e-6)
    if rel_err > tolerance:
        return False

    if norm_a == 0.0 and norm_b == 0.0:
        cos_sim = 1.0
    elif norm_a == 0.0 or norm_b == 0.0:
        cos_sim = 0.0
    else:
        dot = torch.sum(a_f * b_f).item()
        cos_sim = dot / (norm_a * norm_b)
        cos_sim = max(-1.0, min(1.0, cos_sim))

    return cos_sim >= (1.0 - (tolerance / 10.0))


class Verifier:
    """Decides which requests to spot-check: each request is checked with probability
    ``rate`` (0 disables), choosing one pass of it uniformly. Deterministic given ``rng``."""

    def __init__(self, rate: float, *, rng: Callable[[], float] | None = None) -> None:
        """Initialize spot-check verifier with rate and optional RNG."""
        if type(rate) is bool or not (0.0 <= rate <= 1.0) or math.isnan(rate):
            raise ValueError("rate must be in [0, 1]")
        self.rate = rate
        self._rng = random.random if rng is None else rng
        self._lock = threading.Lock()

    def check_this_request(self) -> bool:
        """Decide whether to spot-check this request."""
        if self.rate == 0.0:
            return False
        if self.rate == 1.0:
            return True
        with self._lock:
            return self._rng() < self.rate

    def pick_pass(self, max_new_tokens: int) -> int:
        """Which pass index (0 = prefill) of a request to check."""
        with self._lock:
            val = int(self._rng() * max_new_tokens)
            high = max(0, max_new_tokens - 1)
            return min(high, max(0, val))
