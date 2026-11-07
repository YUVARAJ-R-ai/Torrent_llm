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

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch


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
        raise NotImplementedError

    def get(self, peer_id: str) -> Reputation:
        raise NotImplementedError

    def all(self) -> list[Reputation]:
        raise NotImplementedError

    def record_check(self, peer_id: str, ok: bool) -> Reputation:
        raise NotImplementedError

    def record_served(self, peer_id: str, ms: float) -> None:
        raise NotImplementedError

    def record_used(self, peer_id: str, ms: float) -> None:
        raise NotImplementedError

    def ban(self, peer_id: str, reason: str) -> None:
        raise NotImplementedError

    def unban(self, peer_id: str) -> None:
        raise NotImplementedError

    def is_banned(self, peer_id: str) -> bool:
        raise NotImplementedError

    def gatekeeper(self, peer_id: str) -> str | None:
        raise NotImplementedError

    def priority(self, peer_id: str) -> float:
        raise NotImplementedError

    def flush(self) -> None:
        raise NotImplementedError


def outputs_agree(a: torch.Tensor, b: torch.Tensor, *, tolerance: float = 0.02) -> bool:
    """Whether two devices' outputs for the same input are the same answer.

    Integer tensors (chosen token ids) must be equal. Floating tensors (hidden states)
    differ slightly between hardware and kernels, so they agree when the shapes match,
    neither has NaN/inf, the relative error ``||a-b|| / max(||a||, 1e-6)`` is at most
    ``tolerance`` and the cosine similarity is at least ``1 - tolerance / 10``. A shape or
    dtype-kind mismatch is disagreement. Compared in float32 on the CPU.
    """
    raise NotImplementedError


class Verifier:
    """Decides which requests to spot-check: each request is checked with probability
    ``rate`` (0 disables), choosing one pass of it uniformly. Deterministic given ``rng``."""

    def __init__(self, rate: float, *, rng: Callable[[], float] | None = None) -> None:
        if not (0.0 <= rate <= 1.0) or math.isnan(rate):
            raise ValueError("rate must be in [0, 1]")
        self.rate = rate
        self._rng = rng
        self._lock = threading.Lock()

    def check_this_request(self) -> bool:
        raise NotImplementedError

    def pick_pass(self, max_new_tokens: int) -> int:
        """Which pass index (0 = prefill) of a request to check."""
        raise NotImplementedError
