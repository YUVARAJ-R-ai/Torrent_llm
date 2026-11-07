"""Choosing how hard to compress each hop, from how fast that link has been (issue #18).

On the AWS run (docs/aws-multi-machine.md) compression saved nothing on a fast link
and a quarter of the time at 2 Mbit/s, while costing answer quality everywhere. So
the right codec depends on the link, and a pool with devices on fibre and devices on
a phone hotspot should not use one setting for all of them.

``LinkEstimator`` keeps a smoothed bandwidth and round-trip time per peer from what
each hop actually cost. ``CodecChooser`` walks the pool's ladder of profiles (best
quality first) and picks the first whose estimated transfer time fits the budget.
With ``prefill_only`` the decode passes (one token, a few KiB) always use the best
profile: they are bound by latency, which compression cannot help, so compressing
them would cost quality for nothing.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from torrent_llm.codec.profiles import CodecSet

#: Payloads smaller than this say more about latency than bandwidth; not sampled.
MIN_SAMPLE_BYTES = 16 * 1024
SMOOTHING = 0.3
#: Expected size relative to float32 before any measurement, by codec name.
DEFAULT_RATIOS = {"raw": 1.0, "int8": 0.27, "lowrank": 0.125}
_SIXTEEN_BIT_DTYPES = {"float16", "bfloat16", "half"}


def _is_finite_non_negative(val: Any) -> bool:
    return (
        not isinstance(val, bool)
        and isinstance(val, (int, float))
        and math.isfinite(val)
        and val >= 0
    )


def _is_finite_positive(val: Any) -> bool:
    return (
        not isinstance(val, bool)
        and isinstance(val, (int, float))
        and math.isfinite(val)
        and val > 0
    )


def _is_16bit(wire_dtype: Any) -> bool:
    if wire_dtype is None:
        return False
    val = str(wire_dtype).lower()
    if val.startswith("torch."):
        val = val[6:]
    return val in _SIXTEEN_BIT_DTYPES


@dataclass
class _PeerMetrics:
    bandwidth: float | None = None
    rtt: float | None = None


class LinkEstimator:
    """Smoothed bandwidth (bytes/s) and RTT (ms) per peer. Thread-safe.

    * ``observe(peer, sent_bytes, transport_ms, rtt_ms=None)``: when ``rtt_ms`` is given it
      updates the RTT average; when ``sent_bytes >= MIN_SAMPLE_BYTES`` it takes a bandwidth
      sample ``sent_bytes / max(transport_ms - rtt, 1) * 1000`` (rtt = the current average
      or 0) and folds it in with ``SMOOTHING`` (first sample taken as is). Negative, NaN
      or infinite inputs are ignored.
    * ``bandwidth(peer)`` / ``rtt(peer)``: the averages, or ``None`` before any sample.
    * ``forget(peer)``; at most ``max_peers`` peers (least recently updated dropped).
    """

    def __init__(self, *, max_peers: int = 4096) -> None:
        if max_peers < 1:
            raise ValueError(f"max_peers must be positive, got {max_peers}")
        self._max_peers = max_peers
        self._peers: OrderedDict[str, _PeerMetrics] = OrderedDict()
        self._lock = threading.Lock()

    def observe(
        self, peer: str, sent_bytes: int, transport_ms: float, rtt_ms: float | None = None
    ) -> None:
        valid_rtt = rtt_ms is not None and _is_finite_non_negative(rtt_ms)
        valid_bw = (
            _is_finite_non_negative(sent_bytes)
            and sent_bytes >= MIN_SAMPLE_BYTES
            and _is_finite_non_negative(transport_ms)
        )
        if not valid_rtt and not valid_bw:
            return

        with self._lock:
            entry = self._peers.get(peer)
            if entry is None:
                entry = _PeerMetrics()
                self._peers[peer] = entry

            if valid_rtt and rtt_ms is not None:
                rtt_val = float(rtt_ms)
                if entry.rtt is None:
                    entry.rtt = rtt_val
                else:
                    entry.rtt = (1.0 - SMOOTHING) * entry.rtt + SMOOTHING * rtt_val

            if valid_bw:
                current_rtt = entry.rtt if entry.rtt is not None else 0.0
                effective_ms = max(float(transport_ms) - current_rtt, 1.0)
                bw_sample = (float(sent_bytes) / effective_ms) * 1000.0
                if entry.bandwidth is None:
                    entry.bandwidth = bw_sample
                else:
                    entry.bandwidth = (1.0 - SMOOTHING) * entry.bandwidth + SMOOTHING * bw_sample

            self._peers.move_to_end(peer)
            while len(self._peers) > self._max_peers:
                self._peers.popitem(last=False)

    def bandwidth(self, peer: str) -> float | None:
        with self._lock:
            entry = self._peers.get(peer)
            return entry.bandwidth if entry is not None else None

    def rtt(self, peer: str) -> float | None:
        with self._lock:
            entry = self._peers.get(peer)
            return entry.rtt if entry is not None else None

    def forget(self, peer: str) -> None:
        with self._lock:
            self._peers.pop(peer, None)

    def snapshot(self) -> dict[str, dict[str, float | None]]:
        """``{peer: {"bandwidth": ..., "rtt": ...}}`` for display."""
        with self._lock:
            return {p: {"bandwidth": m.bandwidth, "rtt": m.rtt} for p, m in self._peers.items()}


def _lowrank_rank(profile_id: str, args: Mapping[str, Any]) -> int:
    """The rank of a lowrank profile; ``ValueError`` naming the profile if it is not a
    positive integer."""
    rank = args.get("rank", 128)
    if type(rank) is not int or rank <= 0:
        raise ValueError(f"codec profile {profile_id!r} has invalid rank: {rank!r}")
    return rank


class CodecChooser:
    """Picks a profile id per hop. ``policy`` is ``PoolConfig.policy()``.

    ``choose(peer, raw_bytes, seq_len)``:

    * mode ``fixed`` -> ``policy["fixed"]``;
    * mode ``adaptive``: if ``prefill_only`` and ``seq_len <= 1`` -> ``ladder[0]``; if the
      peer's bandwidth is unknown -> ``ladder[0]``; otherwise the first ``p`` in the ladder
      with ``raw_bytes * ratio(p) / bandwidth * 1000 <= target_ms``, else ``ladder[-1]``.

    ``ratio(p)`` is the measured average of ``sent / uncompressed`` reported through
    ``learn(p, sent_bytes, uncompressed_bytes)`` (smoothed like the estimator), or before
    any measurement ``DEFAULT_RATIOS[codec name]`` with ``0.5`` applied for a raw profile
    whose ``wire_dtype`` is 16-bit and, for lowrank, ``rank / 1024`` (halved for a 16-bit
    wire). Unknown codec names default to 1.0.
    """

    def __init__(
        self, policy: Mapping[str, Any], codecs: CodecSet, estimator: LinkEstimator
    ) -> None:
        mode = policy.get("mode", "fixed")
        if mode not in {"fixed", "adaptive"}:
            raise ValueError(f"codec policy mode must be fixed or adaptive, got {mode!r}")
        self._mode = str(mode)

        ladder = policy.get("ladder")
        if not ladder:
            raise ValueError("codec policy ladder must be non-empty")
        self._ladder = list(ladder)

        fixed = policy.get("fixed") or self._ladder[0]
        for pid in [*self._ladder, fixed]:
            if pid not in codecs:
                raise ValueError(f"codec policy names unknown profile {pid!r}")
            profile = codecs.profile(pid)
            if profile.name == "lowrank":
                _lowrank_rank(pid, profile.args)
        self._fixed: str = fixed

        self._prefill_only = bool(policy.get("prefill_only", False))
        target = policy.get("target_ms", 50.0)
        if type(target) not in (int, float) or not math.isfinite(target) or target <= 0:
            raise ValueError(f"codec policy target_ms must be a positive number, got {target!r}")
        self._target_ms = float(target)
        self._codecs = codecs
        self._estimator = estimator
        self._learned_ratios: dict[str, float] = {}
        self._lock = threading.Lock()

    def choose(self, peer: str, raw_bytes: int, seq_len: int) -> str:
        if self._mode == "fixed":
            return self._fixed

        if not _is_finite_non_negative(raw_bytes):
            return self._ladder[0]

        if self._prefill_only and seq_len <= 1:
            return self._ladder[0]

        bw = self._estimator.bandwidth(peer)
        if bw is None or bw <= 0:
            return self._ladder[0]

        for p in self._ladder:
            estimated_ms = (float(raw_bytes) * self.ratio(p) / bw) * 1000.0
            if estimated_ms <= self._target_ms:
                return p

        return self._ladder[-1]

    def learn(self, profile_id: str, sent_bytes: int, uncompressed_bytes: int) -> None:
        if not _is_finite_non_negative(sent_bytes) or not _is_finite_positive(uncompressed_bytes):
            return

        sample = float(sent_bytes) / float(uncompressed_bytes)
        with self._lock:
            prev = self._learned_ratios.get(profile_id)
            if prev is None:
                self._learned_ratios[profile_id] = sample
            else:
                self._learned_ratios[profile_id] = (1.0 - SMOOTHING) * prev + SMOOTHING * sample

    def ratio(self, profile_id: str) -> float:
        with self._lock:
            learned = self._learned_ratios.get(profile_id)
        if learned is not None:
            return learned

        profile = self._codecs.profile(profile_id)
        wire_dtype = profile.args.get("wire_dtype")
        is_16bit = _is_16bit(wire_dtype)

        if profile.name == "raw":
            base = DEFAULT_RATIOS.get("raw", 1.0)
            return base * 0.5 if is_16bit else base

        if profile.name == "lowrank":
            base = _lowrank_rank(profile_id, profile.args) / 1024.0
            return base * 0.5 if is_16bit else base

        if profile.name in DEFAULT_RATIOS:
            return DEFAULT_RATIOS[profile.name]

        return 1.0


__all__ = ["CodecChooser", "LinkEstimator"]
