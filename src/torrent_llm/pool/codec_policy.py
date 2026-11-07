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

import threading
from collections.abc import Mapping
from typing import Any

from torrent_llm.codec.profiles import CodecSet

#: Payloads smaller than this say more about latency than bandwidth; not sampled.
MIN_SAMPLE_BYTES = 16 * 1024
SMOOTHING = 0.3
#: Expected size relative to float32 before any measurement, by codec name.
DEFAULT_RATIOS = {"raw": 1.0, "int8": 0.27, "lowrank": 0.125}


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
        raise NotImplementedError

    def observe(
        self, peer: str, sent_bytes: int, transport_ms: float, rtt_ms: float | None = None
    ) -> None:
        raise NotImplementedError

    def bandwidth(self, peer: str) -> float | None:
        raise NotImplementedError

    def rtt(self, peer: str) -> float | None:
        raise NotImplementedError

    def forget(self, peer: str) -> None:
        raise NotImplementedError

    def snapshot(self) -> dict[str, dict[str, float | None]]:
        """``{peer: {"bandwidth": ..., "rtt": ...}}`` for display."""
        raise NotImplementedError


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
        raise NotImplementedError

    def choose(self, peer: str, raw_bytes: int, seq_len: int) -> str:
        raise NotImplementedError

    def learn(self, profile_id: str, sent_bytes: int, uncompressed_bytes: int) -> None:
        raise NotImplementedError

    def ratio(self, profile_id: str) -> float:
        raise NotImplementedError


__all__ = ["CodecChooser", "LinkEstimator"]
