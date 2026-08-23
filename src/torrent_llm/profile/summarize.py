"""Turning a pile of hop records into the table that goes in the paper."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, median

from torrent_llm.profile.records import HopRecord


@dataclass(frozen=True)
class HopSummary:
    """Aggregate over every record for one hop index."""

    hop: int
    samples: int
    codec: str
    median_sent_bytes: float
    median_wall_ms: float
    median_compute_ms: float
    median_transport_ms: float
    mean_compression_ratio: float
    mean_transport_share: float
    mean_effective_mbps: float

    def as_row(self) -> dict[str, object]:
        return {
            "hop": self.hop,
            "codec": self.codec,
            "n": self.samples,
            "sent_KiB": round(self.median_sent_bytes / 1024, 2),
            "wall_ms": round(self.median_wall_ms, 3),
            "compute_ms": round(self.median_compute_ms, 3),
            "transport_ms": round(self.median_transport_ms, 3),
            "ratio": round(self.mean_compression_ratio, 2),
            "transport_share": round(self.mean_transport_share, 3),
            "eff_Mbps": round(self.mean_effective_mbps, 1),
        }


def summarize_by_hop(records: list[HopRecord]) -> list[HopSummary]:
    """One row per hop.

    Medians for latency because a single scheduling hiccup or page fault skews a
    mean badly on a handful of samples; means for the ratios, which are stable.
    """
    grouped: dict[int, list[HopRecord]] = defaultdict(list)
    for record in records:
        grouped[record.hop].append(record)

    summaries = []
    for hop in sorted(grouped):
        rows = grouped[hop]
        summaries.append(
            HopSummary(
                hop=hop,
                samples=len(rows),
                codec=rows[0].codec,
                median_sent_bytes=median(r.sent_bytes for r in rows),
                median_wall_ms=median(r.wall_ns / 1e6 for r in rows),
                median_compute_ms=median(r.compute_ns / 1e6 for r in rows),
                median_transport_ms=median(r.transport_ns / 1e6 for r in rows),
                mean_compression_ratio=mean(r.compression_ratio for r in rows),
                mean_transport_share=mean(r.transport_share for r in rows),
                mean_effective_mbps=mean(r.effective_mbps for r in rows),
            )
        )
    return summaries


def format_table(summaries: list[HopSummary]) -> str:
    """Fixed-width table for the terminal and for pasting into notes."""
    if not summaries:
        return "(no hops recorded)"

    rows = [s.as_row() for s in summaries]
    columns = list(rows[0])
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in columns}

    header = "  ".join(c.rjust(widths[c]) for c in columns)
    rule = "  ".join("-" * widths[c] for c in columns)
    body = ["  ".join(str(row[c]).rjust(widths[c]) for c in columns) for row in rows]
    return "\n".join([header, rule, *body])


def bandwidth_verdict(summaries: list[HopSummary], *, threshold: float = 0.5) -> str:
    """State plainly whether these hops were bandwidth-bound.

    Exists because the answer determines whether compression can help at all,
    and it is the one conclusion easiest to assume rather than check.
    """
    if not summaries:
        return "no data"

    shares = [s.mean_transport_share for s in summaries]
    worst, best = min(shares), max(shares)
    if best < threshold:
        return (
            f"compute-bound: transport is {best:.0%} of hop latency at most. "
            "Compression cannot improve end-to-end latency in this regime, only "
            "bytes transferred."
        )
    if worst >= threshold:
        return (
            f"bandwidth-bound: transport is {worst:.0%}-{best:.0%} of hop latency. "
            "Compression has headroom to reduce end-to-end latency here."
        )
    return (
        f"mixed: transport ranges from {worst:.0%} to {best:.0%} of hop latency across "
        "hops. Report per-hop rather than as a single figure."
    )
