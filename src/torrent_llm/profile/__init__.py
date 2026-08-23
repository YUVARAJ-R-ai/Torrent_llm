"""Per-hop bandwidth and latency measurement.

The measurement the project's central claim rests on: how many bytes crossed
the wire, how long the hop took, and how much of that time was transport rather
than the peer's compute.
"""

from torrent_llm.profile.estimate import (
    DTYPE_BYTES,
    KNOWN_HIDDEN_SIZES,
    HopBudget,
    activation_bytes,
    decode_budget,
    prefill_budget,
    transfer_ms,
)
from torrent_llm.profile.records import HopRecord, RunMetadata
from torrent_llm.profile.summarize import (
    HopSummary,
    bandwidth_verdict,
    format_table,
    summarize_by_hop,
)
from torrent_llm.profile.writer import HopProfiler, load_records, new_run_id

__all__ = [
    "DTYPE_BYTES",
    "KNOWN_HIDDEN_SIZES",
    "HopBudget",
    "HopProfiler",
    "HopRecord",
    "HopSummary",
    "RunMetadata",
    "activation_bytes",
    "bandwidth_verdict",
    "decode_budget",
    "format_table",
    "load_records",
    "new_run_id",
    "prefill_budget",
    "summarize_by_hop",
    "transfer_ms",
]
