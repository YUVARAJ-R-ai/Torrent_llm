"""What we record about a hop.

One flat record per hop, appended as JSONL. Flat and denormalised on purpose:
the ablation grid (issue #13) wants to load a run straight into a dataframe and
group by codec, rank, seq_len or hop without joining anything.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class HopRecord:
    """Measurements for one activation crossing one hop."""

    request_id: str
    hop: int
    codec: str
    address: str

    # Shape of the activation, so bytes can be normalised per token later.
    batch: int
    seq_len: int
    hidden_size: int
    dtype: str

    # The bandwidth numbers.
    sent_bytes: int
    received_bytes: int
    uncompressed_bytes: int

    # The latency split. wall = transport + remote compute, measured from the
    # sending side; compute is what the peer reported for its forward pass.
    wall_ns: int
    compute_ns: int

    phase: str = "prefill"  # prefill | decode
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def transport_ns(self) -> int:
        """Latency not explained by the peer's compute."""
        return max(0, self.wall_ns - self.compute_ns)

    @property
    def compression_ratio(self) -> float:
        """Uncompressed bytes / bytes on the wire. 1.0 means no compression."""
        return self.uncompressed_bytes / self.sent_bytes if self.sent_bytes else 1.0

    @property
    def bytes_per_token(self) -> float:
        """Wire cost per token of context — the figure that scales to other models."""
        tokens = self.batch * self.seq_len
        return self.sent_bytes / tokens if tokens else 0.0

    @property
    def effective_mbps(self) -> float:
        """Throughput implied by this hop's payload and transport time."""
        if self.transport_ns <= 0:
            return 0.0
        return (self.sent_bytes * 8) / (self.transport_ns / 1e9) / 1e6

    @property
    def transport_share(self) -> float:
        """Fraction of the hop spent on transport rather than compute.

        The single number that decides whether compressing this hop can help at
        all. Near 0 means the hop is compute-bound and no amount of compression
        will move end-to-end latency.
        """
        return self.transport_ns / self.wall_ns if self.wall_ns else 0.0

    def to_json(self) -> str:
        payload = asdict(self)
        payload.update(
            transport_ns=self.transport_ns,
            compression_ratio=self.compression_ratio,
            bytes_per_token=self.bytes_per_token,
            transport_share=self.transport_share,
        )
        return json.dumps(payload)


@dataclass(frozen=True)
class RunMetadata:
    """Context a record set is meaningless without.

    Written as the first line of every profile file. A bandwidth number with no
    record of the model, dtype and machine it came from cannot go in a paper.
    """

    run_id: str
    model_id: str
    dtype: str
    codec: str
    num_shards: int
    note: str = ""
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    host: str = field(default_factory=platform.node)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"record_type": "metadata", **asdict(self)})
