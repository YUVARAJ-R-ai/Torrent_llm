"""Request and response bodies for the HTTP layer.

These are the contract the dashboard codes against, so they are deliberately
flat and explicit rather than clever: a field named ``sent_bytes`` holding an
integer number of bytes beats a nested "metrics" object that has to be
navigated, especially when the consumer is a chart that wants one array per
series.

Every duration is exposed in **milliseconds as a float**, not nanoseconds as an
int. The profiler records nanoseconds internally because that is what
``perf_counter_ns`` gives and rounding early loses information, but every
consumer of this API is either a human reading a number or a chart axis, and
both want milliseconds. The conversion happens here, once, rather than in each
caller.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from torrent_llm.profile import HopRecord, HopSummary


class ShardInfo(BaseModel):
    """One node's slice of the model, as that node itself reports it."""

    address: str
    shard: int
    layers: str = Field(description="Half-open layer range, e.g. '[0:14)'")
    hidden_size: int
    device: str
    dtype: str
    model_id: str


class TopologyResponse(BaseModel):
    """What the whole chain is hosting right now."""

    model_id: str
    num_layers: int
    num_shards: int
    codec: str
    dtype: str
    shards: list[ShardInfo]


class HopMetrics(BaseModel):
    """One activation crossing one hop, as the dashboard consumes it."""

    hop: int
    address: str
    codec: str
    phase: str = Field(description="'prefill' or 'decode' -- the two regimes are not comparable")

    batch: int
    seq_len: int
    hidden_size: int = Field(
        description="0 on hop 0, which carries token ids: the embedding has not happened yet"
    )
    dtype: str

    sent_bytes: int
    received_bytes: int
    uncompressed_bytes: int
    compression_ratio: float

    wall_ms: float
    compute_ms: float
    transport_ms: float
    transport_share: float = Field(
        description="Transport over wall time. Decides whether compressing this hop could help."
    )

    @classmethod
    def from_record(cls, record: HopRecord) -> HopMetrics:
        return cls(
            hop=record.hop,
            address=record.address,
            codec=record.codec,
            phase=record.phase,
            batch=record.batch,
            seq_len=record.seq_len,
            hidden_size=record.hidden_size,
            dtype=record.dtype,
            sent_bytes=record.sent_bytes,
            received_bytes=record.received_bytes,
            uncompressed_bytes=record.uncompressed_bytes,
            compression_ratio=record.compression_ratio,
            wall_ms=record.wall_ns / 1e6,
            compute_ms=record.compute_ns / 1e6,
            transport_ms=record.transport_ns / 1e6,
            transport_share=record.transport_share,
        )


class HopSummaryMetrics(BaseModel):
    """Aggregate over every record for one hop index."""

    hop: int
    codec: str
    samples: int
    median_sent_bytes: float
    median_wall_ms: float
    median_compute_ms: float
    median_transport_ms: float
    mean_compression_ratio: float
    mean_transport_share: float
    mean_effective_mbps: float

    @classmethod
    def from_summary(cls, summary: HopSummary) -> HopSummaryMetrics:
        return cls(
            hop=summary.hop,
            codec=summary.codec,
            samples=summary.samples,
            median_sent_bytes=summary.median_sent_bytes,
            median_wall_ms=summary.median_wall_ms,
            median_compute_ms=summary.median_compute_ms,
            median_transport_ms=summary.median_transport_ms,
            mean_compression_ratio=summary.mean_compression_ratio,
            mean_transport_share=summary.mean_transport_share,
            mean_effective_mbps=summary.mean_effective_mbps,
        )


class GenerateRequest(BaseModel):
    """Ask the chain to generate."""

    prompt: str
    max_new_tokens: int = Field(default=16, ge=1, le=512)
    use_cache: bool = Field(
        default=True,
        description=(
            "Server-side KV cache (issue #24). False re-sends the whole growing "
            "prefix every step -- far more bandwidth, but the control condition "
            "that makes the cache's payoff measurable rather than assumed."
        ),
    )


class GenerateResponse(BaseModel):
    """Generated text plus what it cost on the wire."""

    prompt: str
    completion: str = Field(description="Newly generated text only, without the prompt echoed back")
    full_text: str
    tokens_generated: int
    use_cache: bool

    total_sent_bytes: int
    total_wall_ms: float
    total_compute_ms: float
    total_transport_ms: float

    hops: list[HopMetrics]

    @property
    def bytes_per_token(self) -> float:
        return self.total_sent_bytes / self.tokens_generated if self.tokens_generated else 0.0


class ProfileRequest(BaseModel):
    """Run a prefill sweep and report per-hop cost at each context length."""

    seq_lens: list[int] = Field(default=[128, 512], min_length=1)
    repeats: int = Field(default=3, ge=1, le=20)


class SeqLenProfile(BaseModel):
    """The hop table for one context length, plus the regime verdict."""

    seq_len: int
    verdict: str = Field(
        description="Whether these hops were bandwidth-bound or compute-bound, in words"
    )
    hops: list[HopSummaryMetrics]


class ProfileResponse(BaseModel):
    """A whole prefill sweep."""

    model_id: str
    codec: str
    repeats: int
    profiles: list[SeqLenProfile]


class CacheComparisonResponse(BaseModel):
    """Cached vs uncached generation of the same prompt, side by side.

    Exists because the KV cache's payoff (issue #24) is a *ratio* between two
    runs, and computing that ratio in the dashboard would mean trusting the
    dashboard to pair up the right two runs. Doing both runs here, in one
    request, makes the comparison impossible to assemble incorrectly.
    """

    prompt: str
    tokens_generated: int
    cached: GenerateResponse
    uncached: GenerateResponse
    bandwidth_reduction: float = Field(
        description="uncached bytes / cached bytes. The headline number for issue #24."
    )
    same_output: bool = Field(
        description=(
            "Whether both paths produced identical text. False means a real bug: "
            "the two differ only in how much they recompute, never in what they compute."
        )
    )
