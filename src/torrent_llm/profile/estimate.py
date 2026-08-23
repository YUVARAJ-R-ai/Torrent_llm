"""Projecting hop cost to models and links we cannot run.

The rig tops out well below 70B, so the headline bandwidth story has to be part
measured and part projected. Activation size is exactly linear in
``batch x seq_len x hidden_size x dtype_bytes``, so projection is arithmetic, not
extrapolation — the measurements validate the model at 0.6-8B and the arithmetic
carries it to 70B.

The distinction this module exists to make explicit:

* **prefill** ships the whole sequence, so payload grows with context and the
  hop is bandwidth-bound at realistic context lengths
* **cached decode** ships one position, a few KiB, so the hop is dominated by
  round-trip latency and compression cannot help

Both are real regimes. Claiming a compression win without saying which regime it
was measured in is the easiest way for this project to publish a wrong number.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Bytes per element for the dtypes activations travel in.
DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2, "float64": 8, "uint8": 1}

#: hidden_size by model. Qwen3 values read from the configs on the Hub;
#: Llama-3 values are from the published architecture, as those repos are gated
#: and could not be verified here.
KNOWN_HIDDEN_SIZES = {
    "Qwen/Qwen3-0.6B": 1024,
    "Qwen/Qwen3-1.7B": 2048,
    "Qwen/Qwen3-8B": 4096,
    "meta-llama/Meta-Llama-3-8B": 4096,
    "meta-llama/Meta-Llama-3-70B": 8192,
}


def activation_bytes(
    hidden_size: int, seq_len: int, *, batch: int = 1, dtype: str = "bfloat16"
) -> int:
    """Bytes one uncompressed activation tensor occupies on the wire."""
    try:
        width = DTYPE_BYTES[dtype]
    except KeyError:
        raise ValueError(f"unknown dtype {dtype!r}; known: {sorted(DTYPE_BYTES)}") from None
    return batch * seq_len * hidden_size * width


def transfer_ms(nbytes: int, link_mbps: float) -> float:
    """Serialisation time for ``nbytes`` on a link of ``link_mbps`` megabits/sec."""
    if link_mbps <= 0:
        raise ValueError(f"link_mbps must be positive, got {link_mbps}")
    return (nbytes * 8) / (link_mbps * 1e6) * 1e3


@dataclass(frozen=True)
class HopBudget:
    """Where one hop's latency goes, under a stated link and compute cost."""

    payload_bytes: int
    link_mbps: float
    rtt_ms: float
    compute_ms: float

    @property
    def wire_ms(self) -> float:
        """Time spent pushing bytes, excluding round-trip latency."""
        return transfer_ms(self.payload_bytes, self.link_mbps)

    @property
    def total_ms(self) -> float:
        return self.wire_ms + self.rtt_ms + self.compute_ms

    @property
    def wire_share(self) -> float:
        """Fraction of the hop that compression could in principle remove."""
        return self.wire_ms / self.total_ms if self.total_ms else 0.0

    @property
    def is_bandwidth_bound(self) -> bool:
        """True when pushing bytes dominates the hop.

        The precondition for compression helping at all. If this is False,
        a compressor can only make the hop slower — it adds two projections of
        compute to save time that was not being spent on the wire.
        """
        return self.wire_ms > (self.rtt_ms + self.compute_ms)

    def with_compression(self, ratio: float, *, codec_ms: float = 0.0) -> HopBudget:
        """This hop under a codec with the given compression ratio.

        Args:
            ratio: uncompressed / compressed bytes. 4.0 means a 4x reduction.
            codec_ms: encode + decode cost, charged to compute. A compressor is
                not free, and ignoring its cost is how a paper claims a win that
                does not exist end to end.
        """
        if ratio <= 0:
            raise ValueError(f"compression ratio must be positive, got {ratio}")
        return HopBudget(
            payload_bytes=int(self.payload_bytes / ratio),
            link_mbps=self.link_mbps,
            rtt_ms=self.rtt_ms,
            compute_ms=self.compute_ms + codec_ms,
        )

    def speedup_from(self, other: HopBudget) -> float:
        """How many times faster this hop is than ``other``. Below 1.0 is a loss."""
        return other.total_ms / self.total_ms if self.total_ms else float("inf")


def prefill_budget(
    *,
    hidden_size: int,
    seq_len: int,
    link_mbps: float,
    rtt_ms: float,
    compute_ms: float,
    batch: int = 1,
    dtype: str = "bfloat16",
) -> HopBudget:
    """Budget for a prefill hop, which ships the whole sequence."""
    return HopBudget(
        payload_bytes=activation_bytes(hidden_size, seq_len, batch=batch, dtype=dtype),
        link_mbps=link_mbps,
        rtt_ms=rtt_ms,
        compute_ms=compute_ms,
    )


def decode_budget(
    *,
    hidden_size: int,
    link_mbps: float,
    rtt_ms: float,
    compute_ms: float,
    batch: int = 1,
    dtype: str = "bfloat16",
) -> HopBudget:
    """Budget for a cached decode hop, which ships a single position.

    Assumes a KV cache is in place. Without one the chain re-sends the whole
    prefix every step and the prefill budget applies instead.
    """
    return HopBudget(
        payload_bytes=activation_bytes(hidden_size, 1, batch=batch, dtype=dtype),
        link_mbps=link_mbps,
        rtt_ms=rtt_ms,
        compute_ms=compute_ms,
    )
