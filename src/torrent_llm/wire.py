"""Wire types shared by the codec and transport layers.

Kept dependency-free on purpose: both ``torrent_llm.codec`` and
``torrent_llm.transport`` import from here, so this module must not import
either of them.

The split between :class:`ActivationHeader` and the payload bytes is what makes
the codec seam work. The header always describes the *logical* tensor — the
shape and dtype the receiving shard must end up with — while ``codec_meta``
carries whatever the specific codec needs to rebuild it. A low-rank codec can
therefore ship a payload of a completely different shape than ``shape`` without
the transport layer knowing or caring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Wire dtype names are torch dtype names without the "torch." prefix.
DTYPE_NAMES = ("float32", "float16", "bfloat16", "float64", "int64", "int32", "uint8")


@dataclass(frozen=True)
class ActivationHeader:
    """Everything needed to interpret an activation payload.

    Attributes:
        request_id: Correlates every hop of one inference request, for the profiler.
        hop: 0-based index of this hop in the shard chain.
        codec: Registry name of the codec that produced the payload.
        dtype: Logical dtype name (see :data:`DTYPE_NAMES`).
        shape: Logical tensor shape after decoding, e.g. ``(batch, seq, hidden)``.
        codec_meta: Codec-private fields (rank, scales, payload shape, ...).
    """

    request_id: str
    hop: int
    codec: str
    dtype: str
    shape: tuple[int, ...]
    codec_meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dtype not in DTYPE_NAMES:
            raise ValueError(
                f"unsupported wire dtype {self.dtype!r}; expected one of {DTYPE_NAMES}"
            )
        if self.hop < 0:
            raise ValueError(f"hop must be non-negative, got {self.hop}")

    @property
    def logical_numel(self) -> int:
        """Element count of the decoded tensor — the denominator for compression ratio."""
        n = 1
        for d in self.shape:
            n *= d
        return n


@dataclass(frozen=True)
class ActivationMessage:
    """One activation crossing one network hop."""

    header: ActivationHeader
    payload: bytes

    @property
    def payload_bytes(self) -> int:
        """Actual bytes on the wire — the numerator the profiler reports."""
        return len(self.payload)
