"""8-bit hidden states with the outlier channels kept exact-ish.

Hidden states of current LLMs are not low-dimensional (``docs/compression.md``:
128 directions explain only 72% of the variance at layer 14), which is why
dropping dimensions costs so much quality. Quantising every dimension to 8 bits
keeps all of them. The catch is a handful of "massive activation" channels whose
values are hundreds of times larger than the rest: one scale per token would then
waste the int8 range on them. So the ``outliers`` largest channels of each payload
travel as float16, and every token's remaining channels share one absmax scale.

Bytes for an ``(N, H)`` payload with ``k`` outliers: ``N*H + 2N + 2Nk + 4k``,
about 4x smaller than float32 and 2x smaller than bfloat16.

Payload layout (little-endian), described by ``codec_meta = {"v": 1, "rows": N,
"cols": H, "k": k}``::

    int32[k]       outlier channel indices, strictly increasing, each in [0, H)
    float16[N, k]  outlier values
    float16[N]     per-row scale (absmax of the non-outlier channels / 127)
    int8[N, H]     quantised values; the outlier columns are 0

Integer tensors (token ids on the first hop) are sent exactly, as the raw codec
would, with ``codec_meta = {"v": 1, "passthrough": true}``.
"""

from __future__ import annotations

import torch

from torrent_llm.codec.base import Codec, register_codec
from torrent_llm.wire import ActivationMessage


@register_codec("int8")
class Int8Codec(Codec):
    """Per-row absmax int8 with ``outliers`` channels kept in float16.

    Contract (tests pin these down):

    * decode(encode(x)) has exactly ``x.shape`` and ``x.dtype`` for any float dtype
      (float32, float16, bfloat16) and any rank >= 1 (the last dim is the channel dim);
    * for ``x ~ N(0, 1)`` of shape (4, 37, 1024), relative error ||x-y|| / ||x|| < 0.01;
    * a channel 1000x larger than the rest survives with relative error < 1e-3
      (it is an outlier) and does not ruin the other channels (overall rel. error < 0.01);
    * all-zero rows decode to zeros; ``outliers=0`` works; ``outliers >= H`` keeps every
      channel in float16;
    * integer tensors round-trip bit-exactly;
    * ``payload_bytes`` equals the formula above;
    * a payload whose length, ``k``, indices (unsorted, duplicated, out of range),
      ``rows``/``cols`` (not matching ``shape``) or ``v`` are wrong raises ``ValueError``
      (bytes from the network are untrusted);
    * the decoder copies the payload into a ``bytearray`` before ``torch.frombuffer``.
    """

    def __init__(self, outliers: int = 8) -> None:
        if outliers < 0:
            raise ValueError("outliers must be >= 0")
        self.outliers = int(outliers)

    def encode(self, tensor: torch.Tensor, *, request_id: str, hop: int) -> ActivationMessage:
        raise NotImplementedError

    def decode(self, message: ActivationMessage) -> torch.Tensor:
        raise NotImplementedError

    def describe(self) -> dict[str, object]:
        return {"codec": self.name, "outliers": self.outliers}
