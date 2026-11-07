"""8-bit hidden states with the outlier channels kept exact-ish.

Hidden states of current LLMs are not low-dimensional (``docs/compression.md``:
128 directions explain only 72% of the variance at layer 14), which is why
dropping dimensions costs so much quality. Quantising every dimension to 8 bits
keeps all of them. The catch is a handful of "massive activation" channels whose
values are hundreds of times larger than the rest: one scale per token would then
waste the int8 range on them. So the ``outliers`` largest channels of each payload
travel exactly (float32), and every token's remaining channels share one absmax scale.

Scales and outliers are float32, not float16: a float16 scale overflows for large
rows and loses precision (subnormals) for small ones, and the few extra bytes per
token do not matter next to the ``H`` quantised values.

Bytes for an ``(N, H)`` payload with ``k`` outliers: ``N*H + 4N + 4Nk + 4k``,
about 3.9x smaller than float32 and 1.9x smaller than bfloat16 at ``H=1024, k=8``.

Payload layout (little-endian), described by ``codec_meta = {"v": 1, "rows": N,
"cols": H, "k": k}``::

    int32[k]       outlier channel indices, strictly increasing, each in [0, H)
    float32[N, k]  outlier values
    float32[N]     per-row scale (absmax of the non-outlier channels / 127)
    int8[N, H]     quantised values; the outlier columns are 0

A tensor with NaN or infinite values is refused (``ValueError``): a corrupted hidden
state must fail loudly, not travel on as plausible numbers.

Integer tensors (token ids on the first hop) are sent exactly, as the raw codec
would, with ``codec_meta = {"v": 1, "passthrough": true}``.
"""

from __future__ import annotations

import torch

from torrent_llm.codec.base import Codec, register_codec
from torrent_llm.codec.raw import bytes_to_tensor, tensor_to_bytes
from torrent_llm.wire import ActivationHeader, ActivationMessage


@register_codec("int8")
class Int8Codec(Codec):
    """Per-row absmax int8 with ``outliers`` channels kept exactly.

    Contract (tests pin these down):

    * decode(encode(x)) has exactly ``x.shape`` and ``x.dtype`` for any float dtype
      (float32, float16, bfloat16) and any rank >= 1 (the last dim is the channel dim);
    * for ``x ~ N(0, 1)`` of shape (4, 37, 1024), relative error ||x-y|| / ||x|| < 0.01;
    * a channel 1000x larger than the rest survives with relative error < 1e-3
      (it is an outlier) and does not ruin the other channels (overall rel. error < 0.01);
    * rows of any magnitude (1e-6 to 1e8) keep relative error < 0.01;
    * all-zero rows decode to zeros; ``outliers=0`` works; ``outliers >= H`` keeps every
      channel exactly; NaN or infinite inputs raise ``ValueError``;
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
        """Quantise a floating-point tensor to int8 with its top outlier channels exact."""
        if not tensor.is_floating_point():
            header = ActivationHeader(
                request_id=request_id,
                hop=hop,
                codec=self.name,
                dtype=str(tensor.dtype).removeprefix("torch."),
                shape=tuple(tensor.shape),
                codec_meta={"v": 1, "passthrough": True},
            )
            return ActivationMessage(header=header, payload=tensor_to_bytes(tensor))

        if tensor.ndim < 1:
            raise ValueError("tensor must have rank >= 1")

        cols = tensor.shape[-1]
        rows = tensor.numel() // cols if cols > 0 else 0
        k = min(self.outliers, cols)

        if rows == 0:
            indices = torch.arange(k, dtype=torch.int32)
            outliers_vals = torch.empty((0, k), dtype=torch.float32)
            scales = torch.empty((0,), dtype=torch.float32)
            q = torch.empty((0, cols), dtype=torch.int8)
        else:
            x = tensor.detach().to("cpu", torch.float32).reshape(rows, cols)
            if not torch.isfinite(x).all():
                raise ValueError("refusing to encode a hidden state with NaN or infinite values")
            if k > 0:
                col_max = x.abs().amax(dim=0)
                indices = torch.topk(col_max, k).indices.sort().values.to(torch.int32)
            else:
                indices = torch.empty(0, dtype=torch.int32)

            outliers_vals = x[:, indices.long()].contiguous()

            x_non_outlier = x.clone()
            x_non_outlier[:, indices.long()] = 0.0
            absmax = x_non_outlier.abs().amax(dim=-1)
            scales = torch.where(absmax == 0, torch.ones_like(absmax), absmax / 127.0)

            q = torch.round(x_non_outlier / scales[:, None]).clamp(-127, 127).to(torch.int8)

        payload = (
            tensor_to_bytes(indices)
            + tensor_to_bytes(outliers_vals)
            + tensor_to_bytes(scales)
            + tensor_to_bytes(q)
        )

        header = ActivationHeader(
            request_id=request_id,
            hop=hop,
            codec=self.name,
            dtype=str(tensor.dtype).removeprefix("torch."),
            shape=tuple(tensor.shape),
            codec_meta={"v": 1, "rows": rows, "cols": cols, "k": k},
        )
        return ActivationMessage(header=header, payload=payload)

    def decode(self, message: ActivationMessage) -> torch.Tensor:
        """Decode an activation message produced by :meth:`encode`."""
        meta = message.header.codec_meta
        if not isinstance(meta, dict):
            raise ValueError("codec_meta must be a dict")

        v = meta.get("v")
        if type(v) is not int or v != 1:
            raise ValueError(f"unsupported or invalid codec version: {v!r}")

        logical_dtype = getattr(torch, message.header.dtype)

        if meta.get("passthrough") is True:
            return bytes_to_tensor(message.payload, logical_dtype, message.header.shape)

        shape = message.header.shape
        if len(shape) < 1:
            raise ValueError("shape must have rank >= 1")

        rows = meta.get("rows")
        cols = meta.get("cols")
        k = meta.get("k")

        for name, val in [("rows", rows), ("cols", cols), ("k", k)]:
            if type(val) is not int:
                raise ValueError(f"codec_meta field {name!r} must be an int, got {val!r}")

        if cols != shape[-1]:
            raise ValueError(f"codec_meta cols={cols} does not match shape[-1]={shape[-1]}")

        if rows * cols != message.header.logical_numel:
            raise ValueError(
                f"rows * cols ({rows} * {cols} = {rows * cols}) does not match "
                f"logical_numel {message.header.logical_numel}"
            )

        if not (0 <= k <= cols):
            raise ValueError(f"k must be in range [0, cols], got k={k}, cols={cols}")

        if rows < 0 or cols < 0:
            raise ValueError(f"rows and cols must be non-negative, got {rows}, {cols}")
        expected_len = 4 * k + 4 * rows * k + 4 * rows + rows * cols
        if len(message.payload) != expected_len:
            raise ValueError(
                f"payload length {len(message.payload)} does not match expected {expected_len}"
            )

        p = message.payload
        off0 = 0
        off1 = off0 + 4 * k
        off2 = off1 + 4 * rows * k
        off3 = off2 + 4 * rows
        off4 = off3 + rows * cols

        indices = bytes_to_tensor(p[off0:off1], torch.int32, (k,))
        outliers_vals = bytes_to_tensor(p[off1:off2], torch.float32, (rows, k))
        scales = bytes_to_tensor(p[off2:off3], torch.float32, (rows,))
        q = bytes_to_tensor(p[off3:off4], torch.int8, (rows, cols))

        if k > 0:
            if (indices < 0).any() or (indices >= cols).any():
                raise ValueError(f"indices out of range [0, {cols}): {indices.tolist()}")
            if k > 1 and not (indices[1:] > indices[:-1]).all():
                raise ValueError(f"indices must be strictly increasing: {indices.tolist()}")

        if not torch.isfinite(scales).all() or not torch.isfinite(outliers_vals).all():
            raise ValueError("payload carries NaN or infinite scales or outliers")
        recon = q.to(torch.float32) * scales[:, None]
        if k > 0:
            recon[:, indices.long()] = outliers_vals

        return recon.reshape(shape).to(logical_dtype)

    def describe(self) -> dict[str, object]:
        return {"codec": self.name, "outliers": self.outliers}
