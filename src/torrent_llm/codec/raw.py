"""Passthrough codec — full-precision activations, no compression.

This is the ``naive latent-passing`` condition of the headline benchmark
(issue #12) and the control against which every compressor is measured. It is
also what the transport layer is developed against, so networking work does not
block on the compressor existing.
"""

from __future__ import annotations

import torch

from torrent_llm.codec.base import Codec, register_codec
from torrent_llm.wire import ActivationHeader, ActivationMessage


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Raw little-endian buffer of ``tensor``.

    Goes through torch's own byte view rather than numpy, because numpy has no
    bfloat16 and bf16 is the dtype most current checkpoints ship in.
    """
    return tensor.detach().to("cpu").contiguous().flatten().view(torch.uint8).numpy().tobytes()


def bytes_to_tensor(payload: bytes, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
    """Inverse of :func:`tensor_to_bytes`.

    ``torch.frombuffer`` needs a writable buffer, and it aliases rather than
    copies, so the immutable ``bytes`` is copied into a ``bytearray`` first.
    It also rejects a zero-length buffer outright, so an empty activation --
    a legitimate zero-length sequence -- is built directly instead.
    """
    if not payload:
        return torch.empty(shape, dtype=dtype)
    flat = torch.frombuffer(bytearray(payload), dtype=dtype)
    return flat.reshape(shape)


@register_codec("raw")
class RawCodec(Codec):
    """Ships the activation verbatim.

    Args:
        wire_dtype: Optional dtype to cast to before transmission. Casting
            bf16 -> fp16 is *not* free of information loss, so the default is
            to leave the tensor alone and let compression be the codec's job.
    """

    def __init__(self, wire_dtype: str | None = None) -> None:
        self.wire_dtype = wire_dtype

    def encode(self, tensor: torch.Tensor, *, request_id: str, hop: int) -> ActivationMessage:
        if self.wire_dtype is not None:
            tensor = tensor.to(getattr(torch, self.wire_dtype))
        header = ActivationHeader(
            request_id=request_id,
            hop=hop,
            codec=self.name,
            dtype=str(tensor.dtype).removeprefix("torch."),
            shape=tuple(tensor.shape),
        )
        return ActivationMessage(header=header, payload=tensor_to_bytes(tensor))

    def decode(self, message: ActivationMessage) -> torch.Tensor:
        dtype = getattr(torch, message.header.dtype)
        return bytes_to_tensor(message.payload, dtype, message.header.shape)

    def describe(self) -> dict[str, object]:
        return {"codec": self.name, "wire_dtype": self.wire_dtype}
