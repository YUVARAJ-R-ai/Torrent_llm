"""Putting a codec in the middle of a model, to measure what it costs.

Serving a model across a network and running it in one process give the same
numbers at a shard boundary: the hidden states that leave layer ``k - 1`` are exactly
what layer ``k`` receives. So quality can be measured without servers by wrapping
that one hand-over: take the hidden states, push them through a codec's encode and
decode (which is what the wire does), and hand the result on.

That is what makes it cheap to sweep many compression settings on a real model: one
model in memory, no sockets, and the byte count comes from the codec's real payload.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

from torrent_llm.codec.base import Codec


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    """The model's stack of decoder layers."""
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise ValueError("could not find the model's decoder layers")
    return layers


def _hidden_of(output):
    """The hidden-state tensor in a layer's output, whichever shape it comes in."""
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output, hidden: torch.Tensor):
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


@contextmanager
def boundary_hook(
    model: nn.Module, boundary: int, transform: Callable[[torch.Tensor], torch.Tensor]
) -> Iterator[None]:
    """Apply ``transform`` to the hidden states leaving layer ``boundary - 1``.

    ``boundary`` is the first layer of the *next* shard, so ``boundary=14`` splits a
    28-layer model into layers 0-13 and 14-27, matching a two-stage pool.
    """
    layers = decoder_layers(model)
    if not 1 <= boundary < len(layers):
        raise ValueError(f"boundary must be in [1, {len(layers) - 1}], got {boundary}")

    def hook(_module, _inputs, output):
        return _replace_hidden(output, transform(_hidden_of(output)))

    handle = layers[boundary - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@dataclass
class WireTap:
    """A transform that sends hidden states through a codec and counts the bytes."""

    codec: Codec
    wire_bytes: int = 0
    raw_bytes: int = 0
    calls: int = 0
    squared_error: float = 0.0
    squared_norm: float = 0.0

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        message = self.codec.encode(hidden, request_id="eval", hop=1)
        restored = self.codec.decode(message).to(hidden.device, hidden.dtype)
        self.calls += 1
        self.wire_bytes += message.payload_bytes
        self.raw_bytes += hidden.numel() * hidden.element_size()
        diff = (restored.float() - hidden.float()).pow(2).sum().item()
        norm = hidden.float().pow(2).sum().item()
        self.squared_error += diff
        self.squared_norm += norm
        return restored

    @property
    def ratio(self) -> float:
        """Raw bytes over wire bytes: how many times smaller the hand-over became."""
        return self.raw_bytes / self.wire_bytes if self.wire_bytes else 1.0

    @property
    def relative_error(self) -> float:
        """How far, as a share of the signal, what arrived is from what was sent."""
        return (self.squared_error / self.squared_norm) ** 0.5 if self.squared_norm else 0.0


@torch.inference_mode()
def capture_boundary(model: nn.Module, boundary: int, input_ids: torch.Tensor) -> torch.Tensor:
    """The hidden states that cross ``boundary`` for ``input_ids``, shaped (batch, seq, hidden)."""
    captured: list[torch.Tensor] = []

    def record(hidden: torch.Tensor) -> torch.Tensor:
        captured.append(hidden.detach())
        return hidden

    with boundary_hook(model, boundary, record):
        model(input_ids=input_ids)
    return captured[0]
