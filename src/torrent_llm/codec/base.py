"""The codec seam.

A codec is the only thing that decides how an activation tensor becomes bytes.
Transport moves opaque payloads; shards call ``encode``/``decode`` and never
look inside. That boundary is what lets the learned low-rank compressor
(issue #8) land without touching networking code, and what makes the headline
benchmark (issue #12) a config change rather than a rewrite.

Adding a codec::

    @register_codec("lowrank")
    class LowRankCodec(Codec):
        def encode(self, tensor, *, request_id, hop): ...
        def decode(self, message): ...

Then reference it by name in a topology config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

from torrent_llm.wire import ActivationMessage

_REGISTRY: dict[str, type[Codec]] = {}


class Codec(ABC):
    """Turns an activation tensor into wire bytes and back.

    Implementations must round-trip shape and dtype exactly: a decoded tensor
    has the same ``shape`` and ``dtype`` as the tensor that was encoded, even
    when the codec is lossy about the *values*. Lossy value reconstruction is
    expected and is precisely what the quality harness (issue #10) measures.
    """

    #: Registry name, set by :func:`register_codec`.
    name: str = "unset"

    @abstractmethod
    def encode(self, tensor: torch.Tensor, *, request_id: str, hop: int) -> ActivationMessage:
        """Serialise ``tensor`` for transmission across one hop."""

    @abstractmethod
    def decode(self, message: ActivationMessage) -> torch.Tensor:
        """Reconstruct the tensor a peer encoded."""

    def describe(self) -> dict[str, object]:
        """Codec configuration, recorded in profiler output so runs are reproducible."""
        return {"codec": self.name}


def register_codec(name: str) -> Callable[[type[Codec]], type[Codec]]:
    """Class decorator that makes a codec constructible by name."""

    def wrap(cls: type[Codec]) -> type[Codec]:
        if name in _REGISTRY:
            raise ValueError(f"codec {name!r} is already registered to {_REGISTRY[name].__name__}")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return wrap


def get_codec(name: str, **kwargs: object) -> Codec:
    """Construct a registered codec.

    Raises:
        KeyError: if ``name`` was never registered — usually a missing import
            of the module that defines it.
    """
    try:
        cls = _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"unknown codec {name!r}; registered codecs: {known}") from None
    return cls(**kwargs)  # type: ignore[arg-type]


def available_codecs() -> tuple[str, ...]:
    """Names of every registered codec."""
    return tuple(sorted(_REGISTRY))
