"""Codec profiles: the named codec settings a pool agrees on.

A pool used to have exactly one codec, fixed in its config file. Choosing the
codec per hop (fast link: full precision; slow link: compress) needs every device
to be able to decode any of a known set, and each message to say which one it
used. A *profile* is one named setting (``"bf16"`` = the raw codec with bfloat16
on the wire; ``"pca256"`` = low rank 256 with these weights). The message header's
``codec`` field carries the profile id instead of the codec's registry name, and a
receiver decodes with its own copy of that profile. An id it does not know is
refused (``UnknownProfile``), never guessed.

Every device in a pool must hold the same profiles, which :meth:`CodecSet.signature`
makes cheap to check.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from torrent_llm.codec.base import Codec, get_codec
from torrent_llm.wire import ActivationMessage

_PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")

#: The profiles every pool has unless its config says otherwise.
DEFAULT_PROFILES: dict[str, dict[str, Any]] = {
    "raw32": {"name": "raw"},
    "bf16": {"name": "raw", "wire_dtype": "bfloat16"},
    "int8": {"name": "int8", "outliers": 8},
}


class UnknownProfile(KeyError):
    """A message named a codec profile this device does not have."""


@dataclass(frozen=True)
class CodecProfile:
    id: str
    name: str
    args: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _PROFILE_ID.match(self.id):
            raise ValueError(f"codec profile id {self.id!r} must match {_PROFILE_ID.pattern}")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"codec profile {self.id!r} needs a codec name")

    def canonical(self) -> dict[str, Any]:
        """What identifies the profile across devices. A weights file counts by its
        file name, not its path, since every device keeps the repository elsewhere."""
        args = dict(self.args)
        if "weights_path" in args and args["weights_path"]:
            args["weights_path"] = str(args["weights_path"]).replace("\\", "/").rsplit("/", 1)[-1]
        return {"id": self.id, "name": self.name, "args": args}


class ProfiledCodec(Codec):
    """A codec instance labelled with its profile id on the wire."""

    def __init__(self, profile: CodecProfile, inner: Codec) -> None:
        self.profile = profile
        self.inner = inner
        self.name = profile.id

    def encode(self, tensor: torch.Tensor, *, request_id: str, hop: int) -> ActivationMessage:
        message = self.inner.encode(tensor, request_id=request_id, hop=hop)
        header = dataclasses.replace(message.header, codec=self.profile.id)
        return ActivationMessage(header=header, payload=message.payload)

    def decode(self, message: ActivationMessage) -> torch.Tensor:
        header = dataclasses.replace(message.header, codec=self.inner.name)
        return self.inner.decode(ActivationMessage(header=header, payload=message.payload))

    def describe(self) -> dict[str, object]:
        return {"profile": self.profile.id, **self.inner.describe()}


class CodecSet:
    """The profiles one pool uses, built lazily and shared between threads."""

    def __init__(self, profiles: Iterable[CodecProfile]) -> None:
        self._profiles: dict[str, CodecProfile] = {}
        for profile in profiles:
            if profile.id in self._profiles:
                raise ValueError(f"codec profile {profile.id!r} is defined twice")
            self._profiles[profile.id] = profile
        if not self._profiles:
            raise ValueError("a pool needs at least one codec profile")
        self._built: dict[str, ProfiledCodec] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, raw: Mapping[str, Mapping[str, Any]] | None) -> CodecSet:
        """From a pool file's ``codecs:`` block (``{id: {name: ..., **args}}``);
        ``None`` gives :data:`DEFAULT_PROFILES`."""
        raw = DEFAULT_PROFILES if raw is None else raw
        profiles = []
        for profile_id, spec in raw.items():
            if not isinstance(spec, Mapping) or "name" not in spec:
                raise ValueError(f"codec profile {profile_id!r} needs a 'name'")
            args = {k: v for k, v in spec.items() if k != "name"}
            profiles.append(CodecProfile(str(profile_id), str(spec["name"]), args))
        return cls(profiles)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._profiles)

    def __contains__(self, profile_id: object) -> bool:
        return profile_id in self._profiles

    def profile(self, profile_id: str) -> CodecProfile:
        try:
            return self._profiles[profile_id]
        except KeyError:
            raise UnknownProfile(f"unknown codec profile {profile_id!r}") from None

    def codec(self, profile_id: str) -> ProfiledCodec:
        """The codec for ``profile_id``; :class:`UnknownProfile` if there is none."""
        built = self._built.get(profile_id)
        if built is not None:
            return built
        profile = self.profile(profile_id)
        with self._lock:
            built = self._built.get(profile_id)
            if built is None:
                built = ProfiledCodec(profile, get_codec(profile.name, **dict(profile.args)))
                self._built[profile_id] = built
        return built

    def decode(self, message: ActivationMessage) -> torch.Tensor:
        """Decode with the profile the message names."""
        return self.codec(message.header.codec).decode(message)

    def signature(self) -> str:
        """Short hash of every profile, so devices with different sets never mix."""
        canonical = [self._profiles[k].canonical() for k in sorted(self._profiles)]
        blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]
