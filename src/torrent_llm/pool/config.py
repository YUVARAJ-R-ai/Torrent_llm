"""What every device in a pool agrees on, and what each device says about itself.

A pool is a set of devices that together serve one model. Two kinds of fact
describe it, and they are kept apart on purpose:

* :class:`PoolConfig` is shared by everyone: which model, how many stages it is
  cut into, how many copies of each stage should exist. Every device must see
  the same values, or two of them would cut the model differently.
* :class:`DeviceProfile` is one device's own claim about itself: what it is
  called, what hardware it has, how much of it the pool may use, and how well it
  is connected. Nobody else can change it.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from torrent_llm.codec.profiles import CodecProfile, CodecSet
from torrent_llm.link import LinkProfile
from torrent_llm.shard import ShardSpec, plan_even


@dataclass(frozen=True)
class PoolConfig:
    """The facts every device in the pool must agree on.

    Attributes:
        model_id: The model the pool serves.
        num_layers: Decoder depth of that model.
        stages: How many contiguous pieces the model is cut into. A request
            passes through every stage once per token.
        replicas: How many devices should hold each stage. Two means any single
            device can disappear without a stage going dark.
        shard_mb: Rough memory one stage costs, used to count how many stages a
            device has room for. An estimate: the point is that a device
            offering 3 GB can hold two stages and one offering 1.5 GB only one.
        dtype: Weight precision, shared so activations are compatible everywhere.
    """

    model_id: str
    num_layers: int
    stages: int = 2
    replicas: int = 2
    shard_mb: int = 1500
    dtype: str = "float32"
    trust_remote_code: bool = False
    #: Let a device with spare room take extra copies of a stage when it is faster
    #: than the slowest device already holding it, so idle capacity is used.
    use_spare_capacity: bool = True
    #: How hidden states are shrunk on the wire. Part of the shared settings because
    #: both ends of a hop must use the same one, and two devices with different
    #: codecs would hand each other bytes they cannot read.
    codec: str = "raw"
    codec_args: dict[str, Any] = field(default_factory=dict)
    #: Named codec settings every device holds (``{id: {name, **args}}``). ``None``
    #: means one profile made from ``codec``/``codec_args`` (older pool files).
    codecs: dict[str, dict[str, Any]] | None = None
    #: How a hop's profile is chosen: ``{"mode": "fixed", "fixed": id}`` or
    #: ``{"mode": "adaptive", "ladder": [ids, best quality first], "prefill_only":
    #: bool, "target_ms": float}``. ``None``: fixed on the first profile.
    codec_policy: dict[str, Any] | None = None
    #: ``requester``: the asking device sends every hop (and gets every reply);
    #: ``relay``: each stage forwards to the next (issue #37).
    routing: str = "requester"
    #: Share of passes re-checked on another copy to catch a lying device. ``None``
    #: picks the swarm's default (some in a public swarm, none in an invite swarm).
    verify_rate: float | None = None
    #: SHA-256 (hex) of the model's file manifest (``blobs.manifest_digest``). Set, it
    #: lets a device fetch the weights from other devices instead of the internet and
    #: check every file against what the whole swarm agreed on.
    model_manifest: str | None = None

    def __post_init__(self) -> None:
        if self.stages < 1 or self.stages > self.num_layers:
            raise ValueError(f"stages must be in [1, {self.num_layers}], got {self.stages}")
        if self.replicas < 1:
            raise ValueError(f"replicas must be at least 1, got {self.replicas}")
        if self.shard_mb < 1:
            raise ValueError(f"shard_mb must be positive, got {self.shard_mb}")
        if self.routing not in ("requester", "relay"):
            raise ValueError(f"routing must be 'requester' or 'relay', got {self.routing!r}")
        if self.model_manifest is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.model_manifest
        ):
            raise ValueError("model_manifest must be a sha256 hex digest")
        if self.verify_rate is not None and not 0.0 <= self.verify_rate <= 1.0:
            raise ValueError(f"verify_rate must be in [0, 1], got {self.verify_rate}")
        codecs = self.codec_set()  # validates the profiles
        policy = self.policy()
        for profile_id in [policy.get("fixed"), *policy.get("ladder", [])]:
            if profile_id is not None and profile_id not in codecs:
                raise ValueError(f"codec policy names unknown profile {profile_id!r}")
        if policy["mode"] not in ("fixed", "adaptive"):
            raise ValueError(f"codec policy mode must be fixed or adaptive, not {policy['mode']!r}")

    def codec_set(self) -> CodecSet:
        """The pool's codec profiles."""
        if self.codecs is not None:
            return CodecSet.from_config(self.codecs)
        legacy_id = re.sub(r"[^a-z0-9_-]", "-", self.codec_signature.lower())[:32] or "raw"
        return CodecSet([CodecProfile(legacy_id, self.codec, dict(self.codec_args))])

    def policy(self) -> dict[str, Any]:
        """The codec policy with defaults filled in."""
        first = self.codec_set().ids[0]
        raw = dict(self.codec_policy or {})
        raw.setdefault("mode", "fixed")
        raw.setdefault("fixed", first)
        raw.setdefault("ladder", [raw["fixed"]])
        raw.setdefault("prefill_only", False)
        raw.setdefault("target_ms", 50.0)
        return raw

    def specs(self) -> list[ShardSpec]:
        """The layer range of every stage, in pipeline order."""
        return plan_even(self.num_layers, self.stages)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PoolConfig:
        missing = {"model_id", "num_layers"} - raw.keys()
        if missing:
            raise ValueError(f"pool config is missing required keys: {sorted(missing)}")
        known = {
            "model_id",
            "num_layers",
            "stages",
            "replicas",
            "shard_mb",
            "dtype",
            "trust_remote_code",
            "use_spare_capacity",
            "codecs",
            "codec_policy",
            "routing",
            "verify_rate",
            "model_manifest",
        }
        fields = {k: v for k, v in raw.items() if k in known}
        codec = raw.get("codec", "raw")
        if isinstance(codec, dict):
            fields["codec_args"] = {k: v for k, v in codec.items() if k != "name"}
            codec = codec["name"]
        fields["codec"] = codec
        return cls(**fields)

    @property
    def signature(self) -> str:
        """Everything two devices must agree on to work together, in one string."""
        policy = json.dumps(self.policy(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(
            f"{self.codec_set().signature()}|{policy}|{self.routing}"
            f"{'|' + self.model_manifest if self.model_manifest else ''}".encode()
        ).hexdigest()[:12]
        return f"{self.model_id}/{self.stages}/{self.replicas}/{self.dtype}/{digest}"

    @property
    def codec_signature(self) -> str:
        """A short description of the codec, so devices with different ones never mix."""
        if self.codecs is not None:
            return f"profiles-{self.codec_set().signature()[:8]}"
        name = self.codec
        weights = self.codec_args.get("weights_path")
        rank = self.codec_args.get("rank")
        wire = self.codec_args.get("wire_dtype")
        parts = [name]
        if rank:
            parts.append(f"r{rank}")
        if weights:
            parts.append(Path(str(weights)).stem)
        if wire:
            parts.append(str(wire))
        return "-".join(parts)


@dataclass(frozen=True)
class DeviceProfile:
    """One device's description of itself.

    Attributes:
        name: Unique within the pool. Also the device's identity in gossip.
        label: What a person would call it, for the dashboard.
        device: Torch device the shards run on: ``cpu``, ``cuda``, ``mps`` or
            ``auto``.
        memory_mb: How much memory the owner is willing to lend the pool.
        link: This device's connection, as the requester sees it.
        speed: Relative speed, only used to rank devices against each other when
            several could take the same stage. 1.0 is a plain CPU.
    """

    name: str
    label: str = ""
    device: str = "auto"
    memory_mb: int = 1500
    link: LinkProfile | None = None
    speed: float = 1.0
    #: Back off automatically when other programs take this much GPU memory.
    watch_owner: bool = False
    owner_busy_mb: int = 1500
    #: Questions this device will start at once before refusing more. ``None`` picks
    #: a default for the hardware (more for a GPU than a CPU).
    max_requests: int | None = None

    def request_limit(self, resolved_device: str) -> int:
        """How many questions to start at once. A GPU copes with far more than a CPU."""
        if self.max_requests is not None:
            return self.max_requests
        return 8 if resolved_device.startswith(("cuda", "mps")) else 3

    def slots(self, shard_mb: int) -> int:
        """How many stages this device has memory for."""
        return max(0, self.memory_mb // shard_mb)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DeviceProfile:
        fields = dict(raw)
        fields["link"] = LinkProfile.from_dict(fields.get("link"))
        known = {
            "name",
            "label",
            "device",
            "memory_mb",
            "link",
            "speed",
            "watch_owner",
            "owner_busy_mb",
            "max_requests",
        }
        unknown = fields.keys() - known
        if unknown:
            raise ValueError(
                f"device {fields.get('name', '?')} has unknown keys: {sorted(unknown)}"
            )
        return cls(**fields)


@dataclass(frozen=True)
class LabConfig:
    """A pool plus the list of simulated devices that make it up.

    Only the lab (see ``torrent_llm.pool.lab``) reads the device list; a real
    device is started with its own flags and learns about the others by gossip.
    """

    pool: PoolConfig
    devices: list[DeviceProfile] = field(default_factory=list)
    #: Devices started automatically; the rest are available to add later.
    start: int | None = None
    #: Shared secret for the pool, or ``None`` for an open one.
    token: str | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> LabConfig:
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            pool=PoolConfig.from_dict(raw),
            devices=[DeviceProfile.from_dict(d) for d in raw.get("devices", [])],
            start=raw.get("start"),
            token=raw.get("token"),
        )
