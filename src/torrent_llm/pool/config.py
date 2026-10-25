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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

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

    def __post_init__(self) -> None:
        if self.stages < 1 or self.stages > self.num_layers:
            raise ValueError(f"stages must be in [1, {self.num_layers}], got {self.stages}")
        if self.replicas < 1:
            raise ValueError(f"replicas must be at least 1, got {self.replicas}")
        if self.shard_mb < 1:
            raise ValueError(f"shard_mb must be positive, got {self.shard_mb}")

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
        }
        return cls(**{k: v for k, v in raw.items() if k in known})


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
