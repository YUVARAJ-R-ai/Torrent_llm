"""Topology configuration.

A topology file is the static stand-in for the DHT layer registry (issue #15):
it says which address hosts which layer range. Keeping it declarative means the
same file drives the single-machine rig and the two-machine LAN rig — only the
addresses change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from torrent_llm.shard import ShardSpec, plan_even, plan_explicit, plan_weighted, validate_plan


@dataclass(frozen=True)
class NodeConfig:
    """One node in the chain."""

    address: str
    device: str = "cpu"

    @property
    def port(self) -> int:
        return int(self.address.rsplit(":", 1)[1])

    @property
    def host(self) -> str:
        return self.address.rsplit(":", 1)[0]


@dataclass(frozen=True)
class TopologyConfig:
    """A complete description of a run: model, split, codec, and who hosts what."""

    model_id: str
    num_layers: int
    nodes: list[NodeConfig]
    dtype: str = "float32"
    codec: str = "raw"
    codec_args: dict[str, Any] = field(default_factory=dict)
    boundaries: list[int] | None = None
    #: Relative per-node capability for a proportional split (issue #17), e.g.
    #: ``[1.0, 2.5]`` to give a second node roughly 2.5x the layers of the
    #: first. Mutually exclusive with ``boundaries`` -- both being ways to
    #: override the even split, picking between them silently would hide a
    #: config mistake rather than catch one.
    weights: list[float] | None = None
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if self.boundaries and self.weights:
            raise ValueError(
                "topology sets both 'boundaries' and 'weights'; these are two "
                "different ways to override the even split and only one may be "
                "given, or it is ambiguous which one actually decided the plan"
            )

    def shard_plan(self) -> list[ShardSpec]:
        """Layer ranges for this topology, validated for full coverage."""
        if self.boundaries:
            specs = plan_explicit(self.num_layers, self.boundaries)
        elif self.weights:
            specs = plan_weighted(self.num_layers, self.weights)
        else:
            specs = plan_even(self.num_layers, len(self.nodes))
        if len(specs) != len(self.nodes):
            raise ValueError(
                f"topology has {len(self.nodes)} nodes but the plan produced {len(specs)} shards"
            )
        validate_plan(specs)
        return specs

    @classmethod
    def from_file(cls, path: str | Path) -> TopologyConfig:
        raw = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TopologyConfig:
        missing = {"model_id", "num_layers", "nodes"} - raw.keys()
        if missing:
            raise ValueError(f"topology is missing required keys: {sorted(missing)}")

        codec = raw.get("codec", "raw")
        codec_args: dict[str, Any] = {}
        if isinstance(codec, dict):
            codec_args = {k: v for k, v in codec.items() if k != "name"}
            codec = codec["name"]

        return cls(
            model_id=raw["model_id"],
            num_layers=int(raw["num_layers"]),
            nodes=[NodeConfig(**n) for n in raw["nodes"]],
            dtype=raw.get("dtype", "float32"),
            codec=codec,
            codec_args=codec_args,
            boundaries=raw.get("boundaries"),
            weights=raw.get("weights"),
            trust_remote_code=bool(raw.get("trust_remote_code", False)),
        )
