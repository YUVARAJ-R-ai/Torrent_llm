"""Deciding which layers live on which node.

A shard plan is a list of contiguous, gapless, non-overlapping ``[start, end)``
layer ranges covering the whole model. Keeping this as plain data — no torch, no
model — means the runner, the profiler and the eventual DHT registry (issue #15)
can all reason about topology without loading weights.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShardSpec:
    """One node's slice of the model.

    Attributes:
        index: Position in the chain; hop ``i`` delivers activations to shard ``i``.
        start: First decoder layer index, inclusive.
        end: Last decoder layer index, exclusive.
        num_layers: Total decoder layers in the model, so a shard can tell
            whether it owns the embedding or the LM head without extra context.
    """

    index: int
    start: int
    end: int
    num_layers: int

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.end <= self.num_layers:
            raise ValueError(
                f"shard {self.index} has invalid range [{self.start}, {self.end}) "
                f"for a {self.num_layers}-layer model"
            )

    @property
    def is_first(self) -> bool:
        """True if this shard owns the token embedding."""
        return self.start == 0

    @property
    def is_last(self) -> bool:
        """True if this shard owns the final norm and the LM head."""
        return self.end == self.num_layers

    @property
    def depth(self) -> int:
        """How many decoder layers this shard runs."""
        return self.end - self.start

    def __str__(self) -> str:
        return f"shard{self.index}[{self.start}:{self.end}]"


def plan_even(num_layers: int, num_shards: int) -> list[ShardSpec]:
    """Split ``num_layers`` as evenly as possible across ``num_shards``.

    Remainder layers go to the earliest shards, so with 33 layers over 2 shards
    you get ``[0:17)`` and ``[17:33)``. This is the homogeneous-hardware default;
    heterogeneous weighting is issue #17 and belongs in :func:`plan_explicit`.
    """
    if num_shards < 1:
        raise ValueError(f"need at least one shard, got {num_shards}")
    if num_shards > num_layers:
        raise ValueError(
            f"cannot split {num_layers} layers across {num_shards} shards; "
            "every shard must own at least one layer"
        )

    base, remainder = divmod(num_layers, num_shards)
    specs: list[ShardSpec] = []
    cursor = 0
    for i in range(num_shards):
        width = base + (1 if i < remainder else 0)
        specs.append(ShardSpec(index=i, start=cursor, end=cursor + width, num_layers=num_layers))
        cursor += width
    return specs


def plan_explicit(num_layers: int, boundaries: list[int]) -> list[ShardSpec]:
    """Build a plan from explicit cut points.

    ``boundaries`` are the interior cuts only: ``[8, 20]`` on a 32-layer model
    yields ``[0:8)``, ``[8:20)``, ``[20:32)``. This is how a config pins more
    layers onto the stronger machine.
    """
    if sorted(boundaries) != boundaries or len(set(boundaries)) != len(boundaries):
        raise ValueError(f"boundaries must be strictly increasing, got {boundaries}")
    if boundaries and (boundaries[0] <= 0 or boundaries[-1] >= num_layers):
        raise ValueError(f"boundaries must lie strictly inside (0, {num_layers}), got {boundaries}")

    cuts = [0, *boundaries, num_layers]
    return [
        ShardSpec(index=i, start=a, end=b, num_layers=num_layers)
        for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:], strict=True))
    ]


def validate_plan(specs: list[ShardSpec]) -> None:
    """Assert a plan covers every layer exactly once.

    Raises:
        ValueError: on a gap, an overlap, or a chain that does not start at
            layer 0 or end at the last layer. A silent gap would produce
            plausible-looking logits from a model that skipped layers, which is
            the worst possible failure mode for a benchmark.
    """
    if not specs:
        raise ValueError("shard plan is empty")

    num_layers = specs[0].num_layers
    if specs[0].start != 0:
        raise ValueError(f"plan starts at layer {specs[0].start}, expected 0")
    if specs[-1].end != num_layers:
        raise ValueError(f"plan ends at layer {specs[-1].end}, expected {num_layers}")

    for i, (left, right) in enumerate(zip(specs[:-1], specs[1:], strict=True)):
        if left.end != right.start:
            gap = "gap" if left.end < right.start else "overlap"
            raise ValueError(f"{gap} between {left} and {right}: layer {left.end} != {right.start}")
        if right.index != i + 1:
            raise ValueError(f"shard indices are not consecutive: {left} then {right}")
