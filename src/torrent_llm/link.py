"""Simulated network link between the runner and each shard.

On one machine every hop runs over loopback, where moving even a full prefill
activation takes about a millisecond. That is the right number for measuring
compute, but it hides the thing this project is about: what the network costs
once the nodes are on different machines. Shaping each hop to a chosen latency
and bandwidth lets a single box stand in for a real link, so the byte counts the
codec saves turn into time a viewer can actually see.

The delay is applied on the client, inside the window ``wall_ns`` measures, so it
lands in ``transport_ns`` exactly where a real link's cost would. Compute stays
untouched because it is reported by the shard itself.

Nothing here touches the bytes. A shaped run sends exactly the same payloads as
an unshaped one; only the clock changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LinkProfile:
    """One simulated link, applied to every hop in the chain.

    Attributes:
        latency_ms: Round-trip time added to every hop, before any payload is
            counted. Half is charged on the way out and half on the way back.
        bandwidth_mbps: Link capacity in megabits per second, the unit ISPs and
            ``iperf`` quote. Both the request and the reply are charged against
            it, since both cross the link.
    """

    latency_ms: float = 0.0
    bandwidth_mbps: float = 0.0

    def __post_init__(self) -> None:
        if self.latency_ms < 0:
            raise ValueError(f"latency_ms must not be negative, got {self.latency_ms}")
        if self.bandwidth_mbps <= 0:
            raise ValueError(f"bandwidth_mbps must be positive, got {self.bandwidth_mbps}")

    def one_way_s(self, payload_bytes: int) -> float:
        """Seconds for one direction: half the round trip plus serialisation."""
        bits = payload_bytes * 8
        return self.latency_ms / 2_000 + bits / (self.bandwidth_mbps * 1e6)

    def describe(self) -> dict[str, float]:
        return {"latency_ms": self.latency_ms, "bandwidth_mbps": self.bandwidth_mbps}

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> LinkProfile | None:
        """Build a profile from a topology file's ``link`` block, or ``None``."""
        if not raw:
            return None
        unknown = raw.keys() - {"latency_ms", "bandwidth_mbps"}
        if unknown:
            raise ValueError(f"link block has unknown keys: {sorted(unknown)}")
        if "bandwidth_mbps" not in raw:
            raise ValueError("link block needs 'bandwidth_mbps'; a link needs a capacity")
        return cls(
            latency_ms=float(raw.get("latency_ms", 0.0)),
            bandwidth_mbps=float(raw["bandwidth_mbps"]),
        )
