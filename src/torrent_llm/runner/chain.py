"""Driving a request through the whole shard chain.

The runner is the only component that knows the chain exists as a chain. Shards
know their own layers; clients know one hop. Sequencing lives here, which is
where rerouting on node failure (issue #16) will eventually go.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

import torch

from torrent_llm.codec import get_codec
from torrent_llm.runner.config import TopologyConfig
from torrent_llm.transport import HopResult, ShardClient

logger = logging.getLogger(__name__)


@dataclass
class ChainResult:
    """Output of one pass through the chain, plus per-hop measurements."""

    logits: torch.Tensor
    hops: list[HopResult]
    request_id: str

    @property
    def total_wall_ns(self) -> int:
        return sum(h.wall_ns for h in self.hops)

    @property
    def total_compute_ns(self) -> int:
        return sum(h.compute_ns for h in self.hops)

    @property
    def total_transport_ns(self) -> int:
        """End-to-end latency not explained by shard compute."""
        return sum(h.transport_ns for h in self.hops)

    @property
    def total_sent_bytes(self) -> int:
        """Every byte this request put on the wire, across all hops."""
        return sum(h.sent_bytes for h in self.hops)


class ChainRunner:
    """Walks token ids through every shard and returns logits."""

    def __init__(self, config: TopologyConfig) -> None:
        self.config = config
        self.clients = [
            ShardClient(node.address, get_codec(config.codec, **config.codec_args))
            for node in config.nodes
        ]

    def forward(
        self, input_ids: torch.Tensor, *, position_ids: torch.Tensor | None = None
    ) -> ChainResult:
        """One full forward pass over the chain."""
        request_id = uuid.uuid4().hex
        hops: list[HopResult] = []

        result = self.clients[0].forward(
            input_ids,
            request_id=request_id,
            hop=0,
            is_token_ids=True,
            position_ids=position_ids,
        )
        hops.append(result)

        for hop, client in enumerate(self.clients[1:], start=1):
            result = client.forward(
                result.tensor, request_id=request_id, hop=hop, position_ids=position_ids
            )
            hops.append(result)

        if not result.is_final:
            raise RuntimeError(
                "chain ended without reaching the shard that owns the LM head; "
                "the topology is probably missing its tail shard"
            )
        return ChainResult(logits=result.tensor, hops=hops, request_id=request_id)

    def generate(
        self, input_ids: torch.Tensor, *, max_new_tokens: int = 16
    ) -> tuple[torch.Tensor, list[ChainResult]]:
        """Greedy decode.

        There is no KV cache yet, so every step re-runs the entire prefix through
        every shard. That is correct but quadratic, and it means per-step wire
        payloads look like prefill payloads rather than the ~8 KB a cached decode
        step would send. **Decode-time bandwidth numbers from this method are not
        meaningful** — they measure repeated prefill. Prefill measurements are
        unaffected. Adding the cache is tracked as its own follow-up.
        """
        ids = input_ids
        passes: list[ChainResult] = []
        for _ in range(max_new_tokens):
            result = self.forward(ids)
            passes.append(result)
            next_token = result.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_token.to(ids.device)], dim=1)
        return ids, passes

    def describe(self) -> list[dict[str, object]]:
        """What every node reports hosting. Verifies the chain before a real run."""
        out = []
        for client in self.clients:
            info = client.info()
            out.append(
                {
                    "address": client.address,
                    "shard": info.shard_index,
                    "layers": f"[{info.layer_start}:{info.layer_end})",
                    "hidden_size": info.hidden_size,
                    "device": info.device,
                    "dtype": info.dtype,
                    "model_id": info.model_id,
                }
            )
        return out

    def close(self) -> None:
        for client in self.clients:
            client.close()

    def __enter__(self) -> ChainRunner:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
