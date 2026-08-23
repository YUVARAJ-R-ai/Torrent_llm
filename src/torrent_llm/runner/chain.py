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
from torrent_llm.profile import HopProfiler
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

    def __init__(self, config: TopologyConfig, *, profiler: HopProfiler | None = None) -> None:
        self.config = config
        self.profiler = profiler
        self.clients = [
            ShardClient(node.address, get_codec(config.codec, **config.codec_args))
            for node in config.nodes
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        phase: str = "prefill",
        logits_keep_last: int = 0,
        request_id: str | None = None,
        use_cache: bool = False,
        end_of_request: bool = False,
    ) -> ChainResult:
        """One full forward pass over the chain.

        Every hop is handed to the profiler, if one is attached, with the shape
        of what was actually sent — token ids on hop 0, activations after that.

        Args:
            logits_keep_last: Trailing logit positions to return; ``0`` means
                all. Greedy decoding needs only ``1``, and asking for all of
                them puts a batch x seq x vocab tensor on the wire, which on a
                modern tokenizer is far larger than any activation in the chain.
            request_id: Correlates this call with earlier ones under the same
                KV cache session (issue #24). Only meaningful together with
                ``use_cache=True``; a fresh id is generated when omitted, which
                is correct for a one-off, uncached call.
            use_cache: Reuse (or start, on the first call for this
                ``request_id``) a server-side KV cache on every shard, so
                ``input_ids``/``hidden_states`` only need to carry the new
                positions rather than the whole sequence. See
                :meth:`generate` for the loop that actually exploits this —
                calling ``forward`` directly with caching on is mostly useful
                for building a custom decode loop.
            end_of_request: Tell every shard this is the last call for
                ``request_id``, so each frees its cached state immediately. See
                the ``end_of_request`` field in ``activation.proto`` for why
                this exists alongside a TTL-based backstop rather than instead
                of one.
        """
        request_id = request_id or uuid.uuid4().hex
        hops: list[HopResult] = []

        sent = input_ids
        result = self.clients[0].forward(
            input_ids,
            request_id=request_id,
            hop=0,
            is_token_ids=True,
            position_ids=position_ids,
            use_cache=use_cache,
            end_of_request=end_of_request,
        )
        hops.append(result)
        self._profile(result, request_id, 0, sent, phase)

        for hop, client in enumerate(self.clients[1:], start=1):
            sent = result.tensor
            result = client.forward(
                sent,
                request_id=request_id,
                hop=hop,
                position_ids=position_ids,
                logits_keep_last=logits_keep_last,
                use_cache=use_cache,
                end_of_request=end_of_request,
            )
            hops.append(result)
            self._profile(result, request_id, hop, sent, phase)

        if not result.is_final:
            raise RuntimeError(
                "chain ended without reaching the shard that owns the LM head; "
                "the topology is probably missing its tail shard"
            )
        return ChainResult(logits=result.tensor, hops=hops, request_id=request_id)

    def _profile(
        self,
        result: HopResult,
        request_id: str,
        hop: int,
        sent: torch.Tensor,
        phase: str,
    ) -> None:
        if self.profiler is None:
            return
        self.profiler.record(
            result,
            request_id=request_id,
            address=self.clients[hop].address,
            shape=tuple(sent.shape),
            dtype=str(sent.dtype).removeprefix("torch."),
            phase=phase,
        )

    def generate(
        self, input_ids: torch.Tensor, *, max_new_tokens: int = 16, use_cache: bool = True
    ) -> tuple[torch.Tensor, list[ChainResult]]:
        """Greedy decode.

        With ``use_cache=True`` (the default, issue #24), every shard keeps a
        KV cache for this generation's ``request_id``: the first step sends the
        whole prompt, and every step after that sends only the single newly
        generated token. Per-step wire payloads are therefore real cached-decode
        payloads — a few KiB, not a repeated prefill — which is what makes
        decode-regime bandwidth something this project can actually measure
        instead of only projecting analytically (see docs/bandwidth-regimes.md).

        ``use_cache=False`` keeps the original behaviour: every step re-runs the
        whole growing prefix through every shard with no server-side state at
        all. Slower and far more bandwidth, but it is still there as an explicit
        control — the cleanest way to *measure* the cache's payoff is to run
        both and diff the profiler output, not to trust that caching helped.

        Cleanup on the happy path is automatic: the last step is sent with
        ``end_of_request=True``, so every shard drops its cache for this request
        as soon as generation finishes. If this method raises partway through
        (a shard error, a dropped connection), that signal never goes out and
        the abandoned sessions are freed later by each shard's own TTL sweep
        rather than by anything this method does — see ``SessionStore`` for why
        a best-effort signal plus a backstop beats trying to guarantee cleanup
        from the calling side, which a network partition can defeat anyway.
        """
        request_id = uuid.uuid4().hex
        ids = input_ids
        # What gets sent *this* step: the whole prompt once, then one token at a
        # time once the cache is carrying the rest of the context.
        sent = input_ids
        passes: list[ChainResult] = []
        for step in range(max_new_tokens):
            is_last_step = step == max_new_tokens - 1
            result = self.forward(
                sent,
                phase="prefill" if step == 0 else "decode",
                logits_keep_last=1,
                request_id=request_id,
                use_cache=use_cache,
                end_of_request=use_cache and is_last_step,
            )
            passes.append(result)
            next_token = result.logits[:, -1, :].argmax(dim=-1, keepdim=True).to(ids.device)
            ids = torch.cat([ids, next_token], dim=1)
            sent = next_token if use_cache else ids
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
