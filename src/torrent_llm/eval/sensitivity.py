"""Per-layer-depth compression sensitivity analyzer (Issue #10).

Evaluates how compression at different layer depths/hops impacts hidden state
cosine similarity, relative L2 reconstruction error, and downstream loss.
Supports both end-to-end multi-hop comparison and isolated single-hop perturbation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from torrent_llm.codec import get_codec
from torrent_llm.runner import ChainRunner

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HopSensitivity:
    """Sensitivity and distortion metrics captured at a single hop."""

    hop: int
    layer_range: str | None
    cosine_similarity: float
    l2_relative_error: float
    mse: float
    immediate_cosine_similarity: float | None = None
    immediate_l2_relative_error: float | None = None
    isolated_loss: float | None = None
    isolated_loss_delta: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hop": self.hop,
            "layer_range": self.layer_range,
            "cosine_similarity": round(self.cosine_similarity, 6),
            "l2_relative_error": round(self.l2_relative_error, 6),
            "mse": round(self.mse, 6),
            "immediate_cosine_similarity": (
                round(self.immediate_cosine_similarity, 6)
                if self.immediate_cosine_similarity is not None
                else None
            ),
            "immediate_l2_relative_error": (
                round(self.immediate_l2_relative_error, 6)
                if self.immediate_l2_relative_error is not None
                else None
            ),
            "isolated_loss": (
                round(self.isolated_loss, 4) if self.isolated_loss is not None else None
            ),
            "isolated_loss_delta": (
                round(self.isolated_loss_delta, 4) if self.isolated_loss_delta is not None else None
            ),
        }


@dataclass(frozen=True)
class SensitivityReport:
    """Comprehensive sensitivity report across all chain hops and final output."""

    model_id: str
    rank: int | None
    baseline_loss: float
    compressed_loss: float
    loss_delta: float
    top1_agreement: float
    kl_divergence: float
    hops: list[HopSensitivity] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "rank": self.rank,
            "baseline_loss": round(self.baseline_loss, 4),
            "compressed_loss": round(self.compressed_loss, 4),
            "loss_delta": round(self.loss_delta, 4),
            "top1_agreement": round(self.top1_agreement, 4),
            "kl_divergence": round(self.kl_divergence, 6),
            "hops": [h.to_dict() for h in self.hops],
        }


class LayerSensitivityAnalyzer:
    """Analyzes compression degradation across network hops and layer depths."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def compare_chain_runs(
        self,
        raw_runner: ChainRunner,
        comp_runner: ChainRunner,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> SensitivityReport:
        """Compare uncompressed reference execution vs compressed execution end-to-end.

        Evaluates per-hop hidden state cosine similarity, MSE, relative L2 error,
        final prediction top-1 agreement, KL divergence, and cross-entropy loss delta.
        """
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        # Baseline uncompressed forward pass
        raw_res = raw_runner.forward(input_ids, logits_keep_last=0)
        # Compressed forward pass
        comp_res = comp_runner.forward(input_ids, logits_keep_last=0)

        # Setup targets for cross-entropy
        if targets is None:
            raw_eval_logits = raw_res.logits[:, :-1, :].contiguous()
            comp_eval_logits = comp_res.logits[:, :-1, :].contiguous()
            eval_targets = input_ids[:, 1:].contiguous()
        else:
            raw_eval_logits = raw_res.logits
            comp_eval_logits = comp_res.logits
            eval_targets = targets

        raw_loss = F.cross_entropy(
            raw_eval_logits.reshape(-1, raw_eval_logits.shape[-1]),
            eval_targets.reshape(-1),
        ).item()

        comp_loss = F.cross_entropy(
            comp_eval_logits.reshape(-1, comp_eval_logits.shape[-1]),
            eval_targets.reshape(-1),
        ).item()

        # Prediction top-1 agreement
        raw_top1 = raw_res.logits.argmax(dim=-1)
        comp_top1 = comp_res.logits.argmax(dim=-1)
        top1_agreement = (raw_top1 == comp_top1).float().mean().item()

        # Logits distribution KL divergence
        raw_probs = F.softmax(raw_res.logits.float(), dim=-1)
        comp_logprobs = F.log_softmax(comp_res.logits.float(), dim=-1)
        kl_div = F.kl_div(comp_logprobs, raw_probs, reduction="batchmean").item()

        # Per-hop layer descriptions if available
        config = comp_runner.config
        shard_plan = config.shard_plan() if hasattr(config, "shard_plan") else []

        hop_metrics: list[HopSensitivity] = []
        num_hops = min(len(raw_res.hops), len(comp_res.hops))
        for h in range(num_hops):
            raw_t = raw_res.hops[h].tensor.float()
            comp_t = comp_res.hops[h].tensor.float()

            cos_sim = F.cosine_similarity(raw_t, comp_t, dim=-1).mean().item()
            mse = F.mse_loss(raw_t, comp_t).item()
            norm_raw = torch.norm(raw_t)
            l2_err = (torch.norm(raw_t - comp_t) / (norm_raw + 1e-12)).item()

            layer_range = (
                f"[{shard_plan[h].start}:{shard_plan[h].end})" if h < len(shard_plan) else None
            )

            hop_metrics.append(
                HopSensitivity(
                    hop=h,
                    layer_range=layer_range,
                    cosine_similarity=cos_sim,
                    l2_relative_error=l2_err,
                    mse=mse,
                )
            )

        rank = (
            comp_runner.config.codec_args.get("rank")
            if comp_runner.config.codec == "lowrank"
            else None
        )

        return SensitivityReport(
            model_id=comp_runner.config.model_id,
            rank=rank,
            baseline_loss=raw_loss,
            compressed_loss=comp_loss,
            loss_delta=comp_loss - raw_loss,
            top1_agreement=top1_agreement,
            kl_divergence=kl_div,
            hops=hop_metrics,
        )

    def analyze_isolated_hops(
        self,
        raw_runner: ChainRunner,
        rank: int,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> SensitivityReport:
        """Measure isolated per-hop compression distortion.

        Compresses activations *only* at hop h, then propagates the reconstructed
        tensor through the rest of the chain to isolate individual hop sensitivity.
        """
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        raw_res = raw_runner.forward(input_ids, logits_keep_last=0)

        if targets is None:
            raw_eval_logits = raw_res.logits[:, :-1, :].contiguous()
            eval_targets = input_ids[:, 1:].contiguous()
        else:
            raw_eval_logits = raw_res.logits
            eval_targets = targets

        raw_loss = F.cross_entropy(
            raw_eval_logits.reshape(-1, raw_eval_logits.shape[-1]),
            eval_targets.reshape(-1),
        ).item()

        config = raw_runner.config
        shard_plan = config.shard_plan() if hasattr(config, "shard_plan") else []
        hop_metrics: list[HopSensitivity] = []

        # Iterate over intermediate hops that produce activations
        for h, hop_res in enumerate(raw_res.hops):
            raw_t = hop_res.tensor
            layer_range = (
                f"[{shard_plan[h].start}:{shard_plan[h].end})" if h < len(shard_plan) else None
            )

            # If hop produces final logits or token IDs (hop 0 input), skip or evaluate passthrough
            if hop_res.is_final or raw_t.ndim < 3 or not raw_t.is_floating_point():
                hop_metrics.append(
                    HopSensitivity(
                        hop=h,
                        layer_range=layer_range,
                        cosine_similarity=1.0,
                        l2_relative_error=0.0,
                        mse=0.0,
                        immediate_cosine_similarity=1.0,
                        immediate_l2_relative_error=0.0,
                        isolated_loss=raw_loss,
                        isolated_loss_delta=0.0,
                    )
                )
                continue

            # Simulate isolated low-rank projection at hop h
            codec = get_codec("lowrank", rank=rank, seed=self.seed)
            encoded = codec.encode(raw_t, request_id="iso-test", hop=h + 1)
            decoded = codec.decode(encoded)

            imm_cos = F.cosine_similarity(raw_t.float(), decoded.float(), dim=-1).mean().item()
            imm_mse = F.mse_loss(raw_t.float(), decoded.float()).item()
            norm_raw = torch.norm(raw_t.float())
            imm_l2 = (torch.norm(raw_t.float() - decoded.float()) / (norm_raw + 1e-12)).item()

            # Forward reconstructed tensor through remaining shards if any
            current = decoded
            for remaining_hop, client in enumerate(raw_runner.clients[h + 1 :], start=h + 1):
                fwd_res = client.forward(
                    current,
                    request_id="iso-test",
                    hop=remaining_hop,
                    logits_keep_last=0,
                )
                current = fwd_res.tensor

            # current is now the final logits resulting from isolated compression at hop h
            iso_eval_logits = current[:, :-1, :].contiguous() if targets is None else current
            iso_loss = F.cross_entropy(
                iso_eval_logits.reshape(-1, iso_eval_logits.shape[-1]),
                eval_targets.reshape(-1),
            ).item()

            hop_metrics.append(
                HopSensitivity(
                    hop=h,
                    layer_range=layer_range,
                    cosine_similarity=imm_cos,
                    l2_relative_error=imm_l2,
                    mse=imm_mse,
                    immediate_cosine_similarity=imm_cos,
                    immediate_l2_relative_error=imm_l2,
                    isolated_loss=iso_loss,
                    isolated_loss_delta=iso_loss - raw_loss,
                )
            )

        return SensitivityReport(
            model_id=raw_runner.config.model_id,
            rank=rank,
            baseline_loss=raw_loss,
            compressed_loss=hop_metrics[-1].isolated_loss or raw_loss,
            loss_delta=(hop_metrics[-1].isolated_loss or raw_loss) - raw_loss,
            top1_agreement=1.0,
            kl_divergence=0.0,
            hops=hop_metrics,
        )


def analyze_layer_sensitivity(
    raw_runner: ChainRunner,
    comp_runner: ChainRunner | None = None,
    input_ids: torch.Tensor | None = None,
    *,
    rank: int = 128,
    seed: int = 42,
) -> SensitivityReport:
    """Convenience function to analyze layer-depth sensitivity."""
    if input_ids is None:
        input_ids = torch.randint(0, 256, (1, 64), dtype=torch.long)

    analyzer = LayerSensitivityAnalyzer(seed=seed)
    if comp_runner is not None:
        return analyzer.compare_chain_runs(raw_runner, comp_runner, input_ids)
    return analyzer.analyze_isolated_hops(raw_runner, rank, input_ids)
