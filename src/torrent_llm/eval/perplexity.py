"""Cross-entropy loss and perplexity evaluation harness (Issue #10).

Calculates chunked and sliding-window cross-entropy loss and perplexity
using ChainRunner over distributed shards, comparing uncompressed baseline
against low-rank compressed runs across arbitrary ranks.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from torrent_llm.runner import ChainRunner


@dataclass(frozen=True)
class PerplexityResult:
    """Outcome of a perplexity evaluation pass."""

    loss: float
    perplexity: float
    total_tokens: int
    num_chunks: int
    codec: str
    rank: int | None = None
    wall_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": round(self.loss, 4),
            "perplexity": round(self.perplexity, 4),
            "total_tokens": self.total_tokens,
            "num_chunks": self.num_chunks,
            "codec": self.codec,
            "rank": self.rank,
            "wall_time_s": round(self.wall_time_s, 4),
        }


#: Curated deterministic synthetic corpus for offline evaluation without HuggingFace downloads.
SYNTHETIC_EVAL_CORPUS: list[str] = [
    (
        "Distributed transformer inference partitions decoder layers across distinct nodes "
        "connected by standard network interfaces. In peer-to-peer topologies, activation "
        "tensors sent between pipeline hops represent the primary communication bottleneck. "
        "Applying low-rank matrix decomposition compresses intermediate hidden states, "
        "reducing serialized byte volume proportionally to rank over hidden dimension."
    ),
    (
        "Low-rank compression projects high-dimensional activation matrices into a compact "
        "latent subspace before network serialization. On the receiving node, activation "
        "reconstruction approximates the original hidden state before propagating through "
        "subsequent attention and feed-forward layers. We evaluate cross-entropy distortion."
    ),
    (
        "Language models exhibit varying degrees of sensitivity to activation compression "
        "depending on layer depth. Early layers extract fundamental syntactic representations, "
        "while intermediate layers capture long-range contextual dependencies and abstract logic. "
        "Evaluating perplexity across rank bottlenecks maps the rate-distortion frontier."
    ),
    (
        "A formal evaluation harness measures both language modeling perplexity and downstream "
        "reasoning accuracy. When rank is constrained, reconstruction error compounds across hops, "
        "leading to higher next-token perplexity and degraded greedy reasoning accuracy."
    ),
]


def get_synthetic_corpus() -> list[str]:
    """Return the deterministic synthetic text evaluation corpus."""
    return list(SYNTHETIC_EVAL_CORPUS)


def generate_synthetic_tokens(
    vocab_size: int = 256,
    seq_len: int = 128,
    num_sequences: int = 4,
    seed: int = 42,
) -> list[torch.Tensor]:
    """Generate deterministic synthetic token sequences for offline unit tests.

    Args:
        vocab_size: Token vocabulary ceiling (kept <= 256 for tiny test models).
        seq_len: Number of tokens per sequence.
        num_sequences: Number of distinct sequences to generate.
        seed: Random seed for repeatability.
    """
    gen = torch.Generator().manual_seed(seed)
    sequences: list[torch.Tensor] = []
    for _ in range(num_sequences):
        ids = torch.randint(0, vocab_size, (1, seq_len), generator=gen, dtype=torch.long)
        sequences.append(ids)
    return sequences


def _prepare_sequences(
    sequences: torch.Tensor | list[torch.Tensor] | list[str] | None,
    tokenizer: Any = None,
    default_vocab_size: int = 256,
    default_seq_len: int = 64,
) -> list[torch.Tensor]:
    """Normalize input sequences into a list of 2D LongTensor batches."""
    if sequences is None:
        if tokenizer is not None:
            sequences = get_synthetic_corpus()
        else:
            return generate_synthetic_tokens(vocab_size=default_vocab_size, seq_len=default_seq_len)

    if isinstance(sequences, torch.Tensor):
        if sequences.ndim == 1:
            return [sequences.unsqueeze(0).to(torch.long)]
        elif sequences.ndim == 2:
            return [sequences[i : i + 1].to(torch.long) for i in range(sequences.shape[0])]
        else:
            raise ValueError(f"Tensor sequences must be 1D or 2D, got shape {sequences.shape}")

    tensors: list[torch.Tensor] = []
    for item in sequences:
        if isinstance(item, torch.Tensor):
            tensors.append(
                item.unsqueeze(0).to(torch.long) if item.ndim == 1 else item.to(torch.long)
            )
        elif isinstance(item, str):
            if tokenizer is not None:
                encoded = tokenizer(item, return_tensors="pt").input_ids.to(torch.long)
                tensors.append(encoded)
            else:
                # Deterministic ASCII byte fallback for tokenizer-free offline tests
                byte_ids = [min(ord(c), default_vocab_size - 1) for c in item]
                ids = torch.tensor([byte_ids], dtype=torch.long)
                tensors.append(ids)
        else:
            raise TypeError(f"Unsupported sequence item type: {type(item)}")
    return tensors


def compute_perplexity(
    runner: ChainRunner,
    sequences: torch.Tensor | list[torch.Tensor] | list[str] | None = None,
    *,
    tokenizer: Any = None,
    max_chunk_size: int = 512,
    stride: int | None = None,
) -> PerplexityResult:
    """Compute cross-entropy loss and perplexity across sequences using ChainRunner.

    Uses memory-safe chunked / sliding-window evaluation so long sequences do not
    allocate huge intermediate logit buffers across the distributed shard chain.

    Args:
        runner: Connected ChainRunner instance.
        sequences: Input token tensors or text strings. If None, uses synthetic corpus.
        tokenizer: Optional tokenizer for string encoding.
        max_chunk_size: Maximum sequence length evaluated in a single forward pass.
        stride: Step size for sliding-window evaluation. If None or >= max_chunk_size,
            non-overlapping chunking is used.

    Returns:
        PerplexityResult containing mean loss, perplexity, token count, and metadata.
    """
    token_sequences = _prepare_sequences(sequences, tokenizer)
    total_loss = 0.0
    total_tokens = 0
    num_chunks = 0
    start_time = time.perf_counter()

    for seq in token_sequences:
        seq_len = seq.shape[1]
        if seq_len < 2:
            continue

        effective_stride = stride if (stride is not None and stride > 0) else (max_chunk_size - 1)
        prev_end_idx = 0

        for i in range(0, seq_len - 1, effective_stride):
            end_idx = min(i + max_chunk_size, seq_len)
            chunk = seq[:, i:end_idx]
            chunk_len = chunk.shape[1]
            if chunk_len < 2:
                break

            result = runner.forward(chunk, logits_keep_last=0)
            logits = result.logits[:, :-1, :].contiguous()
            targets = chunk[:, 1:].contiguous()

            if i == 0:
                eval_logits = logits
                eval_targets = targets
                prev_end_idx = end_idx
            else:
                target_len = end_idx - prev_end_idx
                if target_len <= 0:
                    break
                eval_logits = logits[:, -target_len:, :]
                eval_targets = targets[:, -target_len:]
                prev_end_idx = end_idx

            loss = F.cross_entropy(
                eval_logits.reshape(-1, eval_logits.shape[-1]),
                eval_targets.reshape(-1),
                reduction="sum",
            )

            total_loss += loss.item()
            total_tokens += eval_targets.numel()
            num_chunks += 1

            if end_idx >= seq_len:
                break

    wall_time = time.perf_counter() - start_time
    mean_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    ppl = math.exp(min(mean_loss, 100.0)) if total_tokens > 0 else 1.0

    codec_name = runner.config.codec
    rank = runner.config.codec_args.get("rank") if runner.config.codec == "lowrank" else None

    return PerplexityResult(
        loss=mean_loss,
        perplexity=ppl,
        total_tokens=total_tokens,
        num_chunks=num_chunks,
        codec=codec_name,
        rank=rank,
        wall_time_s=wall_time,
    )


def evaluate_perplexity_across_ranks(
    runners_by_rank: dict[str | int, ChainRunner],
    sequences: torch.Tensor | list[torch.Tensor] | list[str] | None = None,
    *,
    tokenizer: Any = None,
    max_chunk_size: int = 512,
    stride: int | None = None,
) -> dict[str | int, PerplexityResult]:
    """Evaluate perplexity across multiple runners configured with different ranks.

    Args:
        runners_by_rank: Mapping from rank identifier (e.g., 'raw', 128, 64) to ChainRunner.
        sequences: Token sequences or text strings.
        tokenizer: Optional tokenizer.
        max_chunk_size: Maximum chunk size per pass.
        stride: Sliding window stride.

    Returns:
        Mapping from rank identifier to PerplexityResult.
    """
    token_sequences = _prepare_sequences(sequences, tokenizer)
    results: dict[str | int, PerplexityResult] = {}
    for rank_key, runner in runners_by_rank.items():
        results[rank_key] = compute_perplexity(
            runner,
            token_sequences,
            max_chunk_size=max_chunk_size,
            stride=stride,
        )
    return results
