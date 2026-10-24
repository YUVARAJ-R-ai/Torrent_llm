"""Choosing the next token, wherever the logits happen to be.

Lives at the top of the package because both ends of a hop need it: the runner
builds a :class:`Sampling` rule, and the shard that owns the LM head applies it
(issue #33). Applying it on the shard is the point. The logits are
``batch x seq x vocab``, and on a modern tokenizer vocab dwarfs everything else
on the wire -- 151936 x 4 bytes = 593.5 KiB per decode step on Qwen3-0.6B,
against a 4 KiB hidden state. The token the caller actually wants is 8 bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class Sampling:
    """How to pick the next token from the final position's logits.

    Attributes:
        temperature: ``0`` is greedy (argmax). Anything above samples from the
            softmax of ``logits / temperature``.
        top_k: Sample only among the ``top_k`` most likely tokens; ``0`` means
            the whole vocabulary. Ignored when greedy.
        seed: Seeds the draw, so a run is reproducible. A generation loop must
            vary it per step (see :meth:`for_step`), or every step would reuse
            the same random number.
    """

    temperature: float = 0.0
    top_k: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(f"temperature must not be negative, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must not be negative, got {self.top_k}")

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0

    def for_step(self, step: int) -> Sampling:
        """This rule with a seed unique to one generation step."""
        return replace(self, seed=self.seed + step)


GREEDY = Sampling()


def select_tokens(logits: torch.Tensor, sampling: Sampling) -> torch.Tensor:
    """Pick one token per sequence from the **last** position.

    Args:
        logits: ``(batch, seq, vocab)``; only ``[:, -1, :]`` is read.

    Returns:
        ``(batch, 1)`` int64 token ids, on the CPU.
    """
    last = logits[:, -1, :]
    if sampling.is_greedy:
        # The same argmax the runner used to do on its side of the wire, on the
        # same lossless logits, so greedy output does not change by moving it.
        return last.argmax(dim=-1, keepdim=True).to("cpu", torch.long)

    scores = last.float() / sampling.temperature
    if sampling.top_k:
        kth = torch.topk(scores, min(sampling.top_k, scores.shape[-1]), dim=-1).values[:, -1:]
        scores = scores.masked_fill(scores < kth, float("-inf"))
    probs = torch.softmax(scores, dim=-1).cpu()
    generator = torch.Generator().manual_seed(sampling.seed)
    return torch.multinomial(probs, num_samples=1, generator=generator).to(torch.long)
