"""Fitting a low-rank projection to what the model actually sends.

The uncalibrated codec projects onto random directions. That keeps the energy of a
vector but not the *information* the next layers need, and on a real model it breaks
the output. Here the directions are chosen from data instead: collect the hidden
states that cross a shard boundary, and keep the directions in which they vary most
(principal components).

Two details matter on real models:

* **Centering.** Activations carry large near-constant offsets in a few dimensions.
  They are removed before projecting and restored after, which costs nothing on the
  wire because the offset is part of the (static) weights.
* **The first token.** Many models put enormous values on the first position (an
  "attention sink"). Left in the fit, that one position decides the principal
  directions. :meth:`CovarianceAccumulator.add` can leave it out of the *fit*; it is
  still compressed at inference like any other position.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Projection:
    """A fitted projection: what to subtract, what to project onto, what it keeps."""

    mean: torch.Tensor  # (hidden,)
    down_proj: torch.Tensor  # (hidden, rank), orthonormal columns
    up_proj: torch.Tensor  # (rank, hidden)
    #: Share of the data's variance the kept directions explain, in [0, 1].
    explained_variance: float

    @property
    def rank(self) -> int:
        return int(self.down_proj.shape[1])


class CovarianceAccumulator:
    """Streams activations in and keeps just enough to fit a projection later.

    Holds the running sum and the running sum of outer products, so any amount of
    data fits in ``hidden x hidden`` numbers, in float64 so that adding millions of
    rows does not lose the small directions.
    """

    def __init__(self, hidden_size: int, device: torch.device | str = "cpu") -> None:
        self.hidden_size = hidden_size
        self.count = 0
        self._sum = torch.zeros(hidden_size, dtype=torch.float64, device=device)
        self._outer = torch.zeros(hidden_size, hidden_size, dtype=torch.float64, device=device)

    def add(self, activations: torch.Tensor, *, skip_first_position: bool = False) -> None:
        """Add a batch shaped ``(batch, seq, hidden)`` or ``(rows, hidden)``."""
        if activations.shape[-1] != self.hidden_size:
            raise ValueError(
                f"expected hidden size {self.hidden_size}, got {activations.shape[-1]}"
            )
        if skip_first_position and activations.ndim == 3:
            activations = activations[:, 1:, :]
        rows = activations.reshape(-1, self.hidden_size).to(self._sum.device, torch.float64)
        if rows.shape[0] == 0:
            return
        self.count += rows.shape[0]
        self._sum += rows.sum(dim=0)
        self._outer += rows.T @ rows

    def fit(self, rank: int) -> Projection:
        """The best ``rank``-dimensional projection of everything added so far."""
        if self.count < 2:
            raise ValueError("need at least two activation rows to fit a projection")
        if not 1 <= rank <= self.hidden_size:
            raise ValueError(f"rank must be in [1, {self.hidden_size}], got {rank}")
        mean = self._sum / self.count
        covariance = self._outer / self.count - torch.outer(mean, mean)
        # Symmetrise: rounding makes it very slightly asymmetric, and eigh wants it exact.
        covariance = (covariance + covariance.T) / 2
        values, vectors = torch.linalg.eigh(covariance)
        order = torch.argsort(values, descending=True)
        values, vectors = values[order].clamp(min=0), vectors[:, order]
        basis = vectors[:, :rank]
        total = float(values.sum())
        explained = float(values[:rank].sum()) / total if total > 0 else 1.0
        return Projection(
            mean=mean.to(torch.float32).cpu(),
            down_proj=basis.to(torch.float32).cpu().contiguous(),
            up_proj=basis.T.to(torch.float32).cpu().contiguous(),
            explained_variance=explained,
        )

    def variance_curve(self) -> torch.Tensor:
        """Cumulative share of variance explained by the first ``k`` directions, for all ``k``."""
        mean = self._sum / self.count
        covariance = self._outer / self.count - torch.outer(mean, mean)
        values = torch.linalg.eigvalsh((covariance + covariance.T) / 2).flip(0).clamp(min=0)
        total = values.sum()
        return (values.cumsum(0) / total).cpu() if total > 0 else torch.ones_like(values).cpu()
