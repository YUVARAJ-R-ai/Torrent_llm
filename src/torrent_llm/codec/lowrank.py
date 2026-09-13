"""Low-rank activation compressor (Issue #8).

Implements MLA-inspired low-rank projection at the network boundary.
Hidden activations ``(batch, seq, hidden)`` are down-projected to ``(batch, seq, rank)``
before wire serialization, and up-projected back to ``(batch, seq, hidden)`` on the
receiving shard.

Design decisions:
1. Token ID / 2D Passthrough: On hop 0, token IDs of shape ``(batch, seq)`` are
   passed through uncompressed via raw byte framing to prevent dimensional crashes.
2. Multi-Process Weight Synchronization: If weights are not explicitly supplied,
   projection matrices are deterministically derived using a shared pseudo-random
   seed (default: 42) via QR orthonormal decomposition, ensuring separate shard
   processes generate bit-compatible projection weights.
3. Orthonormal Scaling: For uncalibrated baseline execution, ``W_down`` has orthonormal
   columns (``Q[:, :r]``) and ``W_up = W_down.T``. This forms an exact orthogonal projector
   ``P = W_down @ W_down.T`` that preserves activation energy without norm divergence.
4. Device & Dtype Coercion: Projection matrices are lazily moved and cast to match
   ``tensor.device`` and ``tensor.dtype`` before matmul operations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from torrent_llm.codec.base import Codec, register_codec
from torrent_llm.codec.raw import RawCodec, bytes_to_tensor, tensor_to_bytes
from torrent_llm.wire import ActivationHeader, ActivationMessage


def _generate_orthonormal_projections(
    hidden_size: int, rank: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically generate orthonormal down- and up-projection matrices.

    Generates a Gaussian matrix on CPU using a fixed seed, then uses QR
    decomposition so columns of W_down are orthonormal. Setting W_up = W_down.T
    ensures W_down @ W_up is an orthogonal projection matrix of rank min(rank, hidden_size).
    """
    effective_rank = min(rank, hidden_size)
    gen = torch.Generator().manual_seed(seed)
    # Generate on CPU in float32 for deterministic cross-device math
    gaussian = torch.randn(hidden_size, effective_rank, generator=gen, dtype=torch.float32)
    q, _ = torch.linalg.qr(gaussian)
    w_down = q[:, :effective_rank].contiguous()
    w_up = w_down.t().contiguous()
    return w_down, w_up


@register_codec("lowrank")
class LowRankCodec(Codec):
    """Compresses activations by projecting onto a low-rank latent subspace.

    Args:
        rank: Latent bottleneck dimension (e.g., 64, 128, 256).
        seed: Random seed for deterministic weight generation across processes.
        wire_dtype: Optional dtype to cast the latent tensor to before transmission.
        weights_path: Optional path to a checkpoint containing 'down_proj' and 'up_proj'.
        down_proj: Explicit down-projection matrix ``(hidden_size, rank)``.
        up_proj: Explicit up-projection matrix ``(rank, hidden_size)``.
    """

    def __init__(
        self,
        rank: int = 128,
        seed: int = 42,
        wire_dtype: str | None = None,
        weights_path: str | Path | None = None,
        down_proj: torch.Tensor | None = None,
        up_proj: torch.Tensor | None = None,
    ) -> None:
        self.rank = int(rank)
        if self.rank < 1:
            raise ValueError(f"rank must be at least 1, got {self.rank}")
        self.seed = int(seed)
        self.wire_dtype = wire_dtype
        self.weights_path = Path(weights_path) if weights_path is not None else None

        self._raw = RawCodec(wire_dtype=wire_dtype)
        self._weights_cache: dict[
            tuple[int, torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]
        ] = {}

        # Base CPU weights if explicitly passed or loaded from file
        self._base_down_proj: torch.Tensor | None = None
        self._base_up_proj: torch.Tensor | None = None

        if down_proj is not None and up_proj is not None:
            self._base_down_proj = down_proj.detach().cpu().to(torch.float32)
            self._base_up_proj = up_proj.detach().cpu().to(torch.float32)
        elif self.weights_path is not None:
            self._load_weights(self.weights_path)

    def _load_weights(self, path: Path) -> None:
        """Load down_proj and up_proj from safetensors or PyTorch checkpoint."""
        if not path.exists():
            raise FileNotFoundError(f"weights file not found: {path}")

        loaded: dict[str, Any]
        if path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
                loaded = load_file(str(path))
            except ImportError:
                raise ImportError("safetensors is required to load .safetensors weights") from None
        else:
            loaded = torch.load(path, map_location="cpu", weights_only=True)

        if "down_proj" not in loaded or "up_proj" not in loaded:
            keys = list(loaded.keys())
            raise KeyError(f"weights checkpoint must contain 'down_proj' and 'up_proj', got {keys}")

        self._base_down_proj = loaded["down_proj"].detach().cpu().to(torch.float32)
        self._base_up_proj = loaded["up_proj"].detach().cpu().to(torch.float32)

    def save_weights(self, path: str | Path, hidden_size: int) -> None:
        """Save calibrated projection weights to file for deployment."""
        target_path = Path(path)
        w_down, w_up = self._get_projections(hidden_size, torch.device("cpu"), torch.float32)
        state = {"down_proj": w_down, "up_proj": w_up}
        if target_path.suffix == ".safetensors":
            from safetensors.torch import save_file
            save_file(state, str(target_path))
        else:
            torch.save(state, str(target_path))

    def _get_projections(
        self, hidden_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch or generate projections matching target hidden_size, device, and dtype."""
        cache_key = (hidden_size, device, dtype)
        if cache_key in self._weights_cache:
            return self._weights_cache[cache_key]

        if self._base_down_proj is not None and self._base_up_proj is not None:
            if self._base_down_proj.shape[0] != hidden_size:
                raise ValueError(
                    f"configured down_proj hidden_size {self._base_down_proj.shape[0]} "
                    f"does not match tensor hidden_size {hidden_size}"
                )
            w_down = self._base_down_proj.to(device=device, dtype=dtype)
            w_up = self._base_up_proj.to(device=device, dtype=dtype)
        else:
            cpu_down, cpu_up = _generate_orthonormal_projections(hidden_size, self.rank, self.seed)
            w_down = cpu_down.to(device=device, dtype=dtype)
            w_up = cpu_up.to(device=device, dtype=dtype)

        self._weights_cache[cache_key] = (w_down, w_up)
        return w_down, w_up

    def encode(self, tensor: torch.Tensor, *, request_id: str, hop: int) -> ActivationMessage:
        """Project activation down to latent subspace and serialize to bytes.

        If tensor is not 3D floating-point activations (e.g. 2D token IDs on hop 0),
        it is passed through verbatim via raw byte framing.
        """
        # Hop 0 passthrough: token_ids or non-floating-point tensors
        if tensor.ndim < 3 or not tensor.is_floating_point():
            raw_msg = self._raw.encode(tensor, request_id=request_id, hop=hop)
            header = ActivationHeader(
                request_id=request_id,
                hop=hop,
                codec=self.name,
                dtype=raw_msg.header.dtype,
                shape=raw_msg.header.shape,
                codec_meta={"passthrough": True},
            )
            return ActivationMessage(header=header, payload=raw_msg.payload)

        # Zero-sequence edge case
        if tensor.numel() == 0:
            header = ActivationHeader(
                request_id=request_id,
                hop=hop,
                codec=self.name,
                dtype=str(tensor.dtype).removeprefix("torch."),
                shape=tuple(tensor.shape),
                codec_meta={"passthrough": False, "rank": self.rank, "empty": True},
            )
            return ActivationMessage(header=header, payload=b"")

        hidden_size = tensor.shape[-1]
        w_down, _ = self._get_projections(hidden_size, tensor.device, tensor.dtype)

        # Down-project: (..., hidden) @ (hidden, rank) -> (..., rank)
        latent = tensor @ w_down

        if self.wire_dtype is not None:
            latent = latent.to(getattr(torch, self.wire_dtype))

        header = ActivationHeader(
            request_id=request_id,
            hop=hop,
            codec=self.name,
            dtype=str(tensor.dtype).removeprefix("torch."),
            shape=tuple(tensor.shape),
            codec_meta={
                "passthrough": False,
                "rank": w_down.shape[1],
                "latent_shape": list(latent.shape),
                "latent_dtype": str(latent.dtype).removeprefix("torch."),
            },
        )
        return ActivationMessage(header=header, payload=tensor_to_bytes(latent))

    def decode(self, message: ActivationMessage) -> torch.Tensor:
        """Reconstruct the original activation tensor from compressed latent bytes."""
        meta = message.header.codec_meta
        logical_dtype = getattr(torch, message.header.dtype)

        # Passthrough handling
        if meta.get("passthrough"):
            return bytes_to_tensor(message.payload, logical_dtype, message.header.shape)

        # Empty tensor handling
        if meta.get("empty") or not message.payload:
            return torch.empty(message.header.shape, dtype=logical_dtype)

        latent_shape = tuple(meta["latent_shape"])
        latent_dtype = getattr(torch, meta.get("latent_dtype", message.header.dtype))

        latent = bytes_to_tensor(message.payload, latent_dtype, latent_shape)

        hidden_size = message.header.shape[-1]
        # Reconstruct on CPU matching latent dtype
        _, w_up = self._get_projections(hidden_size, latent.device, latent.dtype)

        # Up-project: (..., rank) @ (rank, hidden) -> (..., hidden)
        reconstructed = latent @ w_up

        if reconstructed.dtype != logical_dtype:
            reconstructed = reconstructed.to(logical_dtype)

        return reconstructed.reshape(message.header.shape)

    def describe(self) -> dict[str, object]:
        return {
            "codec": self.name,
            "rank": self.rank,
            "seed": self.seed,
            "wire_dtype": self.wire_dtype,
            "weights_path": str(self.weights_path) if self.weights_path else None,
        }
