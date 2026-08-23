"""Running one shard's slice of the forward pass.

A :class:`ShardRuntime` owns a contiguous range of decoder layers, plus the
embedding if it is first in the chain and the final norm + LM head if it is
last. It takes hidden states in and gives hidden states out, which is exactly
the tensor the codec compresses and the transport ships.

Correctness anchor: running every shard of a plan in sequence must reproduce the
monolithic ``model(input_ids)`` logits *bit-exactly*. ``tests/test_shard_runtime.py``
asserts that, because a sharded pipeline that is merely close would quietly
poison every downstream quality measurement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.masking_utils import create_causal_mask

from torrent_llm.shard.plan import ShardSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShardOutput:
    """What a shard hands to the next hop, or to the caller if it is last.

    Exactly one of ``hidden_states`` and ``logits`` is set: intermediate shards
    produce hidden states to forward on, the final shard produces logits.
    """

    hidden_states: torch.Tensor | None = None
    logits: torch.Tensor | None = None

    @property
    def tensor(self) -> torch.Tensor:
        """Whichever of the two this shard actually produced."""
        out = self.hidden_states if self.hidden_states is not None else self.logits
        if out is None:
            raise ValueError("shard produced neither hidden states nor logits")
        return out


def load_shard(
    model_id: str,
    spec: ShardSpec,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    trust_remote_code: bool = False,
) -> ShardRuntime:
    """Load ``model_id`` and keep only the parts ``spec`` owns.

    The full checkpoint is materialised on CPU and then pruned. That is wasteful
    of host RAM at load time — peak is the whole model, not the shard — but it is
    correct for any architecture and it is not the bottleneck at the 1-2B dev
    sizes this harness iterates on. Lazily reading only the needed tensors out of
    the safetensors index is the obvious optimisation once shard sizes actually
    strain the box; it does not change this function's contract.
    """
    logger.info("loading %s for %s on %s (%s)", model_id, spec, device, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, trust_remote_code=trust_remote_code
    )
    return ShardRuntime(model, spec, device=device)


class ShardRuntime:
    """One node's slice of a decoder-only causal LM."""

    def __init__(self, model: nn.Module, spec: ShardSpec, *, device: str = "cpu") -> None:
        base = _decoder_stack(model)
        if len(base.layers) != spec.num_layers:
            raise ValueError(
                f"{spec} was planned for {spec.num_layers} layers but the model has "
                f"{len(base.layers)}"
            )

        self.spec = spec
        self.device = torch.device(device)
        self.config = model.config
        self.dtype = next(model.parameters()).dtype

        # Rotary embeddings are position-dependent, not layer-dependent, so every
        # shard needs its own copy to build cos/sin for its slice.
        self.rotary_emb = base.rotary_emb
        self.layers = nn.ModuleList(base.layers[spec.start : spec.end])
        self.embed_tokens = base.embed_tokens if spec.is_first else None
        self.norm = base.norm if spec.is_last else None
        self.lm_head = model.lm_head if spec.is_last else None

        self._modules_in_use = nn.ModuleList(
            m
            for m in (self.rotary_emb, self.layers, self.embed_tokens, self.norm, self.lm_head)
            if m is not None
        )
        # Drop the parent's references so the layers this shard does not own
        # become garbage; otherwise every node holds the entire model.
        del model
        self._modules_in_use.to(self.device).eval()

    @property
    def hidden_size(self) -> int:
        """Width of the activation tensor crossing the wire."""
        return int(self.config.hidden_size)

    @torch.inference_mode()
    def forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> ShardOutput:
        """Run this shard's layers.

        Args:
            input_ids: Token ids. Required on the first shard, rejected elsewhere.
            hidden_states: Activations from the previous hop. Required on every
                shard but the first.
            position_ids: Absolute positions. Defaults to ``0..seq_len-1``. Every
                shard must be given the *same* positions for rotary embeddings to
                line up across the chain.
            attention_mask: Optional padding mask, ``(batch, seq)``.

        Returns:
            Hidden states, or logits if this is the last shard.
        """
        h = self._entry_hidden_states(input_ids, hidden_states)
        h = h.to(self.device, self.dtype)

        if position_ids is None:
            position_ids = torch.arange(h.shape[1], device=self.device).unsqueeze(0)
        position_ids = position_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        position_embeddings = self.rotary_emb(h, position_ids)
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=h,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )

        for layer in self.layers:
            h = layer(
                h,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )

        if not self.spec.is_last:
            return ShardOutput(hidden_states=h)

        assert self.norm is not None and self.lm_head is not None
        return ShardOutput(logits=self.lm_head(self.norm(h)))

    def _entry_hidden_states(
        self, input_ids: torch.Tensor | None, hidden_states: torch.Tensor | None
    ) -> torch.Tensor:
        """Resolve this shard's input, rejecting the two easy caller mistakes."""
        # Check for the wrong argument before the missing one: a caller who sent
        # the wrong kind of input has misunderstood the topology, and saying so
        # is more useful than telling them what is absent.
        if self.spec.is_first:
            if hidden_states is not None:
                raise ValueError(
                    f"{self.spec} owns the embedding; pass input_ids, not hidden_states"
                )
            if input_ids is None:
                raise ValueError(f"{self.spec} owns the embedding and needs input_ids")
            assert self.embed_tokens is not None
            return self.embed_tokens(input_ids.to(self.device))

        if input_ids is not None:
            raise ValueError(f"{self.spec} does not own the embedding; pass hidden_states")
        if hidden_states is None:
            raise ValueError(f"{self.spec} is not the first shard and needs hidden_states")
        return hidden_states

    def __repr__(self) -> str:
        return (
            f"ShardRuntime({self.spec}, layers={self.spec.depth}, "
            f"hidden={self.hidden_size}, device={self.device}, dtype={self.dtype})"
        )


def _decoder_stack(model: nn.Module) -> nn.Module:
    """Find the module holding ``.layers``, ``.embed_tokens``, ``.norm``.

    For Llama/Qwen/Mistral this is ``model.model``. Checking by attribute rather
    than by class keeps the harness working across architectures without a
    per-model table.
    """
    for candidate in (getattr(model, "model", None), model):
        if candidate is not None and all(
            hasattr(candidate, attr) for attr in ("layers", "embed_tokens", "norm", "rotary_emb")
        ):
            return candidate
    raise TypeError(
        f"{type(model).__name__} does not expose the decoder stack this harness shards "
        "(expected .layers, .embed_tokens, .norm and .rotary_emb)"
    )


def num_layers_of(model_id: str, *, trust_remote_code: bool = False) -> int:
    """Read a model's decoder depth from its config, without loading weights.

    Lets the runner build a shard plan before any node has downloaded anything.
    """
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    return int(config.num_hidden_layers)
