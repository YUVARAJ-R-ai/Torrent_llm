"""Shared fixtures.

Everything here builds a randomly-initialised tiny Llama from a config, so the
suite runs offline and in seconds. No test downloads a checkpoint.
"""

import copy

import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig

TINY_CONFIG = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=6,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=256,
    max_position_embeddings=128,
)


@pytest.fixture(scope="session")
def tiny_model():
    """One reference model. Do not shard this instance — sharding prunes it."""
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(LlamaConfig(**TINY_CONFIG)).eval()


@pytest.fixture
def model_factory(tiny_model):
    """Fresh copies of the reference weights.

    A ShardRuntime prunes the model it is handed, so building an N-shard chain
    needs N independent copies that nonetheless carry identical weights.
    """

    def make():
        return copy.deepcopy(tiny_model)

    return make


@pytest.fixture(scope="session")
def num_layers():
    return TINY_CONFIG["num_hidden_layers"]
