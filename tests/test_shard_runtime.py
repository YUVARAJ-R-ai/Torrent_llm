"""The sharded forward pass must equal the monolithic one bit-exactly.

This is the anchor for everything downstream: if the chain is merely *close* to
the reference model, every compression-quality number the project publishes is
measuring shard drift as well as compression loss.
"""

import pytest
import torch

from torrent_llm.shard import ShardRuntime, plan_even, plan_explicit


def run_chain(model_factory, specs, input_ids):
    """Walk the whole chain, passing hidden states hop to hop."""
    shards = [ShardRuntime(model_factory(), spec) for spec in specs]

    out = shards[0].forward(input_ids=input_ids)
    for shard in shards[1:]:
        out = shard.forward(hidden_states=out.hidden_states)
    return out


@pytest.fixture
def input_ids():
    torch.manual_seed(1)
    return torch.randint(0, 256, (1, 12))


def test_two_shard_chain_matches_the_monolithic_model_exactly(
    tiny_model, model_factory, num_layers, input_ids
):
    with torch.inference_mode():
        reference = tiny_model(input_ids=input_ids).logits

    out = run_chain(model_factory, plan_even(num_layers, 2), input_ids)

    assert out.logits is not None
    assert torch.equal(out.logits, reference)


def test_three_shard_chain_matches_too(tiny_model, model_factory, num_layers, input_ids):
    with torch.inference_mode():
        reference = tiny_model(input_ids=input_ids).logits

    out = run_chain(model_factory, plan_even(num_layers, 3), input_ids)

    assert torch.equal(out.logits, reference)


def test_uneven_shard_boundaries_match_too(tiny_model, model_factory, num_layers, input_ids):
    # A lopsided split is what heterogeneous hardware (issue #17) will produce,
    # so it must be just as exact as an even one.
    with torch.inference_mode():
        reference = tiny_model(input_ids=input_ids).logits

    out = run_chain(model_factory, plan_explicit(num_layers, [1]), input_ids)

    assert torch.equal(out.logits, reference)


def test_batched_input_matches(tiny_model, model_factory, num_layers):
    torch.manual_seed(2)
    ids = torch.randint(0, 256, (3, 9))
    with torch.inference_mode():
        reference = tiny_model(input_ids=ids).logits

    out = run_chain(model_factory, plan_even(num_layers, 2), ids)

    assert torch.equal(out.logits, reference)


def test_intermediate_shard_emits_hidden_states_not_logits(model_factory, num_layers, input_ids):
    first, _ = plan_even(num_layers, 2)
    shard = ShardRuntime(model_factory(), first)

    out = shard.forward(input_ids=input_ids)

    assert out.logits is None
    assert out.hidden_states.shape == (1, 12, shard.hidden_size)


def test_first_shard_rejects_hidden_states(model_factory, num_layers, input_ids):
    first, _ = plan_even(num_layers, 2)
    shard = ShardRuntime(model_factory(), first)

    with pytest.raises(ValueError, match="pass input_ids, not hidden_states"):
        shard.forward(hidden_states=torch.zeros(1, 12, shard.hidden_size))


def test_later_shard_rejects_input_ids(model_factory, num_layers, input_ids):
    _, second = plan_even(num_layers, 2)
    shard = ShardRuntime(model_factory(), second)

    with pytest.raises(ValueError, match="does not own the embedding"):
        shard.forward(input_ids=input_ids)


def test_later_shard_requires_hidden_states(model_factory, num_layers):
    _, second = plan_even(num_layers, 2)
    shard = ShardRuntime(model_factory(), second)

    with pytest.raises(ValueError, match="needs hidden_states"):
        shard.forward()


def test_shard_only_holds_the_layers_it_owns(model_factory, num_layers):
    first, second = plan_even(num_layers, 2)

    head = ShardRuntime(model_factory(), first)
    tail = ShardRuntime(model_factory(), second)

    assert len(head.layers) == num_layers // 2
    assert head.embed_tokens is not None and head.lm_head is None
    assert tail.embed_tokens is None and tail.lm_head is not None


def test_plan_must_match_the_model_depth(model_factory, num_layers):
    from torrent_llm.shard import ShardSpec

    wrong = ShardSpec(index=0, start=0, end=99, num_layers=99)
    with pytest.raises(ValueError, match=f"the model has {num_layers}"):
        ShardRuntime(model_factory(), wrong)


def test_activation_on_the_wire_is_hidden_size_wide(model_factory, num_layers, input_ids):
    # The tensor whose byte cost the whole thesis is about.
    first, _ = plan_even(num_layers, 2)
    shard = ShardRuntime(model_factory(), first)

    hidden = shard.forward(input_ids=input_ids).hidden_states

    assert hidden.shape == (1, input_ids.shape[1], shard.hidden_size)
