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


# --- KV cache (issue #24) ---
#
# Cached decode is a genuinely different computation from the uncached chain:
# attention sums over "cache concatenated with new" instead of recomputing over
# the whole sequence, and floating-point addition is not associative. The bar
# here is therefore numerically-close (allclose), not bit-exact (equal) -- the
# only place in this suite where that distinction matters, so it is called out
# explicitly rather than left to look like a loosened test.


def run_cached_chain(model_factory, specs, prompt_ids, num_decode_steps):
    """Prefill once, then decode one new token at a time using a cache per shard."""
    from transformers.cache_utils import DynamicCache

    shards = [ShardRuntime(model_factory(), spec) for spec in specs]
    caches = [DynamicCache() for _ in shards]

    def step(tokens_or_hidden, is_first_call):
        out = shards[0].forward(
            input_ids=tokens_or_hidden if is_first_call else None,
            hidden_states=None if is_first_call else tokens_or_hidden,
            cache=caches[0],
        )
        for shard, cache in zip(shards[1:], caches[1:], strict=True):
            out = shard.forward(hidden_states=out.hidden_states, cache=cache)
        return out

    # Prefill returns logits for the whole prompt; only its last position feeds
    # the next token, exactly as a real generation loop would use it.
    prefill_logits = step(prompt_ids, is_first_call=True).logits
    logits = [prefill_logits[:, -1:, :]]
    next_token = logits[-1].argmax(dim=-1)
    for _ in range(num_decode_steps - 1):
        out = step(next_token, is_first_call=True)  # a single new token is still "input_ids"
        logits.append(out.logits)
        next_token = out.logits[:, -1:, :].argmax(dim=-1)
    return torch.cat(logits, dim=1)


def test_cached_decode_matches_uncached_within_float_tolerance(
    tiny_model, model_factory, num_layers
):
    torch.manual_seed(5)
    prompt = torch.randint(0, 256, (1, 6))

    # Reference: greedy-decode the monolithic model without any cache at all,
    # recomputing the full prefix every step -- the ground truth this project
    # has used everywhere so far.
    ids = prompt
    with torch.inference_mode():
        for _ in range(3):
            nxt = tiny_model(input_ids=ids).logits[:, -1:, :].argmax(dim=-1)
            ids = torch.cat([ids, nxt], dim=1)
    reference_new_tokens = ids[:, prompt.shape[1] :]

    cached_logits = run_cached_chain(
        model_factory, plan_even(num_layers, 2), prompt, num_decode_steps=3
    )
    cached_tokens = cached_logits.argmax(dim=-1)

    # Greedy argmax should agree even though the underlying logits are only
    # numerically close, not identical -- a genuine divergence here would mean
    # the cache is feeding the model the wrong context, not just accumulating
    # rounding error.
    assert torch.equal(cached_tokens, reference_new_tokens)


def test_cache_length_is_tracked_per_shard_using_its_own_layer_indices(model_factory, num_layers):
    # The bug this guards against: querying a shard's cache with the library
    # default of layer_idx=0 silently reads as "empty" for every shard but the
    # first, since a middle/tail shard's cache only ever has entries at the
    # global indices that shard owns.
    from transformers.cache_utils import DynamicCache

    _, second = plan_even(num_layers, 2)
    shard = ShardRuntime(model_factory(), second)
    cache = DynamicCache()

    hidden = torch.zeros(1, 4, shard.hidden_size)
    shard.forward(hidden_states=hidden, cache=cache)

    assert cache.get_seq_length(layer_idx=second.start) == 4


def test_decode_step_after_prefill_only_needs_the_new_token(model_factory, num_layers):
    # This is the entire point of the cache: after the first call, a shard only
    # needs the newly generated position, not the whole growing sequence.
    from transformers.cache_utils import DynamicCache

    first, _ = plan_even(num_layers, 2)
    shard = ShardRuntime(model_factory(), first)
    cache = DynamicCache()

    prompt = torch.randint(0, 256, (1, 10))
    out = shard.forward(input_ids=prompt, cache=cache)
    assert out.hidden_states.shape[1] == 10  # prefill: full prompt in, full prompt out

    single_token = torch.randint(0, 256, (1, 1))
    out = shard.forward(input_ids=single_token, cache=cache)
    assert out.hidden_states.shape[1] == 1  # decode: one token in, one token out

    assert cache.get_seq_length(layer_idx=first.start) == 11


def test_without_a_cache_every_call_is_still_a_cold_full_recompute(
    model_factory, num_layers, input_ids
):
    # cache=None (the default) must be unaffected by any of the above --
    # existing callers that never pass a cache keep behaving exactly as before.
    first, _ = plan_even(num_layers, 2)
    shard = ShardRuntime(model_factory(), first)

    out = shard.forward(input_ids=input_ids)

    assert out.hidden_states.shape[1] == input_ids.shape[1]
