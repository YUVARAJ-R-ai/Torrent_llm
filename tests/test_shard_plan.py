"""Shard plans must cover every layer exactly once — a silent gap would produce
plausible logits from a model that skipped layers."""

import pytest

from torrent_llm.shard import ShardSpec, plan_even, plan_explicit, validate_plan


def test_even_split_divides_cleanly():
    specs = plan_even(32, 2)

    assert [(s.start, s.end) for s in specs] == [(0, 16), (16, 32)]
    validate_plan(specs)


def test_remainder_layers_go_to_the_earliest_shards():
    specs = plan_even(33, 2)

    assert [(s.start, s.end) for s in specs] == [(0, 17), (17, 33)]
    validate_plan(specs)


def test_even_split_across_three_shards_covers_everything():
    specs = plan_even(10, 3)

    assert [(s.start, s.end) for s in specs] == [(0, 4), (4, 7), (7, 10)]
    validate_plan(specs)


def test_first_and_last_flags_identify_embedding_and_head_owners():
    first, middle, last = plan_even(9, 3)

    assert first.is_first and not first.is_last
    assert not middle.is_first and not middle.is_last
    assert last.is_last and not last.is_first


def test_single_shard_owns_both_ends():
    (only,) = plan_even(4, 1)

    assert only.is_first and only.is_last
    assert only.depth == 4


def test_cannot_have_more_shards_than_layers():
    with pytest.raises(ValueError, match="every shard must own at least one layer"):
        plan_even(3, 4)


def test_explicit_boundaries_are_interior_cuts():
    specs = plan_explicit(32, [8, 20])

    assert [(s.start, s.end) for s in specs] == [(0, 8), (8, 20), (20, 32)]
    validate_plan(specs)


def test_explicit_boundaries_must_be_inside_the_model():
    with pytest.raises(ValueError, match="strictly inside"):
        plan_explicit(32, [0, 20])
    with pytest.raises(ValueError, match="strictly inside"):
        plan_explicit(32, [8, 32])


def test_explicit_boundaries_must_increase():
    with pytest.raises(ValueError, match="strictly increasing"):
        plan_explicit(32, [20, 8])


def test_validate_rejects_a_gap():
    specs = [
        ShardSpec(index=0, start=0, end=8, num_layers=32),
        ShardSpec(index=1, start=9, end=32, num_layers=32),
    ]
    with pytest.raises(ValueError, match="gap between"):
        validate_plan(specs)


def test_validate_rejects_an_overlap():
    specs = [
        ShardSpec(index=0, start=0, end=10, num_layers=32),
        ShardSpec(index=1, start=8, end=32, num_layers=32),
    ]
    with pytest.raises(ValueError, match="overlap between"):
        validate_plan(specs)


def test_validate_rejects_a_chain_that_does_not_reach_the_last_layer():
    specs = [ShardSpec(index=0, start=0, end=30, num_layers=32)]
    with pytest.raises(ValueError, match="ends at layer 30"):
        validate_plan(specs)
