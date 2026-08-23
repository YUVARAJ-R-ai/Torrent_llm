"""Shard plans must cover every layer exactly once — a silent gap would produce
plausible logits from a model that skipped layers."""

import pytest

from torrent_llm.shard import ShardSpec, plan_even, plan_explicit, plan_weighted, validate_plan


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


# --- weighted split (issue #17) ---


def test_equal_weights_matches_the_even_split():
    weighted = plan_weighted(32, [1.0, 1.0])
    even = plan_even(32, 2)

    assert [(s.start, s.end) for s in weighted] == [(s.start, s.end) for s in even]


def test_weights_are_relative_not_absolute():
    small_numbers = plan_weighted(30, [1.0, 2.0])
    large_numbers = plan_weighted(30, [10.0, 20.0])

    assert [(s.start, s.end) for s in small_numbers] == [(s.start, s.end) for s in large_numbers]


def test_heavier_weight_gets_proportionally_more_layers():
    specs = plan_weighted(30, [1.0, 2.0])  # a 2x-as-capable second node

    a, b = specs
    assert a.depth == 10
    assert b.depth == 20
    validate_plan(specs)


def test_three_way_weighted_split_covers_everything():
    specs = plan_weighted(28, [1.0, 2.0, 1.0])

    assert sum(s.depth for s in specs) == 28
    validate_plan(specs)
    # The 2x node should end up with noticeably more than either 1x node.
    assert specs[1].depth > specs[0].depth
    assert specs[1].depth > specs[2].depth


def test_every_shard_gets_at_least_one_layer_even_with_a_very_skewed_weight():
    # weights [100, 1, 1] would floor to [4, 0, 0] under naive proportional
    # rounding on 5 layers -- exactly the case a mandatory minimum exists for.
    specs = plan_weighted(5, [100.0, 1.0, 1.0])

    assert all(s.depth >= 1 for s in specs)
    validate_plan(specs)
    # The dominant weight still gets the lion's share of what's left.
    assert specs[0].depth > specs[1].depth
    assert specs[0].depth > specs[2].depth


def test_weighted_split_always_sums_to_num_layers_even_with_awkward_ratios():
    # Ratios chosen to produce ugly fractional shares, so the largest-remainder
    # rounding actually gets exercised rather than landing on whole numbers.
    for num_layers, weights in [
        (17, [1.0, 1.0, 1.0]),
        (37, [3.0, 5.0, 7.0]),
        (100, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        (11, [0.3, 0.7]),
    ]:
        specs = plan_weighted(num_layers, weights)
        assert sum(s.depth for s in specs) == num_layers
        validate_plan(specs)


def test_weighted_split_is_deterministic():
    a = plan_weighted(50, [1.0, 3.0, 2.0])
    b = plan_weighted(50, [1.0, 3.0, 2.0])

    assert [(s.start, s.end) for s in a] == [(s.start, s.end) for s in b]


def test_single_weighted_shard_owns_the_whole_model():
    (only,) = plan_weighted(10, [1.0])

    assert only.is_first and only.is_last
    assert only.depth == 10


def test_more_weighted_shards_than_layers_is_rejected():
    with pytest.raises(ValueError, match="every shard must own at least one layer"):
        plan_weighted(3, [1.0, 1.0, 1.0, 1.0])


def test_nonpositive_weights_are_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        plan_weighted(10, [1.0, 0.0])
    with pytest.raises(ValueError, match="must be positive"):
        plan_weighted(10, [1.0, -2.0])


def test_empty_weights_is_rejected():
    with pytest.raises(ValueError, match="at least one weight"):
        plan_weighted(10, [])
