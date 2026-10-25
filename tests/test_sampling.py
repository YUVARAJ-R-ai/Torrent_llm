"""Choosing the next token (issue #33).

The rule runs on the shard that owns the LM head, so whatever it picks has to
match what the runner used to pick from the same logits on its own side.
"""

import pytest
import torch

from torrent_llm.sampling import GREEDY, Sampling, select_tokens


@pytest.fixture
def logits():
    torch.manual_seed(5)
    return torch.randn(2, 7, 50)


def test_greedy_is_the_argmax_of_the_last_position(logits):
    chosen = select_tokens(logits, GREEDY)

    assert chosen.shape == (2, 1)
    assert chosen.dtype == torch.long
    assert torch.equal(chosen, logits[:, -1, :].argmax(dim=-1, keepdim=True))


def test_the_same_seed_draws_the_same_tokens(logits):
    rule = Sampling(temperature=1.0, seed=11)

    assert torch.equal(select_tokens(logits, rule), select_tokens(logits, rule))


def test_top_k_of_one_is_greedy_whatever_the_temperature(logits):
    rule = Sampling(temperature=5.0, top_k=1, seed=3)

    assert torch.equal(select_tokens(logits, rule), select_tokens(logits, GREEDY))


def test_top_k_never_picks_outside_the_top_k(logits):
    allowed = torch.topk(logits[:, -1, :], 3, dim=-1).indices
    for seed in range(20):
        chosen = select_tokens(logits, Sampling(temperature=2.0, top_k=3, seed=seed))
        assert all(chosen[b, 0] in allowed[b] for b in range(2))


def test_each_step_gets_its_own_seed():
    assert Sampling(seed=7).for_step(3).seed == 10
    assert GREEDY.for_step(4).is_greedy


@pytest.mark.parametrize("kwargs", [{"temperature": -0.1}, {"top_k": -1}])
def test_nonsense_rules_are_refused(kwargs):
    with pytest.raises(ValueError):
        Sampling(**kwargs)
