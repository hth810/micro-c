"""Positive minibatch RNG must exactly match the historical baseline stream."""
import numpy as np
import pytest
import torch

from microc.deep import GenomeMLP, seed_all
from scripts.task6_zero_residual_controlled_followup import RANKING_SEED_OFFSET, sample_batches


@pytest.mark.parametrize("seed", [20261006, 20261007, 20261008])
def test_all_initial_parameters_and_twenty_positive_draws_match_baseline(seed):
    seed_all(seed)
    baseline = GenomeMLP(13)
    expected_states = {k: v.clone() for k, v in baseline.state_dict().items()}
    expected = [torch.randint(101, (37,)) for _ in range(20)]
    seed_all(seed)
    corrected = GenomeMLP(13)
    for k, value in corrected.state_dict().items():
        assert torch.equal(value, expected_states[k])
    rng = torch.Generator().manual_seed(seed + RANKING_SEED_OFFSET)
    actual = [sample_batches(101, 79, 37, rng)[0] for _ in range(20)]
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


def test_ranking_draw_does_not_change_global_rng_and_is_reproducible():
    seed_all(17)
    expected = torch.randint(29, (29,))
    expected_after = torch.get_rng_state()
    seed_all(17)
    p, r = sample_batches(29, 31, 20000, torch.Generator().manual_seed(99))
    assert torch.equal(p, expected)
    assert torch.equal(torch.get_rng_state(), expected_after)
    seed_all(17)
    _, other = sample_batches(29, 31, 20000, torch.Generator().manual_seed(99))
    assert torch.equal(r, other)


@pytest.mark.parametrize("counts", [(0, 3, 4), (3, 0, 4), (3, 4, 0)])
def test_empty_sampling_budget_rejected(counts):
    with pytest.raises(ValueError, match="positive edge/ranking"):
        sample_batches(*counts, torch.Generator().manual_seed(2))
