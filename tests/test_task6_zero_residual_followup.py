"""Exact-offset matching and explicitly limited zero-residual ranking loss."""
import numpy as np
import pytest
import torch

from microc.circular import circular_pairs, ring_offsets
from scripts.task6_zero_residual_followup import matched_ranking_pairs, ranking_loss


def synthetic_target(n=12):
    i, j, offset = circular_pairs(n)
    target = np.zeros((n, n))
    # Half of each supported offset positive, half clipped zero.
    for d in np.unique(offset):
        members = np.flatnonzero(offset == d)
        pos = members[::2]
        target[i[pos], j[pos]] = target[j[pos], i[pos]] = .7
    return target


def test_matched_pairs_are_exact_ring_offset_no_neighbors_and_correct_labels():
    target = synthetic_target()
    (pi, pj, zi, zj, shared), receipt = matched_ranking_pairs(target)
    offset = ring_offsets(len(target))
    np.testing.assert_array_equal(offset[pi, pj], shared)
    np.testing.assert_array_equal(offset[zi, zj], shared)
    assert np.all(shared >= 2)
    assert np.all(pi < pj) and np.all(zi < zj)
    assert np.all(target[pi, pj] > 0) and np.all(target[zi, zj] == 0)
    assert len(set(zip(pi, pj))) == len(pi)
    assert len(set(zip(zi, zj))) == len(zi)
    assert sum(row["matched_count"] for row in receipt) == len(pi)


def test_fixed_seed_pairing_is_reproducible_and_unsupported_offsets_omitted():
    target = synthetic_target()
    offsets = ring_offsets(len(target))
    target[offsets == 3] = 1.
    first, _ = matched_ranking_pairs(target, 19)
    second, _ = matched_ranking_pairs(target, 19)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    assert not np.any(first[-1] == 3)


@pytest.mark.parametrize("fill", [0., 1.])
def test_no_supported_positive_zero_offset_fails_explicitly(fill):
    with pytest.raises(ValueError, match="No ring offset"):
        matched_ranking_pairs(np.full((8, 8), fill))


@pytest.mark.parametrize("target", [np.zeros((5, 6)), np.full((8, 8), -1.),
                                   np.eye(8) + np.triu(np.ones((8, 8)), 1),
                                   np.full((8, 8), np.nan)])
def test_invalid_targets_rejected(target):
    with pytest.raises(ValueError, match="finite nonnegative symmetric square"):
        matched_ranking_pairs(target)


def test_ranking_loss_is_finite_differentiable_and_prefers_more_distant_zero():
    close = torch.tensor([[0., 0., 0.], [1., 0., 0.], [1.5, 0., 0.]], requires_grad=True)
    farther = close.detach().clone()
    farther[2, 0] = 3.
    args = ([0], [1], [0], [2])
    loss = ranking_loss(close, *args)
    assert torch.isfinite(loss)
    assert ranking_loss(farther, *args) < loss
    loss.backward()
    assert torch.isfinite(close.grad).all()


def test_antipodal_pairs_appear_once_with_equal_labels_for_even_ring():
    target = synthetic_target(10)
    pairs, receipt = matched_ranking_pairs(target)
    antipodal = pairs[-1] == 5
    assert int(antipodal.sum()) == 2
    row = next(r for r in receipt if r["exact_ring_offset"] == 5)
    assert row == {"exact_ring_offset": 5, "positive_available": 3,
                   "zero_available": 2, "matched_count": 2}
