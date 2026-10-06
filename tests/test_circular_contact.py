import numpy as np
import pytest

from microc.circular import circular_contact_target, circular_matched_auc, circular_pairs, ring_offsets, cyclic_chain_variance


def test_ring_target_rotates_with_chromosome_origin():
    rng = np.random.default_rng(2)
    raw = rng.uniform(0, 20, (9, 9))
    raw = (raw + raw.T) / 2
    target = circular_contact_target(raw)
    rolled = np.roll(np.roll(raw, 3, axis=0), 3, axis=1)
    assert np.allclose(circular_contact_target(rolled), np.roll(np.roll(target, 3, axis=0), 3, axis=1))
    assert np.allclose(target, target.T) and np.isfinite(target).all()


def test_uniform_ring_decay_has_no_residual_and_wrap_shares_baseline():
    offsets = ring_offsets(8)
    raw = 100 / (1 + offsets)
    assert not circular_contact_target(raw).any()
    raw[0, 7] = raw[7, 0] = 100
    target = circular_contact_target(raw)
    assert target[0, 7] > 0
    assert offsets[0, 7] == offsets[0, 1] == 1


def test_unordered_pairs_exclude_both_linear_and_wrap_neighbors():
    i, j, offsets = circular_pairs(8)
    assert len(i) == 8 * 7 // 2 - 8
    assert (offsets >= 2).all()
    assert (0, 7) not in set(zip(i, j))
    assert sum(offsets == 4) == 4


def test_exact_ring_offset_matching_balances_each_stratum():
    rng = np.random.default_rng(4)
    raw = rng.uniform(0, 20, (12, 12))
    raw = raw + raw.T
    target = circular_contact_target(raw)
    offsets = ring_offsets(12).astype(float)
    result = circular_matched_auc(target, {"separation": offsets})
    assert result["matched_pairs_per_class"] > 0
    assert all(row["positive"] == row["zero"] for row in result["offsets"])
    assert result["auc"]["exact_ring_offset_baseline"] == .5
    assert result["auc"]["separation"] == .5


def test_empty_signal_and_invalid_contacts_are_explicit():
    assert circular_matched_auc(np.zeros((8, 8)), {})["matched_pairs_per_class"] == 0
    for invalid in (np.ones((3, 3)), np.ones((5, 6)), np.full((8, 8), np.nan), -np.ones((8, 8))):
        with pytest.raises(ValueError):
            circular_contact_target(invalid)


def test_cyclic_chain_includes_wrap_and_is_origin_invariant():
    import torch
    xyz = torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [2., 2., 0.]])
    lengths = torch.tensor([1., 1., 2., 8 ** .5])
    expected = ((lengths - lengths.mean()) ** 2).mean()
    assert torch.allclose(cyclic_chain_variance(xyz), expected)
    assert torch.allclose(cyclic_chain_variance(torch.roll(xyz, 2, 0)), expected)
