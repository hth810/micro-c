import numpy as np

from scripts.task2_resampling_comparison_20261006 import (
    block_bootstrap_difference, paired_correlation,
)


def test_paired_correlation_is_scale_invariant_and_zero_signal_is_not_agreement():
    a = np.array([[1., 2., 4.], [0., 0., 0.]])
    b = np.array([[7., 9., 13.], [0., 0., 0.]])
    assert np.allclose(paired_correlation(a, b), [1, 0])


def test_genomic_block_bootstrap_is_reproducible_and_keeps_group_count():
    diff = np.array([.2, .2, .4, .4, .4, .3])
    groups = np.array([1, 1, 2, 2, 2, 3])
    one = block_bootstrap_difference(diff, groups, seed=73)
    two = block_bootstrap_difference(diff, groups, seed=73)
    assert one == two
    assert one["genomic_blocks"] == 3
    lo, hi = one["mean_difference_ci95"]
    assert .2 - 1e-12 <= lo <= hi <= .4 + 1e-12


def test_single_genomic_block_does_not_claim_independent_interval():
    result = block_bootstrap_difference(np.array([.2, .3]), np.array([1, 1]))
    assert result["mean_difference_ci95"] is None
