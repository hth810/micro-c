import numpy as np
import pytest

from scripts.task2_course_verification_20261014 import (
    match_controls, robust_covariate_scales, score_free_independent_indices,
    shape_features, thin_symmetric_counts,
)


def test_score_free_thinning_is_coordinate_only_and_spacing_safe():
    centers = np.array([32000, 0, 6000, 12000, 24000])
    ids = score_free_independent_indices(centers, np.array([True] * 5))
    assert centers[ids].tolist() == [0, 12000, 24000]
    assert np.all(np.diff(centers[ids]) >= 10240)
    with pytest.raises(ValueError):
        score_free_independent_indices([0, 0], [True, True])


def test_robust_scale_handles_zero_iqr_without_silently_granting_support():
    pool = np.zeros((100, 7))
    pool[-8:, 2] = np.linspace(.1, .4, 8)
    scale, iqr, _, _ = robust_covariate_scales(pool)
    assert iqr[2] == 0
    assert scale[2] >= 1 / 512
    centers = np.arange(100) * 20000
    target = np.zeros(7)
    target[2] = .3
    _, audit = match_controls(pool, target, centers, [-100000])
    assert not audit["evaluable"]
    assert audit["within_factor_caliper_counts"][2] < 20


def test_matching_keeps_all_factors_and_does_not_force_far_target():
    pool = np.column_stack([np.linspace(-1, 1, 120)] * 7)
    centers = np.arange(120) * 20000
    ids, audit = match_controls(pool, np.zeros(7), centers, [-100000])
    assert len(ids) >= 20 and audit["evaluable"]
    target = np.zeros(7); target[6] = 50
    ids, audit = match_controls(pool, target, centers, [-100000])
    assert not len(ids) and not audit["evaluable"]
    assert audit["within_factor_caliper_counts"][6] == 0


def test_shapes_are_finite_explainable_and_contact_position_changes_centroid():
    maps = np.zeros((2, 64, 64))
    maps[0, 10, 35] = maps[0, 35, 10] = 4
    maps[1, 20, 45] = maps[1, 45, 20] = 4
    features = shape_features(maps)
    assert features.shape == (2, 10)
    assert np.isfinite(features).all()
    assert features[1, 5] > features[0, 5]
    assert np.isfinite(shape_features(np.zeros((1, 64, 64)))).all()


def test_thinning_mirrors_once_and_never_inflates_or_interpolates_counts():
    counts = np.ones((64, 64)) * 10
    np.fill_diagonal(counts, 0)
    a, p, supported = thin_symmetric_counts(counts, 3000, 7)
    b, _, _ = thin_symmetric_counts(counts, 3000, 7)
    assert supported and 0 < p < 1
    assert np.array_equal(a, b) and np.array_equal(a, a.T)
    assert (a <= counts).all() and (np.diag(a) == 0).all()
    with pytest.raises(ValueError, match="integer"):
        thin_symmetric_counts(counts + .2, 3000, 7)
    low, p, supported = thin_symmetric_counts(np.ones((64, 64)), 100000, 7)
    assert not supported and p == 1


def test_thinning_invalid_count_inputs_rejected():
    with pytest.raises(ValueError):
        thin_symmetric_counts(np.eye(4), 100, 7)
    with pytest.raises(ValueError):
        thin_symmetric_counts(np.ones((64, 64)) * -1, 100, 7)
    with pytest.raises(ValueError):
        thin_symmetric_counts(np.ones((64, 64)), 0, 7)
