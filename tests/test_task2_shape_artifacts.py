import numpy as np

from scripts.audit_task2_shape_artifacts import (
    match_controls, masked_vectors, pair_pool,
)


def test_far_masks_have_fixed_size_and_ignore_near_diagonal_changes():
    maps = np.zeros((2, 64, 64), dtype=np.float32)
    maps[1] = np.eye(64, dtype=np.float32) * 100
    assert masked_vectors(maps, "far8").shape == (2, len(np.triu_indices(64, k=8)[0]))
    assert masked_vectors(maps, "far16").shape[1] < masked_vectors(maps, "far8").shape[1]
    assert np.allclose(masked_vectors(maps, "far8")[0], masked_vectors(maps, "far8")[1])


def test_pair_pool_excludes_frozen_and_nonindependent_positions():
    centers = np.array([0, 5000, 20000, 40000, 60000])
    a, b = pair_pool(np.arange(5), centers, np.array([40000]))
    assert all(3 not in (i, j) for i, j in zip(a, b))
    assert all(abs(centers[i] - centers[j]) >= 10240 for i, j in zip(a, b))


def test_coverage_matching_keeps_only_broad_caliper_controls():
    centers = np.array([0, 100000, 200000, 300000, 400000, 500000])
    coverage = np.array([.3, .4, .3, .4, .35, 20.0])
    a = np.array([2, 2, 3, 4]); b = np.array([3, 4, 4, 5])
    picked_a, picked_b, diagnostic = match_controls(a, b, 0, 1, centers, coverage)
    assert diagnostic["pool_pairs"] == 4
    assert len(picked_a) == len(picked_b)
    assert all(5 != index for index in np.r_[picked_a, picked_b])
