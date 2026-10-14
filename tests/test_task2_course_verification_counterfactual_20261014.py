import numpy as np
import pytest

from scripts.task2_course_qc_counterfactual_20261014 import apply_fixed_coverage_mask, zero_row_mask


def test_zero_rows_exclude_diagonal_only_coverage():
    matrix = np.eye(512) * 100
    matrix[3, 20] = matrix[20, 3] = 9
    keep = zero_row_mask(matrix)
    assert keep.sum() == 2 and keep[3] and keep[20]
    assert not keep[4]


def test_fixed_mask_preserves_symmetry_and_never_adds_counts():
    matrix = np.ones((512, 512))
    keep = np.ones(512, bool); keep[100:130] = False
    masked = apply_fixed_coverage_mask(matrix, keep)
    assert np.array_equal(masked, masked.T)
    assert (masked[100:130] == 0).all()
    assert (masked[:, 100:130] == 0).all()
    assert (masked <= matrix).all()
    assert np.array_equal(apply_fixed_coverage_mask(matrix, np.ones(512, bool)), matrix)


def test_mask_input_validation_rejects_unaligned_shape_and_negative_counts():
    with pytest.raises(ValueError):
        zero_row_mask(np.ones((64, 64)))
    with pytest.raises(ValueError):
        zero_row_mask(-np.eye(512))
    with pytest.raises(ValueError):
        apply_fixed_coverage_mask(np.eye(512), np.ones(511, bool))
