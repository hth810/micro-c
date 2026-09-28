"""Controls for the held-out Task 1 interval-CNN perturbation audit."""
from __future__ import annotations

import numpy as np

from task1.audit_interval_cnn_occlusion import margins, matched_long_contact_occlusions


def test_matched_contact_occlusion_preserves_masks_and_equal_offsets() -> None:
    sample = np.ones((4, 16, 16), dtype=np.float32)
    sample[2] = 0
    sample[2, 4:12, 4:12] = 1
    within, outside, count = matched_long_contact_occlusions(sample, np.random.default_rng(17))
    assert count > 0
    np.testing.assert_array_equal(within[2:], sample[2:])
    np.testing.assert_array_equal(outside[2:], sample[2:])
    np.testing.assert_array_equal(within[0], sample[0])
    np.testing.assert_array_equal(outside[0], sample[0])
    for offset in range(2, 16):
        assert np.count_nonzero(np.diagonal(within[1], offset) == 0) == (
            np.count_nonzero(np.diagonal(outside[1], offset) == 0))


def test_true_class_margin() -> None:
    logits = np.array([[2., 1., 0.], [0., 1., 3.]], dtype=np.float32)
    np.testing.assert_array_equal(margins(logits, np.array([0, 2])), [1., 2.])
