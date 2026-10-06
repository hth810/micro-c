"""Train-only preprocessing and fixed inner choice safeguards."""
import numpy as np
import pandas as pd
import pytest

from task1.evaluate_nested_neural_20261006 import EPOCHS, POWERS, choose_plan, fit_standardizer
from task1.evaluate_roi_cnn_20261006 import make_images, reverse_boundary_features
from task1.audit_roi_cnn_20261006 import matched_contact_masks


def test_scaler_does_not_use_held_out_extreme_values():
    x = np.array([[1., 3.], [3., 3.], [1e6, -1e6]], dtype=np.float32)
    mean, scale = fit_standardizer(x, np.array([0, 1]))
    np.testing.assert_allclose(mean, [2., 3.])
    np.testing.assert_allclose(scale, [1., 1.])


def test_inner_selection_prefers_shorter_milder_tied_plan():
    truth = np.array([0, 1, 2, 0, 1, 2])
    p = np.tile(np.eye(3)[truth][None], (2, 1, 1))
    family = {(power, epoch): p for power in POWERS for epoch in EPOCHS}
    power, epoch, rows = choose_plan(family, truth)
    assert (power, epoch) == (0.5, 80)
    assert len(rows) == 6


def test_inner_selection_rejects_partial_oof_predictions():
    family = {(power, epoch): np.ones((2, 3, 3)) / 3 for power in POWERS for epoch in EPOCHS}
    family[0.5, 80][0, 1, 2] = np.nan
    with pytest.raises(ValueError, match="Incomplete"):
        choose_plan(family, np.array([0, 1, 2]))


def test_genomic_axis_reversal_transforms_directional_boundary_features():
    side = np.arange(15, dtype=np.float32)[None]
    reverse = reverse_boundary_features(side)
    np.testing.assert_array_equal(reverse_boundary_features(reverse), side)
    assert reverse[0, 2] == side[0, 12]
    assert reverse[0, 4] == side[0, 10]
    assert reverse[0, 14] == side[0, 14]


def test_inside_outside_occlusion_matches_global_diagonal_distribution():
    inside, outside, offsets = matched_contact_masks(6000, 12000, 9000, 160,
                                                    np.random.default_rng(3))
    ai, bi = np.where(np.triu(inside))
    ao, bo = np.where(np.triu(outside))
    np.testing.assert_array_equal(np.sort(bi - ai), np.sort(bo - ao))
    assert len(offsets) == len(ai) > 0
    assert np.array_equal(inside, inside.T)
    assert np.array_equal(outside, outside.T)
    assert not np.any(inside & outside)
    assert np.all(bi - ai >= 2)


def test_roi_channels_preserve_replicate_order_and_ignore_class_labels():
    ann = pd.DataFrame({"id": ["A", "B"], "start": [2000, 1000], "end": [4000, 5000],
                        "center": [3000, 3000], "type": ["CHIN", "CHID"]})
    x = np.ones((4, 2, 128, 128), dtype=np.float32) * np.arange(1, 5, dtype=np.float32)[:, None, None, None]
    images = make_images(x, ann)
    assert images["interval_roi64"].shape == (4, 5, 64, 64)
    np.testing.assert_allclose(images["interval_roi64"][:, 0, 0, 0], [1, 2, 3, 4])
    np.testing.assert_allclose(images["interval_roi64"][:, 4, 0, 0], [1, 2, 3, 4])
    swapped = ann.copy()
    swapped["type"] = ["CHID", "CHIN"]
    np.testing.assert_array_equal(make_images(x, swapped)["interval_roi64"], images["interval_roi64"])


def test_side_scaler_uses_both_training_repeats_and_excludes_both_test_repeats():
    x = np.array([[2., 4.], [1e6, -1e6], [6., 4.], [-1e6, 1e6]], dtype=np.float32)
    # Two structures: indices 0/2 are both repeats of the training structure.
    mean, scale = fit_standardizer(x, np.array([0, 2]))
    np.testing.assert_allclose(mean, [4., 4.])
    np.testing.assert_allclose(scale, [2., 1.])
