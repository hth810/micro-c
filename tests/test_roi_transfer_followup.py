"""Source-only fitting and paired spatial-fold safeguards for replicate transfer."""
import numpy as np
import pandas as pd
import pytest
import torch

import task1.evaluate_roi_transfer_followup as transfer
from task1.evaluate_nested_neural_20261006 import fit_standardizer


def test_source_indices_exclude_other_rep_and_both_heldout_counterparts():
    groups = np.array([0, 0, 1, 2])
    for rep in (0, 1):
        fit, held = transfer.transfer_indices(np.array([0, 1]), np.array([2, 3]), 4, rep, groups)
        np.testing.assert_array_equal(fit, np.array([0, 1]) + 4 * rep)
        np.testing.assert_array_equal(held, [2, 3, 6, 7])
        assert not np.intersect1d(fit % 4, held % 4).size
        assert np.all(fit // 4 == rep)


def test_transfer_rejects_shared_spatial_group_and_structure():
    with pytest.raises(ValueError, match="spatial group leakage"):
        transfer.transfer_indices(np.array([0]), np.array([1]), 3, 0, np.array([7, 7, 8]))
    with pytest.raises(ValueError, match="structure leakage"):
        transfer.transfer_indices(np.array([0, 1]), np.array([1, 2]), 3, 0, np.arange(3))


@pytest.mark.parametrize("train,test,rep", [([0, 0], [1], 0), ([0], [3], 0), ([0], [1], 2)])
def test_transfer_rejects_invalid_indices_and_replicate(train, test, rep):
    with pytest.raises(ValueError):
        transfer.transfer_indices(np.array(train), np.array(test), 3, rep, np.arange(3))


def test_source_scaler_ignores_target_training_and_both_heldout_replicates():
    # Structures 0/1 train, 2 held out. Only source-rep1 indices 0/1 may fit.
    side = np.array([[2., 3.], [6., 3.], [1e6, -1e6],
                     [-1e7, 1e7], [1e7, -1e7], [-1e8, 1e8]], dtype=np.float32)
    fit, _ = transfer.transfer_indices(np.array([0, 1]), np.array([2]), 3, 0, np.arange(3))
    mean, scale = fit_standardizer(side, fit)
    np.testing.assert_allclose(mean, [4., 3.])
    np.testing.assert_allclose(scale, [2., 1.])


def test_fit_is_invariant_to_target_replicate_training_features(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(transfer, "EPOCHS", 1)
    monkeypatch.setattr(transfer, "INITIALIZATION_OFFSETS", (0,))
    rng = np.random.default_rng(7)
    image = rng.random((8, 5, 64, 64), dtype=np.float32)
    side = rng.normal(size=(8, 15)).astype(np.float32)
    y = np.array([0, 1, 2, 0], dtype=np.int64)
    train, test, groups = np.array([0, 1, 2]), np.array([3]), np.arange(4)
    first, receipt = transfer.fit_transfer(image, side, y, train, test, 0, groups, 11)
    other_image, other_side = image.copy(), side.copy()
    other_image[4:7] *= 1e6
    other_side[4:7] *= -1e6
    second, altered = transfer.fit_transfer(other_image, other_side, y, train, test, 0, groups, 11)
    np.testing.assert_array_equal(first, second)
    assert receipt["side_mean"] == altered["side_mean"]
    assert receipt["class_weights"] == altered["class_weights"]
    assert receipt["training_sample_indices"] == [0, 1, 2]


def test_published_comparator_rejects_different_outer_fold(tmp_path):
    ann = pd.DataFrame({"id": ["a", "b"], "type": ["CHIN", "CHID"]})
    saved = pd.DataFrame({"id": ["a", "b"], "type": ["CHIN", "CHID"],
                          "spatial_group": [0, 1], "outer_fold": [0, 0]})
    path = tmp_path / "published.csv"
    saved.to_csv(path, index=False)
    with pytest.raises(ValueError, match="outer folds differ"):
        transfer.load_published(path, ann, np.array([0, 1]), np.array([0, 1]))
