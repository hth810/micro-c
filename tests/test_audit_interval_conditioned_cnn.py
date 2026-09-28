"""Small deterministic checks for Task 1 paired and overlap audit."""
from __future__ import annotations

import numpy as np
import pandas as pd

from task1.audit_interval_conditioned_cnn import (
    group_confusions, nearest_opposite, overlap_audit, paired_bootstrap, score_matrices,
)


def test_confusion_bootstrap_identical_models_have_zero_difference() -> None:
    truth = np.array([0, 1, 2, 0, 2, 1])
    pred = np.array([0, 1, 0, 0, 2, 2])
    groups = np.array([1, 1, 2, 3, 3, 4])
    _, cm = group_confusions(truth, pred, groups)
    values = score_matrices(cm.sum(axis=0))
    assert values["accuracy"] == 4 / 6
    assert values["chid_recall"] == .5
    paired = paired_bootstrap(truth, groups, {"a": pred, "b": pred.copy()},
                              "a", ("b",), 50, 7)
    assert all(value == 0 for value in paired["a_minus_b"]["point_difference"].values())
    assert all(value == [0, 0] for value in paired["a_minus_b"]["paired_spatial_bootstrap_95ci"].values())


def test_nearest_chid_chin_does_not_cross_chromosomes() -> None:
    ann = pd.DataFrame({"chrom": ["a", "a", "b", "b"],
                        "center": [10, 30, 11, 12],
                        "type": ["CHID", "CHIN", "CHIN", "CHID"]})
    d_chid, d_chin = nearest_opposite(ann)
    np.testing.assert_array_equal(d_chid[[0, 3]], [20, 1])
    np.testing.assert_array_equal(d_chin[[1, 2]], [20, 1])
    overlap = overlap_audit(ann, {"m": np.array([2, 2, 0, 0])})
    assert overlap["within_1000_bp"]["modes"]["m"]["chid_near_true_positive"] == 1
    assert overlap["within_1000_bp"]["modes"]["m"]["chin_near_false_chid"] == 1
