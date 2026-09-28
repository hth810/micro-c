"""Leakage-guard tests for the validation-selected Task 1 CNN."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from task1.evaluate_val_selected_cnn import read_fixed_split, score_structures


def fixture_paths(tmp_path):
    ann = pd.DataFrame({
        "id": [f"s{i}" for i in range(6)],
        "type": ["CHID", "CHIN", "CHID", "OPCID", "CHID", "CHIN"],
        "chrom": ["MG1655"] * 6,
        "center": np.arange(6) * 10000,
    })
    manifest = ann.copy()
    manifest["split"] = ["train", "train", "validation", "validation", "test", "test"]
    ann_path, manifest_path, windows_path = (tmp_path / name for name in
                                             ("annotations.csv", "manifest.csv", "windows.npz"))
    ann.to_csv(ann_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    labels = ann.type.map({"CHIN": 0, "OPCID": 1, "CHID": 2}).to_numpy()
    np.savez_compressed(windows_path,
                        X=np.zeros((12, 1, 128, 128), dtype=np.float32),
                        y=np.tile(labels, 2), group=np.tile(ann.id.to_numpy(dtype=str), 2),
                        is_train=np.tile(np.array([1, 1, 0, 0, 0, 0], bool), 2),
                        is_validation=np.tile(np.array([0, 0, 1, 1, 0, 0], bool), 2),
                        is_test=np.tile(np.array([0, 0, 0, 0, 1, 1], bool), 2))
    return ann_path, manifest_path, windows_path


def test_fixed_split_uses_both_replicates_and_saved_masks(tmp_path):
    ann, manifest, windows = fixture_paths(tmp_path)
    x, labels, saved, masks = read_fixed_split(windows, manifest, ann, window_bp=5120)
    assert x.shape == (12, 1, 128, 128)
    assert labels.tolist() == [2, 0, 2, 1, 2, 0]
    assert saved.loc[4, "split"] == "test"
    assert np.flatnonzero(masks["test"]).tolist() == [4, 5, 10, 11]


def test_fixed_split_rejects_cross_partition_overlap(tmp_path):
    ann, manifest, windows = fixture_paths(tmp_path)
    changed = pd.read_csv(manifest)
    changed.loc[2, "center"] = 13000
    changed.to_csv(manifest, index=False)
    original = pd.read_csv(ann)
    original.loc[2, "center"] = 13000
    original.to_csv(ann, index=False)
    with pytest.raises(ValueError, match="Overlapping input windows"):
        read_fixed_split(windows, manifest, ann, window_bp=5120)


def test_fixed_split_rejects_mismatched_cached_partitions(tmp_path):
    ann, manifest, windows = fixture_paths(tmp_path)
    changed = pd.read_csv(manifest)
    changed.loc[1, "split"] = "validation"
    changed.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="Cached split masks differ"):
        read_fixed_split(windows, manifest, ann, window_bp=5120)


def test_structure_score_averages_replicates_before_prediction():
    logits = np.array([[[0.0, 0.0, 1.0], [0.0, 2.0, 0.0]],
                       [[0.0, 0.0, 3.0], [0.0, 1.0, 0.0]]])
    score = score_structures(logits, np.array([2, 1]))
    assert score["accuracy"] == 1.0
    assert score["chid_recall"] == 1.0
