"""Focused checks for held-out Task 1 attribution reporting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from task1.saliency_audit import (
    _figure_selection,
    distance_matched_enrichment,
    interval_bin_mask,
    run_saliency_audit,
)


def test_interval_bin_mask_uses_annotation_coordinates():
    mask = interval_bin_mask(center_bp=400, start_bp=360, end_bp=440,
                             n_bins=8, bin_bp=20)
    assert np.flatnonzero(mask).tolist() == [2, 3, 4, 5]
    # The raw extraction center need not align to a pooled-bin boundary.
    raw_aligned = interval_bin_mask(center_bp=435, start_bp=360, end_bp=440,
                                    n_bins=8, bin_bp=40, raw_bin_bp=10,
                                    genome_bins=1000)
    assert np.flatnonzero(raw_aligned).tolist() == [2, 3]


def test_same_distance_background_enrichment_and_full_window_limitation():
    bins = np.zeros(16, dtype=bool)
    bins[5:10] = True
    square = np.outer(bins, bins)
    attribution = np.where(square, 4.0, 1.0)
    metrics = distance_matched_enrichment(attribution, np.ones((16, 16)), bins)
    assert metrics["evaluable"]
    assert np.isclose(metrics["attribution_enrichment"], 4.0)
    assert np.isclose(metrics["positive_input_enrichment"], 1.0)
    full = distance_matched_enrichment(attribution, np.ones((16, 16)),
                                       np.ones(16, dtype=bool))
    assert not full["evaluable"]
    assert np.isnan(full["attribution_enrichment"])


def test_misclassified_class_is_not_presented_as_correct(tmp_path):
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, 3))
    with torch.no_grad():
        model[1].weight.fill_(0.01)
        model[1].bias.copy_(torch.tensor([3.0, 0.0, -3.0]))
    x = torch.ones((2, 1, 8, 8), dtype=torch.float32)
    y = torch.tensor([2, 2])
    meta = pd.DataFrame([
        {"id": "CHID_A", "type": "CHID", "center": 400, "start": 360,
         "end": 440, "replicate": rep} for rep in (1, 2)
    ])
    summary = run_saliency_audit(model=model, x_test=x, y_test=y,
                                 sample_metadata=meta, out=tmp_path,
                                 device=torch.device("cpu"), pooled_bin_bp=20)
    assert summary["figures"]["CHID"]["selection"] == "MISCLASSIFIED fallback"
    assert summary["classes"]["CHID"]["correct_only"]["heldout_samples"] == 0
    assert (tmp_path / "saliency_CHID.png").is_file()
    rows = pd.read_csv(tmp_path / "saliency_audit_samples.csv")
    assert rows.predicted_type.tolist() == ["CHIN", "CHIN"]
    assert not rows.correct.any()
    chosen, label = _figure_selection(np.array([False, False]))
    assert chosen.tolist() == [0, 1] and label == "MISCLASSIFIED fallback"
