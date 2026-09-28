import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from scripts.task5_next_peak_ranker import (
    FEATURE_NAMES, FROZEN_VALIDATION_FIELDS, OPTIONAL_FROZEN_VALIDATION_FIELDS,
    choose_model, overlap_audit, proposals, require_frozen_selection,
)


def test_proposals_are_unique_finite_and_image_only():
    base = np.zeros((32, 32), dtype=np.float32)
    base[6, 17] = base[17, 6] = 6
    main = base.copy()
    main[9, 22] = main[22, 9] = 7
    peak = base.copy()
    peak[12, 25] = peak[25, 12] = 8
    coords, features = proposals(base, main, peak)
    assert len(coords) == len(np.unique(coords, axis=0))
    assert features.shape == (len(coords), len(FEATURE_NAMES))
    assert np.isfinite(features).all()
    assert np.all(coords[:, 1] - coords[:, 0] >= 3)


def test_validation_gate_requires_chin_and_tile_noninferiority():
    columns = {}
    for method, known, hit in (
            ("bilinear", [10., 10.], [0., 1.]),
            ("main", [11., 4.], [1., 1.]),
            ("peak", [9., 9.], [1., 1.]),
            ("logistic", [8., 8.], [1., 1.]),
            ("boosted", [11., 6.], [0., 1.])):
        columns[f"{method}_top1_error"] = known
        columns[f"{method}_top5_hit5"] = hit
    known = pd.DataFrame({"type": ["CHIN", "OPCID"], **columns})
    tiles = pd.DataFrame({f"{name}_top1_error": [value] for name, value in (
        ("bilinear", 10.), ("main", 8.), ("peak", 11.),
        ("logistic", 9.), ("boosted", 9.))})
    assert choose_model(tiles, known)["selected"]["name"] == "logistic"
    known["logistic_top1_error"] = [12., 8.]
    assert choose_model(tiles, known)["selected"]["name"] == "bilinear"


def test_genomic_window_overlap_uses_patch_width():
    result = overlap_audit(np.array([0, 1000]), np.array([50, 1640]), 640)
    assert result["overlap_pairs"] == 1
    assert result["minimum_center_distance_bp"] == 50


def frozen_summary():
    summary = {field: {"value": field} for field in (
        FROZEN_VALIDATION_FIELDS + OPTIONAL_FROZEN_VALIDATION_FIELDS)}
    summary["selection"] = {"grid": [{"name": "logistic", "feasible": True}],
                            "selected": {"name": "logistic"}}
    summary["selected_detector"] = "logistic"
    return summary


def test_full_phase_requires_matching_frozen_validation_artifact(tmp_path):
    artifact = tmp_path / "validation_summary.json"
    current = frozen_summary()
    with pytest.raises(FileNotFoundError, match="validation-only"):
        require_frozen_selection(current, artifact)
    artifact.write_text(json.dumps(current), encoding="utf-8")
    selected, digest = require_frozen_selection(current, artifact)
    assert selected == "logistic"
    assert digest == hashlib.sha256(artifact.read_bytes()).hexdigest()


@pytest.mark.parametrize("changed_field", [
    "selection", "main_checkpoint_sha256", "peak_checkpoint_sha256",
    "genomic_split", "metric_protocol", "train",
])
def test_full_phase_rejects_changed_validation_or_checkpoint(tmp_path, changed_field):
    artifact = tmp_path / "validation_summary.json"
    artifact.write_text(json.dumps(frozen_summary()), encoding="utf-8")
    current = frozen_summary()
    current[changed_field] = {"changed": True}
    with pytest.raises(ValueError, match=changed_field):
        require_frozen_selection(current, artifact)


def test_full_phase_rejects_malformed_frozen_detector(tmp_path):
    artifact = tmp_path / "validation_summary.json"
    current = frozen_summary()
    current["selection"]["selected"]["name"] = "boosted"
    artifact.write_text(json.dumps(current), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        require_frozen_selection(current, artifact)


def test_existing_core_snapshot_is_accepted_but_new_optional_fields_are_locked(tmp_path):
    artifact = tmp_path / "validation_summary.json"
    current = frozen_summary()
    older = {field: current[field] for field in FROZEN_VALIDATION_FIELDS}
    artifact.write_text(json.dumps(older), encoding="utf-8")
    assert require_frozen_selection(current, artifact)[0] == "logistic"
    artifact.write_text(json.dumps(current), encoding="utf-8")
    current["genomic_split"] = {"changed": True}
    with pytest.raises(ValueError, match="genomic_split"):
        require_frozen_selection(current, artifact)
