"""Scientific boundary checks for fixed residual Task 2 screening."""
from argparse import Namespace
import json

import numpy as np
import pytest

from scripts.task2_residual_motif_screen_20261006 import (
    enumerate_triads, independent_nearest, match_covariates,
    residual_vectors, sha256, validate,
)


def test_additive_row_column_artifact_has_zero_residual():
    gradient = np.sin(np.arange(64) / 7)
    additive = gradient[:, None] + gradient[None, :] + 3
    residual = residual_vectors(additive[None], "rowcol")
    assert np.max(np.abs(residual)) < 1e-6


def test_views_preserve_local_contact_signal_without_mixing_windows():
    maps = np.zeros((2, 64, 64), np.float32)
    maps[1, 10:15, 40:45] = 5
    maps[1, 40:45, 10:15] = 5
    for view in ("highpass", "rowcol"):
        vectors = residual_vectors(maps, view)
        assert np.allclose(vectors[0], 0)
        assert np.linalg.norm(vectors[1]) > 1
        assert np.allclose(vectors[1], residual_vectors(maps[1:2], view)[0])


def test_triad_requires_each_edge_in_both_views_and_independent_loci():
    centers = np.array([0, 20000, 40000])
    sim = np.full((3, 3), .8)
    views = {"highpass": sim.copy(), "rowcol": sim.copy()}
    cutoff = {"highpass": .7, "rowcol": .7}
    assert len(enumerate_triads(views, np.ones(3, bool), centers, cutoff)) == 1
    views["rowcol"][1, 2] = views["rowcol"][2, 1] = .69
    assert enumerate_triads(views, np.ones(3, bool), centers, cutoff).empty
    views["rowcol"][:] = .8
    assert enumerate_triads(views, np.ones(3, bool), np.array([0, 5000, 40000]), cutoff).empty


def test_known_novelty_reference_excludes_neighboring_annotations():
    points = np.array([[0.], [.01], [1.], [4.]])
    values = independent_nearest(points, np.array([0, 5000, 20000, 60000]))
    assert np.allclose(values, [1, .99, .99, 3])


def test_matching_checks_target_support_not_only_count():
    # Enough controls lie within the generous caliper but the target is outside
    # their empirical support; those controls cannot establish a matched claim.
    pool = np.column_stack([np.linspace(0, 1, 100)] * 7)
    target = np.full(7, 1.5)
    selected, diagnostic = match_covariates(pool, target, np.ones(100, bool))
    assert len(selected) > 20
    assert diagnostic["balanced_all_covariates"] is False
    assert not any(diagnostic["supported_covariates"])


def test_freeze_tampering_is_detected_before_rep2_access(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}", encoding="utf-8")
    receipt = {"frozen_file_sha256": {"protocol.json": sha256(protocol)}}
    (tmp_path / "freeze_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    protocol.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="Freeze hash mismatch"):
        validate(Namespace(discovery=tmp_path, rep2=tmp_path / "does-not-exist.cool", out=tmp_path))
