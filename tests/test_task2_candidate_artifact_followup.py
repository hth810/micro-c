"""Native-contact artifact diagnostics must not manufacture replication."""
import numpy as np
import pandas as pd
import pytest

from microc.data import normalize
from scripts.task2_candidate_artifact_followup import (
    COVARIATES, descriptive_percentile, native_qc, normalization_variants,
    paired_r, qc_covariates, select_controls,
)
from scripts.task2_residual_motif_screen_20261006 import project_novelty


def test_native_qc_counts_pairs_once_and_does_not_rescue_zero_rows_with_diagonal():
    a = np.zeros((100, 100))
    np.fill_diagonal(a, 7)
    a[4, 80] = a[80, 4] = 9
    q = native_qc(a)
    assert q["native_symmetric_total"] == 718
    assert q["native_upper_offdiag_total"] == 9
    assert q["native_zero_offdiag_rows"] == 98
    assert q["native_top_1pct_mass_fraction"] == 1
    assert q["native_upper_nonzero_fraction"] == 1 / 4950


def test_zero_native_counts_are_finite_and_are_not_given_fake_concentration():
    q = native_qc(np.zeros((64, 64)))
    assert q["native_zero_offdiag_row_fraction"] == 1
    assert q["native_top_1pct_mass_fraction"] == 0
    assert all(np.isfinite(x) for x in q.values())
    for invalid in (np.ones((3, 4)), np.eye(3) * -1, np.array([[0., 1.], [0., 0.]])):
        with pytest.raises(ValueError):
            native_qc(invalid)


def test_normalization_sensitivity_is_final_clip_only_and_default_is_exact():
    a = np.zeros((64, 64), dtype=np.float32)
    a[9, 40] = a[40, 9] = 100
    variants, qc = normalization_variants(a)
    assert np.array_equal(variants["default_clip_5"], normalize(a))
    assert variants["unclipped"][9, 40] > 5
    assert variants["default_clip_5"][9, 40] == 5
    assert variants["explicit_clip_2_4"][9, 40] == 4
    assert qc["default_highpass_region_clipped_high_fraction"] > 0


def test_covariates_do_not_use_novelty_score_or_rep2_outcome():
    q = native_qc(np.ones((10, 10)))
    table = pd.DataFrame([{**q, "known_distance_bp": 3000, "candidate_score": .1, "novelty": 99}])
    first = qc_covariates(table)
    table["candidate_score"] = 999
    table["novelty"] = 0
    assert np.array_equal(first, qc_covariates(table))
    assert first.shape == (1, len(COVARIATES))


def test_control_matching_rejects_unsupported_targets_even_with_many_controls():
    pool = np.column_stack([np.linspace(0, 1, 120)] * len(COVARIATES))
    centers = np.arange(120) * 20000
    ids, audit = select_controls(pool, np.full(len(COVARIATES), 1.5), centers, np.array([-100000]))
    assert len(ids) >= 20
    assert audit["evaluable"] is False
    assert not any(audit["supported_covariates"])
    assert descriptive_percentile(np.arange(len(ids)), 1000, audit["evaluable"]) is None


def test_matching_respects_all_candidate_neighborhoods_and_checks_spacing():
    pool = np.column_stack([np.linspace(-1, 1, 120)] * len(COVARIATES))
    centers = np.arange(120) * 20000
    ids, audit = select_controls(pool, np.zeros(len(COVARIATES)), centers, np.array([0, 1000000]))
    assert len(ids) >= 20 and audit["evaluable"]
    assert 0 not in ids and 50 not in ids
    with pytest.raises(ValueError, match="not independent"):
        select_controls(pool, np.zeros(len(COVARIATES)), centers // 20, np.array([-100000]))


def test_constant_support_and_strict_tie_percentile_are_explicit():
    pool = np.zeros((30, len(COVARIATES)))
    ids, audit = select_controls(pool, np.zeros(len(COVARIATES)), np.arange(30) * 20000, np.array([-100000]))
    assert len(ids) == 30 and audit["evaluable"]
    assert descriptive_percentile(np.array([1, 2, 2, 3]), 2, True) == .25


def test_paired_correlations_are_finite_and_do_not_mix_rows():
    a = np.array([[1., 2., 3.], [0., 0., 0.]])
    b = np.array([[3., 2., 1.], [1., 2., 3.]])
    assert np.allclose(paired_r(a, b), [-1, 0])
    with pytest.raises(ValueError):
        paired_r(a, b[:1])


def test_recalibrated_models_and_cutoff_use_known_references_only():
    rng = np.random.default_rng(6)
    vectors = rng.normal(size=(10, 30))
    centers = np.arange(8) * 20000
    _, cutoff1, scaler1, pca1 = project_novelty(vectors, 8, centers)
    vectors[8:] *= 10000
    _, cutoff2, scaler2, pca2 = project_novelty(vectors, 8, centers)
    assert cutoff1 == cutoff2
    assert np.array_equal(scaler1.mean_, scaler2.mean_)
    assert np.array_equal(pca1.components_, pca2.components_)
