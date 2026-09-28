"""Deterministic synthetic checks for frozen-locus thinning."""
import numpy as np
import pytest

from scripts import task2_fixation_depth_power as power


def test_thinning_counts_unique_contacts_once_and_keeps_symmetry():
    raw = np.array([[2, 3, 0], [3, 4, 5], [0, 5, 6]], dtype=np.float32)
    rng = np.random.default_rng(11)
    assert np.array_equal(power.thin_symmetric_counts(raw, 0, rng), np.zeros_like(raw))
    assert np.array_equal(power.thin_symmetric_counts(raw, 1, rng), raw)
    first = power.thin_symmetric_counts(raw, .4, np.random.default_rng(25))
    again = power.thin_symmetric_counts(raw, .4, np.random.default_rng(25))
    assert np.array_equal(first, again)
    assert np.array_equal(first, first.T)
    assert np.all(first <= raw)
    with pytest.raises(ValueError, match="symmetric"):
        power.thin_symmetric_counts(np.array([[0, 2], [1, 0]]), .5, rng)


def test_analysis_map_and_edge_summary_have_frozen_layout():
    raw = np.zeros((64, 64), dtype=np.float32)
    raw[0, 8] = raw[8, 0] = 5
    normalized = power.analysis_map(raw)
    assert normalized.shape == (64, 64)
    assert np.isfinite(normalized).all()
    draws = [{metric: [.1, .2, .3] for metric in power.METRICS},
             {metric: [-.1, .3, .4] for metric in power.METRICS}]
    full = {metric: [.4, .5, .6] for metric in power.METRICS}
    observed = {metric: [0, .25, .35] for metric in power.METRICS}
    report = power.summarize_draws(draws, full, observed, [10, 20, 30])
    for metric in power.METRICS:
        assert [item["centers_bp"] for item in report[metric]["edges"]] == [
            [10, 20], [10, 30], [20, 30]]
        assert report[metric]["thinned_all_three_edges_positive_fraction"] == .5
        assert report[metric]["edges"][0]["fraction_thinned_r_le_fix37"] == .5
