"""Invariant checks for the rep1-only Task 2 candidate split."""
import numpy as np
import pandas as pd

from scripts.task2_rep1_locked_discovery import (
    independent_greedy, nearest_matched_controls, nearest_other_known,
    rank_morphology, rep1_cliques,
)


def test_independent_greedy_keeps_highest_score_without_overlap():
    table = pd.DataFrame({"center": [20000, 11000, 30000, 70000],
                          "candidate_score": [.95, .5, .8, .9]})
    selected = independent_greedy(table)
    assert selected.center.tolist() == [20000, 70000]


def test_known_calibration_excludes_nearby_known_loci():
    x = np.array([[0.0], [.01], [2.0], [4.0]])
    centers = np.array([10000, 11000, 50000, 90000])
    nearest = nearest_other_known(x, centers)
    assert np.allclose(nearest, [2, 1.99, 1.99, 2])


def test_pair_rank_does_not_use_rep2():
    frame = pd.DataFrame({"rep1_upper_r": [.2, .9, .8],
                          "rep1_coarse_r": [.9, .3, .8],
                          "center_a_bp": [10, 20, 30],
                          "center_b_bp": [40, 50, 60]})
    ranked = rank_morphology(frame)
    assert (ranked.center_a_bp.iloc[0], ranked.center_b_bp.iloc[0]) == (30, 60)


def test_matching_rejects_shared_loci_and_bad_balance():
    frame = pd.DataFrame({
        "i": [0, 0, 2, 4], "j": [1, 2, 3, 5],
        "distance_bp": [100000, 110000, 105000, 100000],
        "min_score": [.95, .95, .94, .5],
        "mean_score": [.96, .96, .95, .5],
        "min_novelty": [14, 14, 13.5, 2],
        "mean_novelty": [15, 15, 14.5, 2],
        "rep1_upper_r": [.8, .8, .78, -.8],
        "rep1_coarse_r": [.8, .8, .78, -.8],
    })
    idx, _, diagnostics = nearest_matched_controls(frame, 0)
    assert 1 not in idx
    assert diagnostics["independent_pairs"] == 2
    assert 3 not in idx


def test_three_locus_clique_requires_every_rep1_edge():
    frame = pd.DataFrame({"i": [0, 0, 1], "j": [1, 2, 2],
                          "center_a_bp": [10000, 10000, 30000],
                          "center_b_bp": [30000, 50000, 50000],
                          "rep1_upper_r_pct": [.95, .96, .92],
                          "rep1_coarse_r_pct": [.96, .97, .93],
                          "rep1_joint_percentile_floor": [.95, .96, .92]})
    assert rep1_cliques(frame).iloc[0].centers_bp == "10000,30000,50000"
    frame.loc[2, "rep1_coarse_r_pct"] = .89
    assert rep1_cliques(frame).empty
