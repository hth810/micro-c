import numpy as np
import pandas as pd

from scripts.task2_next_rep2_shape_novelty import novelty_distances, project


def test_novelty_distances_exclude_nearby_known_in_calibration_only():
    known = np.array([[0.0], [0.1], [5.0]])
    candidates = np.array([[0.05], [10.0]])
    known_centers = np.array([0, 100, 20000])
    calibration, candidate_distances = novelty_distances(
        known, candidates, known_centers)
    assert np.allclose(calibration, [5.0, 4.9, 4.9])
    assert np.allclose(candidate_distances.min(axis=1), [.05, 5.0])


def test_frozen_decision_cutoff_ignores_rep2_known_reference_scale():
    known_fit = np.array([[float(i), float(i % 3)] for i in range(12)])
    centers = np.arange(12) * 20000
    candidates = np.array([[15.0, 1.0]])
    ann = pd.DataFrame({"id": [f"k{i}" for i in range(12)],
                        "type": ["CHIN"] * 12, "center": centers})
    _, first = project(known_fit, known_fit, candidates, centers, [300000],
                       "frozen", ann, [True], {300000}, True)
    _, shifted = project(known_fit, known_fit * 3, candidates, centers, [300000],
                         "frozen", ann, [True], {300000}, True)
    assert first["decision_threshold"] == shifted["decision_threshold"]
    assert first["rep2_known_reference_p90"] != shifted["rep2_known_reference_p90"]
