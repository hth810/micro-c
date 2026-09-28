"""Choice rules for Task 1 nested spatial boundary evaluation."""
from __future__ import annotations

import numpy as np
import pytest

from task1.evaluate_next_nested_boundaries import MODES, predict_plan, select_plan


def test_default_class_argmax_and_chid_override():
    probs = np.array([[0.45, 0.30, 0.25], [0.20, 0.50, 0.30]])
    assert predict_plan(probs, None).tolist() == [0, 1]
    assert predict_plan(probs, 0.2).tolist() == [2, 2]


def test_selection_uses_only_given_inner_labels_and_prefers_simple_tie():
    truth = np.array([0, 1, 2, 0, 1, 2])
    perfect = np.eye(3)[truth] * 0.9 + 0.1 / 3
    wrong = np.eye(3)[(truth + 1) % 3] * 0.9 + 0.1 / 3
    probs = {MODES[0]: perfect, MODES[1]: wrong, MODES[2]: wrong}
    mode, threshold, options = select_plan(probs, truth)
    assert (mode, threshold) == (MODES[0], None)
    assert len(options) == 12
    assert options[0]["macro_f1"] == 1.0


def test_selection_rejects_incomplete_crossfit():
    truth = np.array([0, 1, 2])
    probs = {mode: np.full((3, 3), 1 / 3) for mode in MODES}
    probs[MODES[1]][1, 2] = np.nan
    with pytest.raises(ValueError, match="Incomplete"):
        select_plan(probs, truth)
