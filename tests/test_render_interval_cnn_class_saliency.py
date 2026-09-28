"""Deterministic, explicitly post-hoc class-example selection."""
from __future__ import annotations

import pandas as pd

from task1.render_interval_cnn_class_saliency import select_examples


def test_select_examples_requires_correct_prediction_and_prefers_median_length() -> None:
    ann = pd.DataFrame({"id": ["c1", "c2", "c3", "o1", "h1"],
                        "type": ["CHIN", "CHIN", "CHIN", "OPCID", "CHID"],
                        "length_bp": [100, 200, 300, 500, 700]})
    pred = pd.DataFrame({"id": ann.id,
                         "predicted_type": ["CHIN", "CHIN", "CHIN", "OPCID", "CHIN"]})
    choice = select_examples(ann, pred)
    assert choice == {"CHIN": 1, "OPCID": 3, "CHID": None}
