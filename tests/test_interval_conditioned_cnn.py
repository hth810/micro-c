"""Leakage and geometry checks for the Task 1 interval-conditioned ablation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from task1.evaluate_interval_conditioned_cnn import interval_channels, make_modes


def test_interval_channels_repeat_and_depend_on_coordinates_not_type() -> None:
    ann = pd.DataFrame({"id": ["a", "b"], "type": ["CHIN", "CHID"],
                        "start": [0, 0], "end": [320, 640],
                        "center": [160, 320]})
    mask = interval_channels(ann)
    assert mask.shape == (4, 2, 128, 128)
    np.testing.assert_array_equal(mask[:2], mask[2:])
    assert mask[0, 0].sum() < mask[1, 0].sum()
    assert not np.any(mask[0, 0] * mask[0, 1])
    switched = ann.copy()
    switched["type"] = switched["type"].iloc[::-1].to_numpy()
    np.testing.assert_array_equal(mask, interval_channels(switched))


def test_modes_separate_contact_and_interval_information() -> None:
    ann = pd.DataFrame({"id": ["a"], "type": ["CHIN"],
                        "start": [0], "end": [320], "center": [160]})
    x = np.zeros((2, 2, 128, 128), dtype=np.float32)
    x[0, 0] = 2
    x[1, 1] = 3
    modes = make_modes(x, ann)
    assert tuple(modes) == ("contact_short", "contact_dual", "contact_interval", "interval_only")
    assert modes["contact_short"].shape == (2, 1, 128, 128)
    assert modes["contact_interval"].shape == (2, 4, 128, 128)
    np.testing.assert_array_equal(modes["contact_interval"][:, :2], x)
    np.testing.assert_array_equal(modes["contact_interval"][:, 2:], modes["interval_only"])
    assert not np.any(modes["interval_only"] == 3)
