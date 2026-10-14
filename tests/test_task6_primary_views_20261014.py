import numpy as np
import pandas as pd
import pytest

from scripts.task6_primary_views_20261014 import draw_primary_views


def frame():
    boundaries = np.linspace(0, 4641652, 13).astype(int)
    return pd.DataFrame({"start": boundaries[:-1], "end": boundaries[1:]})


def test_primary_two_views_preserve_circular_labels_and_create_image(tmp_path):
    theta = np.arange(12) * 2 * np.pi / 12
    xyz = np.c_[np.cos(theta), np.sin(theta), np.sin(2 * theta)]
    target = tmp_path / "two_views.png"
    labels = draw_primary_views(xyz, frame(), 3925859, target)
    assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert len(labels) == 12 and set(labels) == {"Ori", "NSR", "Right", "Ter", "Left", "NSL"}
    assert labels[-1] == "Ori"


def test_primary_views_reject_missing_coordinate(tmp_path):
    with pytest.raises(ValueError, match="finite full-chromosome"):
        draw_primary_views(np.zeros((11, 3)), frame(), 3925859, tmp_path / "bad.png")
