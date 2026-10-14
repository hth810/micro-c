import pandas as pd
import pytest

from scripts.task2_course_classifier_context_20261014 import provisional_roi


def test_provisional_roi_uses_fixed_original_window_not_inferred_biological_boundaries():
    annotations = pd.DataFrame({"id": ["A1", "A2"], "center": [729600, 1469440],
                                "start": [100, 200], "end": [101, 900]})
    original = annotations.copy()
    output = provisional_roi(annotations)
    assert output.start.tolist() == [727040, 1466880]
    assert output.end.tolist() == [732160, 1472000]
    assert output.length_bp.tolist() == [5120, 5120]
    assert annotations.equals(original)


def test_provisional_roi_rejects_new_width_or_duplicate_ids():
    table = pd.DataFrame({"id": ["A1"], "center": [5000]})
    with pytest.raises(ValueError):
        provisional_roi(table, 3000)
    with pytest.raises(ValueError):
        provisional_roi(pd.concat([table, table], ignore_index=True))
