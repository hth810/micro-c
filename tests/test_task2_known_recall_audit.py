import numpy as np
import pandas as pd

from scripts.audit_task2_known_recall import (
    merge_annotated_loci, overlap_matrix, shifted_recall, summarize_hits,
)


def test_overlap_matrix_respects_chromosome_and_intervals():
    annotations = pd.DataFrame({"chrom": ["MG1655", "other"], "start": [0, 0],
                                "end": [10, 10]})
    windows = pd.DataFrame({"chrom": ["NC_000913.3", "NC_000913.3", "other"],
                            "start": [0, 10, 5], "end": [10, 20, 15]})
    assert overlap_matrix(annotations, windows).tolist() == [
        [True, False, False], [False, False, True]]


def test_merge_annotated_loci_collapses_overlapping_aliases_only():
    ann = pd.DataFrame({"chrom": ["MG1655", "NC_000913.3", "MG1655"],
                        "start": [0, 5, 20], "end": [10, 15, 30]})
    merged = merge_annotated_loci(ann)
    assert merged.to_dict("records") == [
        {"chrom": "NC_000913.3", "start": 0, "end": 15},
        {"chrom": "NC_000913.3", "start": 20, "end": 30},
    ]


def test_circular_shift_preserves_selected_count_and_excludes_identity():
    overlap = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=bool)
    selected = np.array([1, 1, 0, 0], dtype=bool)
    observed, null = shifted_recall(overlap, selected)
    assert observed.tolist() == [True, True]
    assert null.tolist() == [[False, True], [False, False], [True, False]]
    summary = summarize_hits(observed, null)
    assert summary["observed_recall"] == 1
    assert summary["circular_shift_median"] == .5
