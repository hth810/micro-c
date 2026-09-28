from pathlib import Path

import pandas as pd
import pytest

from task1.run_task1 import split_from_manifest


def test_manifest_reuses_disjoint_genomic_partitions(tmp_path: Path) -> None:
    annotations = pd.DataFrame({"id": ["A", "B", "C"], "type": ["CHIN", "OPCID", "CHID"],
                                "chrom": ["NC_000913.3"] * 3, "center": [5_000, 20_000, 35_000]})
    manifest = annotations.assign(split=["train", "validation", "test"])
    path = tmp_path / "manifest.csv"
    manifest.to_csv(path, index=False)
    assert split_from_manifest(annotations, path, 5_120) == ({"A"}, {"B"}, {"C"})


def test_manifest_rejects_cross_partition_window_overlap(tmp_path: Path) -> None:
    annotations = pd.DataFrame({"id": ["A", "B", "C"], "type": ["CHIN", "OPCID", "CHID"],
                                "chrom": ["NC_000913.3"] * 3, "center": [5_000, 7_000, 35_000]})
    path = tmp_path / "manifest.csv"
    annotations.assign(split=["train", "validation", "test"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="overlapping windows"):
        split_from_manifest(annotations, path, 5_120)
