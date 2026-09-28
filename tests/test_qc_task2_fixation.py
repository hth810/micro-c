"""Synthetic checks for frozen-window depth QC."""
from contextlib import contextmanager

import numpy as np
import pytest

from scripts import qc_task2_fixation as qc


class FakeCool:
    chromnames = ["MG1655"]
    binsize = 80


@contextmanager
def fake_open_cool(_path):
    yield FakeCool()


def test_qc_counts_each_symmetric_contact_once(monkeypatch, tmp_path):
    raw = np.zeros((64, 64), dtype=np.float32)
    raw[0, 0] = 2
    raw[0, 8] = raw[8, 0] = 3
    raw[0, 9] = raw[9, 0] = 4

    def fake_window(_cool, chrom, center, size_bp, output_bins):
        assert (chrom, center, size_bp, output_bins) == ("MG1655", 2042880, 5120, 64)
        return raw.copy()

    monkeypatch.setattr(qc, "open_cool", fake_open_cool)
    monkeypatch.setattr(qc, "window", fake_window)
    report = qc.summary_for_sample(tmp_path / "unused.cool", [2042880], "MG1655")
    locus = report["loci"][0]
    assert report["native_bin_bp"] == 80
    assert locus["raw_cis_upper_count"] == 9
    assert locus["raw_far_640bp_upper_count"] == 7
    assert locus["raw_cis_upper_nonzero_fraction"] == pytest.approx(3 / 2080)
    assert locus["raw_far_640bp_upper_nonzero_fraction"] == pytest.approx(2 / 1596)
    assert locus["normalized_64bin_far8_finite_fraction"] == 1
