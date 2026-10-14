import gzip
import json
import numpy as np
import pandas as pd
import pytest

from scripts.task6_full_geometry_verification_20261014 import (
    bh, bin_table, circular_pair_index, dump, index_for_bp, load_models, matched_selection,
    ori_from_gff, pair_auc,
)


def test_explicit_coordinate_keys_prevent_legacy_model_substitution(tmp_path):
    path = tmp_path / "models.npz"
    np.savez(path, final=np.zeros((12, 3)), mlp_coordinates=np.ones((12, 3)))
    assert np.array_equal(load_models(path, ("final",), 12)["final"], np.zeros((12, 3)))
    with pytest.raises(ValueError, match="Missing requested"):
        load_models(path, ("rank_mlp_seed20261006",), 12)
    with pytest.raises(ValueError, match="distinct"):
        load_models(path, ("final", "final"), 12)


def test_nonfinite_coordinates_rejected(tmp_path):
    path = tmp_path / "models.npz"
    a = np.zeros((12, 3)); a[3, 1] = np.nan
    np.savez(path, final=a)
    with pytest.raises(ValueError, match="finite"):
        load_models(path, ("final",), 12)


def test_ring_pairs_exclude_origin_crossing_neighbor():
    i, j, offsets = circular_pair_index(12)
    assert offsets.min() == 2
    assert not np.any((i == 0) & (j == 11))
    assert np.any((i == 0) & (j == 10))


def test_exact_offset_match_balances_categories_and_pure_circle():
    n = 48
    i, j, offsets = circular_pair_index(n)
    labels = np.repeat(np.arange(6), 8)
    a, b = matched_selection(labels[i] == labels[j], labels[i] != labels[j], offsets, n, 9)
    assert len(a) == len(b) > 0
    for step in np.unique(offsets):
        assert np.sum(offsets[a] == step) == np.sum(offsets[b] == step)
    assert pair_auc(offsets.astype(float), a, b) == .5
    strong_domains = offsets * .01 + (labels[i] != labels[j]).astype(float)
    assert pair_auc(strong_domains, a, b) == 1.


def test_bin_boundary_assignment_and_last_partial_bin(tmp_path):
    path = tmp_path / "bins.csv"
    start = np.arange(12) * 5000
    end = np.minimum(start + 5000, 55002)
    pd.DataFrame({"chrom": "NC_000913.3", "start": start, "end": end}).to_csv(path, index=False)
    frame = bin_table(path)
    assert index_for_bp(frame, 5000) == 1
    assert index_for_bp(frame, 55001) == 11
    with pytest.raises(ValueError, match="outside"):
        index_for_bp(frame, 55002)


def test_gff_coordinate_conversion_is_one_based_inclusive(tmp_path):
    path = tmp_path / "gff.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("NC_000913.3\tRefSeq\torigin_of_replication\t101\t200\t.\t+\t.\tNote=oriC\n")
    result = ori_from_gff(path)
    assert result["start_0based"] == 100
    assert result["end_exclusive"] == 200
    assert result["bp"] == 150


def test_bh_is_order_preserving_and_monotone_in_sorted_p():
    p = [.04, .001, .02, .3]
    q = bh(p)
    assert np.all(q >= p)
    assert np.all(np.diff(q[np.argsort(p)]) >= 0)
    assert q[1] == .004


def test_pair_categories_cannot_overlap():
    with pytest.raises(ValueError, match="disjoint"):
        matched_selection(np.ones(6, bool), np.ones(6, bool), np.full(6, 2), 12)


def test_json_keeps_absent_baseline_probability_as_null(tmp_path):
    path = tmp_path / "result.json"
    dump(path, {"baseline_p": np.nan, "score": np.float64(.6)})
    assert json.loads(path.read_text(encoding="utf-8")) == {"baseline_p": None, "score": .6}
    with pytest.raises(ValueError, match="Nonfinite"):
        dump(path, {"score": np.inf})
