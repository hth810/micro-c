from pathlib import Path

import cooler
import numpy as np
import pandas as pd
from argparse import Namespace

from microc.analysis import _features, _replicate_direction_agreement, _shape_score, differential, discover, tracks
from microc.data import annotations, normalize, open_cool, upper_correlation, window
from microc.deep import _coarsen, _consistency, degrade, prepare_sr_pair, reconstruct3d, superresolve
from scripts.validate_discovery import shape_features
from scripts.validate_hns_candidates import condition_effect
from scripts.validate_novel_cluster import components
from scripts.validate_novel_condition_transfer import correlation_matrix
from scripts.validate_3d_replicate import separation_matched_auc, stratified_correlations
from scripts.validate_sr_known_shape import hotspot
from task1.evaluate_chid_overlap import overlap_targets, spatial_groups
from task1.evaluate_pairwise import PairAwareCNN, nearby_chid_chin_pairs
from task1.evaluate_length_context import LengthAwareCNN
from task1.evaluate_interval_boundaries import interval_features
from task1.run_task1 import StructureCNN, select_validation_bias, spatial_split


def test_spatial_split_keeps_overlapping_windows_together():
    centers = [1000, 3500, 8500, 20000, 24000, 33000, 38000, 50000,
               57000, 66000, 75000, 84000, 93000, 102000, 111000]
    ann = pd.DataFrame({"id": [f"id_{i}" for i in range(len(centers))],
                        "chrom": "chr", "center": centers,
                        "label": [i % 3 for i in range(len(centers))]})
    train, val, test = spatial_split(ann, 12, 5120)
    assert train | val | test == set(ann.id)
    assert not (train & val or train & test or val & test)
    assignments = {id_: part for part, ids in enumerate((train, val, test)) for id_ in ids}
    for i, left in ann.iterrows():
        for _, right in ann.iloc[i + 1:].iterrows():
            if right.center - left.center < 5120:
                assert assignments[left.id] == assignments[right.id]


def test_structure_cnn_spatial_grid():
    import torch

    x = torch.zeros(2, 1, 128, 128)
    assert StructureCNN(3, spatial_grid=1)(x).shape == (2, 3)
    assert StructureCNN(3, spatial_grid=4)(x).shape == (2, 3)


def test_overlap_aware_chid_targets_and_groups():
    frame = pd.DataFrame({"chrom": ["chr", "chr", "chr", "other"],
                          "center": [1000, 1800, 9000, 1800],
                          "type": ["CHID", "CHIN", "OPCID", "CHIN"]})
    assert overlap_targets(frame, 1000).tolist() == [1, 1, 0, 0]
    groups = spatial_groups(frame, 5120)
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]
    assert groups[0] != groups[3]


def test_nearby_chid_chin_pairs():
    import torch

    frame = pd.DataFrame({"chrom": ["chr", "chr", "chr", "other"],
                          "center": [1000, 1800, 9000, 1800],
                          "type": ["CHID", "CHIN", "CHIN", "CHIN"]})
    assert nearby_chid_chin_pairs(frame, 2000).tolist() == [[0, 1]]
    assert PairAwareCNN()(torch.zeros(2, 1, 128, 128)).shape == (2, 3)
    assert PairAwareCNN(input_channels=2)(torch.zeros(2, 2, 128, 128)).shape == (2, 3)
    assert LengthAwareCNN()(torch.zeros(2, 1, 128, 128), torch.zeros(2)).shape == (2, 3)


def test_hns_effect_subtracts_replicate_noise():
    base = np.eye(8, dtype=np.float32)
    changed = base.copy()
    changed[0, 3] = changed[3, 0] = 2
    unchanged = condition_effect(base, base, base, base)
    altered = condition_effect(base, base, changed, changed)
    assert unchanged["effect_corrected"] == 0
    assert altered["effect_corrected"] > 0


def test_differential_selection_is_replicate_order_invariant(tmp_path):
    # The old (treatment1-WT1, treatment2-WT2) sign test changed on a
    # treatment-replicate swap despite the underlying data being identical.
    assert not _replicate_direction_agreement([0, 4], [5, 1])
    assert not _replicate_direction_agreement([4, 0], [1, 5])
    assert _replicate_direction_agreement([0, 1], [3, 2])
    assert _replicate_direction_agreement([1, 0], [2, 3])

    bins = pd.DataFrame({"chrom": ["MG1655"] * 128, "start": np.arange(128) * 10,
                         "end": (np.arange(128) + 1) * 10})
    a, b = np.triu_indices(128)
    base = np.maximum(1, 12 - np.abs(a - b))
    paths = []
    for name, depth, extra in (("wt1", 3, 0), ("wt2", 4, 1),
                               ("treatment1", 5, 3), ("treatment2", 6, 4)):
        path = tmp_path / f"{name}.cool"
        counts = base * depth + extra * ((a > 30) & (a < 70) & (b > 45) & (b < 90))
        cooler.create_cooler(str(path), bins, pd.DataFrame({
            "bin1_id": a, "bin2_id": b, "count": counts.astype(int)}))
        paths.append(path)
    labels = tmp_path / "structures.csv"
    pd.DataFrame({"id": ["one", "two"], "chrom": ["MG1655"] * 2,
                  "type": ["CHIN", "CHID"], "start": [290, 610],
                  "end": [350, 670]}).to_csv(labels, index=False)
    common = dict(wt=paths[0], wt2=paths[1], annotations=labels,
                  name="swapped", window_bp=640, heatmaps=2, seed=123)
    first, second = tmp_path / "first", tmp_path / "second"
    differential(Namespace(**common, condition=paths[2], condition2=paths[3], out=first))
    differential(Namespace(**common, condition=paths[3], condition2=paths[2], out=second))
    a_table = pd.read_csv(first / "structure_changes.csv").sort_values("id").reset_index(drop=True)
    b_table = pd.read_csv(second / "structure_changes.csv").sort_values("id").reset_index(drop=True)
    for field in ("wt_strength", "condition_strength", "delta", "wt_repeat_difference",
                  "condition_repeat_difference", "oe_correlation"):
        assert np.allclose(a_table[field], b_table[field])
    for field in ("repeat_direction_agreement", "exceeds_repeat_variation",
                  "exceeds_global_repeat_p95", "reproducible_change"):
        assert a_table[field].equals(b_table[field])


def test_interval_boundary_features_follow_annotated_interval():
    mat = np.zeros((128, 128), dtype=np.float32)
    mat[60:68, 60:68] = 2
    features = interval_features(mat, start=0, end=1280, center=640)
    assert len(features) == 14
    assert np.isfinite(features).all()
    assert features[8] > features[2]


def test_novel_cluster_requires_independent_pair_edge():
    assert components(4, []) == []
    assert components(4, [(0, 2)]) == [[0, 2]]


def test_3d_stratified_consistency_detects_distance_order():
    positions = np.arange(12, dtype=float) ** 1.2
    xyz = np.column_stack((positions, np.zeros((12, 2))))
    distances = np.abs(np.subtract.outer(positions, positions))
    contact = 1 / (distances + 1)
    contact[distances > 6] = 0
    scores = stratified_correlations(contact, {"ordered": xyz})
    assert scores["all_positive"]["ordered"] > .99
    assert scores["top_20_percent"]["pairs"] > 3
    assert scores["weaker_positive"]["pairs"] > 3
    assert scores["positive_vs_zero_distance_auc"]["ordered"] > .99


def test_exact_separation_matching_removes_genomic_distance_shortcut():
    n = 24
    signal = np.zeros((n, n), dtype=float)
    for offset in range(2, n - 1):
        for i in range(n - offset):
            signal[i, i + offset] = signal[i + offset, i] = float(i % 2 == 0)
    theta = 2 * np.pi * np.arange(n) / n
    points = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(n)))
    from scipy.spatial.distance import pdist, squareform
    matched = separation_matched_auc(signal, {"circle": squareform(pdist(points))})
    assert matched["matched_pairs_per_class"] > 0
    assert abs(matched["circle"] - .5) < .01


def test_condition_shape_correlation_and_sr_hotspot():
    rows = np.array([[1., 2., 3.], [2., 4., 6.], [3., 2., 1.]])
    correlations = correlation_matrix(rows)
    assert np.isclose(correlations[0, 1], 1)
    assert np.isclose(correlations[0, 2], -1)
    patch = np.zeros((16, 16), dtype=float)
    patch[3, 10] = 10
    upper = np.triu(np.ones_like(patch, dtype=bool), k=3)
    assert tuple(hotspot(patch, upper).astype(int)) == (3, 10)


def test_real_annotation_workbook_if_present():
    book = Path("data/raw/标注数据.xlsx")
    if not book.exists():
        return
    table = annotations(book)
    assert table.groupby("type").size().to_dict() == {"CHID": 26, "CHIN": 250, "OPCID": 68}
    assert table.center.notna().all()
    assert (table.end > table.start).all()


def test_validation_only_logit_calibration():
    logits = np.array([[2.0, 0.0, 1.9], [0.0, 2.0, 1.9], [0.0, 0.0, 2.0]])
    labels = np.array([0, 1, 2])
    bias, report = select_validation_bias(logits, labels)
    assert len(bias) == 3
    assert report["calibrated"]["macro_f1"] >= report["uncalibrated"]["macro_f1"]


def test_synthetic_cooler_and_geometry(tmp_path):
    path = tmp_path / "small.cool"
    bins = pd.DataFrame({"chrom": ["MG1655"] * 128, "start": np.arange(128) * 10,
                         "end": (np.arange(128) + 1) * 10})
    a, b = np.triu_indices(128)
    counts = np.maximum(1, 20 - np.abs(a - b)).astype(int)
    counts[(a > 40) & (a < 60) & (b > 50) & (b < 80)] += 20
    pixels = pd.DataFrame({"bin1_id": a, "bin2_id": b, "count": counts})
    cooler.create_cooler(str(path), bins, pixels)
    with open_cool(path) as c:
        patch = window(c, "MG1655", 640, 640, 64)
        norm = normalize(patch)
        assert patch.shape == (64, 64)
        assert np.allclose(patch, patch.T)
        assert upper_correlation(norm, norm) > .99
        assert len(_features(norm)) == 105 + 5
        assert len(shape_features(norm)) == 118
        assert np.isfinite(shape_features(norm)).all()
        assert _shape_score(norm) >= 0
        assert degrade(norm, 4).shape == (64, 64)
        low, high = prepare_sr_pair(patch, 4)
        assert np.allclose(low, normalize(degrade(patch, 4)))
        assert np.allclose(high, norm)
        coarse = _coarsen(c, "MG1655", 160)
        assert coarse.shape == (8, 8)
        assert np.allclose(coarse, coarse.T)
        coords = np.arange(8)[:, None] * np.array([[1, 0, 0]])
        assert np.isfinite(_consistency(coords, coarse / coarse.max()))


def test_end_to_end_small_data(tmp_path):
    path = tmp_path / "run.cool"
    count = 512
    bins = pd.DataFrame({"chrom": ["MG1655"] * count, "start": np.arange(count) * 10,
                         "end": (np.arange(count) + 1) * 10})
    a, b = np.triu_indices(count)
    counts = np.maximum(1, 12 - np.abs(a - b)).astype(int)
    counts[(a > 150) & (a < 200) & (b > 180) & (b < 250)] += 20
    cooler.create_cooler(str(path), bins, pd.DataFrame({"bin1_id": a, "bin2_id": b, "count": counts}))
    labels = tmp_path / "structures.csv"
    pd.DataFrame({"id": ["one", "two", "three"], "chrom": ["MG1655"] * 3,
        "type": ["CHIN", "OPCID", "CHID"], "start": [1700, 2900, 4100],
        "end": [1900, 3100, 4300]}).to_csv(labels, index=False)
    discover(Namespace(rep1=path, rep2=path, annotations=labels, out=tmp_path / "discovery",
        window_bp=640, stride_bp=320, top_fraction=.8, min_replicate_r=-1,
        clusters=2, min_cluster=1, max_heatmaps=2, seed=1, ae_epochs=1))
    assert (tmp_path / "discovery" / "all_candidates.csv").exists()
    assert {"candidate_individual", "candidate_cluster_supported", "candidate_new"}.issubset(
        pd.read_csv(tmp_path / "discovery" / "all_candidates.csv").columns)
    differential(Namespace(wt=path, condition=path, annotations=labels, name="same",
        out=tmp_path / "differential", window_bp=640, heatmaps=2))
    changes = pd.read_csv(tmp_path / "differential" / "structure_changes.csv")
    assert len(changes) == 3
    assert np.allclose(changes.delta, 0)
    assert (tmp_path / "differential" / "reproducible_changes.csv").exists()
    tracks(Namespace(cool=[path, path, path], annotations=labels, gff=None,
        out=tmp_path / "tracks", interval_bp=2560))
    assert len(list((tmp_path / "tracks").glob("*.png"))) == 2
    superresolve(Namespace(cool=path, out=tmp_path / "sr", annotations=labels, chrom="MG1655",
        patch_bins=32, factor=4, max_patches=16, epochs=1, batch_size=4, examples=1, seed=1))
    assert (tmp_path / "sr" / "super_resolution.pt").exists()
    reconstruct3d(Namespace(cool=path, out=tmp_path / "3d", annotations=labels,
        chrom="MG1655", bin_bp=320,
        epochs=2, pretrain_epochs=2, anchor_weight=.5, pairs_per_epoch=50, lr=.001, seed=1))
    assert len(pd.read_csv(tmp_path / "3d" / "coordinates.csv")) == 16
