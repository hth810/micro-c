from pathlib import Path

import cooler
import numpy as np
import pandas as pd
from argparse import Namespace

from microc.analysis import _features, _shape_score, differential, discover, tracks
from microc.data import annotations, normalize, open_cool, upper_correlation, window
from microc.deep import _coarsen, _consistency, degrade, prepare_sr_pair, reconstruct3d, superresolve
from scripts.validate_discovery import shape_features
from scripts.validate_hns_candidates import condition_effect
from task1.evaluate_chid_overlap import overlap_targets, spatial_groups
from task1.evaluate_pairwise import PairAwareCNN, nearby_chid_chin_pairs
from task1.evaluate_length_context import LengthAwareCNN
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
