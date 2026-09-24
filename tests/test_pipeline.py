from pathlib import Path

import cooler
import numpy as np
import pandas as pd
from argparse import Namespace

from microc.analysis import _features, _shape_score, differential, discover, tracks
from microc.data import annotations, normalize, open_cool, upper_correlation, window
from microc.deep import _coarsen, _consistency, degrade, reconstruct3d, superresolve


def test_real_annotation_workbook_if_present():
    book = Path("data/raw/标注数据.xlsx")
    if not book.exists():
        return
    table = annotations(book)
    assert table.groupby("type").size().to_dict() == {"CHID": 26, "CHIN": 250, "OPCID": 68}
    assert table.center.notna().all()
    assert (table.end > table.start).all()


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
        assert _shape_score(norm) >= 0
        assert degrade(norm, 4).shape == (64, 64)
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
    differential(Namespace(wt=path, condition=path, annotations=labels, name="same",
        out=tmp_path / "differential", window_bp=640, heatmaps=2))
    assert len(pd.read_csv(tmp_path / "differential" / "structure_changes.csv")) == 3
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
