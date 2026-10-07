"""Deployment input checks are independent of frozen experiment scores."""
import numpy as np
import pandas as pd
import pytest

from task1.predict_roi_cnn_20261006 import load_inputs, predict


def write_inputs(tmp_path, **changes):
    row = {"id": "known_1", "start": 100, "end": 200, "center": 150}
    row.update(changes)
    ann = tmp_path / "ann.csv"
    pd.DataFrame([row]).to_csv(ann, index=False)
    cache = tmp_path / "windows.npz"
    np.savez(cache, X=np.zeros((2, 2, 128, 128), np.float32), group=["known_1", "known_1"])
    return cache, ann


def test_inference_derives_length(tmp_path):
    x, ann = load_inputs(*write_inputs(tmp_path))
    assert x.shape == (2, 2, 128, 128)
    assert ann.length_bp.tolist() == [100]


@pytest.mark.parametrize("changes", [
    {"start": -1}, {"end": 100}, {"center": 200}, {"center": float("inf")},
    {"start": 100.5}, {"length_bp": 99}, {"id": ""},
])
def test_invalid_inference_intervals_are_rejected(tmp_path, changes):
    with pytest.raises(ValueError):
        load_inputs(*write_inputs(tmp_path, **changes))


def test_missing_columns_have_clear_error(tmp_path):
    cache, ann = write_inputs(tmp_path)
    pd.DataFrame({"id": ["known_1"]}).to_csv(ann, index=False)
    with pytest.raises(ValueError, match="start/end/center"):
        load_inputs(cache, ann)


@pytest.mark.parametrize("shape,groups", [
    ((2, 2, 64, 64), ["known_1", "known_1"]),
    ((2, 2, 128, 128), ["known_1", "different"]),
])
def test_inference_cache_alignment_is_checked(tmp_path, shape, groups):
    cache, ann = write_inputs(tmp_path)
    np.savez(cache, X=np.zeros(shape, np.float32), group=groups)
    with pytest.raises(ValueError):
        load_inputs(cache, ann)


def test_nonfinite_cache_is_rejected(tmp_path):
    cache, ann = write_inputs(tmp_path)
    x = np.zeros((2, 2, 128, 128), np.float32)
    x[0, 0, 0, 0] = np.nan
    np.savez(cache, X=x, group=["known_1", "known_1"])
    with pytest.raises(ValueError, match="Nonfinite"):
        load_inputs(cache, ann)


def test_annotation_ids_preserve_leading_zeroes(tmp_path):
    cache, ann = write_inputs(tmp_path, id="001")
    np.savez(cache, X=np.zeros((2, 2, 128, 128), np.float32), group=["001", "001"])
    _, table = load_inputs(cache, ann)
    assert table.id.tolist() == ["001"]


def test_published_ensemble_can_run_inference(tmp_path):
    from argparse import Namespace
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    cache, ann = write_inputs(tmp_path)
    args = Namespace(windows=cache, annotations=ann,
                     checkpoint=root / "results/task1_improve_20261006_final_interval_roi_cnn.pt",
                     config=root / "results/task1_improve_20261006_final_interval_roi_cnn.json",
                     out=tmp_path / "predictions.csv")
    result = predict(args)
    assert len(result) == 1
    assert args.out.is_file()
    assert np.isclose(result[["p_CHIN", "p_OPCID", "p_CHID"]].to_numpy().sum(), 1)
    args.out = ann
    before = ann.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        predict(args)
    assert ann.read_bytes() == before
