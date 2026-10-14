import numpy as np
import pandas as pd
import torch

from microc.deep import SuperResolutionCNN
from scripts.task5_full_structure_verification_20261014 import (
    cohort_manifest, full_interval, image_metrics, prepare_full_pair, whole_cnn,
)


def test_halo_tiling_matches_whole_cnn_without_local_transpose():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = SuperResolutionCNN(channels=8, blocks=4).eval()
    torch.nn.init.normal_(model.tail.weight, std=.03)
    torch.nn.init.normal_(model.tail.bias, std=.03)
    rng = np.random.default_rng(9)
    image = rng.normal(size=(143, 143)).astype(np.float32)
    image = (image + image.T) / 2
    with torch.inference_mode():
        direct = model(torch.from_numpy(image[None, None])).numpy()[0, 0]
    tiled = whole_cnn(model, image, core=43, halo=16, batch_size=3)
    np.testing.assert_allclose(tiled, direct, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(tiled, tiled.T, atol=0)


def test_zero_input_is_not_hallucinated():
    model = SuperResolutionCNN(channels=8, blocks=1).eval()
    torch.nn.init.constant_(model.tail.bias, 3.)
    result = whole_cnn(model, np.zeros((80, 80), np.float32), core=32)
    assert np.count_nonzero(result) == 0


def test_full_pair_input_cannot_use_high_resolution_statistics():
    rng = np.random.default_rng(3)
    low = rng.poisson(5, size=(24, 24)).astype(np.float32)
    raw = np.repeat(np.repeat(low, 4, axis=0), 4, axis=1)
    changed = raw.copy()
    changed[4, 8] += 3
    changed[4, 9] -= 3
    base_a, cubic_a, target_a = prepare_full_pair(raw)
    base_b, cubic_b, target_b = prepare_full_pair(changed)
    np.testing.assert_array_equal(base_a, base_b)
    np.testing.assert_array_equal(cubic_a, cubic_b)
    assert not np.array_equal(target_a, target_b)


def test_full_interval_covers_annotation_and_grid():
    record = type("Record", (), {"start": 10123, "end": 13217})()
    left, right = full_interval(record, 4641652)
    assert left <= record.start - 160 and right >= record.end + 160
    assert left % 40 == right % 40 == 0


def test_metrics_identical_whole_maps_have_perfect_morphology():
    rng = np.random.default_rng(22)
    image = rng.normal(size=(96, 96)).astype(np.float32)
    image = (image + image.T) / 2
    scores = image_metrics(image, image)
    assert scores["shape_r"] > .999999
    assert scores["profile_r"] > .999999
    assert scores["peak_error_bp"] == 0
    assert scores["top5_f1"] == 1


def test_split_excludes_full_windows_crossing_validation_boundary():
    frame = pd.DataFrame([
        {"id": "a", "type": "CHIN", "chrom": "MG1655", "start": 3247700, "end": 3248400, "center": 3248050, "length_bp": 700},
        {"id": "b", "type": "OPCID", "chrom": "MG1655", "start": 3800000, "end": 3810000, "center": 3805000, "length_bp": 10000},
    ])
    manifest, _ = cohort_manifest(frame, 4641652)
    assert manifest.loc[manifest.id == "a", "split"].item() == "excluded_boundary_crossing"
    assert manifest.loc[manifest.id == "b", "split"].item() == "test"
