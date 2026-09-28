import numpy as np

from scripts.evaluate_sr_spatial_blend import (
    bootstrap_hotspot_gain, choose_candidate, peak_mask, predict,
)


def test_peak_mask_is_symmetric_and_uses_only_prediction():
    image = np.zeros((16, 16), dtype=np.float32)
    image[4, 12] = image[12, 4] = 10
    upper = np.triu(np.ones_like(image, dtype=bool), k=3)
    mask = peak_mask(image, 2.0, upper)
    assert np.allclose(mask, mask.T)
    assert mask[4, 12] == 1
    assert mask[0, 15] < .05
    base = np.zeros((1, 16, 16), dtype=np.float32)
    main = np.ones_like(base)
    peak = np.ones_like(base) * 3
    peak[0, 4, 12] = peak[0, 12, 4] = 10
    result = predict(base, main, peak, "peak_roi", 1.0, 2.0)
    assert np.allclose(result, result.transpose(0, 2, 1))
    assert result[0, 4, 12] == 10


def test_selection_requires_both_validation_objectives():
    rows = [
        {"method": "global", "weight": 0.0, "val_tile_ssim_baseline": .1,
         "val_tile_ssim_prediction": .14, "val_tile_psnr_baseline": 18,
         "val_tile_psnr_prediction": 20, "val_known_hotspot_baseline_bins": 15,
         "val_known_hotspot_prediction_bins": 16},
        {"method": "peak_roi", "weight": 1.0, "val_tile_ssim_baseline": .1,
         "val_tile_ssim_prediction": .103, "val_tile_psnr_baseline": 18,
         "val_tile_psnr_prediction": 20, "val_known_hotspot_baseline_bins": 15,
         "val_known_hotspot_prediction_bins": 12},
        {"method": "peak_roi", "weight": .5, "val_tile_ssim_baseline": .1,
         "val_tile_ssim_prediction": .13, "val_tile_psnr_baseline": 18,
         "val_tile_psnr_prediction": 20, "val_known_hotspot_baseline_bins": 15,
         "val_known_hotspot_prediction_bins": 13},
    ]
    assert choose_candidate(rows)["weight"] == .5
    assert choose_candidate(rows[:2])["method"] == "global"


def test_site_bootstrap_gain_is_deterministic():
    base = np.array([2., 4., 6.])
    pred = np.array([1., 3., 5.])
    assert bootstrap_hotspot_gain(base, pred, samples=100) == [1., 1.]
