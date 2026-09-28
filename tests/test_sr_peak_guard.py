import numpy as np
import pandas as pd

from scripts.evaluate_sr_peak_guard import against, block_interval, choose, guarded, paired_report, peak_candidates


def test_guarded_anchor_is_symmetric_and_does_not_use_truth():
    base = np.zeros((1, 32, 32), dtype=np.float32)
    base[0, 8, 24] = base[0, 24, 8] = 5
    main = np.ones_like(base)
    peak = np.ones_like(base) * 2
    result = guarded(base, main, peak, "anchor", 1.0, 2.0)
    assert np.allclose(result, result.transpose(0, 2, 1))
    assert np.isclose(result[0, 8, 24], base[0, 8, 24])
    assert np.isclose(result[0, 0, 31], main[0, 0, 31], atol=.01)


def test_validation_selection_requires_chin_protection_and_image_quality():
    common = {"tile_baseline_psnr": 18., "tile_baseline_ssim": .10,
              "main_tile_psnr": 20., "main_tile_ssim": .14,
              "CHIN_baseline_hotspot_error": 15.,
              "known_baseline_hotspot_error": 14.}
    main = {**common, "name": "main", "tile_prediction_psnr": 20.,
            "tile_prediction_ssim": .14, "CHIN_prediction_hotspot_error": 16.,
            "known_prediction_hotspot_error": 13.}
    good = {**main, "name": "anchor", "tile_prediction_psnr": 19.98,
            "tile_prediction_ssim": .1395, "CHIN_prediction_hotspot_error": 14.,
            "known_prediction_hotspot_error": 12.}
    poor_image = {**good, "name": "peak_roi", "tile_prediction_ssim": .13,
                  "known_prediction_hotspot_error": 10.}
    assert choose([main, good, poor_image])["name"] == "anchor"
    assert choose([main, poor_image])["name"] == "main"


def test_genomic_block_interval_is_deterministic_and_uses_whole_blocks():
    gain = np.array([1., 1., -1., -1.])
    positions = np.array([0, 1000, 60000, 61000])
    first = block_interval(gain, positions, n_boot=1000, seed=7)
    second = block_interval(gain, positions, n_boot=1000, seed=7)
    assert first == second
    assert first == [-1., 1.]


def test_top_five_peaks_obey_upper_triangle_mask():
    image = np.zeros((16, 16))
    image[4, 12] = image[12, 4] = 20
    image[14, 3] = 50  # Excluded lower triangle, irrespective of its strength.
    upper = np.triu(np.ones((16, 16), dtype=bool), k=3)
    peaks = peak_candidates(image, upper)
    assert upper[peaks[0, 0], peaks[0, 1]]


def test_against_replaces_only_reference_metrics_at_matching_centers():
    selected = pd.DataFrame({"center_bp": [10, 20], "prediction_psnr": [2., 3.],
                             "baseline_psnr": [0., 0.]})
    reference = selected.copy()
    reference["prediction_psnr"] = [1., 4.]
    with np.testing.assert_raises(KeyError):
        against(selected, reference)
    for metric in ("ssim", "spearman", "hotspot_error", "top5_error", "top5_hit5"):
        selected[f"prediction_{metric}"] = [2., 3.]
        reference[f"prediction_{metric}"] = [1., 4.]
    pair = against(selected, reference)
    assert pair["baseline_psnr"].tolist() == [1., 4.]
    assert pair["prediction_psnr"].tolist() == [2., 3.]


def test_report_names_reference_and_gain_direction():
    frame = pd.DataFrame({"center_bp": [100, 200]})
    for metric in ("psnr", "ssim", "spearman", "hotspot_error", "top5_error", "top5_hit5"):
        frame[f"baseline_{metric}"] = [1., 1.]
        frame[f"prediction_{metric}"] = [2., 2.]
    report = paired_report(frame, "main CNN", "local fusion")
    assert report["reference_method"] == "main CNN"
    assert report["prediction_method"] == "local fusion"
    assert report["psnr"]["mean_gain"] == 1
    assert report["hotspot_error"]["mean_gain"] == -1
