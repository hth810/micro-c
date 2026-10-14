import numpy as np

from scripts.task5_full_structure_repair_20261014 import affine_fit, calibrate, transformed_images


def test_affine_fit_recovers_training_scale():
    x = np.arange(100, dtype=float) / 20
    parameters = affine_fit(x, .4 * x - .2)
    assert abs(parameters["slope"] - .4) < 1e-10
    assert abs(parameters["intercept"] + .2) < 1e-10


def test_negative_contact_scaling_is_not_permitted():
    parameters = affine_fit(np.arange(20), -np.arange(20))
    assert parameters["slope"] == 0


def test_calibration_uses_only_frozen_parameters_and_image():
    x = np.arange(16).reshape(4, 4).astype(np.float32)
    a = calibrate(x, {"slope": .5, "intercept": -.1})
    np.testing.assert_allclose(a, np.clip(.5*x-.1, -5, 5))


def test_cnn_mixture_changes_actual_image_not_peak_list():
    base = np.ones((8, 8), np.float32)
    cubic = base * 2
    cnn = base * 3
    weights = {name: {"slope": .5, "intercept": 0} for name in ("bilinear", "bicubic", "cnn")}
    images = transformed_images(base, cubic, cnn, weights)
    np.testing.assert_array_equal(images["calibrated_cnn_fraction_0.5"], np.ones((8,8)))
    assert set(images) >= {"train_calibrated_bilinear", "train_calibrated_bicubic", "train_calibrated_cnn"}
