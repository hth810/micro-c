from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from scripts.train_sr_hotspot_finetune import (
    annotation_split, choose_stage, peak_density, soft_peak_loss, split_spec,
)


def test_annotation_training_patches_stop_before_validation():
    cooler = SimpleNamespace(chromnames=["MG1655"], binsize=10,
                             chromsizes={"MG1655": 640 * 100})
    ann = pd.DataFrame({"id": ["early", "near", "validation", "test"],
                        "center": [1000, 44000, 45100, 52000]})
    spec = split_spec(cooler, ann, 64, 4, 1200)
    train = annotation_split(ann, spec, "train")
    validation = annotation_split(ann, spec, "validation")
    assert "early" in train.id.tolist()
    assert "near" not in train.id.tolist()
    assert "validation" in validation.id.tolist()
    assert all(train.center + spec["patch_bp"] < spec["centers"][spec["train_end"]])
    assert spec["split_sha256"] == split_spec(cooler, ann, 64, 4, 1200)["split_sha256"]


def test_soft_hotspot_loss_has_finite_gradient():
    truth = np.zeros((1, 16, 16), dtype=np.float32)
    truth[0, 4, 12] = truth[0, 12, 4] = 5
    prob = torch.from_numpy(peak_density(truth))
    assert torch.allclose(prob.sum(dim=1), torch.ones(1))
    pred = torch.zeros((1, 1, 16, 16), requires_grad=True)
    loss = soft_peak_loss(pred, prob)
    loss.backward()
    assert torch.isfinite(loss)
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_selection_rejects_pixel_regression_and_uses_shorter_epoch_tie():
    main = {"stage": "main", "epochs": 0, "tile_psnr": 20., "tile_ssim": .14,
            "known_hotspot_error": 15., "CHIN_hotspot_error": 16.}
    good = {**main, "stage": "epoch2", "epochs": 2,
            "tile_psnr": 19.98, "tile_ssim": .1395,
            "known_hotspot_error": 14., "CHIN_hotspot_error": 15.}
    later = {**good, "stage": "epoch5", "epochs": 5}
    assert choose_stage([main, good, later])["stage"] == "epoch2"
    poor = {**good, "tile_psnr": 19.9}
    assert choose_stage([main, poor])["stage"] == "main"
