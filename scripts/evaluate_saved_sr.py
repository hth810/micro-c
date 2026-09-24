"""Recompute held-out image metrics from the saved numerical SR predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity

from microc.deep import _psnr
from scripts.validate_superresolution import paired_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    data = np.load(args.bundle / "task5_reconstructed_test_patches.npz")
    rows = []
    for center, truth, baseline, cnn in zip(data["center_bp"], data["truth"],
                                             data["bilinear"], data["cnn"]):
        for name, image in (("bilinear", baseline), ("cnn", cnn)):
            rows.append({"center_bp": int(center), "method": name,
                         "psnr": _psnr(truth, image),
                         "ssim": float(structural_similarity(truth, image, data_range=10))})
    scores = pd.DataFrame(rows)
    scores.to_csv(args.bundle / "task5_reconstruction_metrics.csv", index=False)
    upper = np.triu_indices(data["truth"].shape[-1], k=3)
    shape_rows = []
    uninformative = 0
    informative_indices = []
    for index, (center, truth, baseline, cnn) in enumerate(zip(
            data["center_bp"], data["truth"], data["bilinear"], data["cnn"])):
        target = truth[upper]
        if np.std(target) < 1e-8:
            uninformative += 1
            continue
        informative_indices.append(index)
        for name, image in (("bilinear", baseline), ("cnn", cnn)):
            values = image[upper]
            rho = 0.0 if np.std(values) < 1e-8 else float(spearmanr(target, values).statistic)
            shape_rows.append({"center_bp": int(center), "method": name,
                               "upper_contact_spearman": rho})
    shape = pd.DataFrame(shape_rows)
    shape.to_csv(args.bundle / "task5_shape_validation.csv", index=False)
    pivot = shape.pivot(index="center_bp", columns="method", values="upper_contact_spearman").sort_index()
    gain = (pivot["cnn"] - pivot["bilinear"]).to_numpy()
    block_ids = np.asarray(informative_indices) // 10
    blocks = [gain[block_ids == block] for block in np.unique(block_ids)]
    rng = np.random.default_rng(args.seed)
    bootstrap = np.empty(5000)
    for i in range(len(bootstrap)):
        chosen = rng.integers(0, len(blocks), len(blocks))
        bootstrap[i] = np.concatenate([blocks[j] for j in chosen]).mean()
    shape_validation = {
        "informative_test_windows": len(pivot),
        "zero_signal_windows_excluded": uninformative,
        "bilinear_mean_upper_contact_spearman": float(pivot["bilinear"].mean()),
        "cnn_mean_upper_contact_spearman": float(pivot["cnn"].mean()),
        "mean_paired_gain": float(gain.mean()),
        "win_fraction": float(np.mean(gain > 0)),
        "genomic_block_bootstrap_95ci": np.quantile(bootstrap, [.025, .975]).tolist(),
        "note": "Shape-focused contact-rank check on upper triangle k>=3; this is not external Hi-C validation."
    }
    (args.bundle / "task5_shape_validation.json").write_text(
        json.dumps(shape_validation, indent=2), encoding="utf-8")
    known = pd.read_csv(args.bundle / "task5_known_structure_recovery.csv")
    validation = {"test": paired_summary(scores, "center_bp", 10, args.seed),
                  "known_structures": paired_summary(known, "id", 10, args.seed),
                  "note": "Paired 10-window genomic block bootstrap; simulated low-resolution input, not independent Hi-C."}
    (args.bundle / "task5_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))
    print(json.dumps(shape_validation, indent=2))


if __name__ == "__main__":
    main()
