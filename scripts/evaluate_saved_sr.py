"""Recompute held-out image metrics from the saved numerical SR predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
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
    known = pd.read_csv(args.bundle / "task5_known_structure_recovery.csv")
    validation = {"test": paired_summary(scores, "center_bp", 10, args.seed),
                  "known_structures": paired_summary(known, "id", 10, args.seed),
                  "note": "Paired 10-window genomic block bootstrap; simulated low-resolution input, not independent Hi-C."}
    (args.bundle / "task5_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
