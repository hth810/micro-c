"""Paired block-bootstrap checks for held-out super-resolution predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def paired_summary(frame: pd.DataFrame, unit: str, block_size: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    report = {}
    for metric in ("psnr", "ssim"):
        pairs = frame.pivot(index=unit, columns="method", values=metric).sort_index()
        if unit == "id":
            genomic_order = frame.drop_duplicates("id").set_index("id").center_bp.sort_values().index
            pairs = pairs.loc[genomic_order]
        delta = (pairs["cnn"] - pairs["bilinear"]).to_numpy()
        blocks = [delta[i:i + block_size] for i in range(0, len(delta), block_size)]
        boot = np.empty(5000)
        for i in range(len(boot)):
            sampled = rng.integers(0, len(blocks), len(blocks))
            boot[i] = np.concatenate([blocks[j] for j in sampled]).mean()
        report[metric] = {"n": len(delta), "mean_gain": float(delta.mean()),
                          "median_gain": float(np.median(delta)),
                          "win_fraction": float(np.mean(delta > 0)),
                          "block_bootstrap_95ci": np.quantile(boot, [.025, .975]).tolist()}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    scores = pd.read_csv(args.source / "reconstruction_metrics.csv")
    known = pd.read_csv(args.source / "known_structure_recovery.csv")
    result = {"test": paired_summary(scores, "center_bp", 10, args.seed),
              "known_structures": paired_summary(known, "id", 10, args.seed),
              "note": "Confidence intervals resample contiguous genomic blocks, not pixels; "
                      "the low-resolution input is simulated from the target Micro-C map."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
