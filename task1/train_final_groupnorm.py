"""Fit the cross-validated short-window GroupNorm architecture on all annotations.

This produces a deployment weight file only. Its training-set predictions are
not a performance estimate; use evaluate_multiscale.py for spatial CV results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from task1.evaluate_pairwise import fit_fold
from task1.run_task1 import CLASSES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=Path("outputs/task1_spatial/extracted_windows.npz"))
    parser.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/task1_groupnorm_final.pt"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    ann = pd.read_csv(args.annotations)
    with np.load(args.windows) as bundle:
        x = bundle["X"].astype(np.float32)
        ids = bundle["group"].astype(str)
    if len(x) != 2 * len(ann) or not np.array_equal(ids[:len(ann)], ann.id.to_numpy()):
        raise ValueError("Cached windows do not match annotation order")
    y = ann.type.map({name: index for index, name in enumerate(CLASSES)}).to_numpy(dtype=int)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fit_fold(x, y, np.arange(len(ann)), np.empty((0, 2), dtype=int),
                     args.epochs, args.seed, 0.0, device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.cpu().state_dict(), "classes": CLASSES,
                "architecture": "PairAwareCNN(input_channels=1), GroupNorm, 4x4 spatial pooling",
                "window_bp": 5120, "epochs": args.epochs, "seed": args.seed,
                "performance_estimate": "results/task1_multiscale_eval_summary.json (spatial CV)"}, args.out)
    args.out.with_suffix(".json").write_text(json.dumps({"classes": CLASSES,
        "train_structures": len(ann), "train_samples": len(x),
        "window_bp": 5120, "epochs": args.epochs, "seed": args.seed,
        "performance_estimate": "results/task1_multiscale_eval_summary.json"}, indent=2), encoding="utf-8")
    print(f"Saved final GroupNorm weights to {args.out}; training scores are not reported")


if __name__ == "__main__":
    main()
