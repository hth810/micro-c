"""Fit the annotated-interval CNN on all 344 known structures for inference.

The saved weights are *not* a generalization estimate. Use the five spatial
out-of-fold predictions in evaluate_interval_conditioned_cnn.py for metrics.
The model requires a supplied interval (start/end/center), so it cannot scan
for unknown structures without a separate proposal stage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from task1.evaluate_interval_conditioned_cnn import make_modes
from task1.evaluate_pairwise import fit_fold
from task1.run_task1 import CLASSES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", type=Path, default=Path("outputs/task1_multiscale/windows.npz"))
    ap.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/task1_interval_conditioned_cnn/final_annotated_interval_cnn.pt"))
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    ann = pd.read_csv(args.annotations).reset_index(drop=True)
    with np.load(args.windows) as bundle:
        contact = bundle["X"].astype(np.float32)
        ids = bundle["group"].astype(str)
    if len(ann) != 344 or not np.array_equal(ids, np.tile(ann.id.to_numpy(), 2)):
        raise ValueError("Expected both biological replicates of all 344 annotated structures")
    y = ann.type.map({name: i for i, name in enumerate(CLASSES)}).to_numpy(dtype=int)
    x = make_modes(contact, ann)["contact_interval"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fit_fold(x, y, np.arange(len(ann)), np.empty((0, 2), dtype=int),
                     12, args.seed, 0.0, device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "purpose": "inference for supplied annotated intervals only; no in-sample performance claim",
        "classes": CLASSES,
        "model": "PairAwareCNN(input_channels=4), GroupNorm, 4x4 spatial pooling",
        "channel_order": ["5.12 kb contact", "20.48 kb contact", "within-interval square",
                          "interval-boundary-crossing contact indicator"],
        "contact_preprocessing": "per-window log1p then diagonal-median distance normalization, MAD scaling and clipping to [-5,5]",
        "short_window": {"bins": 128, "bin_bp": 40, "context_bp": 5120},
        "long_window": {"bins": 128, "bin_bp": 160, "context_bp": 20480},
        "interval_mask": "start/end relative to supplied center at 160 bp per long-window pixel; square is inside×inside, crossing is XOR",
        "training_structures": len(ann),
        "training_samples": len(x),
        "biological_replicates": 2,
        "epochs": 12,
        "seed": args.seed,
        "class_weight_power": 0.5,
        "generalization_estimate": "results/task1_interval_cnn_summary.json (40 kb five-fold spatial crossfit)",
        "caution": "All annotated labels were used for this weight. Its predictions on them are in-sample, not a held-out score. The shipped architecture was selected retrospectively from the four crossfit comparisons, so that score has selection bias. Interval masks cannot be formed for unknown structures.",
    }
    torch.save({"model_state_dict": model.cpu().state_dict(), "metadata": metadata}, args.out)
    args.out.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"weights": str(args.out), "metadata": str(args.out.with_suffix('.json')),
                      "training_structures": len(ann), "performance_reported": False},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
