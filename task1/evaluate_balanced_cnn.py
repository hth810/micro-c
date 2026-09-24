"""Audit a class-balanced contact-only CNN on fixed 40 kb spatial folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from task1.evaluate_chid_overlap import spatial_groups
from task1.evaluate_pairwise import evaluate, fit_fold
from task1.run_task1 import CLASSES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", type=Path, default=Path("outputs/task1_spatial/extracted_windows.npz"))
    ap.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    ap.add_argument("--out", type=Path, default=Path("outputs/task1_balanced_cnn"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    ann = pd.read_csv(args.annotations).reset_index(drop=True)
    with np.load(args.windows) as bundle:
        x = bundle["X"].astype(np.float32)
        ids = bundle["group"].astype(str)
    if len(x) != 2 * len(ann) or not np.array_equal(ids[:len(ann)], ann.id.to_numpy()):
        raise ValueError("Cached windows do not match annotations")
    y = ann.type.map({name: index for index, name in enumerate(CLASSES)}).to_numpy(dtype=int)
    groups = spatial_groups(ann, 40000)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
                 .split(ann, y, groups))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = np.full((2, len(ann), 3), np.nan, dtype=np.float32)
    for fold, (train, test) in enumerate(folds):
        model = fit_fold(x, y, train, np.empty((0, 2), dtype=int), args.epochs,
                         args.seed + fold, 0.0, device, class_weight_power=1.0)
        with torch.no_grad():
            for replicate in (0, 1):
                tensors = torch.from_numpy(x[test + replicate * len(ann)])
                output = [model(batch.to(device)).cpu().numpy() for batch in tensors.split(64)]
                logits[replicate, test] = np.concatenate(output)
    if not np.isfinite(logits).all():
        raise RuntimeError("Incomplete cross-fitted predictions")
    prediction = logits.mean(axis=0).argmax(axis=1)
    summary = {"model": "contact-only GroupNorm CNN, no annotation span or boundary features",
               "spatial_gap_bp": 40000, "epochs": args.epochs,
               "class_weight_power": 1.0,
               "selection": "fixed inverse-frequency class weighting; no outer-fold threshold tuning",
               **evaluate(y, prediction)}
    rows = [{"id": row.id, "true_type": row.type,
             "predicted_type": CLASSES[prediction[i]], "spatial_group": groups[i],
             **{f"logit_{name}": float(logits[:, i, ci].mean())
                for ci, name in enumerate(CLASSES)}}
            for i, row in ann.iterrows()]
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "crossfit_predictions.csv", index=False)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
