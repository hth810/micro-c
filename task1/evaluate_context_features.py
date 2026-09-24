"""Explore whether independent 20.48 kb contact context aids known-interval classification.

All variants use the same spatial folds and fixed gradient-boosting settings.
The existing short-context plus span classifier is the reference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from scripts.validate_discovery import shape_features
from task1.evaluate_chid_overlap import spatial_groups
from task1.evaluate_nested_threshold import fit_model
from task1.evaluate_pairwise import evaluate
from task1.run_task1 import CLASSES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", type=Path, default=Path("outputs/task1_multiscale/windows.npz"))
    ap.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    ap.add_argument("--out", type=Path, default=Path("outputs/task1_context_features"))
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--spatial-gap-bp", type=int, default=40000)
    ap.add_argument("--only-both", action="store_true", help="evaluate only the two-scale plus span model")
    args = ap.parse_args()
    ann = pd.read_csv(args.annotations)
    with np.load(args.windows) as bundle:
        x = bundle["X"]
        if x.shape != (2 * len(ann), 2, 128, 128) or not np.array_equal(
            bundle["group"][:len(ann)].astype(str), ann.id.to_numpy()
        ):
            raise ValueError("Cached multiscale windows do not match annotations")
    shape = np.stack([shape_features(mat) for mat in x.reshape(-1, 128, 128)])
    shape = shape.reshape(2, len(ann), 2, -1)
    short = shape[:, :, 0].mean(axis=0)
    long = shape[:, :, 1].mean(axis=0)
    short_repeat_difference = np.abs(shape[0, :, 0] - shape[1, :, 0])
    long_repeat_difference = np.abs(shape[0, :, 1] - shape[1, :, 1])
    span = np.log1p(ann.length_bp.to_numpy(dtype=np.float32))[:, None]
    features = {
        "short_plus_span": np.column_stack((short, span)),
        "long_plus_span": np.column_stack((long, span)),
        "both_scales_plus_span": np.column_stack((short, long, span)),
        "both_scales_repeat_stability_plus_span": np.column_stack((short, long,
            short_repeat_difference, long_repeat_difference, span)),
    }
    y = ann.type.map({label: i for i, label in enumerate(CLASSES)}).to_numpy()
    groups = spatial_groups(ann, args.spatial_gap_bp)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
                 .split(ann, y, groups))
    summary = {"spatial_gap_bp": args.spatial_gap_bp, "seed": args.seed,
               "evaluation": "original 3-class labels, fixed 5-fold spatial CV",
               "caution": "Exploratory feature comparison, not independent external validation."}
    rows = []
    for name, values in features.items():
        if args.only_both and name != "both_scales_plus_span":
            continue
        predicted = np.full(len(ann), -1, dtype=int)
        for fold, (train, test) in enumerate(folds):
            model = fit_model(values, y, train, args.seed + fold)
            predicted[test] = model.predict(values[test])
        summary[name] = evaluate(y, predicted)
        rows.extend({"mode": name, "id": ann.iloc[i].id, "true_type": ann.iloc[i].type,
                     "predicted_type": CLASSES[predicted[i]], "spatial_group": groups[i]}
                    for i in range(len(ann)))
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "crossfit_predictions.csv", index=False)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
