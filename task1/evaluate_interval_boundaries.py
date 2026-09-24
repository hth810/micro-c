"""Test annotation-aligned boundary features against the fixed-window baseline."""
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


def interval_features(mat: np.ndarray, start: int, end: int, center: int) -> np.ndarray:
    """Summarize within/between interval and 1.28 kb flank blocks at 160 bp/bin."""
    left = int(np.clip(np.floor(64 + (start - center) / 160), 0, 128))
    right = int(np.clip(np.ceil(64 + (end - center) / 160), 0, 128))
    if right <= left:
        right = min(128, left + 1)
    bands = (np.arange(max(0, left - 8), left), np.arange(left, right),
             np.arange(right, min(128, right + 8)))
    values = [float(left > 0), float(right < 128)]
    for i, a in enumerate(bands):
        for j in range(i, len(bands)):
            b = bands[j]
            if not len(a) or not len(b):
                values.extend((0.0, 0.0))
                continue
            block = mat[np.ix_(a, b)]
            if i == j:
                block = block[np.triu_indices(len(a), k=2)]
            if not len(block):
                values.extend((0.0, 0.0))
            else:
                values.extend((float(np.mean(block)), float(np.quantile(block, .9))))
    return np.asarray(values, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", type=Path, default=Path("outputs/task1_multiscale/windows.npz"))
    ap.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    ap.add_argument("--out", type=Path, default=Path("outputs/task1_interval_boundaries"))
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--spatial-gap-bp", type=int, default=40000)
    args = ap.parse_args()
    ann = pd.read_csv(args.annotations)
    with np.load(args.windows) as bundle:
        x = bundle["X"]
        if x.shape != (2 * len(ann), 2, 128, 128) or not np.array_equal(
            bundle["group"][:len(ann)].astype(str), ann.id.to_numpy()
        ):
            raise ValueError("Cached windows do not match annotations")
    shape = np.stack([shape_features(mat) for mat in x.reshape(-1, 128, 128)])
    shape = shape.reshape(2, len(ann), 2, -1).mean(axis=0)
    boundary = np.stack([interval_features(x[rep * len(ann) + i, 1],
                                         int(row.start), int(row.end), int(row.center))
                         for rep in (0, 1) for i, row in ann.iterrows()]).reshape(
                             2, len(ann), -1).mean(axis=0)
    span = np.log1p(ann.length_bp.to_numpy(dtype=np.float32))[:, None]
    features = {"short_plus_span": np.column_stack((shape[:, 0], span)),
                "short_boundary_plus_span": np.column_stack((shape[:, 0], boundary, span)),
                "both_boundary_plus_span": np.column_stack((shape[:, 0], shape[:, 1], boundary, span))}
    y = ann.type.map({name: i for i, name in enumerate(CLASSES)}).to_numpy()
    groups = spatial_groups(ann, args.spatial_gap_bp)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
                 .split(ann, y, groups))
    summary = {"seed": args.seed, "spatial_gap_bp": args.spatial_gap_bp,
               "evaluation": "original 3-class labels, fixed 5-fold spatial CV",
               "caution": "Exploratory annotation-aligned features, not usable for unknown intervals."}
    rows = []
    for name, values in features.items():
        prediction = np.full(len(ann), -1, dtype=int)
        for fold, (train, test) in enumerate(folds):
            prediction[test] = fit_model(values, y, train, args.seed + fold).predict(values[test])
        summary[name] = evaluate(y, prediction)
        rows.extend({"mode": name, "id": ann.iloc[i].id, "true_type": ann.iloc[i].type,
                     "predicted_type": CLASSES[prediction[i]], "spatial_group": groups[i]}
                    for i in range(len(ann)))
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "crossfit_predictions.csv", index=False)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
