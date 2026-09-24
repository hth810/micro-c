"""Spatial-CV comparison of short-context and equal-weight two-scale voting."""
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
    ap.add_argument("--out", type=Path, default=Path("outputs/task1_scale_ensemble"))
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
    span = np.log1p(ann.length_bp.to_numpy(dtype=np.float32))[:, None]
    short = np.column_stack((shape[:, 0], span))
    long = np.column_stack((shape[:, 1], span))
    y = ann.type.map({name: i for i, name in enumerate(CLASSES)}).to_numpy()
    groups = spatial_groups(ann, args.spatial_gap_bp)
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    probabilities = {name: np.full((len(ann), 3), np.nan) for name in ("short", "long")}
    for fold, (train, test) in enumerate(folds.split(ann, y, groups)):
        for name, values in (("short", short), ("long", long)):
            model = fit_model(values, y, train, args.seed + fold)
            probabilities[name][test] = model.predict_proba(values[test])
    if any(not np.isfinite(value).all() for value in probabilities.values()):
        raise RuntimeError("Incomplete cross-fitted probabilities")
    modes = {"short": probabilities["short"], "long": probabilities["long"],
             "equal_vote": (probabilities["short"] + probabilities["long"]) / 2}
    summary = {"spatial_gap_bp": args.spatial_gap_bp, "seed": args.seed,
               "evaluation": "original 3-class labels, fixed 5-fold spatial CV",
               "caution": "Exploratory ensemble comparison; equal weights fixed without fold tuning."}
    rows = []
    for name, scores in modes.items():
        prediction = scores.argmax(axis=1)
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
