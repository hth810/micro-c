"""Fit the evaluated two-scale annotation-boundary classifier on all known loci."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from scripts.validate_discovery import shape_features
from task1.evaluate_interval_boundaries import interval_features
from task1.run_task1 import CLASSES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", type=Path, default=Path("outputs/task1_multiscale/windows.npz"))
    ap.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    ap.add_argument("--out", type=Path, default=Path("outputs/task1_boundary_final.joblib"))
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()
    ann = pd.read_csv(args.annotations)
    with np.load(args.windows) as bundle:
        x = bundle["X"]
        if x.shape != (2 * len(ann), 2, 128, 128) or not np.array_equal(
            bundle["group"][:len(ann)].astype(str), ann.id.to_numpy()
        ):
            raise ValueError("Cached multiscale windows do not match annotations")
    shape = np.stack([shape_features(mat) for mat in x.reshape(-1, 128, 128)])
    shape = shape.reshape(2, len(ann), 2, -1).mean(axis=0)
    boundary = np.stack([interval_features(x[rep * len(ann) + i, 1],
                                         int(row.start), int(row.end), int(row.center))
                         for rep in (0, 1) for i, row in ann.iterrows()]).reshape(
                             2, len(ann), -1).mean(axis=0)
    values = np.column_stack((shape[:, 0], shape[:, 1], boundary,
                              np.log1p(ann.length_bp.to_numpy(dtype=np.float32))))
    y = ann.type.map({name: i for i, name in enumerate(CLASSES)}).to_numpy()
    counts = np.bincount(y, minlength=3)
    weights = len(y) / (3 * counts[y])
    model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7,
                                           l2_regularization=1.0, random_state=args.seed)
    model.fit(values, y, sample_weight=weights)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "classes": CLASSES, "feature_count": values.shape[1],
                 "feature_set": "short_long_boundary_span", "short_aggregate": 4,
                 "long_aggregate": 16, "window_bins": 128,
                 "requires_known_interval_boundaries": True}, args.out)
    metadata = {"training_annotations": len(ann), "class_counts": dict(zip(CLASSES, counts.tolist())),
                "feature_count": int(values.shape[1]), "feature_set": "short_long_boundary_span",
                "requires_known_interval_boundaries": True,
                "evaluation_source": "outputs/task1_interval_boundaries/summary.json",
                "caution": "All-label fit is for inference only; spatial CV estimates generalization."}
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    assert joblib.load(args.out)["model"].predict(values[:2]).shape == (2,)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
