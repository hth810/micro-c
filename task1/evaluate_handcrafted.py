"""Spatial-CV baselines using WT contact-shape features and annotation spans."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold

from scripts.validate_discovery import shape_features
from task1.evaluate_chid_overlap import spatial_groups
from task1.evaluate_pairwise import evaluate
from task1.run_task1 import CLASSES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", type=Path, default=Path("outputs/task1_spatial/extracted_windows.npz"))
    ap.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    ap.add_argument("--out", type=Path, default=Path("outputs/task1_handcrafted"))
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--spatial-gap-bp", type=int, default=40000)
    ap.add_argument("--only-combined", action="store_true", help="Evaluate only the prespecified combined HGB model")
    args = ap.parse_args()
    ann = pd.read_csv(args.annotations)
    with np.load(args.windows) as bundle:
        x = bundle["X"][:, 0]
        if len(x) != 2 * len(ann) or not np.array_equal(bundle["group"][:len(ann)].astype(str),
                                                        ann.id.to_numpy()):
            raise ValueError("Cached windows do not match annotations")
    # Two replicate feature vectors become one annotation-level observation.
    shape = np.stack([shape_features(mat) for mat in x]).reshape(2, len(ann), -1).mean(axis=0)
    span = np.log1p(ann.length_bp.to_numpy(dtype=np.float32))[:, None]
    features = {"span_only": span, "contact_shape_only": shape,
                "contact_shape_plus_span": np.column_stack((shape, span))}
    y = ann.type.map({label: i for i, label in enumerate(CLASSES)}).to_numpy()
    groups = spatial_groups(ann, args.spatial_gap_bp)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
                 .split(ann, y, groups))
    rows, summary = [], {"evaluation": "original three-class labels; fixed 5-fold spatial CV",
                         "spatial_gap_bp": args.spatial_gap_bp, "fold_seed": args.seed,
                         "caution": "Exploratory model comparison; span known only for annotated intervals."}
    for feature_name, values in features.items():
        for model_name in ("extra_trees", "hist_gradient_boosting"):
            if args.only_combined and (feature_name, model_name) != (
                "contact_shape_plus_span", "hist_gradient_boosting"
            ):
                continue
            prediction = np.full(len(y), -1, dtype=int)
            for fold, (train, test) in enumerate(folds):
                if model_name == "extra_trees":
                    model = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3,
                                                 class_weight="balanced", random_state=args.seed + fold,
                                                 n_jobs=-1)
                    model.fit(values[train], y[train])
                else:
                    counts = np.bincount(y[train], minlength=3)
                    weights = len(train) / (3 * counts[y[train]])
                    model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7,
                                                           l2_regularization=1.0,
                                                           random_state=args.seed + fold)
                    model.fit(values[train], y[train], sample_weight=weights)
                prediction[test] = model.predict(values[test])
            key = f"{model_name}_{feature_name}"
            summary[key] = evaluate(y, prediction)
            rows.extend({"model": key, "id": ann.iloc[i].id, "true_type": CLASSES[y[i]],
                         "predicted_type": CLASSES[prediction[i]], "spatial_group": groups[i]}
                        for i in range(len(y)))
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "crossfit_predictions.csv", index=False)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
