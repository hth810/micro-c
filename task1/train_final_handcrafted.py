"""Fit the evaluated contact-shape-plus-span classifier on all annotations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from scripts.validate_discovery import shape_features
from task1.evaluate_chid_overlap import spatial_groups
from task1.evaluate_nested_threshold import apply_threshold
from task1.run_task1 import CLASSES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", type=Path, default=Path("outputs/task1_spatial/extracted_windows.npz"))
    ap.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    ap.add_argument("--out", type=Path, default=Path("outputs/task1_handcrafted_final.joblib"))
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()
    ann = pd.read_csv(args.annotations)
    with np.load(args.windows) as bundle:
        x = bundle["X"][:, 0]
        if len(x) != 2 * len(ann) or not np.array_equal(bundle["group"][:len(ann)].astype(str),
                                                        ann.id.to_numpy()):
            raise ValueError("Cached windows do not match annotations")
    shape = np.stack([shape_features(mat) for mat in x]).reshape(2, len(ann), -1).mean(axis=0)
    features = np.column_stack((shape, np.log1p(ann.length_bp.to_numpy(dtype=np.float32))))
    y = ann.type.map({label: i for i, label in enumerate(CLASSES)}).to_numpy()
    counts = np.bincount(y, minlength=3)
    weights = len(y) / (3 * counts[y])
    groups = spatial_groups(ann, 40000)
    out_of_fold_prob = np.full((len(y), 3), np.nan)
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (train, validation) in enumerate(folds.split(features, y, groups)):
        fold_counts = np.bincount(y[train], minlength=3)
        fold_weights = len(train) / (3 * fold_counts[y[train]])
        fold_model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7,
                                                    l2_regularization=1.0,
                                                    random_state=args.seed + fold)
        fold_model.fit(features[train], y[train], sample_weight=fold_weights)
        out_of_fold_prob[validation] = fold_model.predict_proba(features[validation])
    if not np.isfinite(out_of_fold_prob).all():
        raise RuntimeError("Incomplete spatial cross-fitted probabilities")
    thresholds = np.linspace(.1, .9, 17)
    scores = [f1_score(y == 2, apply_threshold(out_of_fold_prob, threshold) == 2,
                       zero_division=0) for threshold in thresholds]
    deployment_threshold = float(thresholds[np.flatnonzero(scores == np.max(scores))[-1]])
    model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7,
                                           l2_regularization=1.0, random_state=args.seed)
    model.fit(features, y, sample_weight=weights)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "classes": CLASSES, "feature_count": int(features.shape[1]),
                 "window_bp": 5120, "contact_replicates": 2,
                 "requires_known_interval_span": True,
                 "chid_cv_f1_threshold": deployment_threshold}, args.out)
    metadata = {"training_annotations": len(ann), "class_counts": dict(zip(CLASSES, counts.tolist())),
                "feature_count": int(features.shape[1]), "requires_known_interval_span": True,
                "chid_cv_f1_threshold": deployment_threshold,
                "threshold_selected_from_training_only": True,
                "evaluation_source": "outputs/task1_handcrafted/summary.json",
                "caution": "Training-fit scores are not a generalization estimate; use spatial CV results."}
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    loaded = joblib.load(args.out)
    assert loaded["model"].predict(features[:2]).shape == (2,)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
