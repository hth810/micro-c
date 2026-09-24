"""Nested spatial-CV threshold selection for the rare CHID class."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from scripts.validate_discovery import shape_features
from task1.evaluate_chid_overlap import spatial_groups
from task1.evaluate_pairwise import evaluate
from task1.run_task1 import CLASSES


def build_model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7,
                                          l2_regularization=1.0, random_state=seed)


def fit_model(features: np.ndarray, y: np.ndarray, indices: np.ndarray,
              seed: int) -> HistGradientBoostingClassifier:
    counts = np.bincount(y[indices], minlength=3)
    weights = len(indices) / (3 * counts[y[indices]])
    model = build_model(seed)
    model.fit(features[indices], y[indices], sample_weight=weights)
    return model


def apply_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    prediction = probabilities[:, :2].argmax(axis=1)
    prediction[probabilities[:, 2] >= threshold] = 2
    return prediction


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", type=Path, default=Path("outputs/task1_spatial/extracted_windows.npz"))
    ap.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    ap.add_argument("--out", type=Path, default=Path("outputs/task1_nested_threshold"))
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
    groups = spatial_groups(ann, 40000)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    thresholds = np.linspace(.1, .9, 17)
    predictions = {"default": np.full(len(y), -1, dtype=int),
                   "inner_chid_f1": np.full(len(y), -1, dtype=int)}
    fold_rows = []
    for fold, (train, test) in enumerate(outer.split(ann, y, groups)):
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=args.seed + fold)
        inner_prob = np.full((len(train), 3), np.nan)
        for inner_fold, (it, iv) in enumerate(inner.split(features[train], y[train], groups[train])):
            model = fit_model(features, y, train[it], args.seed + fold * 10 + inner_fold)
            inner_prob[iv] = model.predict_proba(features[train[iv]])
        if not np.isfinite(inner_prob).all():
            raise RuntimeError("Incomplete inner-fold predictions")
        scores = [f1_score(y[train] == 2, apply_threshold(inner_prob, threshold) == 2,
                           zero_division=0) for threshold in thresholds]
        # In exact ties, prefer the more conservative (higher-precision) threshold.
        selected = float(thresholds[np.flatnonzero(scores == np.max(scores))[-1]])
        model = fit_model(features, y, train, args.seed + fold)
        outer_prob = model.predict_proba(features[test])
        predictions["default"][test] = outer_prob.argmax(axis=1)
        predictions["inner_chid_f1"][test] = apply_threshold(outer_prob, selected)
        fold_rows.append({"fold": fold, "threshold_selected_inside_training": selected,
                          "inner_chid_f1": float(max(scores)), "test_structures": len(test),
                          "test_chid": int((y[test] == 2).sum())})
    summary = {"evaluation": "5 outer spatial folds, 3 inner spatial folds for CHID threshold",
               "spatial_gap_bp": 40000, "threshold_grid": thresholds.tolist(),
               "default": evaluate(y, predictions["default"]),
               "inner_chid_f1": evaluate(y, predictions["inner_chid_f1"]),
               "caution": "Exploratory threshold strategy; no outer test label used to choose a fold threshold."}
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(args.out / "fold_thresholds.csv", index=False)
    pd.DataFrame({"id": ann.id, "true_type": ann.type, "spatial_group": groups,
                  **{f"{name}_prediction": [CLASSES[value] for value in pred]
                     for name, pred in predictions.items()}}).to_csv(args.out / "crossfit_predictions.csv", index=False)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
