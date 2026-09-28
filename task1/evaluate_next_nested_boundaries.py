"""Nested spatial evaluation of interval-boundary classification choices.

The published 0.787 macro-F1 model was selected retrospectively from outer
cross-validation comparisons. This follow-up makes feature-set and CHID
threshold decisions inside each outer training partition. All 12 plans and
the five 40 kb outer folds are fixed before this run. It is still exploratory:
these same annotations and outer folds have appeared in previous analyses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from scripts.validate_discovery import shape_features
from task1.audit_interval_conditioned_cnn import paired_bootstrap
from task1.evaluate_chid_overlap import spatial_groups
from task1.evaluate_interval_boundaries import interval_features
from task1.evaluate_nested_threshold import apply_threshold, fit_model
from task1.evaluate_pairwise import evaluate
from task1.run_task1 import CLASSES


# Preference on an exact validation tie: fewer features and no threshold
# adjustment. The ordering is fixed and never depends on outer test scores.
MODES = ("short_plus_span", "short_boundary_plus_span", "both_boundary_plus_span")
THRESHOLDS: tuple[float | None, ...] = (None, 0.4, 0.3, 0.2)


def load_features(windows: Path, annotations: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    ann = pd.read_csv(annotations).reset_index(drop=True)
    n = len(ann)
    with np.load(windows) as data:
        x = data["X"].astype(np.float32)
        ids = data["group"].astype(str)
    if x.shape != (2 * n, 2, 128, 128) or not np.array_equal(ids, np.tile(ann.id.to_numpy(), 2)):
        raise ValueError("Cached multiscale windows do not align with annotation IDs")
    shape = np.stack([shape_features(mat) for mat in x.reshape(-1, 128, 128)])
    shape = shape.reshape(2, n, 2, -1).mean(axis=0)
    boundary = np.stack([
        interval_features(x[rep * n + i, 1], int(row.start), int(row.end), int(row.center))
        for rep in (0, 1) for i, row in ann.iterrows()
    ]).reshape(2, n, -1).mean(axis=0)
    span = np.log1p(ann.length_bp.to_numpy(dtype=np.float32))[:, None]
    features = {
        "short_plus_span": np.column_stack((shape[:, 0], span)),
        "short_boundary_plus_span": np.column_stack((shape[:, 0], boundary, span)),
        "both_boundary_plus_span": np.column_stack((shape[:, 0], shape[:, 1], boundary, span)),
    }
    return ann, features


def predict_plan(probabilities: np.ndarray, threshold: float | None) -> np.ndarray:
    if threshold is None:
        return probabilities.argmax(axis=1)
    return apply_threshold(probabilities, threshold)


def select_plan(inner_probabilities: dict[str, np.ndarray], truth: np.ndarray) -> tuple[str, float | None, list[dict]]:
    """Choose only from cross-fitted outer-training predictions."""
    if tuple(inner_probabilities) != MODES:
        raise ValueError("Expected all candidate modes in predeclared order")
    if any(prob.shape != (len(truth), 3) or not np.isfinite(prob).all()
           for prob in inner_probabilities.values()):
        raise ValueError("Incomplete or malformed inner probabilities")
    options = []
    for mode in MODES:
        for threshold in THRESHOLDS:
            pred = predict_plan(inner_probabilities[mode], threshold)
            options.append({"mode": mode, "threshold": threshold,
                            "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
                            "accuracy": float(accuracy_score(truth, pred)),
                            "chid_f1": float(f1_score(truth, pred, labels=[2], average="macro", zero_division=0))})
    selected = max(enumerate(options), key=lambda pair: (pair[1]["macro_f1"],
                                                         pair[1]["accuracy"], -pair[0]))[1]
    return selected["mode"], selected["threshold"], options


def inner_crossfit(features: dict[str, np.ndarray], y: np.ndarray,
                   groups: np.ndarray, outer_train: np.ndarray,
                   seed: int) -> dict[str, np.ndarray]:
    if len(np.unique(groups[outer_train])) < 3:
        raise ValueError("Need at least three spatial groups in outer training")
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    probs = {mode: np.full((len(outer_train), 3), np.nan) for mode in MODES}
    for inner_fold, (fit, val) in enumerate(inner.split(features[MODES[0]][outer_train],
                                                     y[outer_train], groups[outer_train])):
        if np.intersect1d(groups[outer_train[fit]], groups[outer_train[val]]).size:
            raise ValueError("Inner spatial group leakage")
        for mode in MODES:
            model = fit_model(features[mode], y, outer_train[fit], seed + inner_fold)
            probs[mode][val] = model.predict_proba(features[mode][outer_train[val]])
    if any(not np.isfinite(value).all() for value in probs.values()):
        raise ValueError("Incomplete inner spatial crossfit")
    return probs


def run(args: argparse.Namespace) -> dict:
    ann, features = load_features(args.windows, args.annotations)
    y = ann.type.map({name: index for index, name in enumerate(CLASSES)}).to_numpy(dtype=int)
    groups = spatial_groups(ann, args.spatial_gap_bp)
    if args.spatial_gap_bp < 20480:
        raise ValueError("Spatial group gap must span the 20.48 kb image context")
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    folds = list(outer.split(ann, y, groups))
    pred = {"fixed_published": np.full(len(y), -1, dtype=int),
            "nested_selected": np.full(len(y), -1, dtype=int)}
    outer_fold = np.full(len(y), -1, dtype=int)
    fold_rows, option_rows = [], []
    for fold, (train, test) in enumerate(folds):
        if np.intersect1d(groups[train], groups[test]).size:
            raise ValueError("Outer spatial group leakage")
        if np.any(outer_fold[test] != -1):
            raise ValueError("A structure appears in multiple outer test folds")
        outer_fold[test] = fold
        inner_probs = inner_crossfit(features, y, groups, train, args.seed + fold * 100)
        mode, threshold, options = select_plan(inner_probs, y[train])
        option_rows.extend({"fold": fold, **record} for record in options)
        # Fair same-fold comparator. The fixed baseline does not use outer
        # test labels to fit, but was historically chosen post hoc.
        fixed_model = fit_model(features["both_boundary_plus_span"], y, train, args.seed + fold)
        fixed_probs = fixed_model.predict_proba(features["both_boundary_plus_span"][test])
        pred["fixed_published"][test] = fixed_probs.argmax(axis=1)
        if mode == "both_boundary_plus_span":
            selected_probs = fixed_probs
        else:
            selected_model = fit_model(features[mode], y, train, args.seed + fold)
            selected_probs = selected_model.predict_proba(features[mode][test])
        pred["nested_selected"][test] = predict_plan(selected_probs, threshold)
        fold_rows.append({"fold": fold, "selected_mode": mode, "selected_chid_threshold": threshold,
                          "train_structures": len(train), "outer_test_structures": len(test),
                          "outer_test_chid": int((y[test] == 2).sum()),
                          "inner_macro_f1": next(item["macro_f1"] for item in options
                                                 if item["mode"] == mode and item["threshold"] == threshold)})
        print(f"completed outer fold {fold + 1}/5: {mode}, CHID threshold {threshold}", flush=True)
    if np.any(outer_fold < 0) or any(np.any(values < 0) for values in pred.values()):
        raise RuntimeError("Incomplete outer prediction")
    uncertainty = paired_bootstrap(y, groups, pred, "nested_selected", ("fixed_published",),
                                   resamples=args.bootstrap, seed=args.seed + 1)
    summary = {"protocol": "five 40 kb spatial outer folds; each outer train uses three inner spatial folds",
               "caution": "Same published annotations and outer folds have informed earlier work; not independent confirmation",
               "selection": "three predeclared boundary feature sets x four CHID threshold choices; inner training OOF macro F1, then accuracy, then simpler candidate order",
               "modes": MODES, "thresholds": THRESHOLDS, "seed": args.seed,
               "spatial_gap_bp": args.spatial_gap_bp,
               "fixed_published": evaluate(y, pred["fixed_published"]),
               "nested_selected": evaluate(y, pred["nested_selected"]),
               "paired_spatial_uncertainty": uncertainty,
               "selected_counts": {mode: sum(row["selected_mode"] == mode for row in fold_rows)
                                   for mode in MODES},
               "folds": fold_rows}
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(option_rows).to_csv(args.out / "inner_options.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.out / "fold_choices.csv", index=False)
    pd.DataFrame({"id": ann.id, "true_type": ann.type, "chrom": ann.chrom,
                  "center": ann.center, "spatial_group": groups, "outer_fold": outer_fold,
                  **{f"{name}_prediction": [CLASSES[i] for i in values]
                     for name, values in pred.items()}}).to_csv(args.out / "outer_predictions.csv", index=False)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", type=Path, default=Path("outputs/task1_multiscale/windows.npz"))
    ap.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    ap.add_argument("--out", type=Path, default=Path("outputs/task1_next_nested_boundaries"))
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--spatial-gap-bp", type=int, default=40000)
    ap.add_argument("--bootstrap", type=int, default=4000)
    args = ap.parse_args()
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
