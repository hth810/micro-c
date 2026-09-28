"""Same-fold shortcut, overlap, and paired uncertainty audit for Task 1 CNNs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from task1.evaluate_chid_overlap import spatial_groups
from task1.evaluate_nested_threshold import fit_model
from task1.evaluate_pairwise import evaluate
from task1.run_task1 import CLASSES


def score_matrices(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Metrics from one or many 3x3 confusion matrices."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("Expected 3x3 confusion matrices")
    tp = np.diagonal(matrix, axis1=-2, axis2=-1)
    true = matrix.sum(axis=-1)
    predicted = matrix.sum(axis=-2)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.divide(2 * tp, true + predicted, out=np.zeros_like(tp), where=(true + predicted) > 0)
        precision = np.divide(tp[..., 2], predicted[..., 2],
                              out=np.zeros_like(tp[..., 2]), where=predicted[..., 2] > 0)
        recall = np.divide(tp[..., 2], true[..., 2],
                           out=np.zeros_like(tp[..., 2]), where=true[..., 2] > 0)
        accuracy = np.divide(tp.sum(axis=-1), matrix.sum(axis=(-2, -1)),
                             out=np.zeros_like(precision), where=matrix.sum(axis=(-2, -1)) > 0)
    return {"accuracy": accuracy, "macro_f1": f1.mean(axis=-1),
            "chid_precision": precision, "chid_recall": recall}


def group_confusions(truth: np.ndarray, pred: np.ndarray,
                     group: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique, inverse = np.unique(group, return_inverse=True)
    if len(truth) != len(pred) or len(pred) != len(group):
        raise ValueError("Predictions and groups are not aligned")
    matrices = np.zeros((len(unique), 3, 3), dtype=np.int32)
    np.add.at(matrices, (inverse, truth, pred), 1)
    return unique, matrices


def paired_bootstrap(truth: np.ndarray, group: np.ndarray,
                     predictions: dict[str, np.ndarray], base: str,
                     comparisons: tuple[str, ...], resamples: int, seed: int) -> dict:
    if base not in predictions or any(name not in predictions for name in comparisons):
        raise ValueError("Missing comparison mode")
    unique, first = group_confusions(truth, predictions[base], group)
    matrix = {base: first}
    for mode in comparisons:
        check, matrix[mode] = group_confusions(truth, predictions[mode], group)
        if not np.array_equal(unique, check):
            raise ValueError("Spatial groups differ between modes")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(unique), size=(resamples, len(unique)))
    scores = {mode: score_matrices(values[sampled].sum(axis=1)) for mode, values in matrix.items()}
    points = {mode: score_matrices(values.sum(axis=0)) for mode, values in matrix.items()}
    out = {}
    for mode in comparisons:
        out[f"{base}_minus_{mode}"] = {
            "point_difference": {key: float(points[base][key] - points[mode][key]) for key in points[base]},
            "paired_spatial_bootstrap_95ci": {
                key: np.quantile(scores[base][key] - scores[mode][key], [.025, .975]).tolist()
                for key in points[base]}}
    return out


def nearest_opposite(ann: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Nearest CHIN to each CHID and nearest CHID to each CHIN, in bp."""
    chid = ann.type.eq("CHID").to_numpy()
    chin = ann.type.eq("CHIN").to_numpy()
    d_chid_to_chin = np.full(len(ann), np.inf)
    d_chin_to_chid = np.full(len(ann), np.inf)
    for _, positions in ann.groupby("chrom").indices.items():
        positions = np.asarray(positions)
        c_pos = positions[chid[positions]]
        h_pos = positions[chin[positions]]
        if len(c_pos) and len(h_pos):
            centers = ann.center.to_numpy()
            distance = np.abs(centers[c_pos, None] - centers[h_pos][None, :])
            d_chid_to_chin[c_pos] = distance.min(axis=1)
            d_chin_to_chid[h_pos] = distance.min(axis=0)
    return d_chid_to_chin, d_chin_to_chid


def overlap_audit(ann: pd.DataFrame, predictions: dict[str, np.ndarray]) -> dict:
    chid = ann.type.eq("CHID").to_numpy()
    chin = ann.type.eq("CHIN").to_numpy()
    d_chid, d_chin = nearest_opposite(ann)
    out = {}
    for threshold in (1000, 5120):
        near_chid = chid & (d_chid < threshold)
        near_chin = chin & (d_chin < threshold)
        result = {"threshold_bp": threshold,
                  "chid_near_count": int(near_chid.sum()),
                  "chid_far_count": int((chid & ~near_chid).sum()),
                  "chin_near_count": int(near_chin.sum()),
                  "chin_far_count": int((chin & ~near_chin).sum()),
                  "modes": {}}
        for mode, pred in predictions.items():
            result["modes"][mode] = {
                "chid_near_true_positive": int(((pred == 2) & near_chid).sum()),
                "chid_far_true_positive": int(((pred == 2) & chid & ~near_chid).sum()),
                "chin_near_false_chid": int(((pred == 2) & near_chin).sum()),
                "chin_far_false_chid": int(((pred == 2) & chin & ~near_chin).sum()),
            }
        out[f"within_{threshold}_bp"] = result
    return out


def run(args: argparse.Namespace) -> dict:
    ann = pd.read_csv(args.annotations).reset_index(drop=True)
    saved = pd.read_csv(args.predictions)
    y = ann.type.map({name: i for i, name in enumerate(CLASSES)}).to_numpy(dtype=int)
    groups = spatial_groups(ann, 40000)
    expected = set((mode, identifier) for mode in (
        "contact_short", "contact_dual", "contact_interval", "interval_only") for identifier in ann.id)
    if set(zip(saved["mode"], saved.id)) != expected or len(saved) != len(expected):
        raise ValueError("Expected exactly four full out-of-fold prediction modes")
    predictions = {}
    for mode, frame in saved.groupby("mode"):
        frame = frame.set_index("id").loc[ann.id].reset_index()
        if not np.array_equal(frame.true_type, ann.type) or not np.array_equal(frame.spatial_group, groups):
            raise ValueError(f"Misaligned predictions in {mode}")
        predictions[mode] = frame.predicted_type.map(
            {name: i for i, name in enumerate(CLASSES)}).to_numpy(dtype=int)
    # A shape-free annotation-length baseline, using the same established
    # classifier and identical 40 kb folds as the boundary-feature study.
    length = np.log1p(ann.length_bp.to_numpy(dtype=np.float32))[:, None]
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260921)
    length_pred = np.full(len(ann), -1, dtype=int)
    for fold, (train, test) in enumerate(folds.split(ann, y, groups)):
        length_pred[test] = fit_model(length, y, train, 20260921 + fold).predict(length[test])
    if (length_pred < 0).any():
        raise RuntimeError("Incomplete length-only predictions")
    predictions["length_only"] = length_pred
    hybrid = pd.read_csv(args.hybrid_predictions)
    hybrid = hybrid[hybrid["mode"] == "both_boundary_plus_span"].set_index("id").loc[ann.id].reset_index()
    if not np.array_equal(hybrid.true_type, ann.type) or not np.array_equal(hybrid.spatial_group, groups):
        raise ValueError("Hybrid comparator uses different annotations or folds")
    predictions["existing_hybrid"] = hybrid.predicted_type.map(
        {name: i for i, name in enumerate(CLASSES)}).to_numpy(dtype=int)
    result = {
        "protocol": "same 40 kb spatial folds and labels for all modes, both replicates combined at structure level",
        "caution": "exploratory reuse of previously analyzed annotations; interval channels and length are not available to an unannotated discovery scan",
        "scores": {mode: evaluate(y, pred) for mode, pred in predictions.items()},
        "chid_chin_overlap": overlap_audit(ann, predictions),
        "paired_uncertainty": paired_bootstrap(y, groups, predictions, "contact_interval",
            ("contact_short", "contact_dual", "interval_only", "length_only", "existing_hybrid"),
            args.resamples, args.seed),
        "uncertainty_method": f"{args.resamples} paired resamples of nonoverlapping 40 kb spatial components; exploratory 95% percentile intervals",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path,
                    default=Path("outputs/task1_interval_conditioned_cnn/crossfit_predictions.csv"))
    ap.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    ap.add_argument("--hybrid-predictions", type=Path,
                    default=Path("results/task1_interval_boundaries_crossfit_predictions.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/task1_interval_conditioned_cnn/audit.json"))
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260928)
    result = run(ap.parse_args())
    print(json.dumps({"scores": result["scores"],
                      "paired_uncertainty": result["paired_uncertainty"]}, indent=2))


if __name__ == "__main__":
    main()
