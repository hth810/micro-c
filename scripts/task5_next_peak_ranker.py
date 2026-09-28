"""Train-only hotspot proposal ranking with a frozen validation gate.

The restored image remains the original main CNN output; this *separate*
detector ranks peaks proposed by the bilinear, main, and peak-weighted maps.
It does not claim to change the image's own argmax. The detector is fitted on
the 70% genomic training tiles, selected on the 10% validation tiles and 28
annotated centers, then tested at fixed coordinates on both 37 C replicates.
No test outcome is used to change the candidate features or selection rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from microc.data import open_cool, window
from microc.deep import prepare_sr_pair
from scripts.evaluate_sr_ensemble import infer, load_model, split_centers
from scripts.evaluate_sr_peak_guard import block_interval, paired_report, peak_candidates, per_patch
from scripts.validate_sr_known_shape import hotspot


FEATURE_NAMES = (
    "row", "col", "offdiag", "center_distance", "edge_distance",
    "base_z", "main_z", "peak_z", "base_peak_gap", "main_peak_gap", "peak_peak_gap",
    "base_local_contrast", "main_local_contrast", "peak_local_contrast",
    "base_top1_distance", "main_top1_distance", "peak_top1_distance",
    "base_vote", "main_vote", "peak_vote",
)


def proposals(base: np.ndarray, main: np.ndarray,
              peak: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unique peak coordinates and image-only features for one patch."""
    n = len(base)
    upper = np.triu(np.ones((n, n), dtype=bool), k=3)
    images = (base, main, peak)
    peaks = [peak_candidates(image, upper, k=5) for image in images]
    coords = np.unique(np.concatenate(peaks, axis=0), axis=0)
    smoothed = [gaussian_filter(image, sigma=2) for image in images]
    broad = [gaussian_filter(image, sigma=5) for image in images]
    top1 = [hotspot(image, upper) for image in images]
    features = []
    for row, col in coords:
        row, col = int(row), int(col)
        position = np.array([row, col], dtype=float)
        item = [row / (n - 1), col / (n - 1), (col - row) / (n - 1),
                np.linalg.norm(position - (n - 1) / 2) / n,
                min(row, col, n - row - 1, n - col - 1) / n]
        item.extend((s[row, col] - np.mean(s[upper])) /
                    max(float(np.std(s[upper])), 1e-6) for s in smoothed)
        item.extend(float(s.max() - s[row, col]) for s in smoothed)
        item.extend(float(s[row, col] - b[row, col]) for s, b in zip(smoothed, broad))
        item.extend(float(np.linalg.norm(position - center) / n) for center in top1)
        item.extend(float(np.any(np.all(source == position, axis=1))) for source in peaks)
        features.append(item)
    matrix = np.asarray(features, dtype=np.float32)
    assert matrix.shape == (len(coords), len(FEATURE_NAMES))
    return coords, matrix


def make_training_data(base: np.ndarray, main: np.ndarray,
                       peak: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    upper = np.triu(np.ones(base.shape[-2:], dtype=bool), k=3)
    features, labels, oracle_distances = [], [], []
    for b, m, p, t in zip(base, main, peak, truth):
        coords, x = proposals(b, m, p)
        distances = np.linalg.norm(coords - hotspot(t, upper), axis=1)
        features.append(x)
        labels.append((distances <= 5).astype(int))
        oracle_distances.append(float(distances.min()))
    x = np.concatenate(features)
    y = np.concatenate(labels)
    if not np.any(y) or np.all(y):
        raise ValueError("Training candidate labels have only one class")
    audit = {"patches": len(base), "candidates": len(y),
             "positive_candidate_fraction": float(y.mean()),
             "oracle_top1_mean_error_bins": float(np.mean(oracle_distances)),
             "oracle_top5_hit5": float(np.mean(np.asarray(oracle_distances) <= 5))}
    return x, y, audit


def fit_rankers(x: np.ndarray, y: np.ndarray) -> dict:
    logistic = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, random_state=20260928))
    boosted = HistGradientBoostingClassifier(
        max_iter=100, learning_rate=.05, max_leaf_nodes=15,
        min_samples_leaf=30, l2_regularization=1.0, random_state=20260928)
    logistic.fit(x, y)
    boosted.fit(x, y)
    return {"logistic": logistic, "boosted": boosted}


def ranked_peaks(base: np.ndarray, main: np.ndarray, peak: np.ndarray,
                 model) -> np.ndarray:
    coords, x = proposals(base, main, peak)
    scores = model.predict_proba(x)[:, 1]
    order = np.argsort(-scores, kind="stable")
    return coords[order[:5]]


def detector_frame(truth: np.ndarray, base: np.ndarray, main: np.ndarray,
                   peak: np.ndarray, centers: np.ndarray,
                   models: dict, types: np.ndarray | None = None) -> pd.DataFrame:
    upper = np.triu(np.ones(base.shape[-2:], dtype=bool), k=3)
    rows = []
    for i, (t, b, m, p) in enumerate(zip(truth, base, main, peak)):
        truth_peak = hotspot(t, upper)
        row = {"center_bp": int(centers[i]),
               "type": str(types[i]) if types is not None else "tile"}
        for name, image in (("bilinear", b), ("main", m), ("peak", p)):
            guesses = peak_candidates(image, upper, k=5)
            distances = np.linalg.norm(guesses - truth_peak, axis=1)
            row[f"{name}_top1_error"] = float(np.linalg.norm(hotspot(image, upper) - truth_peak))
            row[f"{name}_top5_error"] = float(distances.min())
            row[f"{name}_top5_hit5"] = float(distances.min() <= 5)
        for name, model in models.items():
            guesses = ranked_peaks(b, m, p, model)
            distances = np.linalg.norm(guesses - truth_peak, axis=1)
            row[f"{name}_top1_error"] = float(distances[0])
            row[f"{name}_top5_error"] = float(distances.min())
            row[f"{name}_top5_hit5"] = float(distances.min() <= 5)
        rows.append(row)
    return pd.DataFrame(rows)


def choose_model(tiles: pd.DataFrame, known: pd.DataFrame) -> dict:
    """Predeclared validation guard; image output is always the main CNN."""
    baseline_top1 = float(known.bilinear_top1_error.mean())
    baseline_chin = float(known.loc[known.type == "CHIN", "bilinear_top1_error"].mean())
    baseline_tile = float(tiles.bilinear_top1_error.mean())
    baseline_top5 = float(known.bilinear_top5_hit5.mean())
    grid = []
    for name in ("bilinear", "main", "peak", "logistic", "boosted"):
        row = {"name": name,
               "val_known_top1_error": float(known[f"{name}_top1_error"].mean()),
               "val_chin_top1_error": float(known.loc[
                   known.type == "CHIN", f"{name}_top1_error"].mean()),
               "val_tile_top1_error": float(tiles[f"{name}_top1_error"].mean()),
               "val_known_top5_hit5": float(known[f"{name}_top5_hit5"].mean())}
        row["feasible"] = bool(
            row["val_known_top1_error"] < baseline_top1 and
            row["val_chin_top1_error"] <= baseline_chin and
            row["val_tile_top1_error"] <= baseline_tile and
            row["val_known_top5_hit5"] >= baseline_top5)
        grid.append(row)
    eligible = [row for row in grid if row["feasible"]]
    selected = min(eligible, key=lambda row: (row["val_known_top1_error"],
                                               row["val_chin_top1_error"])) if eligible else grid[0]
    return {"grid": grid, "selected": selected}


def paired_summary(frame: pd.DataFrame, method: str, reference: str) -> dict:
    out = {"n": len(frame), "genomic_50kb_blocks": int((frame.center_bp // 50000).nunique())}
    for metric in ("top1_error", "top5_error", "top5_hit5"):
        sign = 1 if metric == "top5_hit5" else -1
        delta = sign * (frame[f"{method}_{metric}"].to_numpy() -
                        frame[f"{reference}_{metric}"].to_numpy())
        out[metric] = {"reference_mean": float(frame[f"{reference}_{metric}"].mean()),
                       "prediction_mean": float(frame[f"{method}_{metric}"].mean()),
                       "mean_gain": float(delta.mean()),
                       "win_fraction": float(np.mean(delta > 0)),
                       "block_bootstrap_95ci_gain": block_interval(
                           delta, frame.center_bp.to_numpy())}
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


FROZEN_VALIDATION_FIELDS = (
    "protocol", "features",
    "train", "validation_tiles", "validation_known", "validation_known_by_type",
    "selection_rule", "selection", "selected_detector",
    "main_checkpoint_sha256", "peak_checkpoint_sha256", "caution",
)
OPTIONAL_FROZEN_VALIDATION_FIELDS = (
    "metric_protocol", "uncertainty_protocol", "genomic_split",
)


def require_frozen_selection(current: dict, artifact: Path) -> tuple[str, str]:
    """Fail before reading test matrices unless the validation run is unchanged."""
    if not artifact.is_file():
        raise FileNotFoundError(
            f"Run --validation-only first; frozen selection is missing: {artifact}")
    frozen = json.loads(artifact.read_text(encoding="utf-8"))
    if not isinstance(frozen, dict):
        raise ValueError("Frozen validation artifact must be a JSON object")
    for field in FROZEN_VALIDATION_FIELDS:
        if field not in frozen or field not in current or frozen[field] != current[field]:
            raise ValueError(f"Frozen validation mismatch in {field}; do not evaluate test data")
    # Older pre-test artifacts contain the core frozen selection above. Newer
    # snapshots additionally lock the expanded metric and split audit fields.
    for field in OPTIONAL_FROZEN_VALIDATION_FIELDS:
        if field in frozen and (field not in current or frozen[field] != current[field]):
            raise ValueError(f"Frozen validation mismatch in {field}; do not evaluate test data")
    selected = frozen["selected_detector"]
    if (selected not in ("bilinear", "main", "peak", "logistic", "boosted") or
            frozen["selection"]["selected"]["name"] != selected):
        raise ValueError("Frozen detector selection is malformed")
    return selected, sha256(artifact)


def overlap_audit(first: np.ndarray, second: np.ndarray, patch_bp: int) -> dict:
    """Count intersecting centered half-open genomic patch intervals."""
    left = np.asarray(first, dtype=int)
    right = np.asarray(second, dtype=int)
    if not len(left) or not len(right):
        return {"first_windows": len(left), "second_windows": len(right),
                "overlap_pairs": 0, "first_windows_overlapping": 0,
                "second_windows_overlapping": 0, "minimum_center_distance_bp": None}
    delta = np.abs(left[:, None] - right[None, :])
    intersects = delta < patch_bp
    return {"first_windows": len(left), "second_windows": len(right),
            "overlap_pairs": int(intersects.sum()),
            "first_windows_overlapping": int(np.any(intersects, axis=1).sum()),
            "second_windows_overlapping": int(np.any(intersects, axis=0).sum()),
            "minimum_center_distance_bp": int(delta.min())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cool", type=Path, required=True)
    ap.add_argument("--external-cool", type=Path)
    ap.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    ap.add_argument("--main-checkpoint", type=Path, default=Path("results/task5_super_resolution.pt"))
    ap.add_argument("--peak-checkpoint", type=Path,
                    default=Path("results/task5_peak_weighted_super_resolution.pt"))
    ap.add_argument("--max-patches", type=int, default=1200)
    ap.add_argument("--validation-only", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("outputs/task5_next_peak_ranker"))
    args = ap.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model_main, check_main = load_model(args.main_checkpoint)
    model_peak, check_peak = load_model(args.peak_checkpoint)
    if (check_main["factor"], check_main["patch_bins"]) != (
            check_peak["factor"], check_peak["patch_bins"]):
        raise ValueError("Checkpoint resolutions differ")
    factor, bins = int(check_main["factor"]), int(check_main["patch_bins"])
    ann = pd.read_csv(args.annotations).sort_values("center")

    def load(cooler, chrom: str, patch_bp: int, locations) -> tuple[np.ndarray, np.ndarray]:
        pairs = [prepare_sr_pair(window(cooler, chrom, int(center), patch_bp, bins), factor)
                 for center in locations]
        return (np.stack([pair[0] for pair in pairs]),
                np.stack([pair[1] for pair in pairs]))

    def predict(base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        main_raw, peak_raw = infer(model_main, base), infer(model_peak, base)
        return (base + float(check_main["blend"]) * (main_raw - base),
                base + float(check_peak["blend"]) * (peak_raw - base))

    with open_cool(args.cool) as cooler:
        chrom = cooler.chromnames[0]
        patch_bp = bins * int(cooler.binsize)
        centers, train_end, test_start = split_centers(
            int(cooler.chromsizes[chrom]), patch_bp, args.max_patches)
        train_centers = np.asarray(centers[:train_end])
        val_centers = np.asarray(centers[train_end:test_start])
        test_centers = np.asarray(centers[test_start:])
        val_ann = ann[(ann.center >= centers[train_end]) &
                      (ann.center < centers[test_start])]
        test_ann = ann[ann.center >= centers[test_start]]
        train_base, train_truth = load(cooler, chrom, patch_bp, train_centers)
        val_base, val_truth = load(cooler, chrom, patch_bp, val_centers)
        val_known_base, val_known_truth = load(cooler, chrom, patch_bp, val_ann.center)
    train_main, train_peak = predict(train_base)
    x, y, train_audit = make_training_data(train_base, train_main, train_peak, train_truth)
    models = fit_rankers(x, y)
    val_main, val_peak = predict(val_base)
    val_known_main, val_known_peak = predict(val_known_base)
    val_tile_frame = detector_frame(val_truth, val_base, val_main, val_peak,
                                    val_centers, models)
    val_known_frame = detector_frame(val_known_truth, val_known_base, val_known_main,
                                     val_known_peak, val_ann.center.to_numpy(), models,
                                     val_ann.type.to_numpy())
    selection = choose_model(val_tile_frame, val_known_frame)
    selected = selection["selected"]["name"]
    summary = {"protocol": "Training-only candidate ranker; validation-only selection; frozen rep1/rep2 test. Detector proposals are distinct from main CNN image argmax.",
               "metric_protocol": "Target hotspot and image top1 are Gaussian(sigma=2 bins) upper-triangle (column-row>=3) argmax. Each image baseline's top5 is its five highest 5x5 local maxima on that same smoothed region. The detector ranks the unique union of bilinear/main/peak-weighted top5 candidates and reports its own top1 and top5; hit5 means any of five candidates within Euclidean 5 bins of target. These detector metrics are not the restored image's hotspot metrics.",
               "uncertainty_protocol": "Paired gain intervals use 5000 bootstrap draws of complete fixed 50-kb genomic blocks, not independent individual windows; intervals are descriptive because only one genomic region and two related biological replicates are available.",
               "features": list(FEATURE_NAMES),
               "train": train_audit,
               "validation_tiles": len(val_base),
               "validation_known": len(val_known_base),
               "validation_known_by_type": val_ann.type.value_counts().to_dict(),
               "selection_rule": "Minimize validation known top1 error subject to CHIN and tile top1 no worse than bilinear, known top5 hit5 no worse than bilinear, and overall known top1 strictly better; else bilinear detector.",
               "selection": selection,
               "selected_detector": selected,
               "genomic_split": {"patch_bp": patch_bp,
                                 "train_tile_centers": len(train_centers),
                                 "validation_tile_centers": len(val_centers),
                                 "test_tile_centers": len(test_centers),
                                 "train_to_validation_tiles": overlap_audit(
                                     train_centers, val_centers, patch_bp),
                                 "train_to_validation_known": overlap_audit(
                                     train_centers, val_ann.center.to_numpy(), patch_bp),
                                 "train_to_test_tiles": overlap_audit(
                                     train_centers, test_centers, patch_bp),
                                 "train_to_test_known": overlap_audit(
                                     train_centers, test_ann.center.to_numpy(), patch_bp),
                                 "validation_tiles_to_test_known": overlap_audit(
                                     val_centers, test_ann.center.to_numpy(), patch_bp),
                                 "validation_known_to_test_known": overlap_audit(
                                     val_ann.center.to_numpy(), test_ann.center.to_numpy(), patch_bp),
                                 "note": "Rep2 reuses identical genomic coordinates intentionally and is a replicate transfer, not independent loci."},
               "main_checkpoint_sha256": sha256(args.main_checkpoint),
               "peak_checkpoint_sha256": sha256(args.peak_checkpoint),
               "caution": "Earlier model development inspected this test genomic segment; results remain exploratory. The restored image is unchanged main CNN; detector hotspot metrics must not be reported as image hotspot metrics. Synthetic low-resolution input from the same Micro-C map."}
    if args.validation_only:
        args.out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(selection["grid"]).to_csv(args.out / "validation_grid.csv", index=False)
        (args.out / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"selected_detector": selected, "train": train_audit,
                          "validation_grid": selection["grid"]}, indent=2))
        return
    selected, summary["pretest_validation_artifact_sha256"] = require_frozen_selection(
        summary, args.out / "validation_summary.json")
    pd.DataFrame(selection["grid"]).to_csv(args.out / "validation_grid.csv", index=False)

    def evaluate(path: Path, label: str) -> None:
        with open_cool(path) as cooler:
            if cooler.chromnames[0] != chrom or bins * int(cooler.binsize) != patch_bp:
                raise ValueError("Replicate chromosome or bin size mismatch")
            tile_base, tile_truth = load(cooler, chrom, patch_bp, test_centers)
            known_base, known_truth = load(cooler, chrom, patch_bp, test_ann.center)
        tile_main, tile_peak = predict(tile_base)
        known_main, known_peak = predict(known_base)
        tile = detector_frame(tile_truth, tile_base, tile_main, tile_peak,
                              test_centers, models)
        known = detector_frame(known_truth, known_base, known_main, known_peak,
                               test_ann.center.to_numpy(), models,
                               test_ann.type.to_numpy())
        tile.to_csv(args.out / f"{label}_tiles.csv", index=False)
        known.to_csv(args.out / f"{label}_known.csv", index=False)
        image_tile = per_patch(tile_truth, tile_base, tile_main, test_centers)
        image_known = per_patch(known_truth, known_base, known_main,
                                test_ann.center.to_numpy(), test_ann.type.to_numpy(),
                                test_ann.id.to_numpy())
        summary[label] = {
            "tiles": paired_summary(tile, selected, "bilinear"),
            "known": paired_summary(known, selected, "bilinear"),
            "known_vs_main_detector": paired_summary(known, selected, "main"),
            "known_by_type": {name: paired_summary(group, selected, "bilinear")
                              for name, group in known.groupby("type")},
            "image_tile_psnr_bilinear": float(image_tile.baseline_psnr.mean()),
            "image_tile_psnr_main": float(image_tile.prediction_psnr.mean()),
            "image_tile_ssim_bilinear": float(image_tile.baseline_ssim.mean()),
            "image_tile_ssim_main": float(image_tile.prediction_ssim.mean()),
            "image_known_psnr_bilinear": float(image_known.baseline_psnr.mean()),
            "image_known_psnr_main": float(image_known.prediction_psnr.mean()),
            "image_known_ssim_bilinear": float(image_known.baseline_ssim.mean()),
            "image_known_ssim_main": float(image_known.prediction_ssim.mean()),
            "image_tiles_paired": paired_report(image_tile, "bilinear image", "main CNN image"),
            "image_known_paired": paired_report(image_known, "bilinear image", "main CNN image"),
        }

    evaluate(args.cool, "rep1_test")
    if args.external_cool:
        evaluate(args.external_cool, "rep2_fixed")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"selected_detector": selected,
                      "rep1_known": summary["rep1_test"]["known"],
                      "rep2_known": summary.get("rep2_fixed", {}).get("known")}, indent=2))


if __name__ == "__main__":
    main()
