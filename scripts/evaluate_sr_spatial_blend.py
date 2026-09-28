"""Validation-select a spatial blend of existing super-resolution predictions.

The peak-weighted network can move hotspots while the ordinary network gives
better image-wide SSIM. A local blend applies peak-weighted predictions near a
peak found *without truth*, retaining the ordinary prediction elsewhere. All
candidate choices are fixed using the 10% genomic validation segment. The
20% test segment is read only after the selection has been made. This remains
exploratory: previous development already inspected this same test segment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity

from microc.data import open_cool, window
from microc.deep import _psnr, prepare_sr_pair
from scripts.evaluate_sr_ensemble import infer, load_model, split_centers
from scripts.validate_sr_known_shape import hotspot


def peak_mask(image: np.ndarray, sigma: float, upper: np.ndarray) -> np.ndarray:
    """Symmetric Gaussian ROI around the unlabelled image's strongest hotspot."""
    row, col = hotspot(image, upper).astype(int)
    yy, xx = np.indices(image.shape)
    a = np.exp(-((yy - row) ** 2 + (xx - col) ** 2) / (2 * sigma ** 2))
    b = np.exp(-((yy - col) ** 2 + (xx - row) ** 2) / (2 * sigma ** 2))
    return np.maximum(a, b).astype(np.float32)


def predict(base: np.ndarray, main: np.ndarray, peak: np.ndarray,
            method: str, weight: float, sigma: float | None = None) -> np.ndarray:
    """Apply a fixed global or local prediction blend; no target is consulted."""
    if method == "global":
        return main + weight * (peak - main)
    if method not in {"peak_roi", "base_roi"} or sigma is None:
        raise ValueError("Unknown spatial blend method")
    upper = np.triu(np.ones(base.shape[-2:], dtype=bool), k=3)
    cue = peak if method == "peak_roi" else base
    masks = np.stack([peak_mask(image, sigma, upper) for image in cue])
    return main + weight * masks * (peak - main)


def image_metrics(truth: np.ndarray, base: np.ndarray,
                  pred: np.ndarray) -> dict[str, float]:
    return {
        "psnr_baseline": float(np.mean([_psnr(t, b) for t, b in zip(truth, base)])),
        "psnr_prediction": float(np.mean([_psnr(t, p) for t, p in zip(truth, pred)])),
        "ssim_baseline": float(np.mean([structural_similarity(t, b, data_range=10)
                                          for t, b in zip(truth, base)])),
        "ssim_prediction": float(np.mean([structural_similarity(t, p, data_range=10)
                                            for t, p in zip(truth, pred)])),
    }


def known_metrics(truth: np.ndarray, base: np.ndarray,
                  pred: np.ndarray) -> dict[str, float]:
    base_errors, pred_errors = known_error_arrays(truth, base, pred)
    return {
        "hotspot_baseline_bins": float(base_errors.mean()),
        "hotspot_prediction_bins": float(pred_errors.mean()),
        "hotspot_win_fraction": float(np.mean(pred_errors < base_errors)),
        "hotspot_tie_fraction": float(np.mean(pred_errors == base_errors)),
        "n_known": int(len(truth)),
    }


def known_error_arrays(truth: np.ndarray, base: np.ndarray,
                       pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    upper = np.triu(np.ones(base.shape[-2:], dtype=bool), k=3)
    target = np.stack([hotspot(image, upper) for image in truth])
    base_peaks = np.stack([hotspot(image, upper) for image in base])
    pred_peaks = np.stack([hotspot(image, upper) for image in pred])
    base_errors = np.linalg.norm(base_peaks - target, axis=1)
    pred_errors = np.linalg.norm(pred_peaks - target, axis=1)
    return base_errors, pred_errors


def bootstrap_hotspot_gain(base_errors: np.ndarray, pred_errors: np.ndarray,
                           seed: int = 20260928, samples: int = 10000) -> list[float]:
    """Descriptive site-bootstrap interval; annotations are not independent draws."""
    gain = base_errors - pred_errors
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(gain), size=(samples, len(gain)))
    return [float(x) for x in np.quantile(gain[indices].mean(axis=1), [.025, .975])]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def type_breakdown(frame: pd.DataFrame) -> dict[str, dict]:
    names = ("hotspot_baseline_bins", "hotspot_main_bins",
             "hotspot_peak_bins", "hotspot_prediction_bins")
    return {name: {"n": int(len(group)),
                   **{column: float(group[column].mean()) for column in names}}
            for name, group in frame.groupby("type")}


def candidates() -> list[dict]:
    options = [{"method": "global", "weight": weight, "sigma": None}
               for weight in (0.0, .25, .5, .75, 1.0)]
    options.extend({"method": method, "weight": weight, "sigma": sigma}
                   for method in ("peak_roi", "base_roi")
                   for weight in (.5, 1.0)
                   for sigma in (3.0, 6.0, 12.0))
    return options


def choose_candidate(rows: list[dict]) -> dict:
    """Hotspot-first selection with predeclared image-quality constraints."""
    feasible = [row for row in rows
                if row["val_tile_ssim_prediction"] >= row["val_tile_ssim_baseline"] + .005
                and row["val_tile_psnr_prediction"] > row["val_tile_psnr_baseline"]
                and row["val_known_hotspot_prediction_bins"] < row["val_known_hotspot_baseline_bins"]]
    if not feasible:
        # Preserve the validated pixel-quality model if no candidate satisfies
        # both objectives; do not select a method by looking at test outcomes.
        return next(row for row in rows if row["method"] == "global" and row["weight"] == 0.0)
    return min(feasible, key=lambda row: (row["val_known_hotspot_prediction_bins"],
                                          -row["val_tile_ssim_prediction"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cool", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    ap.add_argument("--main-checkpoint", type=Path, default=Path("results/task5_super_resolution.pt"))
    ap.add_argument("--peak-checkpoint", type=Path,
                    default=Path("results/task5_peak_weighted_super_resolution.pt"))
    ap.add_argument("--main-patches", type=Path,
                    default=Path("results/task5_reconstructed_test_patches.npz"))
    ap.add_argument("--peak-patches", type=Path,
                    default=Path("outputs/task5_peak_weighted/reconstructed_test_patches.npz"))
    ap.add_argument("--max-patches", type=int, default=1200)
    ap.add_argument("--external-cool", type=Path,
                    help="optional untouched replicate used only for post-selection evaluation")
    ap.add_argument("--out", type=Path, default=Path("outputs/task5_spatial_blend"))
    args = ap.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    main_model, main_check = load_model(args.main_checkpoint)
    peak_model, peak_check = load_model(args.peak_checkpoint)
    if (main_check["factor"], main_check["patch_bins"]) != (
            peak_check["factor"], peak_check["patch_bins"]):
        raise ValueError("Checkpoints use different resolutions")
    factor, patch_bins = int(main_check["factor"]), int(main_check["patch_bins"])
    ann = pd.read_csv(args.annotations).sort_values("center")
    with open_cool(args.cool) as cooler:
        chrom = cooler.chromnames[0]
        patch_bp = patch_bins * cooler.binsize
        centers, train_end, test_start = split_centers(
            int(cooler.chromsizes[chrom]), patch_bp, args.max_patches)
        val_ann = ann[(ann.center >= centers[train_end]) &
                      (ann.center < centers[test_start])]
        test_ann = ann[ann.center >= centers[test_start]]
        if val_ann.empty or test_ann.empty:
            raise ValueError("Both validation and test segments need annotations")

        def load_patches(locations: list[int]) -> tuple[np.ndarray, np.ndarray]:
            pairs = [prepare_sr_pair(window(cooler, chrom, center, patch_bp, patch_bins), factor)
                     for center in locations]
            return (np.stack([pair[0] for pair in pairs]),
                    np.stack([pair[1] for pair in pairs]))

        val_tile_base, val_tile_truth = load_patches(centers[train_end:test_start])
        val_known_base, val_known_truth = load_patches(val_ann.center.astype(int).tolist())

    def model_predictions(base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        main_raw, peak_raw = infer(main_model, base), infer(peak_model, base)
        return (base + float(main_check["blend"]) * (main_raw - base),
                base + float(peak_check["blend"]) * (peak_raw - base))

    val_tile_main, val_tile_peak = model_predictions(val_tile_base)
    val_known_main, val_known_peak = model_predictions(val_known_base)
    grid = []
    for option in candidates():
        tile_pred = predict(val_tile_base, val_tile_main, val_tile_peak, **option)
        known_pred = predict(val_known_base, val_known_main, val_known_peak, **option)
        tile = image_metrics(val_tile_truth, val_tile_base, tile_pred)
        known = known_metrics(val_known_truth, val_known_base, known_pred)
        grid.append({**option, **{f"val_tile_{key}": value for key, value in tile.items()},
                     **{f"val_known_{key}": value for key, value in known.items()}})
    selected = choose_candidate(grid)
    settings = {key: selected[key] for key in ("method", "weight", "sigma")}
    # Only the validation-selected configuration is evaluated on held-out data.
    with open_cool(args.cool) as cooler:
        def test_patches(locations: list[int]) -> tuple[np.ndarray, np.ndarray]:
            pairs = [prepare_sr_pair(window(cooler, chrom, center, patch_bp, patch_bins), factor)
                     for center in locations]
            return (np.stack([pair[0] for pair in pairs]),
                    np.stack([pair[1] for pair in pairs]))

        test_known_base, test_known_truth = test_patches(
            test_ann.center.astype(int).tolist())
        test_tile_base, test_tile_truth = test_patches(centers[test_start:])
    test_known_main, test_known_peak = model_predictions(test_known_base)
    test_tile_main, test_tile_peak = model_predictions(test_tile_base)
    test_known_pred = predict(test_known_base, test_known_main, test_known_peak, **settings)
    test_tile_pred = predict(test_tile_base, test_tile_main, test_tile_peak, **settings)
    test_tile_metrics = image_metrics(test_tile_truth, test_tile_base, test_tile_pred)
    with np.load(args.main_patches) as main_bundle, np.load(args.peak_patches) as peak_bundle:
        for key in ("center_bp", "truth", "bilinear"):
            if not np.array_equal(main_bundle[key], peak_bundle[key]):
                raise ValueError(f"Test patch bundles differ in {key}")
        if not np.array_equal(main_bundle["center_bp"], np.asarray(centers[test_start:])):
            raise ValueError("Saved test patch centers do not match the genomic split")
        saved_audit = {
            "main_checkpoint_sha256": sha256(args.main_checkpoint),
            "peak_checkpoint_sha256": sha256(args.peak_checkpoint),
            "main_bundle_sha256": sha256(args.main_patches),
            "peak_bundle_sha256": sha256(args.peak_patches),
            "truth_exact_match": bool(np.array_equal(main_bundle["truth"], test_tile_truth)),
            "baseline_exact_match": bool(np.array_equal(main_bundle["bilinear"], test_tile_base)),
            "main_prediction_max_abs_difference": float(np.max(np.abs(
                main_bundle["cnn"] - test_tile_main))),
            "peak_prediction_max_abs_difference": float(np.max(np.abs(
                peak_bundle["cnn"] - test_tile_peak))),
            "note": "All reported test tile metrics use fresh inference from the listed checkpoints, not these saved bundles.",
        }
    test_known_metrics = known_metrics(test_known_truth, test_known_base, test_known_pred)
    test_base_errors, test_pred_errors = known_error_arrays(
        test_known_truth, test_known_base, test_known_pred)
    _, test_main_errors = known_error_arrays(
        test_known_truth, test_known_base, test_known_main)
    _, test_peak_errors = known_error_arrays(
        test_known_truth, test_known_base, test_known_peak)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(grid).to_csv(args.out / "validation_grid.csv", index=False)
    test_detail = pd.DataFrame({"id": test_ann.id.to_numpy(),
                                "type": test_ann.type.to_numpy(),
                                "center_bp": test_ann.center.to_numpy(),
                                "hotspot_baseline_bins": test_base_errors,
                                "hotspot_main_bins": test_main_errors,
                                "hotspot_peak_bins": test_peak_errors,
                                "hotspot_prediction_bins": test_pred_errors})
    test_detail.to_csv(args.out / "test_known_per_structure.csv", index=False)
    report = {
        "selection_rule": "minimize validation known-center hotspot error, requiring validation tile SSIM at least baseline +0.005, PSNR above baseline, and hotspot below baseline; fallback to main model",
        "selected": settings,
        "validation_selected": selected,
        "validation_candidates": len(grid),
        "validation_tiles": len(val_tile_truth),
        "validation_known_centers": len(val_known_truth),
        "test_tiles": len(test_tile_truth),
        "test_known_centers": len(test_known_truth),
        "test_tiles_selected": test_tile_metrics,
        "test_known_selected": test_known_metrics,
        "test_tiles_main": image_metrics(test_tile_truth, test_tile_base, test_tile_main),
        "test_tiles_peak": image_metrics(test_tile_truth, test_tile_base, test_tile_peak),
        "test_known_main": known_metrics(test_known_truth, test_known_base, test_known_main),
        "test_known_peak": known_metrics(test_known_truth, test_known_base, test_known_peak),
        "test_known_by_type": type_breakdown(test_detail),
        "saved_test_bundle_audit": saved_audit,
        "test_known_hotspot_gain_bins": float(np.mean(test_base_errors - test_pred_errors)),
        "test_known_hotspot_gain_site_bootstrap_95ci_bins": bootstrap_hotspot_gain(
            test_base_errors, test_pred_errors),
        "test_known_hotspot_gain_vs_main_bins": float(np.mean(test_main_errors - test_pred_errors)),
        "test_known_hotspot_gain_vs_main_site_bootstrap_95ci_bins": bootstrap_hotspot_gain(
            test_main_errors, test_pred_errors),
        "caution": "Exploratory validation-selected ablation; the same test chromosome segment was inspected in earlier experiments. The low-resolution input is simulated from the high-resolution Micro-C map. Local 640-bp patches do not verify multi-kb structure recovery.",
    }
    if args.external_cool:
        with open_cool(args.external_cool) as external:
            if external.binsize * patch_bins != patch_bp or external.chromnames[0] != chrom:
                raise ValueError("External replicate must have the same chromosome and resolution")

            def external_patches(locations: list[int]) -> tuple[np.ndarray, np.ndarray]:
                pairs = [prepare_sr_pair(window(external, chrom, center, patch_bp, patch_bins), factor)
                         for center in locations]
                return (np.stack([pair[0] for pair in pairs]),
                        np.stack([pair[1] for pair in pairs]))

            external_base, external_truth = external_patches(
                test_ann.center.astype(int).tolist())
            external_tile_base, external_tile_truth = external_patches(
                centers[test_start:])
        external_main, external_peak = model_predictions(external_base)
        external_tile_main, external_tile_peak = model_predictions(external_tile_base)
        external_pred = predict(external_base, external_main, external_peak, **settings)
        external_tile_pred = predict(external_tile_base, external_tile_main,
                                     external_tile_peak, **settings)
        external_base_errors, external_pred_errors = known_error_arrays(
            external_truth, external_base, external_pred)
        _, external_main_errors = known_error_arrays(
            external_truth, external_base, external_main)
        _, external_peak_errors = known_error_arrays(
            external_truth, external_base, external_peak)
        external_detail = pd.DataFrame({"id": test_ann.id.to_numpy(),
                                        "type": test_ann.type.to_numpy(),
                                        "center_bp": test_ann.center.to_numpy(),
                                        "hotspot_baseline_bins": external_base_errors,
                                        "hotspot_main_bins": external_main_errors,
                                        "hotspot_peak_bins": external_peak_errors,
                                        "hotspot_prediction_bins": external_pred_errors})
        external_detail.to_csv(args.out / "external_known_per_structure.csv", index=False)
        report["external_replicate"] = {
            "cool": str(args.external_cool),
            "test_tiles": len(external_tile_truth),
            "test_known_centers": len(external_truth),
            "tile_metrics": image_metrics(external_tile_truth, external_tile_base,
                                           external_tile_pred),
            "known_metrics": known_metrics(external_truth, external_base, external_pred),
            "tile_main": image_metrics(external_tile_truth, external_tile_base,
                                       external_tile_main),
            "tile_peak": image_metrics(external_tile_truth, external_tile_base,
                                       external_tile_peak),
            "known_main": known_metrics(external_truth, external_base, external_main),
            "known_peak": known_metrics(external_truth, external_base, external_peak),
            "known_by_type": type_breakdown(external_detail),
            "hotspot_gain_bins": float(np.mean(external_base_errors - external_pred_errors)),
            "hotspot_gain_site_bootstrap_95ci_bins": bootstrap_hotspot_gain(
                external_base_errors, external_pred_errors),
            "hotspot_gain_vs_main_bins": float(np.mean(external_main_errors - external_pred_errors)),
            "hotspot_gain_vs_main_site_bootstrap_95ci_bins": bootstrap_hotspot_gain(
                external_main_errors, external_pred_errors),
        }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
