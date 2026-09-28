"""Validation-frozen conservative SR peak guard and paired replicate audit.

This is an exploratory ablation: earlier work inspected the same genomic test
region. The rule is selected from rep1 validation data only, then frozen for
rep1 test and rep2 at identical coordinates. No test metric enters selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import maximum_filter
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity

from microc.data import open_cool, window
from microc.deep import _psnr, prepare_sr_pair
from scripts.evaluate_sr_ensemble import infer, load_model, split_centers
from scripts.evaluate_sr_spatial_blend import peak_mask
from scripts.validate_sr_known_shape import hotspot


def options() -> list[dict]:
    """Small mechanistic grid, not adapted to the test-set outcomes."""
    return ([{"name": "main", "strength": 0.0, "sigma": None}]
            + [{"name": "anchor", "strength": strength, "sigma": sigma}
               for strength in (0.5, 1.0) for sigma in (3.0, 6.0)]
            + [{"name": "peak_roi", "strength": 0.5, "sigma": 3.0}])


def guarded(base: np.ndarray, main: np.ndarray, peak: np.ndarray,
            name: str, strength: float, sigma: float | None) -> np.ndarray:
    if name == "main":
        return main
    if sigma is None or sigma <= 0:
        raise ValueError("positive sigma required")
    upper = np.triu(np.ones(base.shape[-2:], dtype=bool), k=3)
    cue = base if name == "anchor" else peak
    mask = np.stack([peak_mask(image, sigma, upper) for image in cue])
    if name == "anchor":
        # Preserve the observed low-resolution peak in a small, symmetric ROI.
        return main + strength * mask * (base - main)
    if name == "peak_roi":
        return main + strength * mask * (peak - main)
    raise ValueError(name)


def peak_candidates(image: np.ndarray, upper: np.ndarray, k: int = 5) -> np.ndarray:
    """Top local maxima of the same smoothed upper triangle used for top-1."""
    from scipy.ndimage import gaussian_filter
    smoothed = gaussian_filter(image, sigma=2)
    local = (smoothed == maximum_filter(smoothed, size=5)) & upper
    indices = np.argwhere(local)
    if not len(indices):
        indices = np.argwhere(upper)
    scores = smoothed[indices[:, 0], indices[:, 1]]
    order = np.argsort(-scores, kind="stable")[:k]
    return indices[order]


def per_patch(truth: np.ndarray, baseline: np.ndarray, prediction: np.ndarray,
              center_bp: np.ndarray, types: np.ndarray | None = None,
              ids: np.ndarray | None = None) -> pd.DataFrame:
    upper = np.triu(np.ones(truth.shape[-2:], dtype=bool), k=3)
    rows = []
    for i, (target, base, pred) in enumerate(zip(truth, baseline, prediction)):
        target_peak = hotspot(target, upper)
        target_values = target[upper]
        row = {"center_bp": int(center_bp[i]),
               "type": str(types[i]) if types is not None else "tile",
               "id": str(ids[i]) if ids is not None else f"tile_{i}",
               "truth_peak_row": int(target_peak[0]),
               "truth_peak_col": int(target_peak[1])}
        for name, image in (("baseline", base), ("prediction", pred)):
            values = image[upper]
            candidates = peak_candidates(image, upper, k=5)
            row[f"{name}_psnr"] = _psnr(target, image)
            row[f"{name}_ssim"] = float(structural_similarity(target, image, data_range=10))
            row[f"{name}_spearman"] = (0.0 if np.std(values) < 1e-8 or np.std(target_values) < 1e-8
                                        else float(spearmanr(target_values, values).statistic))
            row[f"{name}_hotspot_error"] = float(np.linalg.norm(hotspot(image, upper) - target_peak))
            row[f"{name}_top5_error"] = float(np.linalg.norm(candidates - target_peak, axis=1).min())
            row[f"{name}_top5_hit5"] = float(row[f"{name}_top5_error"] <= 5)
        rows.append(row)
    return pd.DataFrame(rows)


def means(frame: pd.DataFrame) -> dict[str, float]:
    metrics = ("psnr", "ssim", "spearman", "hotspot_error", "top5_error", "top5_hit5")
    return {f"{method}_{metric}": float(frame[f"{method}_{metric}"].mean())
            for method in ("baseline", "prediction") for metric in metrics}


def choose(rows: list[dict]) -> dict:
    """Require validation image gains and no worsening in the common CHIN class."""
    feasible = [row for row in rows if
                row["tile_prediction_psnr"] >= row["tile_baseline_psnr"] + 1.0
                and row["tile_prediction_ssim"] >= row["tile_baseline_ssim"] + 0.01
                and row["tile_prediction_psnr"] >= row["main_tile_psnr"] - 0.05
                and row["tile_prediction_ssim"] >= row["main_tile_ssim"] - 0.001
                and row["CHIN_prediction_hotspot_error"] <= row["CHIN_baseline_hotspot_error"]
                and row["known_prediction_hotspot_error"] <= row["known_baseline_hotspot_error"]]
    # If the common class cannot be protected on validation, retain the
    # preexisting main model and explicitly report the unmet guard criterion.
    return min(feasible, key=lambda r: (r["known_prediction_hotspot_error"],
                                        -r["tile_prediction_ssim"])) if feasible else rows[0]


def block_interval(delta: np.ndarray, positions: np.ndarray, block_bp: int = 50000,
                   n_boot: int = 5000, seed: int = 20260928) -> list[float]:
    """Resample whole fixed genomic blocks; clustered sites are not IID."""
    delta = np.asarray(delta, dtype=float)
    if len(delta) != len(positions) or not len(delta):
        raise ValueError("nonempty, aligned pairs required")
    block_ids = np.asarray(positions, dtype=int) // block_bp
    blocks = [delta[block_ids == block] for block in np.unique(block_ids)]
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        draw = rng.integers(0, len(blocks), size=len(blocks))
        boot[i] = np.concatenate([blocks[j] for j in draw]).mean()
    return [float(x) for x in np.quantile(boot, [.025, .975])]


def paired_report(frame: pd.DataFrame, reference_method: str,
                  prediction_method: str) -> dict:
    report = {"n": int(len(frame)), "genomic_50kb_blocks": int((frame.center_bp // 50000).nunique()),
              "reference_method": reference_method, "prediction_method": prediction_method,
              "gain_definition": "prediction-reference for PSNR, SSIM, Spearman, top5_hit5; reference-prediction for top1 hotspot and top5 nearest distance; positive is better"}
    for metric in ("psnr", "ssim", "spearman", "hotspot_error", "top5_error", "top5_hit5"):
        gain = frame[f"prediction_{metric}"].to_numpy() - frame[f"baseline_{metric}"].to_numpy()
        if metric in ("hotspot_error", "top5_error"):
            gain = -gain
        report[metric] = {"baseline_mean": float(frame[f"baseline_{metric}"].mean()),
                          "prediction_mean": float(frame[f"prediction_{metric}"].mean()),
                          "mean_gain": float(gain.mean()),
                          "win_fraction": float(np.mean(gain > 0)),
                          "block_bootstrap_95ci_gain": block_interval(gain, frame.center_bp.to_numpy())}
    return report


def against(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Compare a selected prediction with the same sites from another method."""
    if not np.array_equal(frame["center_bp"].to_numpy(), reference["center_bp"].to_numpy()):
        raise ValueError("paired patch coordinates differ")
    out = frame.copy()
    for metric in ("psnr", "ssim", "spearman", "hotspot_error", "top5_error", "top5_hit5"):
        out[f"baseline_{metric}"] = reference[f"prediction_{metric}"].to_numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cool", type=Path, required=True)
    ap.add_argument("--external-cool", type=Path)
    ap.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    ap.add_argument("--main-checkpoint", type=Path, default=Path("results/task5_super_resolution.pt"))
    ap.add_argument("--peak-checkpoint", type=Path, default=Path("results/task5_peak_weighted_super_resolution.pt"))
    ap.add_argument("--max-patches", type=int, default=1200)
    ap.add_argument("--validation-only", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("outputs/task5_peak_guard"))
    args = ap.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model_main, check_main = load_model(args.main_checkpoint)
    model_peak, check_peak = load_model(args.peak_checkpoint)
    if (check_main["factor"], check_main["patch_bins"]) != (check_peak["factor"], check_peak["patch_bins"]):
        raise ValueError("checkpoint resolution mismatch")
    factor, bins = int(check_main["factor"]), int(check_main["patch_bins"])
    ann = pd.read_csv(args.annotations).sort_values("center")

    def load(cooler, chrom: str, patch_bp: int, locations) -> tuple[np.ndarray, np.ndarray]:
        pairs = [prepare_sr_pair(window(cooler, chrom, int(center), patch_bp, bins), factor)
                 for center in locations]
        return (np.stack([pair[0] for pair in pairs]),
                np.stack([pair[1] for pair in pairs]))

    def model_predictions(base):
        return (base + float(check_main["blend"]) * (infer(model_main, base) - base),
                base + float(check_peak["blend"]) * (infer(model_peak, base) - base))

    with open_cool(args.cool) as cooler:
        chrom = cooler.chromnames[0]
        patch_bp = bins * int(cooler.binsize)
        centers, train_end, test_start = split_centers(int(cooler.chromsizes[chrom]),
                                                       patch_bp, args.max_patches)
        val_ann = ann[(ann.center >= centers[train_end]) & (ann.center < centers[test_start])]
        test_ann = ann[ann.center >= centers[test_start]]
        val_centers = np.asarray(centers[train_end:test_start])
        val_tile_base, val_tile_truth = load(cooler, chrom, patch_bp, val_centers)
        val_known_base, val_known_truth = load(cooler, chrom, patch_bp, val_ann.center)
    val_tile_main, val_tile_peak = model_predictions(val_tile_base)
    val_known_main, val_known_peak = model_predictions(val_known_base)
    main_tile = means(per_patch(val_tile_truth, val_tile_base, val_tile_main, val_centers))
    val_rows = []
    for option in options():
        tile = per_patch(val_tile_truth, val_tile_base,
                         guarded(val_tile_base, val_tile_main, val_tile_peak, **option), val_centers)
        known = per_patch(val_known_truth, val_known_base,
                          guarded(val_known_base, val_known_main, val_known_peak, **option),
                          val_ann.center.to_numpy(), val_ann.type.to_numpy(), val_ann.id.to_numpy())
        row = {**option, **{f"tile_{k}": v for k, v in means(tile).items()},
               **{f"known_{k}": v for k, v in means(known).items()},
               "main_tile_psnr": main_tile["prediction_psnr"],
               "main_tile_ssim": main_tile["prediction_ssim"]}
        for group, sub in known.groupby("type"):
            row.update({f"{group}_{k}": v for k, v in means(sub).items()})
        val_rows.append(row)
    selected = choose(val_rows)
    selected_option = {key: selected[key] for key in ("name", "strength", "sigma")}
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(val_rows).to_csv(args.out / "validation_grid.csv", index=False)
    report = {"selection_rule": "Validation-only: image PSNR >= baseline+1 dB and main-0.05 dB; SSIM >= baseline+0.01 and main-0.001; CHIN and overall top-1 hotspot error no worse than baseline; minimize overall error",
              "metric_protocol": "Hotspot is Gaussian-smoothed (sigma=2 bins) argmax on upper triangle k>=3; top5 uses five 5x5 local maxima on the same region; top5_hit5 requires one within 5 bins of truth hotspot. All choices fixed before this run's test evaluation.",
              "selected": selected_option,
              "validation_selected": selected,
              "validation_known_n": int(len(val_ann)),
              "validation_known_by_type": val_ann.type.value_counts().to_dict(),
              "validation_tiles_n": int(len(val_centers)),
              "guard_satisfied": bool(selected["CHIN_prediction_hotspot_error"] <=
                                      selected["CHIN_baseline_hotspot_error"] and
                                      selected["known_prediction_hotspot_error"] <=
                                      selected["known_baseline_hotspot_error"]),
              "caution": "Exploratory; prior experiments already inspected the rep1 test region. Simulated low resolution; local 640 bp windows do not prove full structure recovery."}
    if args.validation_only:
        (args.out / "validation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    def evaluate(cool_path: Path, label: str) -> dict:
        with open_cool(cool_path) as cooler:
            if cooler.chromnames[0] != chrom or int(cooler.binsize) * bins != patch_bp:
                raise ValueError("replicate chromosome or bin size mismatch")
            test_centers = np.asarray(centers[test_start:])
            tile_base, tile_truth = load(cooler, chrom, patch_bp, test_centers)
            known_base, known_truth = load(cooler, chrom, patch_bp, test_ann.center)
        tile_main, tile_peak = model_predictions(tile_base)
        known_main, known_peak = model_predictions(known_base)
        tile_pred = guarded(tile_base, tile_main, tile_peak, **selected_option)
        known_pred = guarded(known_base, known_main, known_peak, **selected_option)
        tile = per_patch(tile_truth, tile_base, tile_pred, test_centers)
        known = per_patch(known_truth, known_base, known_pred,
                          test_ann.center.to_numpy(), test_ann.type.to_numpy(), test_ann.id.to_numpy())
        main_tile = per_patch(tile_truth, tile_base, tile_main, test_centers)
        main_known = per_patch(known_truth, known_base, known_main,
                               test_ann.center.to_numpy(), test_ann.type.to_numpy(), test_ann.id.to_numpy())
        tile.to_csv(args.out / f"{label}_tiles.csv", index=False)
        known.to_csv(args.out / f"{label}_known.csv", index=False)
        main_tile.to_csv(args.out / f"{label}_main_tiles.csv", index=False)
        main_known.to_csv(args.out / f"{label}_main_known.csv", index=False)
        selected_name = f"validation-selected {selected_option['name']}"
        return {"tiles": paired_report(tile, "bilinear interpolation", selected_name),
                "known": paired_report(known, "bilinear interpolation", selected_name),
                "main_tiles": paired_report(main_tile, "bilinear interpolation", "main CNN"),
                "main_known": paired_report(main_known, "bilinear interpolation", "main CNN"),
                "selected_vs_main_tiles": paired_report(against(tile, main_tile), "main CNN", selected_name),
                "selected_vs_main_known": paired_report(against(known, main_known), "main CNN", selected_name),
                "known_by_type": {type_name: paired_report(sub, "bilinear interpolation", selected_name)
                                  for type_name, sub in known.groupby("type")}}

    report["rep1_test"] = evaluate(args.cool, "rep1_test")
    if args.external_cool:
        report["rep2_fixed"] = evaluate(args.external_cool, "rep2_fixed")
    (args.out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected_option,
                      "validation_guard_satisfied": report["guard_satisfied"],
                      "rep1_known_hotspot": report["rep1_test"]["known"]["hotspot_error"],
                      "rep2_known_hotspot": report.get("rep2_fixed", {}).get("known", {}).get("hotspot_error")},
                     indent=2))


if __name__ == "__main__":
    main()
