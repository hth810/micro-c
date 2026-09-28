"""Validation-selected blend of two existing SR models, with held-out shape tests.

The blend weight is selected using annotated centers in the validation genomic
segment only. Test patches and test annotations are read solely for reporting.
This is exploratory because earlier experiments already inspected the same
held-out chromosome segment; it is not a pristine external validation set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity

from microc.data import open_cool, window
from microc.deep import SuperResolutionCNN, _psnr, prepare_sr_pair
from scripts.validate_sr_known_shape import hotspot


def load_model(path: Path) -> tuple[SuperResolutionCNN, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = SuperResolutionCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def infer(model: SuperResolutionCNN, patches: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        tensors = torch.from_numpy(patches[:, None].astype(np.float32))
        return np.concatenate([model(batch).numpy()[:, 0] for batch in tensors.split(32)])


def summarize_known(truth: np.ndarray, bilinear: np.ndarray, main: np.ndarray,
                    peak: np.ndarray, alpha: float) -> dict:
    prediction = alpha * main + (1 - alpha) * peak
    upper = np.triu(np.ones(truth.shape[-2:], dtype=bool), k=3)
    rows = []
    for target, base, restored in zip(truth, bilinear, prediction):
        target_peak = hotspot(target, upper)
        rows.append((float(np.linalg.norm(hotspot(base, upper) - target_peak)),
                     float(np.linalg.norm(hotspot(restored, upper) - target_peak)),
                     float(structural_similarity(target, base, data_range=10)),
                     float(structural_similarity(target, restored, data_range=10))))
    scores = np.asarray(rows)
    return {"n": len(rows), "hotspot_baseline_bins": float(scores[:, 0].mean()),
            "hotspot_blend_bins": float(scores[:, 1].mean()),
            "hotspot_blend_win_fraction": float(np.mean(scores[:, 1] < scores[:, 0])),
            "ssim_baseline": float(scores[:, 2].mean()),
            "ssim_blend": float(scores[:, 3].mean())}


def summarize_patches(truth: np.ndarray, bilinear: np.ndarray, main: np.ndarray,
                      peak: np.ndarray, alpha: float) -> dict:
    prediction = alpha * main + (1 - alpha) * peak
    return {"n": len(truth),
            "psnr_baseline": float(np.mean([_psnr(a, b) for a, b in zip(truth, bilinear)])),
            "psnr_blend": float(np.mean([_psnr(a, b) for a, b in zip(truth, prediction)])),
            "ssim_baseline": float(np.mean([structural_similarity(a, b, data_range=10)
                                             for a, b in zip(truth, bilinear)])),
            "ssim_blend": float(np.mean([structural_similarity(a, b, data_range=10)
                                          for a, b in zip(truth, prediction)]))}


def split_centers(chrom_length: int, patch_bp: int, max_patches: int) -> tuple[list[int], int, int]:
    centers = list(range(patch_bp // 2, chrom_length - patch_bp // 2, patch_bp))
    centers = centers[::max(1, int(np.ceil(len(centers) / max_patches)))]
    return centers, int(.7 * len(centers)), int(.8 * len(centers))


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
    ap.add_argument("--out", type=Path, default=Path("outputs/task5_ensemble"))
    args = ap.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    main_model, main_check = load_model(args.main_checkpoint)
    peak_model, peak_check = load_model(args.peak_checkpoint)
    if (main_check["factor"], main_check["patch_bins"]) != (
            peak_check["factor"], peak_check["patch_bins"]):
        raise ValueError("Checkpoints use different resolutions")
    patch_bins, factor = int(main_check["patch_bins"]), int(main_check["factor"])
    ann = pd.read_csv(args.annotations).sort_values("center").reset_index(drop=True)
    with open_cool(args.cool) as cooler:
        chrom = cooler.chromnames[0]
        patch_bp = patch_bins * cooler.binsize
        centers, train_end, test_start = split_centers(
            int(cooler.chromsizes[chrom]), patch_bp, args.max_patches)
        validation_ann = ann[(ann.center >= centers[train_end]) &
                             (ann.center < centers[test_start])]
        test_ann = ann[ann.center >= centers[test_start]]
        if validation_ann.empty or test_ann.empty:
            raise ValueError("Both validation and test segments need known structures")

        def known_patches(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
            pairs = [prepare_sr_pair(window(cooler, chrom, int(row.center), patch_bp,
                                            patch_bins), factor) for row in frame.itertuples()]
            return np.stack([pair[0] for pair in pairs]), np.stack([pair[1] for pair in pairs])

        val_base, val_truth = known_patches(validation_ann)
        test_base, test_truth = known_patches(test_ann)
    val_main, val_peak = infer(main_model, val_base), infer(peak_model, val_base)
    test_main, test_peak = infer(main_model, test_base), infer(peak_model, test_base)
    val_main = val_base + float(main_check["blend"]) * (val_main - val_base)
    val_peak = val_base + float(peak_check["blend"]) * (val_peak - val_base)
    test_main = test_base + float(main_check["blend"]) * (test_main - test_base)
    test_peak = test_base + float(peak_check["blend"]) * (test_peak - test_base)
    alphas = np.linspace(0, 1, 11)
    val_grid = [{"main_weight": float(alpha), **summarize_known(
        val_truth, val_base, val_main, val_peak, float(alpha))} for alpha in alphas]
    feasible = [row for row in val_grid if
                row["ssim_blend"] > row["ssim_baseline"] and
                row["hotspot_blend_bins"] <= row["hotspot_baseline_bins"]]
    # A two-objective validation score rewards improvements in both image
    # structure and local peak position; neither objective can compensate for
    # worsening the other below the bilinear baseline.
    def joint_gain(row: dict) -> float:
        ssim_gain = (row["ssim_blend"] - row["ssim_baseline"]) / max(row["ssim_baseline"], 1e-8)
        peak_gain = ((row["hotspot_baseline_bins"] - row["hotspot_blend_bins"]) /
                     max(row["hotspot_baseline_bins"], 1e-8))
        return float(ssim_gain * peak_gain)

    selected = max(feasible, key=joint_gain)["main_weight"] if feasible else 1.0
    selected_known = summarize_known(test_truth, test_base, test_main, test_peak, selected)
    with np.load(args.main_patches) as main_bundle, np.load(args.peak_patches) as peak_bundle:
        for key in ("center_bp", "truth", "bilinear"):
            if not np.array_equal(main_bundle[key], peak_bundle[key]):
                raise ValueError(f"Test patch bundles differ in {key}")
        test_grid = [{"main_weight": float(alpha), **summarize_patches(
            main_bundle["truth"], main_bundle["bilinear"],
            main_bundle["cnn"], peak_bundle["cnn"], float(alpha))} for alpha in alphas]
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(val_grid).to_csv(args.out / "validation_known_grid.csv", index=False)
    pd.DataFrame(test_grid).to_csv(args.out / "test_patch_grid.csv", index=False)
    report = {"validation_known_centers": len(validation_ann), "test_known_centers": len(test_ann),
              "selection_rule": "largest product of relative validation SSIM and hotspot-error gains; both must beat bilinear",
              "feasible_validation_weights": [row["main_weight"] for row in feasible],
              "selected_main_weight": selected,
              "selection_fallback_to_main": not bool(feasible),
              "validation_selected": next(row for row in val_grid if row["main_weight"] == selected),
              "test_known_selected": selected_known,
              "test_patches_selected": next(row for row in test_grid if row["main_weight"] == selected),
              "caution": "Earlier model development inspected this same test region; exploratory validation-selected ablation, not external confirmation. 640-bp patches cannot establish multi-kb structure recovery."}
    (args.out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), dpi=160)
    validation = pd.DataFrame(val_grid)
    for ax, column, baseline, title in (
            (axes[0], "ssim_blend", "ssim_baseline", "Validation known-center SSIM"),
            (axes[1], "hotspot_blend_bins", "hotspot_baseline_bins", "Validation hotspot error (bins)")):
        ax.plot(validation.main_weight, validation[column], marker="o", label="CNN blend")
        ax.axhline(validation[baseline].iloc[0], color="gray", ls="--", label="bilinear")
        ax.axvline(selected, color="teal", ls=":", label="selected")
        ax.set(xlabel="Main-model weight", title=title)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out / "validation_tradeoff.png")
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
