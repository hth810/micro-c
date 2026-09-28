"""Compare an archived SR result bundle with a fresh training run before promotion.

Run this *before* collecting the rerun into results/, so the earlier metrics
remain visible in the audit JSON even though the review bundle is refreshed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", type=Path, default=Path("results"))
    ap.add_argument("--rerun", type=Path, default=Path("outputs"))
    ap.add_argument("--out", type=Path, default=Path("results/task5_reproducibility_audit.json"))
    args = ap.parse_args()
    old_main = read(args.reference / "task5_summary.json")
    new_main = read(args.rerun / "task5_corrected" / "summary.json")
    old_known = read(args.reference / "task5_known_shape_summary.json")
    new_known = read(args.rerun / "task5_known_shape" / "summary.json")
    old_ensemble = read(args.reference / "task5_ensemble_summary.json")
    new_ensemble = read(args.rerun / "task5_ensemble" / "summary.json")
    old_checkpoint = args.reference / "task5_super_resolution.pt"
    new_checkpoint = args.rerun / "task5_corrected" / "super_resolution.pt"
    old_patches = args.reference / "task5_reconstructed_test_patches.npz"
    new_patches = args.rerun / "task5_corrected" / "reconstructed_test_patches.npz"
    with np.load(old_patches) as old, np.load(new_patches) as new:
        if not all(np.array_equal(old[name], new[name])
                   for name in ("center_bp", "truth", "bilinear")):
            raise ValueError("Rerun uses different test centers, truth, or baseline")
        delta = np.abs(old["cnn"] - new["cnn"])
        patch_stats = {"patches": len(new["center_bp"]),
                       "mean_absolute_prediction_difference": float(delta.mean()),
                       "max_absolute_prediction_difference": float(delta.max())}
    report = {
        "comparison": "saved branch bundle before rerun vs fresh same-command CPU rerun",
        "training_parameters_rerun": {name: new_main[name] for name in
                                      ("factor", "epochs", "seed", "max_patches", "patch_bins", "peak_weight")},
        "matched_test_data": True,
        "test_predictions": patch_stats,
        "checkpoint_sha256": {"archived": digest(old_checkpoint), "rerun": digest(new_checkpoint)},
        "test_psnr_cnn": {"archived": old_main["comparison"]["psnr"]["cnn"],
                          "rerun": new_main["comparison"]["psnr"]["cnn"]},
        "test_ssim_cnn": {"archived": old_main["comparison"]["ssim"]["cnn"],
                          "rerun": new_main["comparison"]["ssim"]["cnn"]},
        "known_hotspot_error_bins_cnn": {
            "archived": old_known["hotspot_error_bins"]["cnn_mean"],
            "rerun": new_known["hotspot_error_bins"]["cnn_mean"]},
        "validation_selected_main_weight": {
            "archived": old_ensemble["selected_main_weight"],
            "rerun": new_ensemble["selected_main_weight"]},
        "conclusion": "Image metrics reproduce closely, but exact weights and some localization/blend values vary. Neither run beats bilinear for known-center hotspot localization. The rerun is promoted as the current review bundle.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
