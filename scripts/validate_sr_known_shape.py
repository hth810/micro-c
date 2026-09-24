"""Check local shape and hotspot location at held-out annotated structure centers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

from microc.data import open_cool, window
from microc.deep import SuperResolutionCNN, prepare_sr_pair


def hotspot(matrix: np.ndarray, upper: np.ndarray) -> np.ndarray:
    smoothed = gaussian_filter(matrix, sigma=2)
    return np.asarray(np.unravel_index(np.argmax(np.where(upper, smoothed, -np.inf)),
                                       matrix.shape), dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cool", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, default=Path("results"))
    ap.add_argument("--known-file", type=Path)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--out", type=Path, default=Path("outputs/task5_known_shape"))
    args = ap.parse_args()
    known_file = args.known_file or args.bundle / "task5_known_structure_recovery.csv"
    selected = (pd.read_csv(known_file)
                .drop_duplicates("id").sort_values("center_bp"))
    checkpoint_path = args.checkpoint or args.bundle / "task5_super_resolution.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = SuperResolutionCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    patch_bins = int(checkpoint["patch_bins"])
    factor = int(checkpoint["factor"])
    blend = float(checkpoint["blend"])
    upper = np.triu(np.ones((patch_bins, patch_bins), dtype=bool), k=3)
    rows = []
    excluded = 0
    with open_cool(args.cool) as cooler:
        chrom = cooler.chromnames[0]
        patch_bp = int(patch_bins * cooler.binsize)
        for record in selected.itertuples():
            raw = window(cooler, chrom, int(record.center_bp), patch_bp, patch_bins)
            baseline, truth = prepare_sr_pair(raw, factor)
            if np.std(truth[upper]) < 1e-8:
                excluded += 1
                continue
            with torch.no_grad():
                raw_pred = model(torch.from_numpy(baseline[None, None])).numpy()[0, 0]
            cnn = baseline + blend * (raw_pred - baseline)
            target_peak = hotspot(truth, upper)
            for method, image in (("bilinear", baseline), ("cnn", cnn)):
                values = image[upper]
                rho = 0.0 if np.std(values) < 1e-8 else float(spearmanr(truth[upper], values).statistic)
                rows.append({"id": record.id, "type": record.type,
                             "center_bp": int(record.center_bp), "method": method,
                             "upper_contact_spearman": rho,
                             "hotspot_error_bins": float(np.linalg.norm(hotspot(image, upper) - target_peak))})
    frame = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "per_structure.csv", index=False)
    summary = {"known_centers": len(selected), "informative_centers": frame.id.nunique(),
               "zero_signal_centers_excluded": excluded, "patch_bp": patch_bp,
               "note": "Local centered patch only; cannot prove recovery of the full multi-kb structure or automated localization."}
    for metric in ("upper_contact_spearman", "hotspot_error_bins"):
        pivot = frame.pivot(index="id", columns="method", values=metric)
        summary[metric] = {"bilinear_mean": float(pivot.bilinear.mean()),
                           "cnn_mean": float(pivot.cnn.mean()),
                           "cnn_better_fraction": float(np.mean(
                               (pivot.cnn > pivot.bilinear) if metric == "upper_contact_spearman"
                               else (pivot.cnn < pivot.bilinear)))}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
