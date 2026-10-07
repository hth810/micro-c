"""Paired reproducibility ablation for count-preserving Task 2 measurement.

Uses the same rep1-frozen annotated and independent unannotated loci in both
arms. Rep2 is descriptive evaluation only. No candidate or threshold is changed
from these results. Bootstrap intervals resample 100-kb genomic blocks, not
individual correlated windows, and concern measurement reproducibility only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from threadpoolctl import threadpool_limits

from microc.data import area_resample, normalize, open_cool, window
from scripts.task2_residual_motif_screen_20261006 import residual_vectors, sha256, write_json


def paired_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Expected paired N x feature vectors")
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    scale = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide((a * b).sum(axis=1), scale, out=np.zeros(len(a)), where=scale > 1e-12)


def block_bootstrap_difference(differences: np.ndarray, groups: np.ndarray,
                               seed: int = 20261006, repeats: int = 2000) -> dict:
    unique = np.unique(groups)
    sums = np.array([differences[groups == g].sum() for g in unique])
    counts = np.array([(groups == g).sum() for g in unique])
    if len(unique) < 2:
        return {"genomic_blocks": len(unique), "mean_difference_ci95": None}
    rng = np.random.default_rng(seed)
    selections = rng.integers(0, len(unique), size=(repeats, len(unique)))
    means = sums[selections].sum(axis=1) / counts[selections].sum(axis=1)
    return {"genomic_blocks": len(unique), "block_size_bp": 100000,
            "bootstrap_repeats": repeats, "seed": seed,
            "mean_difference_ci95": np.quantile(means, [.025, .975]).tolist()}


def check_freeze(folder: Path):
    receipt = json.loads((folder / "freeze_receipt.json").read_text(encoding="utf-8"))
    for name, digest in receipt["frozen_file_sha256"].items():
        if sha256(folder / name) != digest:
            raise ValueError(f"Frozen discovery changed: {folder / name}")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, default=Path("outputs/task2_improve_20261006/legacy"))
    parser.add_argument("--area", type=Path, default=Path("outputs/task2_improve_20261006/area"))
    parser.add_argument("--rep2", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/task2_improve_20261006/resampling_comparison"))
    args = parser.parse_args()
    check_freeze(args.legacy)
    check_freeze(args.area)
    legacy = np.load(args.legacy / "rep1_maps_models.npz")
    area = np.load(args.area / "rep1_maps_models.npz")
    if not np.array_equal(legacy["centers"], area["centers"]):
        raise ValueError("Paired arms have different positions")
    independent1 = pd.read_csv(args.legacy / "rep1_independent_unannotated.csv")
    independent2 = pd.read_csv(args.area / "rep1_independent_unannotated.csv")
    if not independent1.center.equals(independent2.center):
        raise ValueError("Discovery-independent representative priorities differ")
    centers = legacy["centers"]
    known_n = int(legacy["known_n"])
    position = {int(c): i for i, c in enumerate(centers)}
    idx = np.r_[np.arange(known_n), [position[int(c)] for c in independent1.center]]
    chosen = centers[idx]
    # 64-bin linear interpolation and area aggregation use the identical native
    # input read, so the only ablated factor is resampling of contact counts.
    maps2 = {"legacy": [], "area": []}
    retained_counts = []
    chrom = str(pd.read_csv(args.legacy / "rep1_all_windows.csv").chrom.iloc[0])
    evaluation_started_at_utc = datetime.now(timezone.utc).isoformat()
    with open_cool(args.rep2) as cool:
        native_bins = 5120 // int(cool.binsize)
        if native_bins != 512:
            raise ValueError("This locked ablation expects 10-bp source bins")
        for k, center in enumerate(chosen):
            native = window(cool, chrom, int(center), 5120, native_bins)
            interpolated = zoom(native, 64 / native_bins, order=1)
            interpolated = (interpolated + interpolated.T) / 2
            aggregated = area_resample(native, 64)
            maps2["legacy"].append(normalize(interpolated))
            maps2["area"].append(normalize(aggregated))
            retained_counts.append({"center_bp": int(center),
                                    "native_total": float(native.sum()),
                                    "area_total": float(aggregated.sum()),
                                    "legacy_interpolated_total": float(interpolated.sum())})
            if (k + 1) % 100 == 0:
                print(f"paired native rep2 windows: {k + 1}/{len(chosen)}", flush=True)
    maps1 = {"legacy": legacy["maps"][idx], "area": area["maps"][idx]}
    maps2 = {m: np.stack(v) for m, v in maps2.items()}
    rows = []
    upper = np.triu_indices(64, k=2)
    for arm in ("legacy", "area"):
        for metric in ("upper", "highpass", "rowcol"):
            if metric == "upper":
                v1, v2 = (maps1[arm][:, upper[0], upper[1]], maps2[arm][:, upper[0], upper[1]])
            else:
                v1, v2 = residual_vectors(maps1[arm], metric), residual_vectors(maps2[arm], metric)
            r = paired_correlation(v1, v2)
            for i, center in enumerate(chosen):
                rows.append({"arm": arm, "metric": metric, "center_bp": int(center),
                             "reference_index": i, "cohort": "annotated" if i < known_n else "unannotated_independent",
                             "known_type": str(legacy["known_types"][i]) if i < known_n else "",
                             "genomic_block": int(center) // 100000, "replicate_r": float(r[i])})
    table = pd.DataFrame(rows)
    reports = []
    for (cohort, metric), frame in table.groupby(["cohort", "metric"]):
        joined = frame.pivot(index="reference_index", columns="arm", values="replicate_r")
        blocks = frame.drop_duplicates("reference_index").set_index("reference_index").loc[joined.index, "genomic_block"].to_numpy()
        diff = (joined.area - joined.legacy).to_numpy()
        reports.append({"cohort": cohort, "metric": metric, "n_loci": len(joined),
                        "legacy_median_r": float(joined.legacy.median()),
                        "area_median_r": float(joined.area.median()),
                        "legacy_mean_r": float(joined.legacy.mean()), "area_mean_r": float(joined.area.mean()),
                        "mean_paired_difference_area_minus_legacy": float(diff.mean()),
                        "fraction_loci_improved": float((diff > 0).mean()),
                        **block_bootstrap_difference(diff, blocks)})
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "paired_reproducibility.csv", index=False)
    counts = pd.DataFrame(retained_counts)
    counts.to_csv(args.out / "count_conservation.csv", index=False)
    plot(table, args.out / "resampling_reproducibility.png")
    report = {"analysis": "Paired count-preserving measurement ablation; descriptive retrospective rep2 evaluation",
              "evaluation_started_at_utc": evaluation_started_at_utc,
              "run_code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "script_sha256": sha256(Path(__file__)), "data_api_sha256": sha256(Path("microc/data.py")),
              "residual_implementation_sha256": sha256(Path("scripts/task2_residual_motif_screen_20261006.py")),
              "rep2_sha256": sha256(args.rep2),
              "legacy_discovery_receipt_sha256": sha256(args.legacy / "freeze_receipt.json"),
              "area_discovery_receipt_sha256": sha256(args.area / "freeze_receipt.json"),
              "candidate_selection_or_thresholds_changed_using_rep2": False,
              "n_annotated_loci": known_n, "n_independent_unannotated_loci": len(independent1),
              "source_counts_conserved_max_absolute_error": float(np.abs(counts.area_total - counts.native_total).max()),
              "legacy_total_count_ratio_median": float((counts.legacy_interpolated_total / counts.native_total).median()),
              "reports": reports,
              "caution": "Higher reproducibility supports a better measurement, not a new structure type. Annotated references are not a classifier holdout set. Block-bootstrap intervals concern paired measurement agreement and do not adjust the history of novelty screens.",
              "output_sha256": {p.name: sha256(p) for p in args.out.glob("*") if p.is_file()}}
    write_json(args.out / "comparison_summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


def plot(table: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=160)
    for row, cohort in enumerate(("annotated", "unannotated_independent")):
        for col, metric in enumerate(("upper", "highpass", "rowcol")):
            frame = table[(table.cohort == cohort) & (table.metric == metric)]
            pair = frame.pivot(index="reference_index", columns="arm", values="replicate_r")
            ax = axes[row, col]
            ax.scatter(pair.legacy, pair.area, s=10, alpha=.45, color="#2377a0")
            ax.plot([-1, 1], [-1, 1], "--", color="gray", linewidth=1)
            ax.set(xlim=(-.2, 1), ylim=(-.2, 1), xlabel="Legacy interpolation: rep1/rep2 r",
                   ylabel="Count-area pooling: rep1/rep2 r", title=f"{cohort} / {metric}")
    fig.suptitle("Identical loci and raw inputs; count-conserving resampling ablation")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
