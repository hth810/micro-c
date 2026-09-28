"""Audit whether the frozen Task 2 motif is a coverage/band artifact.

All three loci are read from the rep1-only discovery manifest. Four fixed
views are evaluated in *both* replicates: ordinary non-near-diagonal contacts
(k>=8), more distant contacts (k>=16), k>=8 after subtracting a sigma-4
Gaussian background, and k>=8 after removing row/column main effects.
Controls are selected using rep1 raw log coverage and genomic separation only;
rep2 morphology never enters matching. This is retrospective descriptive
analysis, not an independent discovery p-value or genomic FDR.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from microc.data import normalize, open_cool, window
from scripts.task2_rep1_locked_discovery import INDEPENDENCE_BP, corr_matrix


METRICS = ("far8", "far16", "highpass_far8", "rowcol_far8")


def masked_vectors(maps: np.ndarray, metric: str) -> np.ndarray:
    """Return pixels only from a fixed upper-triangle shape view."""
    if maps.ndim != 3 or maps.shape[1:] != (64, 64):
        raise ValueError("Expected N x 64 x 64 normalized contact maps")
    if metric not in METRICS:
        raise ValueError(metric)
    offset = 16 if metric == "far16" else 8
    selector = np.triu_indices(64, k=offset)
    if metric == "highpass_far8":
        viewed = np.stack([mat - gaussian_filter(mat, sigma=4) for mat in maps])
    elif metric == "rowcol_far8":
        mean_row = maps.mean(axis=2, keepdims=True)
        mean_col = maps.mean(axis=1, keepdims=True)
        viewed = maps - mean_row - mean_col + maps.mean(axis=(1, 2), keepdims=True)
    else:
        viewed = maps
    return viewed[:, selector[0], selector[1]]


def pair_pool(indices: np.ndarray, centers: np.ndarray, excluded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    i, j = np.triu_indices(len(indices), k=1)
    a, b = indices[i], indices[j]
    separated = np.abs(centers[a] - centers[b]) >= INDEPENDENCE_BP
    away = np.array([all(abs(int(centers[k]) - int(c)) >= INDEPENDENCE_BP for c in excluded)
                     for k in indices])
    allowed = separated & away[i] & away[j]
    return a[allowed], b[allowed]


def coverage_covariates(a: np.ndarray, b: np.ndarray, centers: np.ndarray,
                        density: np.ndarray) -> np.ndarray:
    d1, d2 = density[a], density[b]
    return np.stack([np.minimum(d1, d2), np.maximum(d1, d2),
                     np.log1p(np.abs(centers[a] - centers[b]))], axis=1)


def match_controls(pool_a: np.ndarray, pool_b: np.ndarray, target_a: int,
                   target_b: int, centers: np.ndarray, density: np.ndarray,
                   limit: int = 200) -> tuple[np.ndarray, np.ndarray, dict]:
    """Nearest controls within a broad two-IQR caliper on all three factors."""
    if len(pool_a) == 0:
        return np.empty(0, int), np.empty(0, int), {"pool_pairs": 0, "caliper_pairs": 0}
    pool = coverage_covariates(pool_a, pool_b, centers, density)
    target = coverage_covariates(np.array([target_a]), np.array([target_b]),
                                 centers, density)[0]
    iqr = np.maximum(np.quantile(pool, .75, axis=0) - np.quantile(pool, .25, axis=0), 1e-6)
    standardized = np.abs((pool - target) / iqr)
    eligible = np.flatnonzero((standardized <= 2).all(axis=1))
    rank = np.linalg.norm(standardized[eligible], axis=1)
    selected = eligible[np.argsort(rank, kind="stable")[:limit]]
    diagnostics = {"pool_pairs": len(pool_a), "caliper_pairs": len(eligible),
                   "selected_pairs": len(selected),
                   "selected_median_covariate_distance_iqr": (
                       float(np.median(np.linalg.norm(standardized[selected], axis=1)))
                       if len(selected) else None),
                   "selected_unique_loci": len(set(pool_a[selected]) | set(pool_b[selected]))}
    return pool_a[selected], pool_b[selected], diagnostics


def extract_rep2_and_rep1_coverage(rep1_path: Path, rep2_path: Path,
                                    centers: np.ndarray, chrom: str) -> tuple[np.ndarray, np.ndarray]:
    with open_cool(rep1_path) as cool:
        density = np.array([
            np.log1p(np.maximum(window(cool, chrom, int(center), 5120, 32), 0)).mean()
            for center in centers], dtype=float)
    with open_cool(rep2_path) as cool:
        rep2 = np.stack([normalize(window(cool, chrom, int(center), 5120, 64))
                         for center in centers]).astype(np.float32)
    return rep2, density


def plot_shape_sensitivity(frame: pd.DataFrame, out: Path) -> None:
    """Display every frozen edge and every fixed preprocessing view."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unique = frame.drop_duplicates(["center_a_bp", "center_b_bp", "metric"])
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    colors = ("#246f9a", "#d87921", "#7851a9")
    for color, ((a, b), group) in zip(colors, unique.groupby(["center_a_bp", "center_b_bp"])):
        group = group.set_index("metric").loc[list(METRICS)]
        label = f"{a:,}–{b:,} bp"
        ax.plot(METRICS, group.rep1_r, "--o", color=color, alpha=.65, label=f"{label} rep1")
        ax.plot(METRICS, group.rep2_r, "-o", color=color, linewidth=2, label=f"{label} rep2")
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(ylabel="Pairwise shape correlation", xlabel="Fixed contact-map view",
           title="Frozen three-locus motif loses coherence after background removal")
    ax.tick_params(axis="x", labelrotation=18)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rep1", type=Path, required=True)
    parser.add_argument("--rep2", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, default=Path("outputs/task2_rep1_locked"))
    parser.add_argument("--out", type=Path, default=Path("outputs/task2_shape_artifact_audit"))
    args = parser.parse_args()
    summary = json.loads((args.discovery / "discovery_summary.json").read_text(encoding="utf-8"))
    frozen = summary.get("rep1_top_three_locus_clique_bp")
    if frozen is None or len(frozen) != 3:
        raise ValueError("No rep1-frozen three-locus clique")
    manifest = np.load(args.discovery / "rep1_locked_maps.npz")
    centers = manifest["centers"].astype(int)
    rep1 = manifest["maps"].astype(np.float32)
    known_n = len(manifest["known_centers"])
    eligible = pd.read_csv(args.discovery / "rep1_eligible_loci.csv")
    if not np.array_equal(centers[known_n:], eligible.center.to_numpy(dtype=int)):
        raise ValueError("Discovery positions and saved maps disagree")
    chrom = str(eligible.chrom.iloc[0])
    rep2, density = extract_rep2_and_rep1_coverage(args.rep1, args.rep2, centers, chrom)
    index = {int(center): i for i, center in enumerate(centers)}
    frozen_idx = [index[int(center)] for center in frozen]
    frozen_a, frozen_b = np.triu_indices(3, k=1)
    target_pairs = [(frozen_idx[a], frozen_idx[b]) for a, b in zip(frozen_a, frozen_b)]
    known_a, known_b = pair_pool(np.arange(known_n), centers, np.asarray(frozen))
    unann_a, unann_b = pair_pool(np.arange(known_n, len(centers)), centers,
                                 np.asarray(frozen))
    pools = {"annotated": (known_a, known_b), "unannotated": (unann_a, unann_b)}
    similarities = {name: (corr_matrix(masked_vectors(rep1, name)),
                           corr_matrix(masked_vectors(rep2, name)))
                    for name in METRICS}
    rows = []
    control_rows = []
    for a, b in target_pairs:
        for group, (pool_a, pool_b) in pools.items():
            control_a, control_b, matching = match_controls(
                pool_a, pool_b, a, b, centers, density)
            for ca, cb in zip(control_a, control_b):
                control_rows.append({
                    "target_center_a_bp": int(centers[a]),
                    "target_center_b_bp": int(centers[b]),
                    "control_group": group,
                    "control_center_a_bp": int(centers[ca]),
                    "control_center_b_bp": int(centers[cb]),
                    "control_coverage_low": float(min(density[ca], density[cb])),
                    "control_coverage_high": float(max(density[ca], density[cb])),
                    "control_separation_bp": abs(int(centers[ca]) - int(centers[cb])),
                    "target_coverage_low": float(min(density[a], density[b])),
                    "target_coverage_high": float(max(density[a], density[b])),
                    "target_separation_bp": abs(int(centers[a]) - int(centers[b])),
                })
            for name, (sim1, sim2) in similarities.items():
                controls1 = sim1[control_a, control_b]
                controls2 = sim2[control_a, control_b]
                observed1, observed2 = float(sim1[a, b]), float(sim2[a, b])
                best_known_a = float(sim2[a, :known_n].max())
                best_known_b = float(sim2[b, :known_n].max())
                rows.append({
                    "center_a_bp": int(centers[a]), "center_b_bp": int(centers[b]),
                    "metric": name, "control_group": group,
                    "rep1_r": observed1, "rep2_r": observed2,
                    "rep2_best_known_a_r": best_known_a,
                    "rep2_best_known_b_r": best_known_b,
                    "rep2_pair_exceeds_both_best_known": bool(observed2 > max(best_known_a, best_known_b)),
                    "rep1_coverage_a": float(density[a]),
                    "rep1_coverage_b": float(density[b]),
                    "matched_control_count": len(control_a),
                    "control_pool_size": matching["pool_pairs"],
                    "control_caliper_pairs": matching["caliper_pairs"],
                    "control_unique_loci": matching.get("selected_unique_loci", 0),
                    "control_median_match_distance_iqr": matching.get(
                        "selected_median_covariate_distance_iqr"),
                    "rep1_matched_percentile": (float(np.mean(controls1 < observed1))
                                                if len(controls1) else None),
                    "rep2_matched_percentile": (float(np.mean(controls2 < observed2))
                                                if len(controls2) else None),
                    "rep2_control_p95": (float(np.quantile(controls2, .95))
                                        if len(controls2) else None),
                })
    frame = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "frozen_clique_artifact_audit.csv", index=False)
    control_frame = pd.DataFrame(control_rows)
    control_frame.to_csv(args.out / "coverage_separation_matched_controls.csv", index=False)
    plot_shape_sensitivity(frame, args.out / "shape_artifact_sensitivity.png")
    balance = []
    for (target_a, target_b, group), controls in control_frame.groupby(
            ["target_center_a_bp", "target_center_b_bp", "control_group"]):
        low_interval = controls.control_coverage_low.quantile([.1, .9]).tolist()
        high_interval = controls.control_coverage_high.quantile([.1, .9]).tolist()
        separation_interval = controls.control_separation_bp.quantile([.1, .9]).tolist()
        target_low = float(controls.target_coverage_low.iloc[0])
        target_high = float(controls.target_coverage_high.iloc[0])
        target_separation = int(controls.target_separation_bp.iloc[0])
        balanced = bool(low_interval[0] <= target_low <= low_interval[1] and
                        high_interval[0] <= target_high <= high_interval[1] and
                        separation_interval[0] <= target_separation <= separation_interval[1])
        balance.append({
            "target_pair_bp": [int(target_a), int(target_b)],
            "control_group": group, "control_pairs": len(controls),
            "target_within_control_q10_q90_all_covariates": balanced,
            "target_coverage_low": target_low,
            "control_coverage_low_median": float(controls.control_coverage_low.median()),
            "control_coverage_low_q10_q90": low_interval,
            "target_coverage_high": target_high,
            "control_coverage_high_median": float(controls.control_coverage_high.median()),
            "control_coverage_high_q10_q90": high_interval,
            "target_separation_bp": target_separation,
            "control_separation_bp_median": float(controls.control_separation_bp.median()),
            "control_separation_bp_q10_q90": separation_interval,
        })
    metric_reports = []
    for name in METRICS:
        portion = frame[frame.metric == name]
        distinct = portion.drop_duplicates(["center_a_bp", "center_b_bp"])
        metric_reports.append({
            "metric": name,
            "rep1_internal_min_r": float(distinct.rep1_r.min()),
            "rep2_internal_min_r": float(distinct.rep2_r.min()),
            "rep2_edges_positive": int((distinct.rep2_r > 0).sum()),
            "rep2_edges_exceed_both_best_known": int(
                distinct.rep2_pair_exceeds_both_best_known.sum()),
            "minimum_rep2_matched_percentile_by_group": {
                group: (float(rows.rep2_matched_percentile.min())
                        if rows.matched_control_count.min() >= 20 else None)
                for group, rows in portion.groupby("control_group")},
            "minimum_matched_control_count_by_group": {
                group: int(rows.matched_control_count.min())
                for group, rows in portion.groupby("control_group")},
        })
    report = {"frozen_clique_bp": frozen,
              "views": {"far8": "upper-triangle distance >=8 bins",
                        "far16": "upper-triangle distance >=16 bins",
                        "highpass_far8": "sigma-4 Gaussian background subtracted, then distance >=8 bins",
                        "rowcol_far8": "row and column effects subtracted, then distance >=8 bins"},
              "control_matching": "rep1 raw log coverage of both loci and log genomic separation; two IQR caliper each; nearest 200; excludes candidate-proximal loci",
              "control_balance": balance,
              "target_within_control_q10_q90_all_covariates_by_group": {
                  group: int(sum(item["target_within_control_q10_q90_all_covariates"]
                                 for item in balance if item["control_group"] == group))
                  for group in pools},
              "metric_reports": metric_reports,
              "caution": "Retrospective artifact/robustness audit. Matched pairs are dependent; no p-value or FDR. Unannotated controls have imperfect coverage overlap and cannot substantiate a matched significance claim. Matching does not equalize novelty distance, which lacks adequate overlap."}
    (args.out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
