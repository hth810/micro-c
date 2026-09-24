"""Evaluate WT-selected loci in independent H-NS mutant Micro-C replicates.

This is a condition-change check, not a test of de novo discovery FDR or
experimental proof that a candidate is a CHID/CHIN/OPCID.
"""
from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd

from microc.data import normalize, open_cool, save_heatmap, upper_correlation, window


def condition_effect(w1: np.ndarray, w2: np.ndarray, h1: np.ndarray,
                     h2: np.ndarray) -> dict[str, float]:
    """Noise-corrected squared difference of condition-mean contact shapes."""
    mask = np.triu_indices(len(w1), k=2)
    a, b, c, d = (x[mask] for x in (w1, w2, h1, h2))
    wt_mean, hns_mean = (a + b) / 2, (c + d) / 2
    effect = np.mean((wt_mean - hns_mean) ** 2)
    noise = (np.mean((a - b) ** 2) + np.mean((c - d) ** 2)) / 4
    return {"effect_corrected": float(effect - noise),
            "effect_raw": float(effect), "noise_estimate": float(noise),
            "wt_replicate_r": upper_correlation(w1, w2),
            "hns_replicate_r": upper_correlation(h1, h2),
            "wt_hns_mean_r": upper_correlation((w1 + w2) / 2, (h1 + h2) / 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wt1", type=Path, required=True)
    ap.add_argument("--wt2", type=Path, required=True)
    ap.add_argument("--hns1", type=Path, required=True)
    ap.add_argument("--hns2", type=Path, required=True)
    ap.add_argument("--loci", type=Path, default=Path("outputs/task2_strict_individual_loci.csv"))
    ap.add_argument("--scan", type=Path, default=Path("outputs/task2_strict_revised/all_windows.csv"))
    ap.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    ap.add_argument("--out", type=Path, default=Path("outputs/task2_hns_validation"))
    ap.add_argument("--controls-per-locus", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()
    loci = pd.read_csv(args.loci)
    scan = pd.read_csv(args.scan)
    annotations = pd.read_csv(args.annotations)
    known_chid_centers = set(annotations.loc[annotations.type == "CHID", "center"].astype(int))
    candidates = scan.set_index("center").loc[loci.representative_center_bp.to_numpy()].copy()
    if not (candidates.candidate_score >= scan.candidate_score.quantile(.8)).all():
        raise ValueError("Candidate table does not match strict discovery scan")
    eligible = scan[(~scan.is_known) & (scan.known_distance_bp >= 2560) &
                    (scan.replicate_oe_r >= .3) & (scan.candidate_score >= scan.candidate_score.quantile(.8))].copy()
    eligible = eligible[~eligible.center.isin(candidates.index)]
    # Match the two WT selection variables. Reuse controls only once in the
    # matrix pass, but retain per-candidate matches for empirical comparison.
    matched = {}
    for center, candidate in candidates.iterrows():
        pool = eligible[(eligible.center - center).abs() >= 10240].copy()
        pool["match_distance"] = (((pool.candidate_score - candidate.candidate_score) / .1) ** 2 +
                                  ((pool.replicate_oe_r - candidate.replicate_oe_r) / .15) ** 2 +
                                  ((np.log1p(pool.known_distance_bp) -
                                    np.log1p(candidate.known_distance_bp)) / 1.5) ** 2)
        matched[int(center)] = pool.nsmallest(args.controls_per_locus, "match_distance").center.astype(int).tolist()
    centers = sorted(set(candidates.index.astype(int)) | set(sum(matched.values(), [])) |
                     known_chid_centers)
    with ExitStack() as stack:
        cools = [stack.enter_context(open_cool(path)) for path in (args.wt1, args.wt2, args.hns1, args.hns2)]
        if len({cool.binsize for cool in cools}) != 1 or any(
            tuple(cool.chromnames) != tuple(cools[0].chromnames) for cool in cools[1:]
        ):
            raise ValueError("WT and H-NS matrices must share a genome and bin resolution")
        chrom = str(candidates.iloc[0].chrom)
        rows = []
        candidate_matrices = {}
        for i, center in enumerate(centers, 1):
            matrices = [normalize(window(cool, chrom, center, 5120, 64)) for cool in cools]
            if center in candidates.index:
                candidate_matrices[center] = matrices
            rows.append({"center_bp": center, "is_candidate": center in candidates.index,
                         "is_known_chid": center in known_chid_centers,
                         **condition_effect(*matrices)})
            if i % 50 == 0:
                print(f"Evaluated {i}/{len(centers)} loci", flush=True)
    frame = pd.DataFrame(rows).set_index("center_bp")
    rng = np.random.default_rng(args.seed)
    evidence = []
    for center, control_centers in matched.items():
        observed = float(frame.loc[center, "effect_corrected"])
        null = frame.loc[control_centers, "effect_corrected"].to_numpy()
        evidence.append({"center_bp": center, "effect_corrected": observed,
                         "matched_control_count": len(null),
                         "matched_control_median": float(np.median(null)),
                         "matched_control_percentile": float(np.mean(null <= observed)),
                         "matched_empirical_upper_p": float((1 + np.sum(null >= observed)) / (1 + len(null))),
                         "wt_replicate_r": float(frame.loc[center, "wt_replicate_r"]),
                         "hns_replicate_r": float(frame.loc[center, "hns_replicate_r"]),
                         "wt_hns_mean_r": float(frame.loc[center, "wt_hns_mean_r"])})
    result = pd.DataFrame(evidence)
    ordered = result.matched_empirical_upper_p.sort_values()
    adjusted = np.minimum.accumulate((ordered.to_numpy() * len(result) /
                                      np.arange(1, len(result) + 1))[::-1])[::-1]
    result["matched_bh_q"] = pd.Series(np.minimum(adjusted, 1), index=ordered.index).reindex(result.index).to_numpy()
    # Matched-set randomization tests the cohort mean; the per-locus p values
    # have low resolution and should not be read as genome-wide q values.
    observed_mean = float(result.effect_corrected.mean())
    draws = np.array([[frame.loc[int(rng.choice(matched[c])), "effect_corrected"]
                       for c in result.center_bp] for _ in range(5000)])
    null_means = draws.mean(axis=1)
    aggregate_p = float((1 + np.sum(null_means >= observed_mean)) / (1 + len(null_means)))
    observed_median = float(result.effect_corrected.median())
    median_p = float((1 + np.sum(np.median(draws, axis=1) >= observed_median)) / (1 + len(draws)))
    leave_one_out_p = {}
    for i, center in enumerate(result.center_bp):
        observed_without = float(result.effect_corrected.drop(i).mean())
        null_without = np.delete(draws, i, axis=1).mean(axis=1)
        leave_one_out_p[str(center)] = float((1 + np.sum(null_without >= observed_without)) /
                                         (1 + len(null_without)))
    args.out.mkdir(parents=True, exist_ok=True)
    top_center = int(result.loc[result.effect_corrected.idxmax(), "center_bp"])
    top_matrices = candidate_matrices[top_center]
    wt_mean = (top_matrices[0] + top_matrices[1]) / 2
    hns_mean = (top_matrices[2] + top_matrices[3]) / 2
    save_heatmap(wt_mean, args.out / "top_candidate_wt.png", f"WT, {top_center:,} bp")
    save_heatmap(hns_mean, args.out / "top_candidate_hns.png", f"delta H-NS, {top_center:,} bp")
    save_heatmap(wt_mean - hns_mean, args.out / "top_candidate_wt_minus_hns.png",
                 f"WT - delta H-NS, {top_center:,} bp", vmin=-2, vmax=2)
    frame.reset_index().to_csv(args.out / "all_evaluated_loci.csv", index=False)
    result.to_csv(args.out / "candidate_evidence.csv", index=False)
    summary = {"n_candidates": len(result),
               "largest_effect_candidate_center_bp": top_center,
               "n_unique_matched_controls": len(set(sum(matched.values(), []))),
               "known_chid_centers": len(known_chid_centers),
               "known_chid_median_corrected_effect": float(frame.loc[list(known_chid_centers), "effect_corrected"].median()),
               "candidate_mean_corrected_effect": observed_mean,
               "candidate_median_corrected_effect": observed_median,
               "matched_null_mean_corrected_effect": float(null_means.mean()),
               "cohort_matched_empirical_upper_p": aggregate_p,
               "cohort_median_matched_empirical_upper_p": median_p,
               "leave_one_candidate_out_cohort_p": leave_one_out_p,
               "individual_loci_bh_q_below_005": int((result.matched_bh_q < .05).sum()),
               "caution": "WT-selected candidates; H-NS perturbation is a condition check, not novel-structure proof. Matched-null tests are exploratory."}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
