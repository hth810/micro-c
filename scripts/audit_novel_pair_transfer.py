"""Audit every WT-selected novel-locus pair in unselected Micro-C conditions.

The seven positions are fixed by the WT discovery. For each condition, compare
all 21 possible pairs with the same WT-gated background used by the original
N1 analysis, in both condition replicates and their mean. This is a
post-discovery robustness audit, not a new genome-wide FDR calculation: WT
background windows cannot be fully matched on the novelty-distance gate.
"""
from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd

from microc.analysis import _features
from microc.data import normalize, open_cool, window
from scripts.validate_novel_condition_transfer import correlation_matrix


def similarity_percentile(value: float, background: np.ndarray) -> float:
    """Empirical percentile; ties count against a potential discovery."""
    return float(np.mean(background < value))


def independent_pairs(centers: list[int], min_distance_bp: int = 10240) -> np.ndarray:
    separated = np.abs(np.subtract.outer(centers, centers)) >= min_distance_bp
    return np.triu(separated, 1)


def condition_audit(
    name: str, rep1: Path, rep2: Path, chrom: str,
    candidate_centers: list[int], background_centers: list[int],
) -> tuple[pd.DataFrame, dict]:
    positions = candidate_centers + background_centers
    with ExitStack() as stack:
        cools = [stack.enter_context(open_cool(path)) for path in (rep1, rep2)]
        if cools[0].binsize != cools[1].binsize or tuple(cools[0].chromnames) != tuple(cools[1].chromnames):
            raise ValueError(f"{name}: replicates have different genome or resolution")
        maps = np.stack([[normalize(window(cool, chrom, center, 5120, 64))
                          for cool in cools] for center in positions])
    upper = np.triu_indices(64, k=2)
    views = {"mean": maps.mean(axis=1), "rep1": maps[:, 0], "rep2": maps[:, 1]}
    n = len(candidate_centers)
    background_mask = independent_pairs(background_centers)
    if background_mask.sum() < 100:
        raise ValueError("Too few independent background pairs")
    similarities = {}
    nulls = {}
    for view_name, matrices in views.items():
        for feature_name, vectors in (
            ("upper", np.stack([mat[upper] for mat in matrices])),
            ("coarse", np.stack([_features(mat) for mat in matrices])),
        ):
            sim = correlation_matrix(vectors)
            key = f"{view_name}_{feature_name}"
            similarities[key] = sim
            nulls[key] = sim[n:, n:][background_mask]
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(candidate_centers[i] - candidate_centers[j]) < 10240:
                continue
            row = {"condition": name, "center_a_bp": candidate_centers[i],
                   "center_b_bp": candidate_centers[j],
                   "distance_bp": candidate_centers[j] - candidate_centers[i]}
            for key, sim in similarities.items():
                row[f"{key}_r"] = float(sim[i, j])
                row[f"{key}_percentile"] = similarity_percentile(sim[i, j], nulls[key])
            row["joint_percentile_floor"] = min(row[f"{key}_percentile"] for key in similarities)
            row["passes_all_six_p99"] = row["joint_percentile_floor"] > .99
            rows.append(row)
    frame = pd.DataFrame(rows)
    best = frame.sort_values("joint_percentile_floor", ascending=False).iloc[0]
    n1 = frame[(frame.center_a_bp == 3307520) & (frame.center_b_bp == 4241920)]
    # Search-space audit: the original pair was nominated after looking at
    # seven novelty loci. Show its descriptive rank among *all* same-gate
    # unannotated pairs, not only those seven loci.
    all_pair_mask = independent_pairs(positions)
    all_i, all_j = np.where(all_pair_mask)
    all_floor = np.minimum.reduce([
        np.searchsorted(np.sort(nulls[key]), sim[all_pair_mask], side="left") / len(nulls[key])
        for key, sim in similarities.items()
    ])
    full_order = np.argsort(-all_floor)
    full_top = [{"center_a_bp": int(positions[all_i[k]]),
                 "center_b_bp": int(positions[all_j[k]]),
                 "joint_percentile_floor": float(all_floor[k])}
                for k in full_order[:5]]
    n1_floor = float(n1.iloc[0].joint_percentile_floor) if len(n1) else None
    report = {"condition": name, "n_candidate_loci": n,
              "n_candidate_pairs": len(frame),
              "n_background_windows": len(background_centers),
              "n_independent_background_pairs": int(background_mask.sum()),
              "pairs_passing_all_six_p99": int(frame.passes_all_six_p99.sum()),
              "best_pair_bp": [int(best.center_a_bp), int(best.center_b_bp)],
              "best_joint_percentile_floor": float(best.joint_percentile_floor),
              "n1_joint_percentile_floor": n1_floor,
              "full_gate_independent_pairs": int(len(all_floor)),
              "full_gate_pairs_passing_all_six_p99": int(np.sum(all_floor > .99)),
              "n1_full_gate_rank_by_joint_floor": (int(1 + np.sum(all_floor > n1_floor))
                                                   if n1_floor is not None else None),
              "full_gate_top_five": full_top,
              "caution": ("WT-selected candidate positions and WT-gated controls; "
                          "controls do not match novelty distance. Percentiles are "
                          "descriptive, not independent discovery FDR.")}
    return frame, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", nargs=3, action="append", metavar=("NAME", "REP1", "REP2"), required=True)
    ap.add_argument("--scan", type=Path, default=Path("outputs/task2_strict_revised/all_windows.csv"))
    ap.add_argument("--loci", type=Path, default=Path("outputs/task2_strict_individual_loci.csv"))
    ap.add_argument("--selected", type=Path,
                    default=Path("outputs/task2_strict_revised/all_candidates.csv"))
    ap.add_argument("--discovery-summary", type=Path,
                    default=Path("outputs/task2_strict_revised/summary.json"))
    ap.add_argument("--out", type=Path, default=Path("outputs/task2_all_pair_transfer"))
    args = ap.parse_args()
    scan = pd.read_csv(args.scan)
    loci = pd.read_csv(args.loci)
    if loci.chrom.nunique() != 1:
        raise ValueError("This audit expects loci on one chromosome")
    chrom = str(loci.chrom.iloc[0])
    centers = sorted(loci.representative_center_bp.astype(int).tolist())
    gate = scan[(~scan.is_known) & (scan.known_distance_bp >= 2560) &
                (scan.replicate_oe_r >= .3) &
                (scan.candidate_score >= scan.candidate_score.quantile(.8))]
    if not set(centers).issubset(set(gate.center.astype(int))):
        raise ValueError("Locus representatives do not pass the WT gate")
    background = sorted(set(gate.center.astype(int)) - set(centers))
    selected = pd.read_csv(args.selected)
    threshold = float(json.loads(args.discovery_summary.read_text(encoding="utf-8"))[
        "known_feature_distance_threshold_p90"])
    novelty_controls = selected[(selected.center.isin(background)) &
                                (selected.known_feature_distance > threshold)].center.astype(int).tolist()
    independent_novelty_controls = [p for p in novelty_controls
                                    if all(abs(p - c) >= 10240 for c in centers)]
    frames, summaries = [], []
    for name, rep1, rep2 in args.condition:
        frame, summary = condition_audit(name, Path(rep1), Path(rep2), chrom, centers, background)
        frames.append(frame)
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)
    all_pairs = pd.concat(frames, ignore_index=True)
    cross = all_pairs.groupby(["center_a_bp", "center_b_bp"], sort=False).agg(
        conditions_passing_all_six_p99=("passes_all_six_p99", "sum"),
        worst_condition_joint_percentile_floor=("joint_percentile_floor", "min"),
        median_joint_percentile_floor=("joint_percentile_floor", "median"),
    ).reset_index().sort_values("worst_condition_joint_percentile_floor", ascending=False)
    args.out.mkdir(parents=True, exist_ok=True)
    all_pairs.to_csv(args.out / "all_condition_pairs.csv", index=False)
    cross.to_csv(args.out / "cross_condition_pairs.csv", index=False)
    report = {"conditions": summaries,
              "selection_data_reuse": ("Both WT replicates enter the replicate-r candidate gate; "
                                       "the same WT matrices enter the pair-morphology audit. "
                                       "The WT pair is therefore not a held-out validation."),
              "novelty_distance_threshold_p90": threshold,
              "novelty_distance_matched_background_windows": len(novelty_controls),
              "independent_novelty_distance_matched_background_windows": len(independent_novelty_controls),
              "pair_passing_every_condition": int((cross.conditions_passing_all_six_p99 == len(frames)).sum()),
              "best_worst_condition_pair_bp": [int(cross.iloc[0].center_a_bp),
                                               int(cross.iloc[0].center_b_bp)],
              "caution": ("Exploratory robustness audit of WT-selected loci. "
                          "The same-gate background is not novelty-distance matched "
                          "and does not support a formal discovery p or FDR; conditions "
                          "are biological perturbations, not independent WT replicates.")}
    (args.out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
