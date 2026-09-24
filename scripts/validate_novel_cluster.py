"""Group strict novelty loci into independently replicated putative new motifs.

A cluster requires at least two non-overlapping loci, each already passing the
strict WT candidate rule, and pairwise shape similarity above the 99th
percentile of same-gate unannotated background pairs. The pair
must also exceed the background shape thresholds in each WT replicate and be
more similar to each other than either is to any known annotation center.
This defines a computational *putative* motif, not biological confirmation.
"""
from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd

from microc.analysis import _features
from microc.data import normalize, open_cool, save_heatmap, window


def components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in edges:
        ra, rb = root(a), root(b)
        parent[rb] = ra
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    return [members for members in groups.values() if len(members) >= 2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rep1", type=Path, required=True)
    ap.add_argument("--rep2", type=Path, required=True)
    ap.add_argument("--loci", type=Path, default=Path("outputs/task2_strict_individual_loci.csv"))
    ap.add_argument("--candidates", type=Path,
                    default=Path("outputs/task2_strict_revised/new_candidates.csv"))
    ap.add_argument("--scan", type=Path,
                    default=Path("outputs/task2_strict_revised/all_windows.csv"))
    ap.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    ap.add_argument("--out", type=Path, default=Path("outputs/task2_novel_cluster"))
    ap.add_argument("--null-draws", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()
    loci = pd.read_csv(args.loci)
    candidate_windows = pd.read_csv(args.candidates).set_index("center")
    scan = pd.read_csv(args.scan)
    known = pd.read_csv(args.annotations)
    centers = loci.representative_center_bp.astype(int).tolist()
    representatives = candidate_windows.loc[centers]
    if not representatives.candidate_new.all() or representatives.is_known.any():
        raise ValueError("Representative loci do not pass strict novelty selection")
    if any(abs(a - b) < 10240 for i, a in enumerate(centers) for b in centers[i + 1:]):
        raise ValueError("Representative loci must be physically independent")
    score_cutoff = float(scan.candidate_score.quantile(.8))
    eligible = scan[(~scan.is_known) & (scan.known_distance_bp >= 2560) &
                    (scan.replicate_oe_r >= .3) & (scan.candidate_score >= score_cutoff)]
    background = eligible[~eligible.center.isin(centers)].center.astype(int).tolist()
    all_centers = centers + background
    chrom = str(loci.iloc[0].chrom)
    with ExitStack() as stack:
        cools = [stack.enter_context(open_cool(path)) for path in (args.rep1, args.rep2)]
        if cools[0].binsize != cools[1].binsize or tuple(cools[0].chromnames) != tuple(cools[1].chromnames):
            raise ValueError("WT replicates must use the same genome and bin size")
        matrices = [[normalize(window(cool, chrom, center, 5120, 64))
                     for cool in cools] for center in all_centers]
        known_average = np.stack([np.mean([normalize(window(cool, chrom, int(center), 5120, 64))
                                            for cool in cools], axis=0)
                                  for center in known.center])
    average = np.stack([(pair[0] + pair[1]) / 2 for pair in matrices])
    upper = np.stack([mat[np.triu_indices(64, k=2)] for mat in average])
    coarse = np.stack([_features(mat) for mat in average])
    known_upper = np.stack([mat[np.triu_indices(64, k=2)] for mat in known_average])
    known_coarse = np.stack([_features(mat) for mat in known_average])
    candidate_known_upper = np.corrcoef(np.concatenate((upper[:len(centers)], known_upper)))[
        :len(centers), len(centers):]
    candidate_known_coarse = np.corrcoef(np.concatenate((coarse[:len(centers)], known_coarse)))[
        :len(centers), len(centers):]
    upper_r = np.corrcoef(upper)
    coarse_r = np.corrcoef(coarse)
    rep_upper_r = [np.corrcoef(np.stack([pair[rep][np.triu_indices(64, k=2)]
                                            for pair in matrices])) for rep in (0, 1)]
    rep_coarse_r = [np.corrcoef(np.stack([_features(pair[rep])
                                             for pair in matrices])) for rep in (0, 1)]
    if not np.isfinite(upper_r).all() or not np.isfinite(coarse_r).all():
        raise ValueError("Non-finite shape similarity")
    n = len(centers)
    separated = np.abs(np.subtract.outer(all_centers, all_centers)) >= 10240
    background_pairs = np.triu(separated[n:, n:], 1)
    threshold_upper = float(np.quantile(upper_r[n:, n:][background_pairs], .99))
    threshold_coarse = float(np.quantile(coarse_r[n:, n:][background_pairs], .99))
    rep_upper_thresholds = [float(np.quantile(mat[n:, n:][background_pairs], .99))
                            for mat in rep_upper_r]
    rep_coarse_thresholds = [float(np.quantile(mat[n:, n:][background_pairs], .99))
                             for mat in rep_coarse_r]
    rows, edges = [], []
    for i in range(n):
        for j in range(i + 1, n):
            rep1_r, rep2_r = (float(mat[i, j]) for mat in rep_upper_r)
            rep1_coarse, rep2_coarse = (float(mat[i, j]) for mat in rep_coarse_r)
            more_similar_than_known = bool(
                upper_r[i, j] > max(candidate_known_upper[i].max(), candidate_known_upper[j].max())
                and coarse_r[i, j] > max(candidate_known_coarse[i].max(), candidate_known_coarse[j].max()))
            supported = bool(separated[i, j] and upper_r[i, j] > threshold_upper and
                             coarse_r[i, j] > threshold_coarse and
                             rep1_r > rep_upper_thresholds[0] and
                             rep2_r > rep_upper_thresholds[1] and
                             rep1_coarse > rep_coarse_thresholds[0] and
                             rep2_coarse > rep_coarse_thresholds[1] and
                             more_similar_than_known)
            rows.append({"center_a_bp": centers[i], "center_b_bp": centers[j],
                         "distance_bp": abs(centers[i] - centers[j]),
                         "mean_map_upper_r": float(upper_r[i, j]),
                         "mean_map_coarse_r": float(coarse_r[i, j]),
                         "rep1_upper_r": rep1_r, "rep2_upper_r": rep2_r,
                         "rep1_coarse_r": rep1_coarse, "rep2_coarse_r": rep2_coarse,
                         "pair_exceeds_each_member_best_known": more_similar_than_known,
                         "supported_pair": supported})
            if supported:
                edges.append((i, j))
    clusters = components(n, edges)
    null_upper = upper_r[n:, n:]
    null_rep2_upper = rep_upper_r[1][n:, n:]
    null_separated = separated[n:, n:]
    candidate_upper = upper_r[:n, :n][np.triu(separated[:n, :n], 1)]
    observed_max = float(candidate_upper.max())
    candidate_rep2_upper = rep_upper_r[1][:n, :n][np.triu(separated[:n, :n], 1)]
    observed_rep2_max = float(candidate_rep2_upper.max())
    rng = np.random.default_rng(args.seed)
    maxima = []
    rep2_maxima = []
    attempts = 0
    while len(maxima) < args.null_draws:
        attempts += 1
        if attempts > args.null_draws * 100:
            raise RuntimeError("Too few independent background sets for the null")
        chosen = rng.choice(len(background), n, replace=False)
        if not np.all(null_separated[np.ix_(chosen, chosen)][np.triu_indices(n, 1)]):
            continue
        mask = np.triu(np.ones((n, n), dtype=bool), 1)
        maxima.append(float(null_upper[np.ix_(chosen, chosen)][mask].max()))
        rep2_maxima.append(float(null_rep2_upper[np.ix_(chosen, chosen)][mask].max()))
    max_pair_p = float((1 + np.sum(np.asarray(maxima) >= observed_max)) / (1 + len(maxima)))
    rep2_max_pair_p = float((1 + np.sum(np.asarray(rep2_maxima) >= observed_rep2_max)) /
                            (1 + len(rep2_maxima)))
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "all_candidate_pairs.csv", index=False)
    cluster_rows = []
    for cluster_no, members in enumerate(clusters, 1):
        medoid_index = max(members, key=lambda index: (
            float(np.mean([upper_r[index, other] for other in members if other != index])),
            float(representatives.iloc[index].candidate_score)))
        for i in members:
            locus = loci.iloc[i]
            candidate = representatives.iloc[i]
            cluster_rows.append({"putative_cluster": f"N{cluster_no}",
                                 "cluster_independent_loci": len(members),
                                 "cluster_medoid_center_bp": centers[medoid_index],
                                 "chrom": locus.chrom, "start": int(locus.start),
                                 "end": int(locus.end), "center": centers[i],
                                 "length_bp": int(locus.end - locus.start),
                                 "candidate_score": float(candidate.candidate_score),
                                 "wt_replicate_oe_r": float(candidate.replicate_oe_r),
                                 "known_distance_bp": int(candidate.known_distance_bp),
                                 "known_feature_distance": float(candidate.known_feature_distance),
                                 "largest_known_upper_r": float(candidate_known_upper[i].max()),
                                 "largest_known_coarse_r": float(candidate_known_coarse[i].max()),
                                 "distinction": "no known annotation overlap; morphology outside known nearest-neighbor p90"})
            for rep, matrix in enumerate(matrices[i], 1):
                save_heatmap(matrix, args.out / f"N{cluster_no}_{centers[i]}_rep{rep}.png",
                             f"N{cluster_no}: {centers[i]:,} bp, WT rep{rep}")
    columns = ["putative_cluster", "cluster_independent_loci", "cluster_medoid_center_bp",
               "chrom", "start", "end",
               "center", "length_bp", "candidate_score", "wt_replicate_oe_r",
               "known_distance_bp", "known_feature_distance", "largest_known_upper_r",
               "largest_known_coarse_r", "distinction"]
    pd.DataFrame(cluster_rows, columns=columns).to_csv(args.out / "putative_novel_clusters.csv", index=False)
    summary = {"strict_novelty_loci": n, "matched_background_windows": len(background),
               "background_definition": "same score top-20% and WT replicate-r>=0.3 gates; unannotated; known distance >=2560 bp",
               "known_annotations_compared": len(known),
               "independent_background_pairs": int(background_pairs.sum()),
               "pair_upper_r_threshold_p99": threshold_upper,
               "pair_coarse_r_threshold_p99": threshold_coarse,
               "replicate_upper_r_thresholds_p99": rep_upper_thresholds,
               "replicate_coarse_r_thresholds_p99": rep_coarse_thresholds,
               "supported_independent_pairs": len(edges), "putative_novel_clusters": len(clusters),
               "cluster_member_counts": [len(members) for members in clusters],
               "largest_candidate_pair_mean_upper_r": observed_max,
               "matched_background_max_pair_empirical_p": max_pair_p,
               "rep2_largest_candidate_pair_upper_r": observed_rep2_max,
               "rep2_matched_background_max_pair_empirical_p": rep2_max_pair_p,
               "null_draws": args.null_draws,
               "caution": "Exploratory WT-selected motif; matched-background max-p is not genome-wide discovery FDR or experimental proof."}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
