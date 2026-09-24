"""Test whether the WT-discovered N1 motif pair recurs in an unselected condition."""
from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd

from microc.analysis import _features
from microc.data import normalize, open_cool, window


def correlation_matrix(vectors: np.ndarray) -> np.ndarray:
    centered = vectors - vectors.mean(axis=1, keepdims=True)
    length = np.linalg.norm(centered, axis=1, keepdims=True)
    scaled = centered / np.maximum(length, 1e-12)
    return np.clip(scaled @ scaled.T, -1, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rep1", type=Path, required=True)
    ap.add_argument("--rep2", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--members", type=Path,
                    default=Path("outputs/task2_novel_cluster/putative_novel_clusters.csv"))
    ap.add_argument("--scan", type=Path,
                    default=Path("outputs/task2_strict_revised/all_windows.csv"))
    ap.add_argument("--loci", type=Path, default=Path("outputs/task2_strict_individual_loci.csv"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    members = pd.read_csv(args.members)
    if members.putative_cluster.nunique() != 1 or len(members) != 2:
        raise ValueError("This validation expects one two-locus putative cluster")
    centers = members.center.astype(int).tolist()
    scan = pd.read_csv(args.scan)
    all_representatives = pd.read_csv(args.loci).representative_center_bp.astype(int).tolist()
    cutoff = float(scan.candidate_score.quantile(.8))
    gate = scan[(~scan.is_known) & (scan.known_distance_bp >= 2560) &
                (scan.replicate_oe_r >= .3) & (scan.candidate_score >= cutoff)]
    background = gate[~gate.center.isin(all_representatives)].center.astype(int).tolist()
    positions = centers + background
    chrom = str(members.chrom.iloc[0])
    with ExitStack() as stack:
        coolers = [stack.enter_context(open_cool(path)) for path in (args.rep1, args.rep2)]
        if coolers[0].binsize != coolers[1].binsize:
            raise ValueError("Replicate bin sizes differ")
        maps = np.stack([[normalize(window(cooler, chrom, center, 5120, 64))
                          for cooler in coolers] for center in positions])
    upper = np.triu_indices(64, k=2)
    representations = []
    for matrices in (maps.mean(axis=1), maps[:, 0], maps[:, 1]):
        representations.append((correlation_matrix(np.stack([m[upper] for m in matrices])),
                                correlation_matrix(np.stack([_features(m) for m in matrices]))))
    separated = np.abs(np.subtract.outer(background, background)) >= 10240
    pairs = np.triu(separated, 1)
    report = {"condition": args.name, "pair_centers_bp": centers,
              "same_WT_selected_background_windows": len(background),
              "independent_background_pairs": int(pairs.sum()), "comparisons": {}}
    for label, (upper_r, coarse_r) in zip(("mean", "rep1", "rep2"), representations):
        upper_null = upper_r[2:, 2:][pairs]
        coarse_null = coarse_r[2:, 2:][pairs]
        pair_upper, pair_coarse = float(upper_r[0, 1]), float(coarse_r[0, 1])
        report["comparisons"][label] = {
            "upper_r": pair_upper, "upper_background_p99": float(np.quantile(upper_null, .99)),
            "upper_background_percentile": float(np.mean(upper_null < pair_upper)),
            "coarse_r": pair_coarse, "coarse_background_p99": float(np.quantile(coarse_null, .99)),
            "coarse_background_percentile": float(np.mean(coarse_null < pair_coarse)),
            "passes_both_p99": bool(pair_upper > np.quantile(upper_null, .99) and
                                    pair_coarse > np.quantile(coarse_null, .99))}
    report["caution"] = ("The pair and background were selected using WT; this condition did not select them. "
                         "Pass/fail is a robustness check, not independent genome-wide discovery FDR.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
