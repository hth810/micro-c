"""Recover per-window novelty outliers from an existing discovery run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    table = pd.read_csv(args.discovery / "all_candidates.csv")
    summary = json.loads((args.discovery / "summary.json").read_text(encoding="utf-8"))
    threshold = summary["known_feature_distance_threshold_p90"]
    if threshold is None or "known_feature_distance" not in table:
        raise ValueError("Discovery run has no calibrated novelty distances")
    individuals = table[(~table.is_known) & (table.known_feature_distance > threshold)].copy()
    individuals["candidate_individual"] = True
    individuals = individuals.sort_values(["chrom", "start", "end"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    individuals.to_csv(args.out, index=False)
    loci = []
    for chrom, frame in individuals.groupby("chrom", sort=False):
        current = []
        current_end = -1
        for _, row in frame.iterrows():
            if current and int(row.start) >= current_end:
                best = max(current, key=lambda item: item.candidate_score)
                loci.append({"chrom": chrom, "start": int(current[0].start),
                             "end": current_end, "representative_center_bp": int(best.center),
                             "windows": len(current), "max_candidate_score": float(best.candidate_score),
                             "representative_wt_replicate_r": float(best.replicate_oe_r)})
                current = []
            current.append(row)
            current_end = max(current_end, int(row.end))
        if current:
            best = max(current, key=lambda item: item.candidate_score)
            loci.append({"chrom": chrom, "start": int(current[0].start),
                         "end": current_end, "representative_center_bp": int(best.center),
                         "windows": len(current), "max_candidate_score": float(best.candidate_score),
                         "representative_wt_replicate_r": float(best.replicate_oe_r)})
    loci_path = args.out.with_name(args.out.stem + "_loci.csv")
    pd.DataFrame(loci).to_csv(loci_path, index=False)
    print(json.dumps({"individual_novelty_windows": len(individuals),
                      "merged_loci": len(loci),
                      "cluster_supported_windows": int(individuals.candidate_new.sum()),
                      "novelty_threshold": threshold}, indent=2))


if __name__ == "__main__":
    main()
