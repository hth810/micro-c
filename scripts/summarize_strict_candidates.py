"""Join strict novelty loci to cross-condition and shift-control evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loci", type=Path, default=Path("outputs/task2_strict_individual_loci.csv"))
    parser.add_argument("--windows", type=Path, default=Path("outputs/task2_strict_individual.csv"))
    parser.add_argument("--conditions", type=Path, default=Path("outputs/task2_strict_individual_conditions.csv"))
    parser.add_argument("--wt-null", type=Path,
                        default=Path("outputs/task2_strict_individual_validation/candidate_shift_null.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/task2_strict_locus_evidence.csv"))
    args = parser.parse_args()
    loci = pd.read_csv(args.loci)
    windows = pd.read_csv(args.windows)
    conditions = pd.read_csv(args.conditions)
    wt_null = pd.read_csv(args.wt_null)
    result = loci.merge(windows[["center", "known_distance_bp", "known_feature_distance",
                                 "novel_cluster_size", "candidate_new"]],
                        left_on="representative_center_bp", right_on="center", validate="one_to_one")
    result = result.merge(conditions.drop(columns="chrom"), left_on="representative_center_bp",
                          right_on="center_bp", validate="one_to_one")
    result = result.merge(wt_null[["center_bp", "bh_q", "local_bh_q"]],
                          on="center_bp", validate="one_to_one")
    result = result.drop(columns="center")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    summary = {"strict_individual_windows": len(windows), "merged_loci": len(result),
               "strict_cluster_supported_loci": int(result.candidate_new.sum()),
               "loci_heat_local_shift_q_below_005": int((result.heat_local_shift_bh_q < .05).sum()),
               "loci_rif_local_shift_q_below_005": int((result.rif_local_shift_bh_q < .05).sum()),
               "minimum_known_annotation_distance_bp": int(result.known_distance_bp.min()),
               "caution": "Candidate-local shift tests are not genome-wide discovery FDR; no orthogonal biological validation."}
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
