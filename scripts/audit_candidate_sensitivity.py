"""Separate candidate score/repeat gates from downstream novelty clustering."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, default=Path("results/task2_all_windows.csv"))
    parser.add_argument("--candidates", type=Path, default=Path("results/task2_new_candidates.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/task2_sensitivity"))
    args = parser.parse_args()
    scan = pd.read_csv(args.scan)
    candidates = pd.read_csv(args.candidates)
    if not set(candidates.center).issubset(set(scan.center)):
        raise ValueError("Candidates must be present in the scan table")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for top_fraction in (.5, .4, .3, .2):
        cutoff = scan.candidate_score.quantile(1 - top_fraction)
        for min_r in (.2, .3, .4, .5):
            selected = scan[(scan.candidate_score >= cutoff) & (scan.replicate_oe_r >= min_r)]
            surviving = sorted(set(selected.center) & set(candidates.center))
            rows.append({"top_fraction": top_fraction, "min_replicate_r": min_r,
                         "score_cutoff": float(cutoff), "all_selected": len(selected),
                         "original_five_surviving": len(surviving),
                         "surviving_centers_bp": ";".join(map(str, surviving))})
    grid = pd.DataFrame(rows)
    grid.to_csv(args.out / "threshold_grid.csv", index=False)
    strict = grid[(grid.top_fraction == .2) & (grid.min_replicate_r == .3)].iloc[0]
    summary = {"original_candidates": len(candidates),
               "strict_score_repeat_survivors": int(strict.original_five_surviving),
               "strict_surviving_centers_bp": strict.surviving_centers_bp,
               "interpretation": "This audits only score and replicate gates; shape novelty and clustering are separate."}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
