"""Summarize sensitivity of heuristic Task 4 hits to depth-thinning draws.

The historical default seed 20260921 is the prespecified primary run; the
other two seeds are robustness checks, not opportunities to select more hits.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd


PRIMARY_SEED = 20260921
SEEDS = (PRIMARY_SEED, 20260922, 20260923)
CONDITIONS = ("heat", "rif")


def output_dir(root: Path, condition: str, seed: int) -> Path:
    suffix = "" if seed == PRIMARY_SEED else f"_seed{seed}"
    return root / "outputs" / f"task4_{condition}_order_invariant{suffix}"


def merged_loci(frame: pd.DataFrame) -> int:
    count, right = 0, -1
    for row in frame.sort_values("start").itertuples():
        if int(row.start) >= right:
            count += 1
        right = max(right, int(row.end))
    return count


def summarize_condition(root: Path, condition: str) -> dict:
    frames = {seed: pd.read_csv(output_dir(root, condition, seed) / "reproducible_changes.csv")
              for seed in SEEDS}
    hits = {seed: set(frame.id.astype(str)) for seed, frame in frames.items()}
    intersection = set.intersection(*hits.values())
    union = set.union(*hits.values())
    old_path = root / "outputs" / f"task4_{condition}_final" / "reproducible_changes.csv"
    old_hits = (set(pd.read_csv(old_path).id.astype(str)) if old_path.exists() else None)
    pairs = {}
    for a, b in combinations(SEEDS, 2):
        shared = hits[a] & hits[b]
        either = hits[a] | hits[b]
        pairs[f"{a}_vs_{b}"] = {"shared_hits": len(shared),
                                 "jaccard": len(shared) / len(either) if either else 1.0}
    return {
        "prespecified_primary_seed": PRIMARY_SEED,
        "seeds": {str(seed): {"selected_annotations": len(frame),
                               "merged_loci": merged_loci(frame),
                               "class_counts": frame.type.value_counts().sort_index().to_dict(),
                               "output_dir": str(output_dir(root, condition, seed).relative_to(root))}
                  for seed, frame in frames.items()},
        "all_seed_intersection_count": len(intersection),
        "all_seed_intersection_ids": sorted(intersection),
        "all_seed_intersection_merged_loci": merged_loci(
            frames[PRIMARY_SEED][frames[PRIMARY_SEED].id.isin(intersection)]),
        "all_seed_union_count": len(union),
        "all_seed_jaccard": len(intersection) / len(union) if union else 1.0,
        "pairwise": pairs,
        "historical_run": ({"selected_annotations": len(old_hits),
                            "shared_with_primary": len(old_hits & hits[PRIMARY_SEED]),
                            "jaccard_with_primary": len(old_hits & hits[PRIMARY_SEED]) /
                            len(old_hits | hits[PRIMARY_SEED])} if old_hits is not None else None),
        "caution": "Heuristic hits are sensitive to random depth thinning; no formal FDR or causal proof.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=Path("outputs/task4_order_invariant_sensitivity.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    report = {condition: summarize_condition(root, condition) for condition in CONDITIONS}
    out = args.out if args.out.is_absolute() else root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    for condition in CONDITIONS:
        selected = pd.read_csv(output_dir(root, condition, PRIMARY_SEED) / "reproducible_changes.csv")
        stable = selected[selected.id.isin(report[condition]["all_seed_intersection_ids"])]
        stable.to_csv(out.parent / f"task4_{condition}_stable_changes.csv", index=False)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
