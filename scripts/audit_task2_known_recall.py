"""Compare Task 2 known-structure recall with chromosome-shifted selections.

All nonzero circular shifts of the fixed selected-window mask preserve its
window count and clustering. This is a descriptive spatial baseline, not an
independent confirmation or a formal biological discovery p-value.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def canonical_chrom(chrom: object) -> str:
    return "NC_000913.3" if str(chrom) in {"MG1655", "NC_000913.3"} else str(chrom)


def overlap_matrix(annotations: pd.DataFrame, windows: pd.DataFrame) -> np.ndarray:
    """Rows are annotated intervals; columns are genome-scan windows."""
    a = annotations.reset_index(drop=True)
    w = windows.reset_index(drop=True)
    a_chrom = np.array([canonical_chrom(chrom) for chrom in a.chrom])
    w_chrom = np.array([canonical_chrom(chrom) for chrom in w.chrom])
    return ((a_chrom[:, None] == w_chrom[None, :]) &
            (a.start.to_numpy()[:, None] < w.end.to_numpy()[None, :]) &
            (a.end.to_numpy()[:, None] > w.start.to_numpy()[None, :]))


def merge_annotated_loci(annotations: pd.DataFrame) -> pd.DataFrame:
    ordered = annotations[["chrom", "start", "end"]].copy()
    ordered["chrom"] = ordered.chrom.map(canonical_chrom)
    ordered = ordered.sort_values(["chrom", "start", "end"])
    merged = []
    for row in ordered.itertuples():
        if merged and merged[-1][0] == row.chrom and int(row.start) < merged[-1][2]:
            merged[-1][2] = max(merged[-1][2], int(row.end))
        else:
            merged.append([row.chrom, int(row.start), int(row.end)])
    return pd.DataFrame(merged, columns=["chrom", "start", "end"])


def shifted_recall(overlap: np.ndarray, selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if overlap.ndim != 2 or selected.shape != (overlap.shape[1],):
        raise ValueError("Annotation-window overlap and selection dimensions differ")
    if not 0 < selected.sum() < len(selected):
        raise ValueError("Selection must include some but not all windows")
    observed = np.any(overlap & selected[None, :], axis=1)
    null = np.stack([np.any(overlap & np.roll(selected, shift)[None, :], axis=1)
                     for shift in range(1, len(selected))])
    return observed, null


def summarize_hits(observed: np.ndarray, null: np.ndarray) -> dict:
    actual = float(observed.mean())
    shifted = null.mean(axis=1)
    return {"annotated_intervals": len(observed),
            "observed_recall": actual,
            "circular_shift_median": float(np.median(shifted)),
            "circular_shift_p05": float(np.quantile(shifted, .05)),
            "circular_shift_p95": float(np.quantile(shifted, .95)),
            "circular_shift_rank_percentile": float(np.mean(shifted < actual)),
            "circular_shift_tail_fraction_ge_observed": float(np.mean(shifted >= actual))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=Path("results/task2_all_windows.csv"))
    parser.add_argument("--selected", type=Path, default=Path("results/task2_all_candidates.csv"))
    parser.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/task2_known_recall_audit.json"))
    args = parser.parse_args()
    windows = pd.read_csv(args.windows).sort_values(["chrom", "center"]).reset_index(drop=True)
    selected = pd.read_csv(args.selected)
    ann = pd.read_csv(args.annotations)
    keys = set(zip(selected.chrom.astype(str), selected.center.astype(int)))
    chosen = np.array([(str(row.chrom), int(row.center)) in keys
                       for row in windows.itertuples()], dtype=bool)
    if chosen.sum() != len(selected):
        raise ValueError("Selected windows are missing or duplicated in full scan")
    overlap = overlap_matrix(ann, windows)
    observed, null = shifted_recall(overlap, chosen)
    merged = merge_annotated_loci(ann)
    locus_observed, locus_null = shifted_recall(overlap_matrix(merged, windows), chosen)
    report = {"windows": len(windows), "selected_windows": int(chosen.sum()),
              "nonzero_circular_shifts": len(windows) - 1,
              "overall": summarize_hits(observed, null),
              "merged_nonoverlapping_loci": summarize_hits(locus_observed, locus_null),
              "by_type": {name: summarize_hits(observed[mask], null[:, mask])
                          for name in sorted(ann.type.unique())
                          for mask in [(ann.type == name).to_numpy()]},
              "caution": "The shift baseline preserves selection clustering but is descriptive; annotations and data have been examined in prior experiments. Multiple overlapping annotations are not independent loci."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
