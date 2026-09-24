"""Check whether candidate patterns replicate under other assayed conditions."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd

from microc.data import normalize, open_cool, upper_correlation, window


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("results/task2_new_candidates.csv"))
    parser.add_argument("--heat1", type=Path, required=True)
    parser.add_argument("--heat2", type=Path, required=True)
    parser.add_argument("--rif1", type=Path, required=True)
    parser.add_argument("--rif2", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/task2_condition_validation.csv"))
    parser.add_argument("--null-shifts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    candidates = pd.read_csv(args.candidates)
    rng = np.random.default_rng(args.seed)
    rows = []
    with ExitStack() as stack:
        cools = {key: stack.enter_context(open_cool(value)) for key, value in
                 (("heat1", args.heat1), ("heat2", args.heat2),
                  ("rif1", args.rif1), ("rif2", args.rif2))}
        for _, candidate in candidates.iterrows():
            matrices = {key: normalize(window(cool, candidate.chrom, int(candidate.center),
                                              int(candidate.length_bp), 64))
                        for key, cool in cools.items()}
            row = {"chrom": candidate.chrom, "center_bp": int(candidate.center),
                         "wt_replicate_r": float(candidate.replicate_oe_r),
                         "heat_replicate_r": upper_correlation(matrices["heat1"], matrices["heat2"]),
                         "rif_replicate_r": upper_correlation(matrices["rif1"], matrices["rif2"]),
                         "heat_vs_rif_rep1_r": upper_correlation(matrices["heat1"], matrices["rif1"])}
            for condition in ("heat", "rif"):
                observed = row[condition + "_replicate_r"]
                offsets = rng.integers(int(candidate.length_bp) // 8,
                                       int(candidate.length_bp) // 2 + 1,
                                       size=args.null_shifts)
                signs = rng.choice([-1, 1], size=args.null_shifts)
                null = [upper_correlation(matrices[condition + "1"], normalize(window(
                    cools[condition + "2"], candidate.chrom,
                    int(candidate.center) + int(offset * sign),
                    int(candidate.length_bp), 64)))
                    for offset, sign in zip(offsets, signs)]
                row[condition + "_local_shift_p95"] = float(np.quantile(null, .95))
                row[condition + "_local_shift_p"] = float((1 + np.sum(np.asarray(null) >= observed)) /
                                                     (len(null) + 1))
            rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    for condition in ("heat", "rif"):
        p_column = condition + "_local_shift_p"
        order = np.argsort(result[p_column].to_numpy())
        ordered_p = result[p_column].to_numpy()[order]
        ranks = np.arange(1, len(result) + 1)
        adjusted = np.minimum.accumulate((ordered_p * len(result) / ranks)[::-1])[::-1]
        q = np.empty(len(result)); q[order] = np.minimum(adjusted, 1)
        result[condition + "_local_shift_bh_q"] = q
    result.to_csv(args.out, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
