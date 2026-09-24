"""Paired spatial-block uncertainty for boundary versus short-only predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.summarize_task1_cv import scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path,
                    default=Path("outputs/task1_interval_boundaries/crossfit_predictions.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/task1_interval_boundaries/paired_uncertainty.json"))
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()
    frame = pd.read_csv(args.predictions)
    baseline = frame[frame["mode"] == "short_plus_span"].set_index("id").sort_index()
    improved = frame[frame["mode"] == "both_boundary_plus_span"].set_index("id").sort_index()
    if not baseline.index.equals(improved.index) or not baseline.true_type.equals(improved.true_type):
        raise ValueError("Prediction tables are not aligned")
    if not baseline.spatial_group.equals(improved.spatial_group):
        raise ValueError("Spatial groups differ")
    fields = ("accuracy", "macro_f1", "chid_precision", "chid_recall")
    point = {field: scores(improved)[field] - scores(baseline)[field] for field in fields}
    groups = baseline.spatial_group.unique()
    indices = {group: np.flatnonzero(baseline.spatial_group.to_numpy() == group) for group in groups}
    rng = np.random.default_rng(args.seed)
    boot = {field: [] for field in fields}
    for _ in range(args.resamples):
        drawn = rng.choice(groups, size=len(groups), replace=True)
        positions = np.concatenate([indices[group] for group in drawn])
        old, new = scores(baseline.iloc[positions]), scores(improved.iloc[positions])
        for field in fields:
            boot[field].append(new[field] - old[field])
    summary = {"comparison": "both_boundary_plus_span minus short_plus_span",
               "resampling_unit": "non-overlapping 40 kb genomic component",
               "resamples": args.resamples, "point_difference": point,
               "paired_spatial_bootstrap_95ci": {field: np.quantile(boot[field], [.025, .975]).tolist()
                                                for field in fields},
               "caution": "Exploratory post-comparison intervals, not an external validation set."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
