"""Spatial-block uncertainty for saved out-of-fold CHID-presence predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path,
                        default=Path("outputs/task1_chid_overlap/crossfit_predictions.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/task1_chid_overlap/bootstrap.json"))
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions)
    groups = frame.spatial_group.unique()
    member = {group: np.flatnonzero(frame.spatial_group.to_numpy() == group) for group in groups}
    rng = np.random.default_rng(args.seed)
    summary = {"unit": "annotation structure; score is mean of two biological replicates",
               "resampling_unit": "non-overlapping genomic component", "resamples": args.resamples}
    for label_name in ("chid_within_radius", "chid_center_label"):
        y = frame[label_name].to_numpy(dtype=int)
        score = frame.mean_score.to_numpy()
        sample_ap, sample_auc = [], []
        for _ in range(args.resamples):
            draw = rng.choice(groups, size=len(groups), replace=True)
            indices = np.concatenate([member[group] for group in draw])
            if len(np.unique(y[indices])) == 2:
                sample_ap.append(average_precision_score(y[indices], score[indices]))
                sample_auc.append(roc_auc_score(y[indices], score[indices]))
        summary[label_name] = {
            "prevalence": float(y.mean()),
            "average_precision": float(average_precision_score(y, score)),
            "average_precision_95ci": np.quantile(sample_ap, [.025, .975]).tolist(),
            "roc_auc": float(roc_auc_score(y, score)),
            "roc_auc_95ci": np.quantile(sample_auc, [.025, .975]).tolist(),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
