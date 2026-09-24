"""Spatial-block uncertainty for saved three-class cross-fitted predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def scores(frame: pd.DataFrame) -> dict:
    true = frame.true_type.to_numpy()
    pred = frame.predicted_type.to_numpy()
    positive = true == "CHID"
    predicted = pred == "CHID"
    return {"accuracy": float(accuracy_score(true, pred)),
            "macro_f1": float(f1_score(true, pred, labels=["CHIN", "OPCID", "CHID"], average="macro")),
            "chid_precision": float(precision_score(positive, predicted, zero_division=0)),
            "chid_recall": float(recall_score(positive, predicted, zero_division=0)),
            "chid_true_positive": int((positive & predicted).sum()),
            "chid_predicted_positive": int(predicted.sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path,
                        default=Path("outputs/task1_multiscale_eval/crossfit_predictions.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/task1_multiscale_eval/uncertainty.json"))
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions)
    rng = np.random.default_rng(args.seed)
    output = {"resampling_unit": "non-overlapping 20.48 kb genomic component",
              "resamples": args.resamples, "modes": {}}
    for mode, subset in frame.groupby("mode"):
        groups = subset.spatial_group.unique()
        indices = {group: np.flatnonzero(subset.spatial_group.to_numpy() == group) for group in groups}
        sampled = {field: [] for field in ("accuracy", "macro_f1", "chid_precision", "chid_recall")}
        for _ in range(args.resamples):
            drawn = rng.choice(groups, size=len(groups), replace=True)
            selection = np.concatenate([indices[group] for group in drawn])
            row = scores(subset.iloc[selection])
            for field in sampled:
                sampled[field].append(row[field])
        point = scores(subset)
        point["spatial_bootstrap_95ci"] = {
            field: np.quantile(values, [.025, .975]).tolist() for field, values in sampled.items()}
        output["modes"][mode] = point
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
