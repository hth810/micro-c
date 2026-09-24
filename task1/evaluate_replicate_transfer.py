"""Spatially held-out transfer between the two WT Micro-C replicates.

Training and test loci never overlap, and cross-replicate modes use different
biological replicates at training and prediction time. Annotation boundaries
remain required inputs, so this is not external-label validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from scripts.validate_discovery import shape_features
from task1.evaluate_chid_overlap import spatial_groups
from task1.evaluate_interval_boundaries import interval_features
from task1.evaluate_nested_threshold import fit_model
from task1.evaluate_pairwise import evaluate
from task1.run_task1 import CLASSES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", type=Path, default=Path("outputs/task1_multiscale/windows.npz"))
    ap.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    ap.add_argument("--out", type=Path, default=Path("outputs/task1_replicate_transfer"))
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--spatial-gap-bp", type=int, default=40000)
    args = ap.parse_args()
    ann = pd.read_csv(args.annotations)
    with np.load(args.windows) as bundle:
        x = bundle["X"]
        if x.shape != (2 * len(ann), 2, 128, 128) or not np.array_equal(
            bundle["group"][:len(ann)].astype(str), ann.id.to_numpy()
        ):
            raise ValueError("Cached windows do not match annotations")
    shapes = np.stack([shape_features(mat) for mat in x.reshape(-1, 128, 128)])
    shapes = shapes.reshape(2, len(ann), 2, -1)
    boundaries = np.stack([interval_features(x[rep * len(ann) + i, 1],
                                           int(row.start), int(row.end), int(row.center))
                           for rep in (0, 1) for i, row in ann.iterrows()]).reshape(2, len(ann), -1)
    span = np.log1p(ann.length_bp.to_numpy(dtype=np.float32))[:, None]
    features = [np.column_stack((shapes[rep, :, 0], shapes[rep, :, 1],
                                 boundaries[rep], span)) for rep in (0, 1)]
    features.append((features[0] + features[1]) / 2)
    y = ann.type.map({name: i for i, name in enumerate(CLASSES)}).to_numpy()
    groups = spatial_groups(ann, args.spatial_gap_bp)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
                 .split(ann, y, groups))
    modes = {"rep1_to_rep2": (0, 1), "rep2_to_rep1": (1, 0),
             "rep1_to_rep1": (0, 0), "rep2_to_rep2": (1, 1),
             "both_replicates": (2, 2)}
    summary = {"evaluation": "original 3-class labels, 5 spatial folds; one replicate train to other-replicate test",
               "spatial_gap_bp": args.spatial_gap_bp, "seed": args.seed,
               "caution": "Same published annotation set; cross-replicate transfer is not independent external-label validation."}
    rows = []
    for name, (train_rep, test_rep) in modes.items():
        predicted = np.full(len(ann), -1, dtype=int)
        for fold, (train, test) in enumerate(folds):
            model = fit_model(features[train_rep], y, train, args.seed + fold)
            predicted[test] = model.predict(features[test_rep][test])
        summary[name] = evaluate(y, predicted)
        rows.extend({"mode": name, "id": ann.iloc[i].id, "true_type": ann.iloc[i].type,
                     "predicted_type": CLASSES[predicted[i]], "spatial_group": groups[i]}
                    for i in range(len(ann)))
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "crossfit_predictions.csv", index=False)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
