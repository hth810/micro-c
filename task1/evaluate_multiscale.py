"""Compare 5.12 kb and 5.12+20.48 kb input on identical spatial folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from task1.evaluate_chid_overlap import spatial_groups
from task1.evaluate_pairwise import evaluate, fit_fold
from task1.run_task1 import CLASSES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=Path("outputs/task1_multiscale/windows.npz"))
    parser.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/task1_multiscale_eval"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    ann = pd.read_csv(args.annotations).reset_index(drop=True)
    with np.load(args.windows) as bundle:
        x = bundle["X"].astype(np.float32)
        ids = bundle["group"].astype(str)
    if x.shape != (2 * len(ann), 2, 128, 128) or not np.array_equal(ids[:len(ann)], ann.id.to_numpy()):
        raise ValueError("Expected 5.12 kb and 20.48 kb windows in annotation order")
    y = ann.type.map({name: index for index, name in enumerate(CLASSES)}).to_numpy(dtype=int)
    groups = spatial_groups(ann, 20480)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
                 .split(ann, y, groups))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = {"epochs": args.epochs, "spatial_group_gap_bp": 20480,
               "evaluation": "original three-class labels; fixed epochs; five spatial-group folds",
               "selection": "no threshold or hyperparameter chosen from outer-fold predictions"}
    rows = []
    for mode, channels in (("short_only", slice(0, 1)), ("dual_scale", slice(0, 2))):
        xx = x[:, channels]
        logits = np.full((2, len(ann), 3), np.nan, dtype=np.float32)
        for fold, (train, test) in enumerate(folds):
            model = fit_fold(xx, y, train, np.empty((0, 2), dtype=int),
                             args.epochs, args.seed + fold, 0.0, device)
            with torch.no_grad():
                for replicate in (0, 1):
                    tensors = torch.from_numpy(xx[test + replicate * len(ann)])
                    output = [model(batch.to(device)).cpu().numpy() for batch in tensors.split(64)]
                    logits[replicate, test] = np.concatenate(output)
        if not np.isfinite(logits).all():
            raise RuntimeError("Incomplete cross-fitted predictions")
        predictions = logits.mean(axis=0).argmax(axis=1)
        summary[mode] = {"structure_level": evaluate(y, predictions),
                         "replicate_sample_level": evaluate(
                             np.tile(y, 2), logits.argmax(axis=2).reshape(-1))}
        for i, row in ann.iterrows():
            rows.append({"mode": mode, "id": row.id, "true_type": row.type,
                         "predicted_type": CLASSES[predictions[i]],
                         "chrom": row.chrom, "center": row.center, "spatial_group": groups[i]})
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "crossfit_predictions.csv", index=False)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
