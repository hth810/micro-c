"""Cross-fitted CHID-presence experiment for overlapping annotations.

This is a separate presence-detection question: a window is positive when a
CHID center lies within ``--label-radius-bp`` of its annotated center. It must
not be compared directly with the old three-way single-label precision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from task1.run_task1 import StructureCNN, seed_all


def overlap_targets(ann: pd.DataFrame, radius_bp: int) -> np.ndarray:
    targets = np.zeros(len(ann), dtype=np.int64)
    for chrom, index in ann.groupby("chrom").groups.items():
        subset = ann.loc[index]
        chid_centers = subset.loc[subset.type == "CHID", "center"].to_numpy()
        if len(chid_centers):
            targets[np.asarray(list(index))] = (
                np.abs(subset.center.to_numpy()[:, None] - chid_centers).min(axis=1) <= radius_bp)
    return targets


def spatial_groups(ann: pd.DataFrame, window_bp: int) -> np.ndarray:
    ordered = ann.sort_values(["chrom", "center"])
    boundary = ordered.chrom.ne(ordered.chrom.shift()) | ordered.center.diff().ge(window_bp)
    return pd.Series(boundary.cumsum().to_numpy(), index=ordered.index).sort_index().to_numpy()


def metrics(y: np.ndarray, score: np.ndarray) -> dict:
    pred = score >= .5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"positives": int(y.sum()), "average_precision": float(average_precision_score(y, score)),
            "roc_auc": float(roc_auc_score(y, score)),
            "precision_at_0_5": float(tp / (tp + fp)) if tp + fp else 0.0,
            "recall_at_0_5": float(tp / (tp + fn)) if tp + fn else 0.0,
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=Path("outputs/task1_spatial/extracted_windows.npz"))
    parser.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/task1_chid_overlap"))
    parser.add_argument("--window-bp", type=int, default=5120)
    parser.add_argument("--label-radius-bp", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    seed_all(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    ann = pd.read_csv(args.annotations).reset_index(drop=True)
    with np.load(args.windows) as bundle:
        x = bundle["X"].astype(np.float32)
        ids = bundle["group"].astype(str)
    if len(x) != 2 * len(ann) or not np.array_equal(ids[:len(ann)], ann.id.to_numpy()):
        raise ValueError("Cached windows must contain both replicates in annotation order")
    targets = overlap_targets(ann, args.label_radius_bp)
    original = (ann.type == "CHID").to_numpy(dtype=np.int64)
    groups = spatial_groups(ann, args.window_bp)
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    scores = np.full((2, len(ann)), np.nan, dtype=np.float32)
    fold_rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fold, (train, test) in enumerate(folds.split(ann, original, groups)):
        train_reps = np.r_[train, train + len(ann)]
        test_reps = np.r_[test, test + len(ann)]
        y_train = np.tile(targets, 2)[train_reps]
        model = StructureCNN(1, spatial_grid=4).to(device)
        generator = torch.Generator().manual_seed(args.seed + fold)
        loader = DataLoader(TensorDataset(torch.from_numpy(x[train_reps]),
                         torch.from_numpy(y_train.astype(np.float32))),
                         batch_size=32, shuffle=True, generator=generator)
        positive_weight = (len(y_train) - y_train.sum()) / max(1, y_train.sum())
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        for _ in range(args.epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(xb).flatten(), yb)
                loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            predicted = []
            for batch in torch.from_numpy(x[test_reps]).split(64):
                predicted.append(model(batch.to(device)).sigmoid().flatten().cpu().numpy())
        predicted = np.concatenate(predicted)
        scores[0, test], scores[1, test] = predicted[:len(test)], predicted[len(test):]
        fold_rows.append({"fold": fold, "train_structures": len(train), "test_structures": len(test),
                          "test_chid_annotations": int(original[test].sum()),
                          "test_overlap_positive": int(targets[test].sum())})
    if not np.isfinite(scores).all():
        raise RuntimeError("Not all structures received cross-fitted scores")
    mean_score = scores.mean(axis=0)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ann.id, "chrom": ann.chrom, "center": ann.center,
                  "type": ann.type, "chid_center_label": original,
                  "chid_within_radius": targets, "rep1_score": scores[0],
                  "rep2_score": scores[1], "mean_score": mean_score,
                  "spatial_group": groups}).to_csv(args.out / "crossfit_predictions.csv", index=False)
    summary = {"question": f"CHID presence within {args.label_radius_bp} bp of annotation center",
               "not_comparable_to_single_label_precision": True,
               "folds": fold_rows, "epochs_per_fold": args.epochs,
               "window_bp": args.window_bp, "label_radius_bp": args.label_radius_bp,
               "overlap_aware": metrics(targets, mean_score),
               "original_center_label_diagnostic": metrics(original, mean_score)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
