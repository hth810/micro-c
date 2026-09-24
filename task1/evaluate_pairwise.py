"""Spatial cross-validation of CHID-vs-nearby-CHIN pairwise training.

The evaluation target remains the original single-label three-class task.
All settings are fixed before outer-fold scoring; no thresholds are selected
using the held-out folds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from task1.evaluate_chid_overlap import spatial_groups
from task1.run_task1 import CLASSES, seed_all


class PairAwareCNN(nn.Module):
    """Retain central spatial context without batch-dependent normalization."""

    def __init__(self, input_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, 5, padding=2), nn.GroupNorm(4, 16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(.2), nn.Linear(64 * 16, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def nearby_chid_chin_pairs(ann: pd.DataFrame, radius_bp: int) -> np.ndarray:
    pairs = []
    for chrom, frame in ann.groupby("chrom"):
        chid = frame[frame.type == "CHID"]
        chin = frame[frame.type == "CHIN"]
        for ci, row in chid.iterrows():
            distance = (chin.center - row.center).abs()
            if len(distance) and distance.min() <= radius_bp:
                pairs.append((ci, int(distance.idxmin())))
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def fit_fold(x: np.ndarray, y: np.ndarray, train: np.ndarray, pairs: np.ndarray,
             epochs: int, seed: int, pair_weight: float, device: torch.device,
             class_weight_power: float = .5) -> PairAwareCNN:
    seed_all(seed)
    model = PairAwareCNN(x.shape[1]).to(device)
    train_indices = np.r_[train, train + len(y)]
    labels = np.tile(y, 2)
    counts = np.bincount(y[train], minlength=3).astype(np.float32)
    class_weights = (counts.sum() / (3 * np.maximum(counts, 1))) ** class_weight_power
    ce = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))
    loader = DataLoader(TensorDataset(torch.from_numpy(x[train_indices]),
                                      torch.from_numpy(labels[train_indices])),
                        batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    pair_indices = pairs[np.isin(pairs[:, 0], train) & np.isin(pairs[:, 1], train)]
    if pair_weight and len(pair_indices) == 0:
        raise ValueError("No CHID/CHIN pairs in training fold")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = ce(model(xb), yb)
            loss.backward(); optimizer.step()
        if pair_weight:
            for replicate in (0, 1):
                offset = replicate * len(y)
                for chunk in np.array_split(pair_indices, max(1, int(np.ceil(len(pair_indices) / 8)))):
                    positive = torch.from_numpy(x[chunk[:, 0] + offset]).to(device)
                    negative = torch.from_numpy(x[chunk[:, 1] + offset]).to(device)
                    optimizer.zero_grad()
                    pos_logits = model(positive)
                    neg_logits = model(negative)
                    pos_margin = pos_logits[:, 2] - pos_logits[:, 0]
                    neg_margin = neg_logits[:, 2] - neg_logits[:, 0]
                    loss = pair_weight * nn.functional.softplus(1 - (pos_margin - neg_margin)).mean()
                    loss.backward(); optimizer.step()
    return model.eval()


def evaluate(y: np.ndarray, predictions: np.ndarray) -> dict:
    cm = confusion_matrix(y, predictions, labels=range(3))
    return {"accuracy": float(accuracy_score(y, predictions)),
            "macro_f1": float(f1_score(y, predictions, average="macro")),
            "chid_precision": float(classification_report(
                y, predictions, labels=range(3), output_dict=True, zero_division=0)["2"]["precision"]),
            "chid_recall": float(classification_report(
                y, predictions, labels=range(3), output_dict=True, zero_division=0)["2"]["recall"]),
            "confusion_matrix": cm.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=Path("outputs/task1_spatial/extracted_windows.npz"))
    parser.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/task1_pairwise"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--pair-radius-bp", type=int, default=2000)
    parser.add_argument("--pair-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    ann = pd.read_csv(args.annotations).reset_index(drop=True)
    with np.load(args.windows) as bundle:
        x = bundle["X"].astype(np.float32)
        ids = bundle["group"].astype(str)
    if len(x) != 2 * len(ann) or not np.array_equal(ids[:len(ann)], ann.id.to_numpy()):
        raise ValueError("Cached windows do not match annotation order")
    y = ann.type.map({name: index for index, name in enumerate(CLASSES)}).to_numpy(dtype=int)
    groups = spatial_groups(ann, 5120)
    pairs = nearby_chid_chin_pairs(ann, args.pair_radius_bp)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
                 .split(ann, y, groups))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    summary = {"epochs": args.epochs, "pair_radius_bp": args.pair_radius_bp,
               "pair_weight": args.pair_weight, "paired_chid_structures": len(pairs),
               "evaluation": "original 3-class labels; 5 spatial-group out-of-fold predictions",
               "model_selection": "fixed hyperparameters and epochs, no outer-fold threshold tuning"}
    for mode, weight in (("control", 0.0), ("pairwise", args.pair_weight)):
        logits = np.full((2, len(ann), 3), np.nan, dtype=np.float32)
        for fold, (train, test) in enumerate(folds):
            model = fit_fold(x, y, train, pairs, args.epochs, args.seed + fold,
                             weight, device)
            for replicate in (0, 1):
                with torch.no_grad():
                    tensors = torch.from_numpy(x[test + replicate * len(ann)])
                    output = [model(batch.to(device)).cpu().numpy() for batch in tensors.split(64)]
                logits[replicate, test] = np.concatenate(output)
        if not np.isfinite(logits).all():
            raise RuntimeError("Incomplete cross-fitted predictions")
        structure_predictions = logits.mean(axis=0).argmax(axis=1)
        sample_predictions = logits.argmax(axis=2).reshape(-1)
        summary[mode] = {"structure_level": evaluate(y, structure_predictions),
                         "replicate_sample_level": evaluate(np.tile(y, 2), sample_predictions)}
        for i, row in ann.iterrows():
            rows.append({"mode": mode, "id": row.id, "type": row.type,
                         "chrom": row.chrom, "center": row.center,
                         "spatial_group": groups[i], "predicted_type": CLASSES[structure_predictions[i]],
                         "chid_margin": float(logits[:, i, 2].mean() - logits[:, i, 0].mean())})
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "crossfit_predictions.csv", index=False)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
