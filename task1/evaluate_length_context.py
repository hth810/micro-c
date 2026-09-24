"""Spatial CV of contact-only, annotation-span-only, and hybrid classifiers.

The annotation span is an auxiliary feature available for known structures,
not for arbitrary genome windows. The span-only baseline prevents attributing
metadata discrimination to the contact-map CNN.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from task1.evaluate_chid_overlap import spatial_groups
from task1.evaluate_pairwise import PairAwareCNN, evaluate, fit_fold
from task1.run_task1 import CLASSES, seed_all


class LengthAwareCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = PairAwareCNN(1).features
        self.classifier = nn.Sequential(nn.Dropout(.2), nn.Linear(64 * 16 + 1, 3))

    def forward(self, x: torch.Tensor, log_span: torch.Tensor) -> torch.Tensor:
        features = self.features(x).flatten(1)
        return self.classifier(torch.cat((features, log_span[:, None]), dim=1))


def fit_hybrid(x: np.ndarray, y: np.ndarray, log_span: np.ndarray, train: np.ndarray,
               epochs: int, seed: int, device: torch.device) -> tuple[LengthAwareCNN, float, float]:
    seed_all(seed)
    model = LengthAwareCNN().to(device)
    mean = float(log_span[train].mean())
    scale = max(float(log_span[train].std()), 1e-6)
    standardized = (log_span - mean) / scale
    train_reps = np.r_[train, train + len(y)]
    labels = np.tile(y, 2)
    span_reps = np.tile(standardized, 2).astype(np.float32)
    counts = np.bincount(y[train], minlength=3).astype(np.float32)
    weights = np.sqrt(counts.sum() / (3 * np.maximum(counts, 1)))
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    loader = DataLoader(TensorDataset(torch.from_numpy(x[train_reps]),
                                      torch.from_numpy(span_reps[train_reps]),
                                      torch.from_numpy(labels[train_reps])),
                        batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for xb, lb, yb in loader:
            xb, lb, yb = xb.to(device), lb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb, lb), yb)
            loss.backward(); optimizer.step()
    return model.eval(), mean, scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=Path("outputs/task1_spatial/extracted_windows.npz"))
    parser.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/task1_length_context"))
    parser.add_argument("--epochs", type=int, default=12)
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
    length = np.log1p(ann.length_bp.to_numpy(dtype=np.float32))
    groups = spatial_groups(ann, 40000)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
                 .split(ann, y, groups))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = {"epochs": args.epochs, "spatial_group_gap_bp": 40000,
               "evaluation": "original three-class labels; five non-overlapping spatial folds",
               "length_feature": "log1p(annotation interval bp), standardized within each training fold",
               "caution": "Annotation span is not known for an arbitrary genome window"}
    rows = []
    for mode in ("contact_only", "length_only", "contact_plus_length"):
        predictions = np.full(len(ann), -1, dtype=int)
        for fold, (train, test) in enumerate(folds):
            if mode == "length_only":
                model = make_pipeline(StandardScaler(), LogisticRegression(
                    class_weight="balanced", max_iter=1000))
                model.fit(length[train, None], y[train])
                predictions[test] = model.predict(length[test, None])
            elif mode == "contact_only":
                model = fit_fold(x, y, train, np.empty((0, 2), dtype=int),
                                 args.epochs, args.seed + fold, 0.0, device)
                logits = np.zeros((len(test), 3), dtype=np.float32)
                with torch.no_grad():
                    for replicate in (0, 1):
                        output = [model(batch.to(device)).cpu().numpy() for batch in
                                  torch.from_numpy(x[test + replicate * len(ann)]).split(64)]
                        logits += np.concatenate(output) / 2
                predictions[test] = logits.argmax(axis=1)
            else:
                model, mean, scale = fit_hybrid(x, y, length, train, args.epochs,
                                                args.seed + fold, device)
                logits = np.zeros((len(test), 3), dtype=np.float32)
                with torch.no_grad():
                    for replicate in (0, 1):
                        indices = test + replicate * len(ann)
                        xb = torch.from_numpy(x[indices]).to(device)
                        lb = torch.from_numpy(((length[test] - mean) / scale).astype(np.float32)).to(device)
                        logits += model(xb, lb).cpu().numpy() / 2
                predictions[test] = logits.argmax(axis=1)
        if (predictions < 0).any():
            raise RuntimeError("Incomplete cross-fitted predictions")
        summary[mode] = evaluate(y, predictions)
        for i, row in ann.iterrows():
            rows.append({"mode": mode, "id": row.id, "true_type": row.type,
                         "predicted_type": CLASSES[predictions[i]], "center": row.center,
                         "length_bp": row.length_bp, "spatial_group": groups[i]})
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "crossfit_predictions.csv", index=False)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
