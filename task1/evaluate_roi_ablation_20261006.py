"""Required fixed-budget CNN image/side-branch ablations, no model selection."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from task1.audit_interval_conditioned_cnn import paired_bootstrap
from task1.audit_roi_cnn_20261006 import side_features
from task1.evaluate_chid_overlap import spatial_groups
from task1.evaluate_nested_neural_20261006 import fit_standardizer, sha256
from task1.evaluate_pairwise import evaluate
from task1.evaluate_roi_cnn_20261006 import EPOCHS, INITIALIZATION_OFFSETS, fit_predict, make_images, reverse_boundary_features
from task1.run_task1 import CLASSES, seed_all


def fit_side_only(side: np.ndarray, y: np.ndarray, train: np.ndarray,
                  test: np.ndarray, seed: int) -> np.ndarray:
    n = len(y)
    fit_ids, test_ids = np.r_[train, train + n], np.r_[test, test + n]
    mean, scale = fit_standardizer(side, fit_ids)
    x = torch.from_numpy(((side - mean) / scale).astype(np.float32))
    labels = torch.from_numpy(np.tile(y, 2))
    counts = np.bincount(y[train], minlength=3).astype(np.float32)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.sqrt(counts.sum() / (3 * counts))))
    output = np.zeros((2, len(test), 3), dtype=np.float32)
    for offset in INITIALIZATION_OFFSETS:
        seed_all(seed + offset)
        model = nn.Sequential(nn.Linear(15, 16), nn.GELU(), nn.Dropout(0.2), nn.Linear(16, 3))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
        rng = torch.Generator().manual_seed(seed + offset)
        for _ in range(EPOCHS):
            model.train()
            for batch in torch.randperm(len(fit_ids), generator=rng).split(32):
                indices = fit_ids[batch.numpy()]
                xb = x[indices]
                if torch.rand((), generator=rng) < 0.5:
                    xb = torch.from_numpy(((reverse_boundary_features(side[indices]) - mean) / scale).astype(np.float32))
                optimizer.zero_grad()
                loss = criterion(model(xb), labels[indices])
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            output += model(x[test_ids]).softmax(dim=1).numpy().reshape(2, len(test), 3) / len(INITIALIZATION_OFFSETS)
    return output


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(2)
    ann = pd.read_csv(args.annotations).reset_index(drop=True)
    n = len(ann)
    with np.load(args.windows) as bundle:
        x = bundle["X"].astype(np.float32)
        if not np.array_equal(bundle["group"].astype(str), np.tile(ann.id.to_numpy(), 2)):
            raise ValueError("Misaligned replicate/annotation inputs")
    images = make_images(x, ann)["interval_roi64"]
    sides = side_features(x, ann)
    empty_sides = np.empty((2 * n, 0), dtype=np.float32)
    y = ann.type.map({label: i for i, label in enumerate(CLASSES)}).to_numpy(dtype=int)
    groups = spatial_groups(ann, 40000)
    args.out.mkdir(parents=True, exist_ok=True)
    protocol_path = args.out / "protocol.json"
    if protocol_path.exists():
        raise ValueError("Ablation protocol already exists")
    protocol = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_sha256": sha256(Path(__file__)),
                "base_source_sha256": sha256(Path("task1/evaluate_roi_cnn_20261006.py")),
                "base_protocol_sha256": sha256(args.cnn / "protocol.json"),
                "input_sha256": {"windows": sha256(args.windows), "annotations": sha256(args.annotations)},
                "reason": "Separate CNN+ROI image contribution from 15-dimensional boundary/length side branch after the main model score was observed",
                "caution": "Retrospective necessary ablation; neither a new holdout nor an architecture selection trial",
                "settings": "24 epochs, AdamW .001/.001, batch32, sqrt train frequency weights, initialization offsets0/31, genomic reversal augmentation",
                "modes": {"roi_images_only": "Identical five CNN contact/mask/ROI channels, side branch removed",
                          "boundary_only_mlp": "Original side branch 15->16 GELU dropout.2->3, no image inputs"},
                "seed": args.seed, "spatial_gap_bp": 40000, "folds": 5,
                "selection": "No choice of model, threshold, settings from ablation results"}
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    probabilities = {mode: np.full((2, n, 3), np.nan, dtype=np.float32)
                     for mode in ("roi_images_only", "boundary_only_mlp")}
    fold_ids = np.full(n, -1)
    for fold, (train, test) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed).split(ann, y, groups)):
        if np.intersect1d(groups[train], groups[test]).size:
            raise ValueError("Spatial component crossed ablation fold")
        fold_ids[test] = fold
        probabilities["roi_images_only"][:, test], _ = fit_predict(images, empty_sides, y, train, test, args.seed + fold * 100)
        probabilities["boundary_only_mlp"][:, test] = fit_side_only(sides, y, train, test, args.seed + fold * 100)
        print(f"completed required ablation fold {fold + 1}/5", flush=True)
    if any(not np.isfinite(p).all() for p in probabilities.values()):
        raise ValueError("Incomplete ablation predictions")
    saved = pd.read_csv(args.cnn / "outer_predictions.csv").set_index("id").loc[ann.id]
    if not np.array_equal(saved.outer_fold, fold_ids) or not np.array_equal(saved.type, ann.type):
        raise ValueError("Ablation folds differ from primary CNN")
    predictions = {mode: p.mean(axis=0).argmax(axis=1) for mode, p in probabilities.items()}
    predictions["cnn_roi_boundary"] = saved.interval_roi64_prediction.map({label: i for i, label in enumerate(CLASSES)}).to_numpy()
    scores = {mode: evaluate(y, p) for mode, p in predictions.items()}
    paired = paired_bootstrap(y, groups, predictions, "cnn_roi_boundary", ("roi_images_only", "boundary_only_mlp"), 4000, args.seed + 97)
    rows = {"id": ann.id, "type": ann.type, "spatial_group": groups, "outer_fold": fold_ids,
            **{f"{mode}_prediction": [CLASSES[i] for i in p] for mode, p in predictions.items()}}
    for mode, p in probabilities.items():
        for rep in (0, 1):
            for cls, label in enumerate(CLASSES):
                rows[f"{mode}_rep{rep + 1}_p_{label}"] = p[rep, :, cls]
    pd.DataFrame(rows).to_csv(args.out / "outer_predictions.csv", index=False)
    summary = {"protocol_sha256": sha256(protocol_path), "caution": protocol["caution"],
               "seed": args.seed, "scores": scores, "paired_spatial_uncertainty": paired,
               "output_sha256": {"outer_predictions.csv": sha256(args.out / "outer_predictions.csv")},
               "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["scores"], indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=Path("outputs/task1_multiscale/windows.npz"))
    parser.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    parser.add_argument("--cnn", type=Path, default=Path("outputs/task1_improve_20261006/roi_cnn_seed21"))
    parser.add_argument("--out", type=Path, default=Path("outputs/task1_improve_20261006/roi_cnn_ablation"))
    parser.add_argument("--seed", type=int, default=20260921)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
