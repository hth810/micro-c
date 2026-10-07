"""Fixed, interval-aligned matrix CNN experiment with spatial held-out folds.

This is a retrospective follow-up using previously inspected data. Settings
are frozen before this run's scores, but cannot remove historical selection
bias. ROI denotes the supplied annotated interval, not a discovered interval.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from task1.audit_interval_conditioned_cnn import paired_bootstrap
from task1.evaluate_chid_overlap import spatial_groups
from task1.evaluate_interval_conditioned_cnn import interval_channels
from task1.evaluate_nested_neural_20261006 import fit_standardizer, load_features, make_protocol, sha256
from task1.evaluate_pairwise import evaluate
from task1.run_task1 import CLASSES, seed_all


MODES = ("contact64", "interval_roi64")
EPOCHS = 24
INITIALIZATION_OFFSETS = (0, 31)


def reverse_boundary_features(side: np.ndarray) -> np.ndarray:
    """Reverse LL/LI/RI/RR and the clipping flags with genomic axes."""
    if side.shape[-1] != 15:
        raise ValueError("Expected 14 directional boundary descriptors and log length")
    return side[..., [1, 0, 12, 13, 10, 11, 6, 7, 8, 9, 4, 5, 2, 3, 14]]


def make_images(x: np.ndarray, ann: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(ann)
    if x.shape != (2 * n, 2, 128, 128):
        raise ValueError("Expected two 128-bin contact scales and two replicates")
    masks = interval_channels(ann)
    compact = nn.functional.avg_pool2d(torch.from_numpy(x), 2).numpy()
    mask64 = nn.functional.avg_pool2d(torch.from_numpy(masks), 2).numpy()
    roi = np.zeros((2 * n, 1, 64, 64), dtype=np.float32)
    for rep in (0, 1):
        for i, row in ann.iterrows():
            left = int(np.clip(np.floor(64 + (row.start - row.center) / 160), 0, 127))
            right = int(np.clip(np.ceil(64 + (row.end - row.center) / 160), left + 1, 128))
            region = torch.from_numpy(x[rep * n + i, 1, left:right, left:right])[None, None]
            roi[rep * n + i] = nn.functional.interpolate(region, size=(64, 64), mode="bilinear",
                                                         align_corners=False).numpy()[0]
    return {"contact64": compact,
            "interval_roi64": np.concatenate((compact, mask64, roi), axis=1)}


class CompactIntervalCNN(nn.Module):
    def __init__(self, channels: int, side_features: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 8, 5, stride=2, padding=2), nn.GroupNorm(2, 8), nn.GELU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(512, 64), nn.GELU())
        self.side = nn.Sequential(nn.Linear(side_features, 16), nn.GELU()) if side_features else None
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(80 if side_features else 64, 3))

    def forward(self, image: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(image)
        if self.side is not None:
            embedding = torch.cat((embedding, self.side(side)), dim=1)
        return self.classifier(embedding)


def fit_predict(image: np.ndarray, side: np.ndarray, y: np.ndarray,
                train: np.ndarray, test: np.ndarray, seed: int) -> tuple[np.ndarray, list[dict]]:
    n = len(y)
    if np.intersect1d(train, test).size:
        raise ValueError("Train/prediction structures overlap")
    fit_ids, test_ids = np.r_[train, train + n], np.r_[test, test + n]
    mean, scale = fit_standardizer(side, fit_ids)
    side_tensor = torch.from_numpy(((side - mean) / scale).astype(np.float32))
    image_tensor = torch.from_numpy(image)
    labels = torch.from_numpy(np.tile(y, 2))
    counts = np.bincount(y[train], minlength=3).astype(np.float32)
    class_weights = np.sqrt(counts.sum() / (3 * np.maximum(counts, 1)))
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights))
    output = np.zeros((2, len(test), 3), dtype=np.float32)
    states = []
    for offset in INITIALIZATION_OFFSETS:
        seed_all(seed + offset)
        model = CompactIntervalCNN(image.shape[1], side.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
        rng = torch.Generator().manual_seed(seed + offset)
        for epoch in range(EPOCHS):
            model.train()
            for batch in torch.randperm(len(fit_ids), generator=rng).split(32):
                ids = fit_ids[batch.numpy()]
                xb = image_tensor[ids]
                # Topology is unaffected by reversing both genomic axes. All
                # five channels undergo the same transform; directional flank
                # summaries and clipping flags must reverse too.
                sb = side_tensor[ids]
                if torch.rand((), generator=rng) < 0.5:
                    xb = xb.flip((2, 3))
                    if side.shape[1]:
                        reversed_raw = reverse_boundary_features(side[ids])
                        sb = torch.from_numpy(((reversed_raw - mean) / scale).astype(np.float32))
                optimizer.zero_grad()
                loss = loss_fn(model(xb, sb), labels[ids])
                loss.backward()
                optimizer.step()
        model.eval()
        predictions = []
        with torch.no_grad():
            for indices in np.array_split(test_ids, max(1, int(np.ceil(len(test_ids) / 64)))):
                predictions.append(model(image_tensor[indices], side_tensor[indices]).softmax(dim=1).numpy())
        output += np.concatenate(predictions).reshape(2, len(test), 3) / len(INITIALIZATION_OFFSETS)
        states.append({"state_dict": model.state_dict(), "initialization_seed": seed + offset,
                       "side_mean": mean, "side_scale": scale})
    return output, states


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(2)
    start = time.perf_counter()
    ann, descriptors = load_features(args.windows, args.annotations)
    with np.load(args.windows) as bundle:
        images = make_images(bundle["X"].astype(np.float32), ann)
    # Retain boundary summaries and length as an explicitly disclosed learned
    # side branch; no precomputed contact-shape descriptors enter the CNN.
    sides = {"contact64": np.empty((2 * len(ann), 0), dtype=np.float32),
             "interval_roi64": descriptors["contact_interval"][:, -15:]}
    y = ann.type.map({label: i for i, label in enumerate(CLASSES)}).to_numpy(dtype=np.int64)
    groups = spatial_groups(ann, 40000)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed).split(ann, y, groups))
    args.out.mkdir(parents=True, exist_ok=True)
    protocol_path = args.out / "protocol.json"
    if protocol_path.exists():
        raise ValueError("Output already has a frozen protocol")
    protocol = make_protocol(args, ann, descriptors, groups, folds)
    protocol.update({"source_sha256": {**protocol["source_sha256"], "task1/evaluate_roi_cnn_20261006.py": sha256(Path(__file__)),
                                        "task1/evaluate_interval_conditioned_cnn.py": sha256(Path("task1/evaluate_interval_conditioned_cnn.py"))},
                     "neural_architecture": "5-channel 64x64 contact+interval+ROI matrix input -> conv8/16/32 stride2 GroupNorm GELU -> spatial4x4 -> dense64; interval boundary/loglength branch dense16; combined dense3",
                     "representation": "64x64 average-pooled short/long contact windows, known interval within/cross masks, bilinear-resized long-scale interval ROI; 15 boundary/span side features",
                     "optimizer": {"name": "AdamW", "lr": 0.001, "weight_decay": 0.001, "batch_size": 32},
                     "selection": "No within-run architecture, epoch, class weight or threshold selection; both fixed ablations retained",
                     "inner_folds": None, "epochs": EPOCHS, "class_weight_power": 0.5,
                     "augmentation": "Random genomic axis reversal applied to every image channel; boundary features must transform directionally",
                     "ensemble_initialization_offsets": INITIALIZATION_OFFSETS,
                     "modes": MODES})
    # make_protocol describes the nested MLP default; remove its unrelated
    # candidate settings instead of presenting them as CNN settings.
    for key in ("class_weight_powers", "epoch_checkpoints", "ablation_settings", "feature_dimensions"):
        protocol.pop(key, None)
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    probabilities = {mode: np.full((2, len(y), 3), np.nan, dtype=np.float32) for mode in MODES}
    fold_ids = np.full(len(y), -1)
    for fold, (train, test) in enumerate(folds):
        if np.intersect1d(groups[train], groups[test]).size:
            raise ValueError("Spatial fold leakage")
        fold_ids[test] = fold
        for mode in MODES:
            probabilities[mode][:, test], _ = fit_predict(images[mode], sides[mode], y, train, test,
                                                         args.seed + fold * 100)
            print(f"completed fold {fold + 1}/5 {mode}; elapsed {time.perf_counter() - start:.1f}s", flush=True)
    if any(not np.isfinite(value).all() for value in probabilities.values()):
        raise RuntimeError("Missing CNN outer predictions")
    predictions = {mode: p.mean(axis=0).argmax(axis=1) for mode, p in probabilities.items()}
    if args.seed == 20260921:
        prior = pd.read_csv(args.previous_cnn)
        prior = prior[prior["mode"] == "contact_interval"].set_index("id").loc[ann.id]
        if len(prior) != len(ann) or not np.array_equal(prior.true_type, ann.type) or not np.array_equal(prior.spatial_group, groups):
            raise ValueError("Historical CNN comparison does not use matching IDs/labels/groups")
        predictions["published_interval_cnn"] = prior.predicted_type.map({label: i for i, label in enumerate(CLASSES)}).to_numpy()
    scores = {mode: {**evaluate(y, p), "classification_report": classification_report(y, p, labels=range(3),
                        target_names=CLASSES, output_dict=True, zero_division=0)} for mode, p in predictions.items()}
    uncertainty = paired_bootstrap(y, groups, predictions, "interval_roi64", tuple(
                                  mode for mode in predictions if mode != "interval_roi64"), 4000, args.seed + 97)
    rows = {"id": ann.id, "type": ann.type, "chrom": ann.chrom, "center": ann.center,
            "length_bp": ann.length_bp, "spatial_group": groups, "outer_fold": fold_ids,
            **{f"{mode}_prediction": [CLASSES[i] for i in p] for mode, p in predictions.items()}}
    for mode, p in probabilities.items():
        for rep in (0, 1):
            for cls, label in enumerate(CLASSES):
                rows[f"{mode}_rep{rep + 1}_p_{label}"] = p[rep, :, cls]
    pd.DataFrame(rows).to_csv(args.out / "outer_predictions.csv", index=False)
    summary = {"protocol_sha256": sha256(protocol_path), "frozen_at_utc": protocol["frozen_at_utc"],
               "evaluation": "Fixed five 40 kb spatial-group folds, two biological replicates stay together",
               "caution": protocol["caution"], "seed": args.seed, "scores": scores,
               "paired_spatial_uncertainty": uncertainty, "wall_seconds": time.perf_counter() - start,
               "output_sha256": {"outer_predictions.csv": sha256(args.out / "outer_predictions.csv")},
               "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"scores": {mode: {key: value for key, value in score.items() if key != "classification_report"}
                                 for mode, score in scores.items()}, "wall_seconds": summary["wall_seconds"]}, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=Path("outputs/task1_multiscale/windows.npz"))
    parser.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    parser.add_argument("--previous-cnn", type=Path, default=Path("results/task1_interval_cnn_crossfit_predictions.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/task1_improve_20261006/roi_cnn_seed21"))
    parser.add_argument("--seed", type=int, default=20260921)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
