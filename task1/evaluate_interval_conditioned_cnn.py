"""Exploratory, spatially grouped CNN ablation for annotated intervals.

All four modes, folds, loss weights, and the 12-epoch budget are fixed in this
file. This is *not* independent confirmation: these annotations and outer-fold
outcomes have informed earlier experiments. It estimates whether a neural
model can use interval alignment without hiding a length-only shortcut.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold

from task1.evaluate_chid_overlap import spatial_groups
from task1.evaluate_pairwise import evaluate, fit_fold
from task1.run_task1 import CLASSES


MODES = ("contact_short", "contact_dual", "contact_interval", "interval_only")


def interval_channels(annotations: pd.DataFrame, bins: int = 128,
                      long_bin_bp: int = 160) -> np.ndarray:
    """Image channels for the annotated interval; never uses its class label.

    Channel 0 is the within-interval square. Channel 1 marks contacts crossing
    an interval boundary. The images are repeated for two biological replicates.
    """
    required = {"start", "end", "center"}
    if not required.issubset(annotations.columns):
        raise ValueError(f"Annotations lack {sorted(required - set(annotations.columns))}")
    coordinate = np.arange(bins, dtype=np.float32)
    output = np.zeros((len(annotations), 2, bins, bins), dtype=np.float32)
    for i, row in annotations.iterrows():
        left = np.floor(bins / 2 + (float(row.start) - float(row.center)) / long_bin_bp)
        right = np.ceil(bins / 2 + (float(row.end) - float(row.center)) / long_bin_bp)
        if not np.isfinite(left + right) or right <= left:
            raise ValueError(f"Invalid interval for {row.get('id', i)}")
        inside = (coordinate >= left) & (coordinate < right)
        output[i, 0] = np.logical_and.outer(inside, inside)
        output[i, 1] = np.logical_xor.outer(inside, inside)
    return np.concatenate((output, output), axis=0)


def make_modes(x: np.ndarray, annotations: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(annotations)
    if x.shape != (2 * n, 2, 128, 128):
        raise ValueError("Expected two independently normalized 128-bin scales per replicate")
    masks = interval_channels(annotations)
    return {
        "contact_short": np.ascontiguousarray(x[:, :1]),
        "contact_dual": np.ascontiguousarray(x[:, :2]),
        "contact_interval": np.ascontiguousarray(np.concatenate((x, masks), axis=1)),
        "interval_only": np.ascontiguousarray(masks),
    }


def crossfit_one(x: np.ndarray, y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]],
                 epochs: int, seed: int, device: torch.device) -> np.ndarray:
    n = len(y)
    logits = np.full((2, n, len(CLASSES)), np.nan, dtype=np.float32)
    for fold, (train, test) in enumerate(folds):
        # Both replicates of a structure stay together. No validation or test
        # samples enter the optimizer; hyperparameters are fixed before scoring.
        model = fit_fold(x, y, train, np.empty((0, 2), dtype=int), epochs,
                         seed + fold, 0.0, device)
        with torch.no_grad():
            for rep in (0, 1):
                batches = torch.from_numpy(x[test + rep * n]).split(64)
                logits[rep, test] = np.concatenate(
                    [model(batch.to(device)).cpu().numpy() for batch in batches])
        print(f"completed fold {fold + 1}/{len(folds)}", flush=True)
    if not np.isfinite(logits).all():
        raise RuntimeError("Incomplete out-of-fold predictions")
    return logits


def summarize(y: np.ndarray, logits: np.ndarray) -> dict:
    pred = logits.mean(axis=0).argmax(axis=1)
    result = evaluate(y, pred)
    result["classification_report"] = classification_report(
        y, pred, labels=range(len(CLASSES)), target_names=CLASSES,
        output_dict=True, zero_division=0)
    result["replicate_sample_level"] = evaluate(
        np.tile(y, 2), logits.argmax(axis=2).reshape(-1))
    return result


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    ann = pd.read_csv(args.annotations).reset_index(drop=True)
    with np.load(args.windows) as bundle:
        x = bundle["X"].astype(np.float32)
        ids = bundle["group"].astype(str)
    if not np.array_equal(ids, np.tile(ann.id.to_numpy(), 2)):
        raise ValueError("Cached replicate IDs do not match annotation order")
    y = ann.type.map({name: i for i, name in enumerate(CLASSES)}).to_numpy(dtype=int)
    if np.any(y < 0) or np.any(pd.isna(y)):
        raise ValueError("Unknown structure class")
    groups = spatial_groups(ann, args.spatial_gap_bp)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
                 .split(ann, y, groups))
    # Full-window non-overlap is mandatory between fit and evaluation groups.
    for train, test in folds:
        if np.intersect1d(groups[train], groups[test]).size:
            raise ValueError("Spatial component crossed a fold")
    data = make_modes(x, ann)
    chosen_modes = tuple(args.modes)
    if not chosen_modes or any(mode not in MODES for mode in chosen_modes):
        raise ValueError("Unknown or empty mode list")
    summary = {
        "protocol": "five fixed 40 kb spatial-group folds; both replicates grouped by structure",
        "caution": "exploratory reuse of annotations and folds reported in earlier analyses; no independent confirmation",
        "selection": "four ablations predeclared, fixed 12 epochs and half-power inverse-frequency loss; each fold trained without its own test labels or threshold tuning, but final shipped architecture chosen retrospectively after comparing outer-fold scores, so its reported score has selection bias",
        "seed": args.seed, "epochs": args.epochs,
        "spatial_gap_bp": args.spatial_gap_bp,
        "modes_run": chosen_modes,
        "class_counts": {name: int((y == i).sum()) for i, name in enumerate(CLASSES)},
        "fold_class_counts": [{"train": np.bincount(y[tr], minlength=3).tolist(),
                               "test": np.bincount(y[te], minlength=3).tolist()}
                              for tr, te in folds],
    }
    rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    for mode in chosen_modes:
        print(f"starting {mode}", flush=True)
        logits = crossfit_one(data[mode], y, folds, args.epochs, args.seed, device)
        pred = logits.mean(axis=0).argmax(axis=1)
        summary[mode] = summarize(y, logits)
        for i, row in ann.iterrows():
            rows.append({"mode": mode, "id": row.id, "true_type": row.type,
                         "predicted_type": CLASSES[pred[i]], "chrom": row.chrom,
                         "center": row.center, "spatial_group": groups[i],
                         "replicate_agreement": bool(logits[0, i].argmax() == logits[1, i].argmax()),
                         **{f"logit_{name}": float(logits[:, i, k].mean())
                            for k, name in enumerate(CLASSES)}})
        # Persist each mode immediately so a long CPU run can be inspected
        # without changing any tracked report or published result.
        pd.DataFrame(rows).to_csv(args.out / "crossfit_predictions.csv", index=False)
        (args.out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=Path("outputs/task1_multiscale/windows.npz"))
    parser.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/task1_interval_conditioned_cnn"))
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--spatial-gap-bp", type=int, default=40000)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    args = parser.parse_args()
    if args.epochs != 12 or args.spatial_gap_bp != 40000:
        raise ValueError("Confirmatory ablation is predeclared at 12 epochs / 40 kb gap")
    result = run(args)
    print(json.dumps({mode: result[mode] for mode in args.modes}, indent=2), flush=True)


if __name__ == "__main__":
    main()
