"""Fixed cross-replicate robustness audit of the published interval ROI CNN.

Each outer fit sees only ONE biological replicate of training regions. Both
replicate views of every held-out region remain outside fitting and scaling.
This is retrospective replicate robustness, not independent-label validation.
No architecture, threshold, epoch, or source-replicate selection is performed.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from task1.audit_interval_conditioned_cnn import paired_bootstrap
from task1.evaluate_chid_overlap import spatial_groups
from task1.evaluate_nested_neural_20261006 import fit_standardizer, load_features, sha256
from task1.evaluate_pairwise import evaluate
from task1.evaluate_roi_cnn_20261006 import (
    CompactIntervalCNN, EPOCHS, INITIALIZATION_OFFSETS, make_images,
    reverse_boundary_features,
)
from task1.run_task1 import CLASSES, seed_all

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260921
RESAMPLES = 4000
MODES = ("train_rep1_test_rep1", "train_rep1_test_rep2",
         "train_rep2_test_rep1", "train_rep2_test_rep2")


def transfer_indices(train: np.ndarray, test: np.ndarray, n: int,
                     source_rep: int, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map structures to samples without including the other replicate in fit."""
    train, test, groups = np.asarray(train), np.asarray(test), np.asarray(groups)
    if source_rep not in (0, 1) or len(groups) != n:
        raise ValueError("Need one of two source replicates and n spatial groups")
    for ids in (train, test):
        if ids.ndim != 1 or not len(ids) or ids.dtype.kind not in "iu":
            raise ValueError("Need nonempty integer structure indices")
        if len(np.unique(ids)) != len(ids) or np.any((ids < 0) | (ids >= n)):
            raise ValueError("Duplicate or out-of-range structure indices")
    if np.intersect1d(train, test).size:
        raise ValueError("Train/test structure leakage")
    if np.intersect1d(groups[train], groups[test]).size:
        raise ValueError("Train/test spatial group leakage")
    fit_ids = train + source_rep * n
    test_ids = np.r_[test, test + n]
    if np.intersect1d(fit_ids % n, test_ids % n).size:
        raise ValueError("Counterpart-replicate structure leakage")
    return fit_ids, test_ids


def fit_transfer(image: np.ndarray, side: np.ndarray, y: np.ndarray,
                 train: np.ndarray, test: np.ndarray, source_rep: int,
                 groups: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    """Fixed published CNN objective, but optimization sees one source only."""
    n = len(y)
    if image.shape != (2 * n, 5, 64, 64) or side.shape != (2 * n, 15):
        raise ValueError("Expected published five-channel ROI and 15 side features")
    if not np.isfinite(image).all() or not np.isfinite(side).all():
        raise ValueError("Nonfinite CNN input")
    fit_ids, test_ids = transfer_indices(train, test, n, source_rep, groups)
    mean, scale = fit_standardizer(side, fit_ids)
    images = torch.from_numpy(image)
    sides = torch.from_numpy(((side - mean) / scale).astype(np.float32))
    labels = torch.from_numpy(np.tile(y, 2))
    counts = np.bincount(y[train], minlength=3).astype(np.float32)
    weights = np.sqrt(counts.sum() / (3 * np.maximum(counts, 1)))
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights))
    probabilities = np.zeros((2, len(test), 3), dtype=np.float32)
    for offset in INITIALIZATION_OFFSETS:
        seed_all(seed + offset)
        model = CompactIntervalCNN(5, 15)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
        rng = torch.Generator().manual_seed(seed + offset)
        for _ in range(EPOCHS):
            model.train()
            for batch in torch.randperm(len(fit_ids), generator=rng).split(32):
                ids = fit_ids[batch.numpy()]
                xb, sb = images[ids], sides[ids]
                if torch.rand((), generator=rng) < 0.5:
                    xb = xb.flip((2, 3))
                    reversed_raw = reverse_boundary_features(side[ids])
                    sb = torch.from_numpy(((reversed_raw - mean) / scale).astype(np.float32))
                optimizer.zero_grad()
                loss = loss_fn(model(xb, sb), labels[ids])
                loss.backward()
                optimizer.step()
        model.eval()
        output = []
        with torch.no_grad():
            for ids in np.array_split(test_ids, max(1, int(np.ceil(len(test_ids) / 64)))):
                output.append(model(images[ids], sides[ids]).softmax(dim=1).numpy())
        probabilities += np.concatenate(output).reshape(2, len(test), 3) / len(INITIALIZATION_OFFSETS)
    receipt = {"source_replicate": source_rep + 1,
               "training_sample_indices": fit_ids.tolist(),
               "held_out_sample_indices": test_ids.tolist(),
               "side_mean": mean.tolist(), "side_scale": scale.tolist(),
               "training_class_counts": counts.astype(int).tolist(),
               "class_weights": weights.tolist(),
               "initialization_seeds": [seed + offset for offset in INITIALIZATION_OFFSETS]}
    return probabilities, receipt


def load_published(path: Path, ann: pd.DataFrame, groups: np.ndarray,
                   fold_ids: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Require exactly the same historical IDs, labels, spatial groups and folds."""
    saved = pd.read_csv(path)
    if saved.id.duplicated().any() or set(saved.id) != set(ann.id) or len(saved) != len(ann):
        raise ValueError("Published comparator must contain exactly every annotation ID once")
    saved = saved.set_index("id").loc[ann.id].reset_index()
    if not np.array_equal(saved.type, ann.type) or not np.array_equal(saved.spatial_group, groups):
        raise ValueError("Published labels or spatial groups differ")
    if not np.array_equal(saved.outer_fold.to_numpy(dtype=int), fold_ids):
        raise ValueError("Published outer folds differ")
    probabilities = np.stack([saved[[f"interval_roi64_rep{rep}_p_{label}"
                                     for label in CLASSES]].to_numpy(dtype=np.float32)
                              for rep in (1, 2)])
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("Invalid published probabilities")
    np.testing.assert_allclose(probabilities.sum(axis=2), 1, atol=2e-6)
    mapping = {label: i for i, label in enumerate(CLASSES)}
    joint = saved.interval_roi64_prediction.map(mapping).to_numpy(dtype=int)
    if not np.array_equal(joint, probabilities.mean(axis=0).argmax(axis=1)):
        raise ValueError("Published combined predictions disagree with frozen probabilities")
    return {"published_joint_training_rep1": probabilities[0].argmax(axis=1),
            "published_joint_training_rep2": probabilities[1].argmax(axis=1),
            "published_joint_training_mean": joint}, probabilities


def freeze_protocol(args: argparse.Namespace, ann: pd.DataFrame, groups: np.ndarray,
                    folds: list) -> dict:
    sources = [Path(__file__).resolve(), ROOT / "task1/evaluate_roi_cnn_20261006.py",
               ROOT / "task1/evaluate_nested_neural_20261006.py",
               ROOT / "task1/evaluate_interval_conditioned_cnn.py",
               ROOT / "task1/evaluate_interval_boundaries.py",
               ROOT / "task1/evaluate_chid_overlap.py", ROOT / "task1/evaluate_pairwise.py",
               ROOT / "task1/audit_interval_conditioned_cnn.py", ROOT / "task1/run_task1.py",
               ROOT / "scripts/validate_discovery.py", ROOT / "microc/analysis.py",
               ROOT / "task1/evaluate_nested_threshold.py",
               ROOT / "tests/test_roi_transfer_followup.py"]
    inputs = {"windows": args.windows, "annotations": args.annotations,
              "published_joint_oof": args.published}
    return {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Task 1 retrospective cross-replicate transfer on annotated intervals",
            "caution": "All labels and biological replicates have previously been inspected. This is replicate robustness, NOT independent-label confirmation; known interval geometry is supplied.",
            "code_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources},
            "input_sha256": {name: sha256(path) for name, path in inputs.items()},
            "input_paths": {name: str(path) for name, path in inputs.items()},
            "environment": {name: importlib.metadata.version(name) for name in
                            ("torch", "numpy", "pandas", "scikit-learn", "matplotlib")},
            "device": "cpu", "torch_threads": torch.get_num_threads(),
            "architecture": "Published CompactIntervalCNN; five 64x64 contact/mask/ROI channels and 15 directional boundary/loglength side features",
            "optimizer": {"name": "AdamW", "lr": 0.001, "weight_decay": 0.001,
                          "batch_size": 32, "epochs": EPOCHS},
            "ensemble_initialization_offsets": list(INITIALIZATION_OFFSETS),
            "outer_seed": SEED, "spatial_gap_bp": 40000, "outer_folds": 5,
            "source_replicates": [1, 2], "evaluation_replicates": [1, 2],
            "preprocessing": "Fit side location/scale using ONLY source-replicate training regions; class weights are sqrt inverse frequency of source training labels. No target-replicate training regions enter fitting.",
            "augmentation": "Same random genomic-axis reversal as published CNN; reverse directional raw side features before applying source training scaler",
            "selection": "None. Fixed published model, 24 epochs, both directions and both replicate views; retain every result. No best direction, seed, epoch, threshold, or architecture selection.",
            "scoring": "Average two initialization probabilities, then argmax separately for each evaluation replicate. No biological-replicate pooling for transfer scores.",
            "comparators": "Source held-out replicate versus target held-out replicate from the SAME fit; previously published joint-training OOF predictions per replicate and combined. Joint training uses twice the training views and is not an equal-resource superiority comparator.",
            "uncertainty": {"method": "paired spatial-component percentile bootstrap",
                            "resamples": RESAMPLES, "seed": SEED + 197,
                            "caution": "Exploratory intervals over reused labels; not independent confirmatory p-values"},
            "structures": len(ann), "spatial_components": int(len(np.unique(groups))),
            "fold_ids": [{"fold": fold, "train_ids": ann.id.iloc[train].tolist(),
                          "test_ids": ann.id.iloc[test].tolist()}
                         for fold, (train, test) in enumerate(folds)]}


def draw_figure(truth: np.ndarray, predictions: dict[str, np.ndarray], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for ax, mode in zip(axes.flat, MODES):
        matrix = confusion_matrix(truth, predictions[mode], labels=range(3))
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=250)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > 140 else "black")
        score = evaluate(truth, predictions[mode])
        ax.set(title=f"{mode.replace('_', ' ')}\naccuracy {score['accuracy']:.3f}; macro F1 {score['macro_f1']:.3f}",
               xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES,
               xlabel="Predicted", ylabel="Historical annotated label")
    fig.suptitle("Fixed CNN: source-only training, spatial held-out regions\n344 structures per panel; replicate robustness, not independent labels", fontsize=12)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    torch.set_num_threads(2)
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use a new, empty output directory; frozen evidence is never overwritten")
    args.out.mkdir(parents=True, exist_ok=True)
    ann, features = load_features(args.windows, args.annotations)
    y = ann.type.map({name: i for i, name in enumerate(CLASSES)}).to_numpy(dtype=np.int64)
    groups = spatial_groups(ann, 40000)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(ann, y, groups))
    for train, test in folds:
        transfer_indices(train, test, len(ann), 0, groups)
    protocol = freeze_protocol(args, ann, groups, folds)
    protocol_path = args.out / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    protocol_hash = sha256(protocol_path)
    print(f"Protocol frozen {protocol['frozen_at_utc']}; SHA256 {protocol_hash}", flush=True)
    # Per-sample deterministic transforms occur only after the run is frozen.
    with np.load(args.windows) as bundle:
        image = make_images(bundle["X"].astype(np.float32), ann)["interval_roi64"]
    side = features["contact_interval"][:, -15:].astype(np.float32)
    probabilities = np.full((2, 2, len(ann), 3), np.nan, dtype=np.float32)
    fold_ids = np.full(len(ann), -1, dtype=int)
    fit_receipts = []
    for fold, (train, test) in enumerate(folds):
        fold_ids[test] = fold
        for source_rep in (0, 1):
            probability, receipt = fit_transfer(image, side, y, train, test, source_rep,
                                                groups, SEED + 100 * fold)
            probabilities[source_rep][:, test] = probability
            receipt.update({"outer_fold": fold, "training_structure_ids": ann.id.iloc[train].tolist(),
                            "held_out_structure_ids": ann.id.iloc[test].tolist(),
                            "training_spatial_groups": sorted(np.unique(groups[train]).tolist()),
                            "held_out_spatial_groups": sorted(np.unique(groups[test]).tolist())})
            fit_receipts.append(receipt)
            print(f"Completed fold {fold + 1}/5 source rep{source_rep + 1}; {time.perf_counter() - started:.1f}s", flush=True)
    if not np.isfinite(probabilities).all() or np.any(fold_ids < 0):
        raise RuntimeError("Missing out-of-fold transfer predictions")
    predictions = {f"train_rep{s + 1}_test_rep{t + 1}": probabilities[s, t].argmax(axis=1)
                   for s in (0, 1) for t in (0, 1)}
    published, published_probability = load_published(args.published, ann, groups, fold_ids)
    predictions.update(published)
    scores = {mode: {**evaluate(y, pred), "classification_report": classification_report(
                    y, pred, labels=range(3), target_names=CLASSES, output_dict=True, zero_division=0)}
              for mode, pred in predictions.items()}
    uncertainty = {}
    for source, target in ((1, 2), (2, 1)):
        base = f"train_rep{source}_test_rep{target}"
        comparisons = (f"train_rep{source}_test_rep{source}",
                       f"published_joint_training_rep{target}", "published_joint_training_mean")
        uncertainty.update(paired_bootstrap(y, groups, predictions, base, comparisons,
                                           RESAMPLES, SEED + 197))
    agreement = {f"train_rep{s + 1}": {"same_vs_cross_prediction_agreement": float(np.mean(
        probabilities[s, 0].argmax(axis=1) == probabilities[s, 1].argmax(axis=1))),
        "mean_probability_l1_between_replicates": float(np.abs(probabilities[s, 0] - probabilities[s, 1]).sum(axis=1).mean())}
        for s in (0, 1)}
    rows = {"id": ann.id, "type": ann.type, "chrom": ann.chrom, "center": ann.center,
            "length_bp": ann.length_bp, "spatial_group": groups, "outer_fold": fold_ids,
            **{f"{mode}_prediction": [CLASSES[i] for i in pred] for mode, pred in predictions.items()}}
    for source in (0, 1):
        for target in (0, 1):
            for cls, label in enumerate(CLASSES):
                rows[f"train_rep{source + 1}_test_rep{target + 1}_p_{label}"] = probabilities[source, target, :, cls]
    for target in (0, 1):
        for cls, label in enumerate(CLASSES):
            rows[f"published_joint_training_rep{target + 1}_p_{label}"] = published_probability[target, :, cls]
    pd.DataFrame(rows).to_csv(args.out / "outer_predictions.csv", index=False)
    (args.out / "fit_receipts.json").write_text(json.dumps(fit_receipts, indent=2), encoding="utf-8")
    pd.DataFrame([{"mode": mode, **{key: value for key, value in score.items()
                                   if key != "classification_report"}} for mode, score in scores.items()]).to_csv(
                                   args.out / "score_table.csv", index=False)
    draw_figure(y, predictions, args.out / "transfer_confusions.png")
    # Detect source or input drift instead of blessing a mixed-source run.
    for relative, expected in protocol["source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen source changed during run: {relative}")
    for name, path in (("windows", args.windows), ("annotations", args.annotations),
                       ("published_joint_oof", args.published)):
        if sha256(path) != protocol["input_sha256"][name]:
            raise RuntimeError(f"Frozen input changed during run: {name}")
    outputs = ("outer_predictions.csv", "fit_receipts.json", "score_table.csv", "transfer_confusions.png")
    summary = {"protocol_sha256": protocol_hash, "frozen_at_utc": protocol["frozen_at_utc"],
               "completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "scope": protocol["scope"], "caution": protocol["caution"],
               "structures_per_evaluation": len(ann), "spatial_components": len(np.unique(groups)),
               "scores": scores, "paired_spatial_bootstrap": uncertainty,
               "within_fit_replicate_agreement": agreement,
               "output_sha256": {name: sha256(args.out / name) for name in outputs},
               "wall_seconds": time.perf_counter() - started,
               "conclusion_rule": "Report both transfer directions, all controls and intervals; no best-direction or new-model selection. Published joint training is a fixed, higher-training-resource historical comparator."}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.publish:
        prefix = "task1_followup_20261006_transfer_"
        names = ("protocol.json", "summary.json", *outputs)
        for name in names:
            destination = ROOT / "results" / (prefix + name)
            if destination.exists():
                raise ValueError(f"Published follow-up evidence already exists: {destination}")
        for name in names:
            shutil.copyfile(args.out / name, ROOT / "results" / (prefix + name))
        receipt = {"published_at_utc": datetime.now(timezone.utc).isoformat(),
                   "output_directory": str(args.out), "protocol_sha256": protocol_hash,
                   "files": {prefix + name: sha256(ROOT / "results" / (prefix + name)) for name in names},
                   "source_sha256": protocol["source_sha256"],
                   "input_sha256": protocol["input_sha256"],
                   "command": "python -m task1.evaluate_roi_transfer_followup --out outputs/task1_followup_20261006_transfer --publish"}
        (ROOT / "results" / (prefix + "evidence_index.json")).write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"scores": {mode: {key: value for key, value in score.items()
                                      if key != "classification_report"} for mode, score in scores.items()},
                      "agreement": agreement, "wall_seconds": summary["wall_seconds"]}, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=Path("outputs/task1_multiscale/windows.npz"))
    parser.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    parser.add_argument("--published", type=Path,
                        default=Path("results/task1_improve_20261006_roi_cnn_seed21_outer_predictions.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/task1_followup_20261006_transfer"))
    parser.add_argument("--publish", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
