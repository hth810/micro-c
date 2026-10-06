"""Retrospective, nested spatial evaluation of a small neural descriptor model.

The fixed candidate family is selected only by inner, spatially grouped
out-of-fold predictions. This strengthens the validation procedure, but the
historically reused annotations cannot become an independent confirmation.
The model is an MLP on pooled contact geometry, not a convolutional network.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from scripts.validate_discovery import shape_features
from task1.audit_interval_conditioned_cnn import paired_bootstrap
from task1.evaluate_chid_overlap import spatial_groups
from task1.evaluate_interval_boundaries import interval_features
from task1.evaluate_nested_threshold import fit_model
from task1.evaluate_pairwise import evaluate
from task1.run_task1 import CLASSES, seed_all


EPOCHS = (80, 160, 240)
POWERS = (0.5, 1.0)
INITIALIZATIONS = (0, 17, 43)
MODES = ("contact_interval", "contact_only", "interval_only")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_standardizer(x: np.ndarray, train_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit each feature's location/scale on training replicates only."""
    mean = x[train_samples].mean(axis=0)
    scale = x[train_samples].std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return mean.astype(np.float32), scale.astype(np.float32)


def load_features(windows: Path, annotations: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    ann = pd.read_csv(annotations).reset_index(drop=True)
    if ann.id.duplicated().any() or not ann.type.isin(CLASSES).all():
        raise ValueError("Duplicate structure IDs or unknown labels")
    n = len(ann)
    with np.load(windows) as bundle:
        x = bundle["X"].astype(np.float32)
        ids = bundle["group"].astype(str)
    if x.shape != (2 * n, 2, 128, 128) or not np.array_equal(ids, np.tile(ann.id.to_numpy(), 2)):
        raise ValueError("Expected exactly both biological replicates in annotation order")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite cached contact windows")
    # Per-sample deterministic descriptors; this step learns no population
    # statistics and uses no class labels. Both replicates remain separate.
    contact = np.stack([shape_features(mat) for mat in x.reshape(-1, 128, 128)])
    contact = contact.reshape(2 * n, -1)
    boundary = np.stack([interval_features(x[rep * n + i, 1], int(row.start),
                                         int(row.end), int(row.center))
                         for rep in (0, 1) for i, row in ann.iterrows()])
    span = np.tile(np.log1p(ann.length_bp.to_numpy(dtype=np.float32)), 2)[:, None]
    return ann, {"contact_interval": np.column_stack((contact, boundary, span)),
                 "contact_only": contact, "interval_only": span}


class DescriptorMLP(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(n_features, 64), nn.GELU(), nn.Dropout(0.15),
                                    nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.15),
                                    nn.Linear(32, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def fit_predict_checkpoints(x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray,
                            seed: int, power: float,
                            checkpoints: tuple[int, ...] = EPOCHS,
                            keep_models: bool = False) -> tuple[dict[int, np.ndarray], list[dict]]:
    """Every returned prediction comes from train-only scaling/optimization.

    The training objective weights both replicates identically. One structure
    therefore has the same total weight as another structure of its class.
    """
    if np.intersect1d(train, test).size:
        raise ValueError("Train and prediction structure IDs overlap")
    if not checkpoints or min(checkpoints) <= 0:
        raise ValueError("Need positive fixed checkpoints")
    n = len(y)
    train_samples = np.r_[train, train + n]
    test_samples = np.r_[test, test + n]
    mean, scale = fit_standardizer(x, train_samples)
    fit_x = torch.from_numpy(((x[train_samples] - mean) / scale).astype(np.float32))
    test_x = torch.from_numpy(((x[test_samples] - mean) / scale).astype(np.float32))
    fit_y = torch.from_numpy(np.tile(y, 2)[train_samples])
    counts = np.bincount(y[train], minlength=3).astype(np.float32)
    weights = (counts.sum() / (3 * np.maximum(counts, 1))) ** power
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights))
    probabilities = {epoch: np.zeros((2, len(test), 3), dtype=np.float32) for epoch in checkpoints}
    states = []
    for offset in INITIALIZATIONS:
        seed_all(seed + offset)
        model = DescriptorMLP(x.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
        # Full-batch steps avoid a noisy majority/minority batch ratio. There is
        # no data-dependent early stopping or test-dependent learning schedule.
        for epoch in range(1, max(checkpoints) + 1):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(model(fit_x), fit_y)
            loss.backward()
            optimizer.step()
            if epoch in checkpoints:
                model.eval()
                with torch.no_grad():
                    probability = model(test_x).softmax(dim=1).numpy()
                probabilities[epoch] += probability.reshape(2, len(test), 3) / len(INITIALIZATIONS)
        if keep_models:
            states.append({"state_dict": model.state_dict(), "initialization_seed": seed + offset})
    if any(not np.isfinite(p).all() for p in probabilities.values()):
        raise ValueError("Incomplete neural probabilities")
    if keep_models:
        states = [{"models": states, "mean": mean, "scale": scale,
                   "power": power, "epochs": max(checkpoints)}]
    return probabilities, states


def choose_plan(inner: dict[tuple[float, int], np.ndarray], truth: np.ndarray) -> tuple[float, int, list[dict]]:
    """Only the supplied inner-fold labels determine the chosen plan."""
    expected = {(power, epoch) for power in POWERS for epoch in EPOCHS}
    if set(inner) != expected or any(p.shape != (2, len(truth), 3) or not np.isfinite(p).all()
                                      for p in inner.values()):
        raise ValueError("Incomplete inner candidate family")
    rows = []
    for power in POWERS:
        for epoch in EPOCHS:
            prediction = inner[power, epoch].mean(axis=0).argmax(axis=1)
            rows.append({"class_weight_power": power, "epochs": epoch,
                         **evaluate(truth, prediction)})
    # Macro F1 is the primary criterion; ties favor fewer steps, then the
    # milder weighting. No outer-fold label or score is an input to this rule.
    chosen = max(rows, key=lambda row: (row["macro_f1"], -row["epochs"], -row["class_weight_power"]))
    return chosen["class_weight_power"], chosen["epochs"], rows


def inner_selection(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    outer_train: np.ndarray, seed: int) -> tuple[float, int, list[dict]]:
    candidate = {(power, epoch): np.full((2, len(outer_train), 3), np.nan, dtype=np.float32)
                 for power in POWERS for epoch in EPOCHS}
    folds = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    for fold, (train, val) in enumerate(folds.split(x[outer_train], y[outer_train], groups[outer_train])):
        if np.intersect1d(groups[outer_train[train]], groups[outer_train[val]]).size:
            raise ValueError("Inner spatial component leakage")
        for power in POWERS:
            predictions, _ = fit_predict_checkpoints(x, y, outer_train[train], outer_train[val],
                                                      seed + 100 * fold, power)
            for epoch in EPOCHS:
                candidate[power, epoch][:, val] = predictions[epoch]
    return choose_plan(candidate, y[outer_train])


def make_protocol(args: argparse.Namespace, ann: pd.DataFrame, data: dict[str, np.ndarray],
                  groups: np.ndarray, folds: list) -> dict:
    sources = [Path(__file__), Path("scripts/validate_discovery.py"), Path("microc/analysis.py"),
               Path("task1/evaluate_interval_boundaries.py"), Path("task1/evaluate_pairwise.py"),
               Path("task1/run_task1.py"), Path("task1/evaluate_chid_overlap.py"),
               Path("task1/audit_interval_conditioned_cnn.py"), Path("task1/evaluate_nested_threshold.py")]
    return {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Task 1 original single-label CHIN/OPCID/CHID; annotated intervals only",
            "caution": "Retrospective reuse of previously inspected annotations; no independent confirmation",
            "code_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": {str(p).replace("\\", "/"): sha256(p) for p in sources},
            "input_sha256": {"windows": sha256(args.windows), "annotations": sha256(args.annotations)},
            "environment": {name: importlib.metadata.version(name) for name in
                            ("torch", "numpy", "pandas", "scikit-learn")},
            "device": "cpu", "torch_threads": torch.get_num_threads(),
            "outer_seed": args.seed, "spatial_gap_bp": 40000, "outer_folds": 5, "inner_folds": 3,
            "representation": "Deterministic 16x16 upper-triangle contact pooling, diagonal bands, quantiles at two scales; 1.28 kb interval flank blocks; log interval length",
            "neural_architecture": "standardized descriptors -> 64 GELU dropout 0.15 -> 32 GELU dropout 0.15 -> 3",
            "optimizer": {"name": "AdamW", "lr": 0.003, "weight_decay": 0.01,
                          "batch": "full outer/inner training replicates"},
            "selection": "6 fixed power/epoch plans; inner spatial OOF macro-F1, then fewer steps, milder weights",
            "class_weight_powers": POWERS, "epoch_checkpoints": EPOCHS,
            "ensemble_initialization_offsets": INITIALIZATIONS,
            "ablation_settings": {"class_weight_power": 0.5, "epochs": 160},
            "preprocessing": "Each inner/outer fit standardizes using only both replicates of its own training structures",
            "scoring": "Mean probabilities of 3 initializations and 2 biological replicates, argmax; original structure-level label",
            "uncertainty": "paired spatial component bootstrap; 4000 fixed resamples; exploratory percentile intervals",
            "feature_dimensions": {mode: x.shape[1] for mode, x in data.items()},
            "structures": len(ann), "spatial_components": int(len(np.unique(groups))),
            "fold_ids": [{"fold": fold, "train_ids": ann.id.iloc[train].tolist(),
                          "test_ids": ann.id.iloc[test].tolist()}
                         for fold, (train, test) in enumerate(folds)]}


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(2)
    started = time.perf_counter()
    ann, data = load_features(args.windows, args.annotations)
    y = ann.type.map({label: i for i, label in enumerate(CLASSES)}).to_numpy(dtype=np.int64)
    groups = spatial_groups(ann, 40000)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
                 .split(ann, y, groups))
    args.out.mkdir(parents=True, exist_ok=True)
    protocol_path = args.out / "protocol.json"
    if protocol_path.exists():
        raise ValueError("Output already contains a frozen protocol; use a new output directory")
    protocol = make_protocol(args, ann, data, groups, folds)
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    protocol_hash = sha256(protocol_path)
    probabilities = {mode: np.full((2, len(y), 3), np.nan, dtype=np.float32) for mode in MODES}
    predicted = {"nested_neural": np.full(len(y), -1), "contact_only": np.full(len(y), -1),
                 "interval_only": np.full(len(y), -1), "same_fold_hgb": np.full(len(y), -1)}
    choices, options = [], []
    fold_indices = np.full(len(y), -1)
    for fold, (train, test) in enumerate(folds):
        if np.intersect1d(groups[train], groups[test]).size:
            raise ValueError("Outer spatial component leakage")
        fold_indices[test] = fold
        power, epochs, selection_rows = inner_selection(data["contact_interval"], y, groups, train,
                                                        args.seed + 1000 * fold)
        options.extend({"outer_fold": fold, **row} for row in selection_rows)
        for mode in MODES:
            settings = (power, epochs) if mode == "contact_interval" else (0.5, 160)
            predicted_checkpoints, _ = fit_predict_checkpoints(data[mode], y, train, test,
                                                               args.seed + 100 * fold, settings[0],
                                                               (settings[1],))
            probabilities[mode][:, test] = predicted_checkpoints[settings[1]]
            key = "nested_neural" if mode == "contact_interval" else mode
            predicted[key][test] = probabilities[mode][:, test].mean(axis=0).argmax(axis=1)
        # Same descriptor inputs and same structure grouping as the MLP.
        average_features = data["contact_interval"].reshape(2, len(y), -1).mean(axis=0)
        predicted["same_fold_hgb"][test] = fit_model(average_features, y, train, args.seed + fold).predict(average_features[test])
        choices.append({"outer_fold": fold, "class_weight_power": power, "epochs": epochs,
                        "train_structures": len(train), "test_structures": len(test),
                        "test_chid": int((y[test] == 2).sum())})
        pd.DataFrame(choices).to_csv(args.out / "fold_choices.csv", index=False)
        pd.DataFrame(options).to_csv(args.out / "inner_options.csv", index=False)
        print(f"outer fold {fold + 1}/5: selected power={power}, epochs={epochs}; elapsed {time.perf_counter() - started:.1f}s", flush=True)
    if any(np.any(p < 0) for p in predicted.values()):
        raise RuntimeError("Incomplete outer predictions")
    # A valid same-fold historical comparison is available only for seed 21.
    if args.seed == 20260921 and args.previous_cnn.is_file():
        prior = pd.read_csv(args.previous_cnn)
        prior = prior[prior["mode"] == "contact_interval"].set_index("id").loc[ann.id]
        if len(prior) != len(ann) or not np.array_equal(prior.true_type, ann.type) or not np.array_equal(prior.spatial_group, groups):
            raise ValueError("Historical CNN prediction IDs/labels/spatial components differ")
        predicted["published_interval_cnn"] = prior.predicted_type.map({label: i for i, label in enumerate(CLASSES)}).to_numpy()
    uncertainty = paired_bootstrap(y, groups, predicted, "nested_neural",
                                   tuple(key for key in predicted if key != "nested_neural"),
                                   resamples=4000, seed=args.seed + 97)
    rows = {"id": ann.id, "type": ann.type, "chrom": ann.chrom, "center": ann.center,
            "length_bp": ann.length_bp, "spatial_group": groups, "outer_fold": fold_indices,
            **{f"{mode}_prediction": [CLASSES[i] for i in p] for mode, p in predicted.items()}}
    for mode, p in probabilities.items():
        for replicate in (0, 1):
            for cls, label in enumerate(CLASSES):
                rows[f"{mode}_rep{replicate + 1}_p_{label}"] = p[replicate, :, cls]
    pd.DataFrame(rows).to_csv(args.out / "outer_predictions.csv", index=False)
    scores = {mode: {**evaluate(y, p), "classification_report": classification_report(
                        y, p, labels=range(3), target_names=CLASSES, output_dict=True, zero_division=0)}
              for mode, p in predicted.items()}
    # Additional per-replicate scores expose benefit from combining repeats.
    replicate_scores = {mode: {f"rep{rep + 1}": evaluate(y, p[rep].argmax(axis=1)) for rep in (0, 1)}
                        for mode, p in probabilities.items()}
    summary = {"protocol_sha256": protocol_hash, "protocol_frozen_at_utc": protocol["frozen_at_utc"],
               "evaluation": "nested five outer / three inner 40 kb spatial-group folds; both replicates grouped",
               "caution": protocol["caution"], "seed": args.seed, "scores": scores,
               "replicate_scores": replicate_scores, "paired_spatial_uncertainty": uncertainty,
               "fold_choices": choices, "wall_seconds": time.perf_counter() - started,
               "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    summary["output_sha256"] = {name: sha256(args.out / name) for name in
                                ("protocol.json", "fold_choices.csv", "inner_options.csv", "outer_predictions.csv")}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"scores": {mode: {k: v for k, v in score.items() if k != "classification_report"}
                                 for mode, score in scores.items()}, "wall_seconds": summary["wall_seconds"]}, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=Path("outputs/task1_multiscale/windows.npz"))
    parser.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    parser.add_argument("--previous-cnn", type=Path, default=Path("results/task1_interval_cnn_crossfit_predictions.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/task1_improve_20261006/nested_neural_seed21"))
    parser.add_argument("--seed", type=int, default=20260921)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
