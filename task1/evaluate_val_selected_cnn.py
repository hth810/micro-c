"""Validation-selected, contact-only CNN on the fixed spatial split.

This is an exploratory follow-up to an already reported test split, not a new
independent confirmation. Architecture, class weighting and augmentation are
selected using the validation structures only. The test partition is loaded
into the evaluation function only after the selected checkpoint is fixed.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from task1.evaluate_pairwise import PairAwareCNN
from task1.run_task1 import CLASSES, StructureCNN, seed_all


@dataclass(frozen=True)
class Candidate:
    name: str
    architecture: str
    weight_power: float
    reverse_augmentation: bool


# This list is fixed before accessing test labels or windows. The four modes
# isolate GroupNorm and class weighting from a plausible orientation symmetry.
CANDIDATES = (
    Candidate("groupnorm_half", "groupnorm", 0.5, False),
    Candidate("groupnorm_full", "groupnorm", 1.0, False),
    Candidate("groupnorm_half_reverse", "groupnorm", 0.5, True),
    Candidate("groupnorm_full_reverse", "groupnorm", 1.0, True),
)


def read_fixed_split(windows: Path, manifest: Path, annotations: Path,
                     window_bp: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, np.ndarray]]:
    ann = pd.read_csv(annotations).reset_index(drop=True)
    saved = pd.read_csv(manifest).reset_index(drop=True)
    if len(ann) != len(saved) or not ann.id.equals(saved.id):
        raise ValueError("Manifest and annotation ID order differ")
    for col in ("type", "chrom", "center"):
        if not ann[col].equals(saved[col]):
            raise ValueError(f"Manifest and annotations differ in {col}")
    if saved.id.duplicated().any() or set(saved.split) != {"train", "validation", "test"}:
        raise ValueError("Manifest must have unique IDs and all three partitions")
    for _, frame in saved.sort_values(["chrom", "center"]).groupby("chrom"):
        near = frame.center.diff().to_numpy()[1:] < window_bp
        crosses = frame.split.to_numpy()[1:] != frame.split.to_numpy()[:-1]
        if np.any(near & crosses):
            raise ValueError("Overlapping input windows cross a partition")
    with np.load(windows) as data:
        x = data["X"].astype(np.float32)
        cached_y = data["y"].astype(int)
        ids = data["group"].astype(str)
        cached_masks = {part: data[f"is_{part}"] for part in ("train", "validation", "test")}
    n = len(ann)
    y = ann.type.map({name: i for i, name in enumerate(CLASSES)}).to_numpy(dtype=int)
    if x.shape != (2 * n, 1, 128, 128) or not np.array_equal(ids, np.tile(ann.id.to_numpy(), 2)):
        raise ValueError("Cached replicate windows do not match current annotations")
    if not np.array_equal(cached_y, np.tile(y, 2)):
        raise ValueError("Cached labels do not match current annotations")
    masks = {part: np.tile(saved.split.eq(part).to_numpy(), 2)
             for part in ("train", "validation", "test")}
    if any(not np.array_equal(masks[part], cached_masks[part]) for part in masks):
        raise ValueError("Cached split masks differ from the audited manifest")
    if any(np.sum(saved.split.eq(part) & saved.type.eq("CHID")) == 0 for part in masks):
        raise ValueError("At least one partition has no CHID annotations")
    return x, y, saved, masks


def build_model(candidate: Candidate) -> nn.Module:
    if candidate.architecture == "groupnorm":
        return PairAwareCNN(1)
    if candidate.architecture == "batchnorm":
        return StructureCNN(3, spatial_grid=4)
    raise ValueError(candidate.architecture)


def score_structures(logits: np.ndarray, truth: np.ndarray) -> dict:
    pred = logits.mean(axis=0).argmax(axis=1)
    return {"macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(truth, pred)),
            "chid_f1": float(f1_score(truth, pred, labels=[2], average="macro", zero_division=0)),
            "chid_recall": float(np.mean(pred[truth == 2] == 2))}


def predict_replicates(model: nn.Module, x: np.ndarray, structure_indices: np.ndarray,
                       n_structures: int, device: torch.device) -> np.ndarray:
    output = []
    model.eval()
    with torch.no_grad():
        for rep in (0, 1):
            tensor = torch.from_numpy(x[structure_indices + rep * n_structures])
            pieces = [model(part.to(device)).cpu().numpy() for part in tensor.split(64)]
            output.append(np.concatenate(pieces))
    return np.stack(output)


def fit_and_select(x: np.ndarray, y: np.ndarray, saved: pd.DataFrame,
                   candidate: Candidate, epochs: int, seed: int,
                   device: torch.device) -> tuple[nn.Module, dict, list[dict]]:
    """Train only on train IDs; select the best epoch on validation IDs."""
    seed_all(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    n = len(y)
    train = np.flatnonzero(saved.split.eq("train"))
    val = np.flatnonzero(saved.split.eq("validation"))
    train_samples = np.r_[train, train + n]
    labels = np.tile(y, 2)
    counts = np.bincount(y[train], minlength=3).astype(np.float32)
    weights = (counts.sum() / (3 * counts)) ** candidate.weight_power
    model = build_model(candidate).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(x[train_samples]),
                                      torch.from_numpy(labels[train_samples])),
                        batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best_key: tuple[float, float, int] = (-1.0, -1.0, -999)
    best_state = None
    best_record: dict = {}
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if candidate.reverse_augmentation:
                flip = torch.rand(len(xb), device=device) < .5
                xb[flip] = xb[flip].flip((-2, -1))
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(xb)
        metrics = score_structures(predict_replicates(model, x, val, n, device), y[val])
        record = {"epoch": epoch, "train_loss": running / len(train_samples), **metrics}
        history.append(record)
        # Chosen criterion is macro F1; accuracy only resolves ties. Earlier
        # epochs win otherwise. CHID metrics are descriptive, not selectors.
        key = (metrics["macro_f1"], metrics["accuracy"], -epoch)
        if key > best_key:
            best_key = key
            best_record = record
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No trained checkpoint")
    model.load_state_dict(best_state)
    return model, best_record, history


def run(args: argparse.Namespace) -> dict:
    x, y, saved, masks = read_fixed_split(args.windows, args.manifest, args.annotations,
                                           args.window_bp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates = []
    selected_model = None
    selected_info = None
    selected_key = (-1.0, -1.0, -999, -999)
    for index, candidate in enumerate(CANDIDATES):
        model, best, history = fit_and_select(x, y, saved, candidate, args.epochs,
                                              args.seed, device)
        info = {"candidate": candidate.__dict__, "best_validation": best, "history": history}
        candidates.append(info)
        key = (best["macro_f1"], best["accuracy"], -best["epoch"], -index)
        if key > selected_key:
            selected_key, selected_info, selected_model = key, info, model
    assert selected_info is not None and selected_model is not None
    # First and only use of the test partition. No test metric informs the
    # candidate set, epoch choice, threshold or following post-processing.
    test = np.flatnonzero(saved.split.eq("test"))
    logits = predict_replicates(selected_model, x, test, len(y), device)
    pred = logits.mean(axis=0).argmax(axis=1)
    cm = confusion_matrix(y[test], pred, labels=range(3))
    report = classification_report(y[test], pred, labels=range(3),
                                   target_names=CLASSES, output_dict=True, zero_division=0)
    sample_truth = np.tile(y[test], 2)
    sample_pred = logits.argmax(axis=2).reshape(-1)
    sample_report = classification_report(sample_truth, sample_pred, labels=range(3),
                                          target_names=CLASSES, output_dict=True, zero_division=0)
    metrics = {"protocol": "fixed 5120 bp spatial train/validation/test manifest; 2 replicates grouped by ID",
               "caution": "Exploratory reuse of previously reported test fold; independent confirmation still required",
               "window_bp": args.window_bp, "seed": args.seed,
               "train_structures": int(masks["train"].sum() // 2),
               "validation_structures": int(masks["validation"].sum() // 2),
               "test_structures": len(test),
               "selection": "best validation structure-level macro F1, then accuracy, then earlier epoch/candidate order; no test tuning",
               "candidates": candidates, "selected": selected_info["candidate"],
               "selected_epoch": selected_info["best_validation"]["epoch"],
               "validation": selected_info["best_validation"],
               "test": {**score_structures(logits, y[test]),
                        "confusion_matrix": cm.tolist(), "classification_report": report},
               "test_sample_level": {
                   "accuracy": float(accuracy_score(sample_truth, sample_pred)),
                   "macro_f1": float(f1_score(sample_truth, sample_pred, average="macro", zero_division=0)),
                   "confusion_matrix": confusion_matrix(sample_truth, sample_pred, labels=range(3)).tolist(),
                   "classification_report": sample_report},
               "device": str(device)}
    args.out.mkdir(parents=True, exist_ok=True)
    rows = saved.iloc[test][["id", "type", "chrom", "center", "split"]].copy()
    rows["predicted_type"] = [CLASSES[i] for i in pred]
    rows["replicate_agreement"] = logits.argmax(axis=2)[0] == logits.argmax(axis=2)[1]
    for ci, name in enumerate(CLASSES):
        rows[f"logit_{name}"] = logits[:, :, ci].mean(axis=0)
    rows.to_csv(args.out / "test_predictions.csv", index=False)
    sample_rows = pd.concat([saved.iloc[test]] * 2, ignore_index=True)[
        ["id", "type", "chrom", "center", "split"]].copy()
    sample_rows["replicate"] = np.repeat([1, 2], len(test))
    sample_rows["predicted_type"] = [CLASSES[i] for i in sample_pred]
    sample_rows.to_csv(args.out / "test_sample_predictions.csv", index=False)
    (args.out / "summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"state_dict": selected_model.state_dict(), "candidate": selected_info["candidate"],
                "epoch": selected_info["best_validation"]["epoch"], "classes": CLASSES},
               args.out / "selected_cnn.pt")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", type=Path, default=Path("outputs/task1_spatial/extracted_windows.npz"))
    ap.add_argument("--manifest", type=Path, default=Path("results/task1_spatial_annotation_manifest.csv"))
    ap.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    ap.add_argument("--out", type=Path, default=Path("outputs/task1_val_selected_cnn"))
    ap.add_argument("--window-bp", type=int, default=5120)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()
    metrics = run(args)
    print(json.dumps({"selected": metrics["selected"], "validation": metrics["validation"],
                      "test": metrics["test"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
