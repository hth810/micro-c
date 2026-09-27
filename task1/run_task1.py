"""Task 1: supervised recognition of known Micro-C structures.

The script extracts fixed-size windows around the 344 annotated structures,
normalizes them with log1p + per-window robust scaling, trains a small CNN,
and writes metrics, a confusion matrix, model weights, and class saliency maps.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import cooler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from microc.data import annotations as load_shared_annotations, normalize, open_cool


CLASSES = ["CHIN", "OPCID", "CHID"]


def seed_all(seed: int) -> None:
    # Required by deterministic CUDA matrix multiplication on CUDA >= 10.2.
    # Set before torch initializes a CuBLAS context.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def load_annotations(xlsx: Path) -> pd.DataFrame:
    """Read the supplied workbook or a unified structures.csv."""
    out = load_shared_annotations(xlsx)
    if len(out) != 344 or set(out["type"]) != set(CLASSES):
        raise ValueError(f"Expected 344 annotations in 3 classes, found {len(out)}")
    out["label"] = out["type"].map({c: i for i, c in enumerate(CLASSES)})
    return out


def spatial_split(ann: pd.DataFrame, seed: int, window_bp: int) -> tuple[set, set, set]:
    """Keep overlapping input windows in the same genomic component.

    A gap of at least one full window between neighboring annotation centers
    is required before starting a new component. Both replicates inherit the
    same split through their annotation ID.
    """
    ordered = ann.sort_values(["chrom", "center"]).reset_index(drop=True)
    new_group = ordered.chrom.ne(ordered.chrom.shift()) | ordered.center.diff().ge(window_bp)
    groups = new_group.cumsum().to_numpy()
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fit_idx, test_idx = next(folds.split(ordered, ordered.label, groups))
    fit = ordered.iloc[fit_idx].reset_index(drop=True)
    fit_groups = groups[fit_idx]
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 1)
    train_idx, val_idx = next(inner.split(fit, fit.label, fit_groups))
    return set(fit.iloc[train_idx].id), set(fit.iloc[val_idx].id), set(ordered.iloc[test_idx].id)


def split_from_manifest(ann: pd.DataFrame, path: Path, window_bp: int) -> tuple[set, set, set]:
    """Reuse an audited split across scikit-learn versions and machines."""
    saved = pd.read_csv(path)
    required = {"id", "type", "chrom", "center", "split"}
    if not required.issubset(saved.columns):
        raise ValueError(f"Split manifest lacks {sorted(required - set(saved.columns))}")
    if saved.id.duplicated().any() or set(saved.id) != set(ann.id):
        raise ValueError("Split manifest must contain each current annotation ID exactly once")
    current = ann.set_index("id")
    recorded = saved.set_index("id").loc[current.index]
    for field in ("type", "chrom", "center"):
        if not np.array_equal(current[field].to_numpy(), recorded[field].to_numpy()):
            raise ValueError(f"Split manifest does not match current annotation {field}")
    if not set(saved.split).issubset({"train", "validation", "test"}):
        raise ValueError("Split manifest contains an unknown partition")
    ordered = saved.sort_values(["chrom", "center"])
    for chrom, subset in ordered.groupby("chrom"):
        near = subset.center.diff().to_numpy()[1:] < window_bp
        crossing = subset.split.to_numpy()[1:] != subset.split.to_numpy()[:-1]
        if np.any(near & crossing):
            raise ValueError(f"Split manifest has overlapping windows across partitions on {chrom}")
    parts = tuple(set(saved.loc[saved.split == name, "id"]) for name in ("train", "validation", "test"))
    if not all(parts):
        raise ValueError("Split manifest has an empty partition")
    return parts


def extract_window(c: cooler.Cooler, center_bp: int, window_bins: int, aggregate: int = 4) -> np.ndarray:
    # Read a wider physical context, then average-pool it to the CNN input.
    # This keeps the tensor small while retaining multi-kb CHID/OPCID shapes.
    raw_bins = window_bins * aggregate
    half = raw_bins // 2
    center_bin = int(center_bp // c.binsize)
    lo = max(0, center_bin - half)
    hi = min(int(c.shape[0]), lo + raw_bins)
    lo = max(0, hi - raw_bins)
    mat = c.matrix(balance=False, sparse=True)[lo:hi, lo:hi].toarray().astype(np.float32)
    if mat.shape != (raw_bins, raw_bins):
        padded = np.zeros((raw_bins, raw_bins), dtype=np.float32)
        padded[: mat.shape[0], : mat.shape[1]] = mat
        mat = padded
    if aggregate > 1:
        mat = mat.reshape(window_bins, aggregate, window_bins, aggregate).mean(axis=(1, 3))
    return normalize(mat)


class StructureCNN(nn.Module):
    def __init__(self, n_classes: int, spatial_grid: int = 1):
        super().__init__()
        if spatial_grid < 1:
            raise ValueError("spatial_grid must be positive")
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((spatial_grid, spatial_grid)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.2),
                                        nn.Linear(64 * spatial_grid * spatial_grid, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def save_confusion(cm: np.ndarray, out: Path) -> None:
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out / "confusion_matrix.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(5, 4), dpi=180)
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted", ylabel="True", title="Held-out confusion matrix")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * .55 else "black")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png"); plt.close(fig)


def select_validation_bias(logits: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, dict]:
    """Tune two small class offsets on validation data, never on test data."""
    raw = logits.argmax(axis=1)
    before = {"accuracy": float(accuracy_score(truth, raw)),
              "macro_f1": float(f1_score(truth, raw, average="macro"))}
    best_bias = np.zeros(3, dtype=np.float32)
    best_key = (-1.0, -1.0, -float("inf"))
    for opc in np.linspace(-1.5, 1.5, 13):
        for chid in np.linspace(-3, 1.5, 19):
            bias = np.array([0, opc, chid], dtype=np.float32)
            prediction = (logits + bias).argmax(axis=1)
            key = (float(f1_score(truth, prediction, average="macro")),
                   float(accuracy_score(truth, prediction)), -float(np.linalg.norm(bias)))
            if key > best_key:
                best_key, best_bias = key, bias
    after = {"accuracy": best_key[1], "macro_f1": best_key[0]}
    return best_bias, {"uncalibrated": before, "calibrated": after}


def save_saliency(model, x_test, y_test, out: Path, device: torch.device,
                  logit_bias: np.ndarray | None = None) -> None:
    model.eval()
    x = x_test.to(device).clone().requires_grad_(True)
    logits = model(x)
    adjusted = logits + torch.tensor(logit_bias if logit_bias is not None else np.zeros(3),
                                      dtype=logits.dtype, device=device)
    pred = adjusted.argmax(1)
    for ci, cname in enumerate(CLASSES):
        idx = torch.where((y_test.to(device) == ci) & (pred == ci))[0]
        if len(idx) == 0:
            idx = torch.where(y_test.to(device) == ci)[0]
        if len(idx) == 0:
            continue
        maps = []
        for k in idx[: min(20, len(idx))]:
            model.zero_grad(set_to_none=True); x.grad = None
            logits[k, ci].backward(retain_graph=True)
            maps.append(x.grad[k].detach().abs().cpu().numpy()[0])
        sal = np.mean(maps, axis=0)
        fig, ax = plt.subplots(figsize=(5, 4), dpi=180)
        im = ax.imshow(sal, cmap="magma", origin="lower")
        fig.colorbar(im, ax=ax, fraction=0.046, label="|d logit / d input|")
        ax.set(title=f"Saliency: {cname} (n={len(maps)})", xlabel="window bin", ylabel="window bin")
        fig.tight_layout(); fig.savefig(out / f"saliency_{cname}.png"); plt.close(fig)


def save_example_heatmaps(x_test, y_test, preds, out: Path) -> None:
    """Save one correctly recognized normalized contact window per class."""
    xx, yy = x_test.cpu().numpy(), y_test.cpu().numpy()
    for ci, cname in enumerate(CLASSES):
        hit = np.where((yy == ci) & (preds == ci))[0]
        if len(hit) == 0:
            continue
        mat = xx[hit[0], 0]
        fig, ax = plt.subplots(figsize=(5, 4), dpi=180)
        im = ax.imshow(mat, cmap="magma", origin="lower", vmin=-1, vmax=4)
        fig.colorbar(im, ax=ax, fraction=0.046, label="robust-normalized log contact")
        ax.set(title=f"Correctly recognized {cname}", xlabel="window bin", ylabel="window bin")
        fig.tight_layout(); fig.savefig(out / f"example_{cname}.png"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--annotations", type=Path)
    ap.add_argument("--rep1", type=Path)
    ap.add_argument("--rep2", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("task1/output"))
    ap.add_argument("--window-bins", type=int, default=128)
    ap.add_argument("--aggregate", type=int, default=4, help="raw bins per CNN bin (default 4 = 5,120 bp context)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--split-mode", choices=("spatial", "structure"), default="spatial",
                    help="spatial prevents overlapping windows crossing splits; structure reproduces legacy result")
    ap.add_argument("--split-gap-bp", type=int,
                    help="minimum group-separating center gap; defaults to the physical input window")
    ap.add_argument("--split-manifest", type=Path,
                    help="reuse an audited annotation manifest to reproduce the exact train/validation/test partition")
    ap.add_argument("--chid-weight-multiplier", type=float, default=1.0,
                    help="experimental additional CHID loss weight; select using validation only")
    ap.add_argument("--spatial-grid", type=int, default=1,
                    help="size of final spatial feature grid; 1 reproduces global average pooling")
    ap.add_argument("--calibrate", action="store_true",
                    help="experimental validation-set logit calibration; disabled by default")
    args = ap.parse_args()
    seed_all(args.seed); args.out_dir.mkdir(parents=True, exist_ok=True)
    xlsx = args.annotations or next(args.data_dir.glob("*.xlsx"))
    if args.rep1 and args.rep2:
        cools = [args.rep1, args.rep2]
    else:
        cools = []
        for tag in ("rep1", "rep2"):
            choices = list(args.data_dir.glob(f"*37C_{tag}*.cool")) or list(
                args.data_dir.glob(f"*37C_{tag}*.cool.gz"))
            if choices:
                cools.append(sorted(choices)[0])
    if len(cools) < 2: raise FileNotFoundError("Need the two 37C .cool biological replicates")
    ann = load_annotations(xlsx)
    if args.split_manifest and args.split_mode != "spatial":
        raise ValueError("--split-manifest requires --split-mode spatial")
    if args.split_mode == "spatial":
        split_gap_bp = args.split_gap_bp or args.window_bins * args.aggregate * 10
        if split_gap_bp < args.window_bins * args.aggregate * 10:
            raise ValueError("spatial split gap must cover the full input window")
        train_ids, val_ids, test_ids = (split_from_manifest(ann, args.split_manifest, split_gap_bp)
                                       if args.split_manifest else spatial_split(ann, args.seed, split_gap_bp))
    else:
        split_gap_bp = None
        ids_fit, ids_test, labels_fit, _ = train_test_split(ann["id"].values, ann["label"].values,
            test_size=.2, random_state=args.seed, stratify=ann["label"].values)
        ids_train, ids_val = train_test_split(ids_fit, test_size=.2, random_state=args.seed + 1,
                                             stratify=labels_fit)
        train_ids, val_ids, test_ids = set(ids_train), set(ids_val), set(ids_test)
    if not (train_ids and val_ids and test_ids):
        raise ValueError("Empty train, validation, or test partition")
    rows = []
    mats, ys, groups = [], [], []
    for rep, cf in enumerate(cools[:2], 1):
        with open_cool(cf) as c:
            for _, r in ann.iterrows():
                mats.append(extract_window(c, int(r.center), args.window_bins, args.aggregate))
                ys.append(int(r.label)); groups.append((r.id, rep))
    X = np.stack(mats)[:, None, :, :].astype(np.float32); y = np.asarray(ys, dtype=np.int64)
    group_ids = np.asarray([g[0] for g in groups])
    tr = np.isin(group_ids, list(train_ids)); va_mask = np.isin(group_ids, list(val_ids))
    te = np.isin(group_ids, list(test_ids))
    # Keep a complete machine-readable manifest for reproducibility.
    manifest = ann.copy()
    manifest["split"] = np.select([manifest.id.isin(list(train_ids)), manifest.id.isin(list(val_ids))],
                                   ["train", "validation"], default="test")
    manifest.to_csv(args.out_dir / "annotation_manifest.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(args.out_dir / "extracted_windows.npz", X=X, y=y, is_train=tr,
                        is_validation=va_mask, is_test=te, group=group_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StructureCNN(3, args.spatial_grid).to(device)
    train_ds = TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(y[tr]))
    val_x, val_y = torch.from_numpy(X[va_mask]).to(device), torch.from_numpy(y[va_mask]).to(device)
    test_x, test_y = torch.from_numpy(X[te]).to(device), torch.from_numpy(y[te]).to(device)
    loader = DataLoader(train_ds, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    counts = np.bincount(y[tr], minlength=3).astype(np.float32)
    class_weights = counts.sum() / (len(CLASSES) * np.maximum(counts, 1))
    if args.chid_weight_multiplier <= 0:
        raise ValueError("--chid-weight-multiplier must be positive")
    class_weights[CLASSES.index("CHID")] *= args.chid_weight_multiplier
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    history = []
    best_state, best_score, best_epoch = None, -1.0, 0
    for epoch in range(args.epochs):
        model.train(); running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device); opt.zero_grad()
            loss = loss_fn(model(xb), yb); loss.backward(); opt.step(); running += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            val_pred = model(val_x).argmax(1)
            va = (val_pred == val_y).float().mean().item()
            vf1 = f1_score(val_y.cpu().numpy(), val_pred.cpu().numpy(), average="macro")
        history.append({"epoch": epoch + 1, "train_loss": running / len(train_ds),
                        "validation_accuracy": va, "validation_macro_f1": float(vf1)})
        if vf1 > best_score:
            best_score, best_epoch = float(vf1), epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        validation_logits = model(val_x).cpu().numpy()
        test_logits = model(test_x).cpu().numpy()
    candidate_bias, calibration = select_validation_bias(validation_logits, val_y.cpu().numpy())
    logit_bias = candidate_bias if args.calibrate else np.zeros(3, dtype=np.float32)
    calibration["applied"] = bool(args.calibrate)
    calibration["candidate_bias"] = candidate_bias.tolist()
    raw_preds = test_logits.argmax(axis=1)
    preds = (test_logits + logit_bias).argmax(axis=1)
    yy = test_y.cpu().numpy(); cm = confusion_matrix(yy, preds, labels=range(3))
    test_indices = np.flatnonzero(te)
    pd.DataFrame({
        "structure_id": group_ids[test_indices],
        "replicate": [groups[i][1] for i in test_indices],
        "true_type": [CLASSES[i] for i in yy],
        "uncalibrated_predicted_type": [CLASSES[i] for i in raw_preds],
        "predicted_type": [CLASSES[i] for i in preds],
        "correct": yy == preds,
    }).to_csv(args.out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    metrics = {"seed": args.seed, "split_mode": args.split_mode,
               "split_manifest": str(args.split_manifest) if args.split_manifest else None,
               "split_gap_bp": split_gap_bp,
               "chid_weight_multiplier": args.chid_weight_multiplier,
               "spatial_grid": args.spatial_grid,
               "class_weights": class_weights.tolist(),
               "window_bins": args.window_bins, "aggregate": args.aggregate,
               "window_bp": args.window_bins * args.aggregate * 10,
               "train_structures": len(train_ids), "validation_structures": len(val_ids),
               "test_structures": len(test_ids), "train_samples": int(tr.sum()),
               "validation_samples": int(va_mask.sum()), "test_samples": int(te.sum()),
               "accuracy": float(accuracy_score(yy, preds)),
               "macro_f1": float(f1_score(yy, preds, average="macro")),
               "uncalibrated_test_accuracy": float(accuracy_score(yy, raw_preds)),
               "uncalibrated_test_macro_f1": float(f1_score(yy, raw_preds, average="macro")),
               "logit_bias": logit_bias.tolist(), "validation_calibration": calibration,
               "best_epoch": best_epoch,
               "classification_report": classification_report(yy, preds, target_names=CLASSES, output_dict=True),
               "history": history, "device": str(device), "replicates": [p.name for p in cools[:2]]}
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"model_state_dict": model.state_dict(), "classes": CLASSES, "window_bins": args.window_bins,
                "aggregate": args.aggregate, "spatial_grid": args.spatial_grid,
                "seed": args.seed, "logit_bias": logit_bias.tolist()},
               args.out_dir / "task1_cnn.pt")
    save_confusion(cm, args.out_dir); save_saliency(model, test_x, test_y, args.out_dir, device, logit_bias)
    save_example_heatmaps(test_x, test_y, preds, args.out_dir)
    print(json.dumps({"accuracy": metrics["accuracy"], "confusion_matrix": cm.tolist(), "output": str(args.out_dir)}, ensure_ascii=False))


if __name__ == "__main__": main()



