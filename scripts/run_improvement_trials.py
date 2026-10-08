"""Fixed retrospective Task 1/5 trials; never replace published experiments.

All arms are reported. Outer tests never choose an architecture, seed or
threshold. SR epochs and residual blend use only the spatial validation split.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedGroupKFold
from skimage.metrics import structural_similarity
from torch import nn

from microc.data import area_window, normalize, open_cool, window
from microc.deep import ResidualBlock, _gradient_loss, _psnr, prepare_sr_pair
from scripts.evaluate_sr_spatial_blend import known_error_arrays
from scripts.verify_improvements_20261006 import digest
from task1.audit_interval_conditioned_cnn import paired_bootstrap
from task1.audit_roi_cnn_20261006 import predict_states, side_features
from task1.evaluate_chid_overlap import spatial_groups
from task1.evaluate_pairwise import evaluate
from task1.evaluate_roi_cnn_20261006 import fit_predict, make_images
from task1.run_task1 import CLASSES, extract_window, seed_all

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def trial_protocol(args, details):
    if args.out.exists():
        raise FileExistsError(f"Keep prior trials; choose a new --out: {args.out}")
    args.out.mkdir(parents=True)
    protocol = {"frozen_utc": datetime.now(timezone.utc).isoformat(),
                "caution": "Retrospective reuse of inspected labels/regions, not new blind validation",
                "selection": "Report all predeclared arms; never select using outer/test results",
                "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                    (Path(__file__), ROOT / "microc/deep.py", ROOT / "microc/data.py",
                     ROOT / "task1/evaluate_roi_cnn_20261006.py")},
                "python_torch": torch.__version__, "seed": args.seed, **details}
    save_json(args.out / "protocol.json", protocol)
    return protocol


def run_task1(args):
    started = time.perf_counter()
    ann = pd.read_csv(ROOT / "data/structures.csv").reset_index(drop=True)
    n = len(ann)
    y = ann.type.map({label: i for i, label in enumerate(CLASSES)}).to_numpy(dtype=int)
    groups = spatial_groups(ann, 40000)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed).split(ann, y, groups))
    trial_protocol(args, {"task": 1, "epochs": 24, "device": "cpu", "torch_threads": 2,
        "initialization_offsets": [0, 31], "gap_bp": 40000,
        "arms": ["side_single", "side_ensemble", "images_single", "images_ensemble"],
        "folds": [{"train_ids": ann.id.iloc[train].tolist(), "test_ids": ann.id.iloc[test].tolist()}
                  for train, test in folds],
        "inputs": {"annotations_sha256": digest(ROOT / "data/structures.csv"),
                   "raw_sha256": {p.name: digest(p) for p in args.rep_paths}}})
    windows = []
    for rep, path in enumerate(args.rep_paths, 1):
        with open_cool(path) as cool:
            for row in ann.itertuples():
                windows.append(np.stack([extract_window(cool, int(row.center), 128, factor)
                                         for factor in (4, 16)]))
        print(f"Task1 prepared replicate {rep}", flush=True)
    x = np.stack(windows).astype(np.float32)
    np.savez_compressed(args.out / "windows.npz", X=x, group=np.tile(ann.id.to_numpy(dtype=str), 2))
    images = make_images(x, ann)["interval_roi64"]
    side = side_features(x, ann)
    empty = np.empty((2 * n, 0), np.float32)
    names = ("side_single", "side_ensemble", "images_single", "images_ensemble")
    probabilities = {name: np.full((2, n, 3), np.nan, np.float32) for name in names}
    fold_ids = np.full(n, -1)
    for fold, (train, test) in enumerate(folds):
        if np.intersect1d(groups[train], groups[test]).size:
            raise ValueError("Spatial leakage")
        fold_ids[test] = fold
        for name, descriptors in (("side", side), ("images", empty)):
            prediction, states = fit_predict(images, descriptors, y, train, test, args.seed + fold * 100)
            probabilities[name + "_ensemble"][:, test] = prediction
            ids = np.r_[test, test + n]
            probabilities[name + "_single"][:, test] = predict_states(
                images[ids], descriptors[ids], states[:1]).reshape(2, len(test), 3)
            torch.save(states, args.out / f"fold{fold}_{name}_states.pt")
            print(f"Task1 fold {fold+1}/5 {name} complete; {time.perf_counter()-started:.1f}s", flush=True)
    if any(not np.isfinite(p).all() for p in probabilities.values()):
        raise ValueError("Incomplete outer predictions")
    predictions = {name: p.mean(axis=0).argmax(axis=1) for name, p in probabilities.items()}
    scores = {name: evaluate(y, p) for name, p in predictions.items()}
    uncertainty = paired_bootstrap(y, groups, predictions, "images_ensemble",
                                   ("side_ensemble", "images_single"), 4000, args.seed + 97)
    rows = {"id": ann.id, "type": ann.type, "center": ann.center,
            "spatial_group": groups, "outer_fold": fold_ids}
    for name, p in probabilities.items():
        rows[name + "_prediction"] = np.asarray(CLASSES)[predictions[name]]
        for rep in (0, 1):
            for cls, label in enumerate(CLASSES):
                rows[f"{name}_rep{rep+1}_p_{label}"] = p[rep, :, cls]
    pd.DataFrame(rows).to_csv(args.out / "outer_predictions.csv", index=False)
    summary = {"scores": scores, "paired_spatial_uncertainty": uncertainty,
               "wall_seconds": time.perf_counter() - started,
               "protocol_sha256": digest(args.out / "protocol.json"),
               "caution": "New fits on a fixed spatial split; historical labels already inspected. No winning arm deployed."}
    save_json(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


class ContextSR(nn.Module):
    """Same-size two-channel control: zero context vs measured context."""
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(2, 48, 5, padding=2), nn.ReLU())
        self.body = nn.Sequential(*[ResidualBlock(48) for _ in range(4)], nn.Conv2d(48, 48, 3, padding=1))
        self.tail = nn.Conv2d(48, 1, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        base = x[:, :1]
        observed = (base.abs().sum((1, 2, 3), keepdim=True) > 0).to(x.dtype)
        result = base + observed * self.tail(self.body(self.head(x)))
        return (result + result.transpose(-1, -2)) / 2


def context_pair(cool, chrom, center):
    base, target = prepare_sr_pair(window(cool, chrom, int(center), 640, 64), 4)
    context = normalize(area_window(cool, chrom, int(center), 2560, 64))
    return np.stack((base, context)), target


def sr_metrics(truth, pred):
    upper = np.triu_indices(64, k=3)
    rows = []
    for a, b in zip(truth, pred):
        rho = float(spearmanr(a[upper], b[upper]).statistic) if np.std(a[upper]) > 1e-8 and np.std(b[upper]) > 1e-8 else None
        rows.append({"psnr": _psnr(a, b), "ssim": float(structural_similarity(a, b, data_range=10)), "rho": rho})
    return rows, {"psnr": float(np.mean([r["psnr"] for r in rows])),
                  "ssim": float(np.mean([r["ssim"] for r in rows])),
                  "rho": float(np.mean([r["rho"] for r in rows if r["rho"] is not None]))}


def block_gain(base, candidate, centers, seed):
    groups = np.asarray(centers) // 100000
    unique = np.unique(groups)
    differences = np.asarray(candidate) - np.asarray(base)
    total = np.asarray([differences[groups == g].sum() for g in unique])
    count = np.asarray([(groups == g).sum() for g in unique])
    picks = np.random.default_rng(seed).integers(0, len(unique), (4000, len(unique)))
    values = total[picks].sum(axis=1) / count[picks].sum(axis=1)
    return {"difference": float(differences.mean()), "ci95": np.quantile(values, [.025, .975]).tolist(), "genomic_blocks": len(unique)}


def run_task5(args):
    started = time.perf_counter()
    with open_cool(args.rep_paths[0]) as cool:
        chrom = cool.chromnames[0]
        if int(cool.binsize) != 10:
            raise ValueError("This fixed physical-context trial requires 10 bp inputs")
        length = int(cool.chromsizes[chrom])
    centers = list(range(320, length - 320, 640))
    centers = centers[::max(1, int(np.ceil(len(centers) / 1200)))]
    train_end, test_start = int(.7 * len(centers)), int(.8 * len(centers))
    if min(centers[train_end] - centers[train_end-1], centers[test_start] - centers[test_start-1]) < 2560:
        raise ValueError("Wide contexts overlap split boundaries")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trial_protocol(args, {"task": 5, "epochs": 12, "seeds": [args.seed, args.seed+1],
        "device": device, "arms": ["zero_context", "wide_context"], "context_bp": 2560,
        "target_bp": 640, "target_bin_bp": 10, "factor": 4,
        "split_centers": {"train": centers[:train_end], "validation": centers[train_end:test_start], "test": centers[test_start:]},
        "model": "2->48, four residual blocks, zero-initialized residual output, symmetry",
        "optimizer": "Adam .001 batch32", "loss": "MSE + .15 gradient smooth-L1",
        "selection": "Within each arm, epoch and blend0:.1:1 minimize validation MSE; report both arms and both seeds",
        "context": "Raw wider counts pooled by area then independently normalized; no target statistics in inputs",
        "inputs": {p.name: digest(p) for p in args.rep_paths}})
    ann = pd.read_csv(ROOT / "data/structures.csv")
    known = ann[ann.center >= centers[test_start]].reset_index(drop=True)
    datasets = {}
    for rep, path in enumerate(args.rep_paths, 1):
        tile_centers = centers if rep == 1 else centers[test_start:]
        with open_cool(path) as cool:
            pairs = [context_pair(cool, chrom, c) for c in tile_centers]
            known_pairs = [context_pair(cool, chrom, c) for c in known.center]
        datasets[rep] = (np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs]),
                         np.stack([p[0] for p in known_pairs]), np.stack([p[1] for p in known_pairs]))
        print(f"Task5 prepared rep{rep}: {len(pairs)} tiles, {len(known)} known centers", flush=True)
    inputs, truth, _, _ = datasets[1]
    rows, summaries, known_rows = [], {}, []
    saved_predictions = {}
    for seed in (args.seed, args.seed + 1):
        for mode in ("zero_context", "wide_context"):
            seed_all(seed)
            x = inputs.copy()
            if mode == "zero_context":
                x[:, 1] = 0
            xt = torch.from_numpy(x)
            yt = torch.from_numpy(truth[:, None])
            model = ContextSR().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            best_loss, best_state, best_epoch = float("inf"), None, None
            def infer(array):
                current = array.copy()
                if mode == "zero_context":
                    current[:, 1] = 0
                model.eval()
                with torch.inference_mode():
                    return np.concatenate([model(piece.to(device)).cpu().numpy()[:, 0]
                                           for piece in torch.from_numpy(current).split(32)])
            for epoch in range(12):
                model.train()
                for ids in torch.randperm(train_end).split(32):
                    pred = model(xt[ids].to(device))
                    target = yt[ids].to(device)
                    loss = nn.functional.mse_loss(pred, target) + .15 * _gradient_loss(pred, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                val = infer(x[train_end:test_start])
                val_loss = float(np.mean((val - truth[train_end:test_start]) ** 2))
                if val_loss < best_loss:
                    best_loss, best_epoch = val_loss, epoch + 1
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if (epoch + 1) % 4 == 0:
                    print(f"Task5 seed{seed} {mode} epoch{epoch+1}/12 valMSE={val_loss:.4f}", flush=True)
            model.load_state_dict(best_state)
            val = infer(x[train_end:test_start])
            base_val = inputs[train_end:test_start, 0]
            blend = float(min(np.linspace(0, 1, 11), key=lambda a: np.mean(
                (truth[train_end:test_start] - (base_val + a * (val - base_val))) ** 2)))
            label = f"{mode}_seed{seed}"
            torch.save({"state_dict": best_state, "blend": blend, "seed": seed, "mode": mode}, args.out / (label + ".pt"))
            summaries[label] = {"validation_mse": best_loss, "best_epoch": best_epoch, "blend": blend}
            for rep in (1, 2):
                raw_x, raw_y, kx, ky = datasets[rep]
                if rep == 1:
                    raw_x, raw_y = raw_x[test_start:], raw_y[test_start:]
                output = raw_x[:, 0] + blend * (infer(raw_x) - raw_x[:, 0])
                saved_predictions[f"rep{rep}_{label}"] = output
                detail, score = sr_metrics(raw_y, output)
                summaries[label][f"rep{rep}"] = score
                for center, metric in zip(centers[test_start:], detail):
                    rows.append({"method": label, "replicate": rep, "center": center, **metric})
                kp = kx[:, 0] + blend * (infer(kx) - kx[:, 0])
                base_errors, errors = known_error_arrays(ky, kx[:, 0], kp)
                summaries[label][f"rep{rep}_hotspot_bins"] = float(errors.mean())
                summaries[label][f"rep{rep}_baseline_hotspot_bins"] = float(base_errors.mean())
                for item, a, b in zip(known.itertuples(), base_errors, errors):
                    known_rows.append({"id": item.id, "type": item.type, "center": item.center,
                                       "method": label, "replicate": rep,
                                       "baseline_error_bins": a, "prediction_error_bins": b})
            print(f"Task5 {label} finished; {time.perf_counter()-started:.1f}s", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "tile_metrics.csv", index=False)
    pd.DataFrame(known_rows).to_csv(args.out / "known_hotspot_metrics.csv", index=False)
    differences = {}
    for seed in (args.seed, args.seed+1):
        for rep in (1, 2):
            base = frame[(frame.method == f"zero_context_seed{seed}") & (frame.replicate == rep)]
            candidate = frame[(frame.method == f"wide_context_seed{seed}") & (frame.replicate == rep)]
            differences[f"seed{seed}_rep{rep}"] = {metric: block_gain(base[metric], candidate[metric], base.center, args.seed)
                                                   for metric in ("psnr", "ssim")}
    np.savez_compressed(args.out / "test_predictions.npz", centers=centers[test_start:], **saved_predictions)
    summary = {"arms": summaries, "context_minus_control_block_bootstrap": differences,
               "train_tiles": train_end, "validation_tiles": test_start-train_end, "test_tiles": len(centers)-test_start,
               "known_test_centers": len(known), "wall_seconds": time.perf_counter()-started,
               "caution": "Retrospective spatial validation/test and rep2 robustness; no new independent Hi-C or physical truth. All arms retained."}
    save_json(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, choices=(1, 5), required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "micro-c数据(1)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()
    args.rep_paths = [args.data_dir / f"GSE272159_37C_rep{rep}.mapq_30.10.cool" for rep in (1, 2)]
    if any(not p.is_file() for p in args.rep_paths):
        raise FileNotFoundError("Both raw WT replicate COOL files are required")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.task == 1:
        run_task1(args)
    else:
        run_task5(args)


if __name__ == "__main__":
    main()
