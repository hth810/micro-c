"""Tasks 5-6: super-resolution and 3D chromosome reconstruction."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom
from scipy.stats import spearmanr
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import nn

from .data import annotation_chrom_mask, annotations, normalize, open_cool, window


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class SuperResolutionCNN(nn.Module):
    """Residual correction network for the interpolated low-resolution map."""
    def __init__(self, channels: int = 48, blocks: int = 4):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(1, channels, 5, padding=2), nn.ReLU(inplace=True))
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)],
                                   nn.Conv2d(channels, channels, 3, padding=1))
        self.tail = nn.Conv2d(channels, 1, 3, padding=1)
        # Start at the interpolation baseline. The network learns only a residual.
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        observed = (x.abs().sum(dim=(1, 2, 3), keepdim=True) > 0).to(x.dtype)
        features = self.head(x)
        correction = self.tail(self.body(features))
        result = x + observed * correction
        return (result + result.transpose(-1, -2)) / 2


def _gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalize loss of contact boundaries and stripes in addition to intensity error."""
    dx_pred = prediction[..., :, 1:] - prediction[..., :, :-1]
    dy_pred = prediction[..., 1:, :] - prediction[..., :-1, :]
    dx_true = target[..., :, 1:] - target[..., :, :-1]
    dy_true = target[..., 1:, :] - target[..., :-1, :]
    return (nn.functional.smooth_l1_loss(dx_pred, dx_true) +
            nn.functional.smooth_l1_loss(dy_pred, dy_true)) / 2


def _psnr(target: np.ndarray, prediction: np.ndarray) -> float:
    if np.mean((target - prediction) ** 2) < 1e-12:
        return 100.0
    return float(peak_signal_noise_ratio(target, prediction, data_range=10))


def degrade(high: np.ndarray, factor: int) -> np.ndarray:
    n = len(high)
    low = high.reshape(n // factor, factor, n // factor, factor).mean(axis=(1, 3))
    return zoom(low, factor, order=1).astype(np.float32)


def prepare_sr_pair(raw: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    """Produce input and target without exposing target normalization to input."""
    target = normalize(raw)
    input_map = normalize(degrade(raw, factor))
    return input_map, target


def superresolve(args) -> None:
    seed_all(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open_cool(args.cool) as c:
        chrom = args.chrom or c.chromnames[0]
        length = int(c.chromsizes[chrom])
        patch_bp = args.patch_bins * c.binsize
        # Non-overlapping genomic tiles, with one held-out chromosome segment.
        centers = list(range(patch_bp // 2, length - patch_bp // 2, patch_bp))
        centers = centers[::max(1, int(np.ceil(len(centers) / args.max_patches)))]
        train_end, test_start = int(.7 * len(centers)), int(.8 * len(centers))
        if train_end < 4 or test_start <= train_end or test_start == len(centers):
            raise ValueError("At least ten non-overlapping patches are required")
        raw = np.stack([window(c, chrom, x, patch_bp, args.patch_bins) for x in centers])
        # Aggregate observed counts first. Normalizing high before degradation
        # would leak high-resolution distance/background statistics to the input.
        pairs = [prepare_sr_pair(m, args.factor) for m in raw]
        low = np.stack([pair[0] for pair in pairs])
        high = np.stack([pair[1] for pair in pairs])
    x = torch.from_numpy(low[:, None]); y = torch.from_numpy(high[:, None])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SuperResolutionCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_state, best_loss = None, float("inf")
    for epoch in range(args.epochs):
        model.train()
        for indices in torch.randperm(train_end).split(args.batch_size):
            prediction = model(x[indices].to(device))
            target = y[indices].to(device)
            loss = nn.functional.mse_loss(prediction, target) + .15 * _gradient_loss(prediction, target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = float(np.mean([nn.functional.mse_loss(model(xi.to(device)), yi.to(device)).item()
                                        for xi, yi in zip(x[train_end:test_start].split(args.batch_size),
                                                          y[train_end:test_start].split(args.batch_size))]))
        if validation < best_loss:
            best_loss = validation
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_pred = np.concatenate([model(chunk.to(device)).cpu().numpy()[:, 0]
            for chunk in x[train_end:test_start].split(args.batch_size)])
        raw_output = np.concatenate([model(chunk.to(device)).cpu().numpy()[:, 0]
            for chunk in x[test_start:].split(args.batch_size)])
    val_target, val_base = high[train_end:test_start], low[train_end:test_start]
    blends = np.linspace(0, 1, 11)
    blend = float(min(blends, key=lambda a: np.mean((val_target - (val_base + a * (validation_pred - val_base))) ** 2)))
    truth, baseline = high[test_start:], low[test_start:]
    output = baseline + blend * (raw_output - baseline)
    rows = []
    for center, target, interp, restored in zip(centers[test_start:], truth, baseline, output):
        for name, pred in (("bilinear", interp), ("cnn", restored)):
            rows.append(dict(center_bp=center, method=name,
                psnr=_psnr(target, pred),
                ssim=float(structural_similarity(target, pred, data_range=10))))
    scores = pd.DataFrame(rows)
    scores.to_csv(out / "reconstruction_metrics.csv", index=False)
    scores.groupby("method")[["psnr", "ssim"]].agg(["mean", "std"]).to_csv(out / "comparison.csv")
    means = scores.groupby("method")[["psnr", "ssim"]].mean().reindex(["bilinear", "cnn"])
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), dpi=150)
    for ax, metric in zip(axes, ("psnr", "ssim")):
        ax.bar(means.index, means[metric], color=["gray", "teal"])
        ax.set(ylabel=metric.upper(), title=f"Held-out {metric.upper()}")
    fig.tight_layout(); fig.savefig(out / "metric_comparison.png"); plt.close(fig)
    if args.annotations:
        ann = annotations(args.annotations)
        held_out = ann[annotation_chrom_mask(ann, chrom) & (ann.center >= min(centers[test_start:]))]
        known_rows = []
        plotted_types = set()
        with open_cool(args.cool) as c:
            for _, structure in held_out.iterrows():
                raw_structure = window(c, chrom, int(structure.center), patch_bp, args.patch_bins)
                interp, target = prepare_sr_pair(raw_structure, args.factor)
                with torch.no_grad():
                    raw_restored = model(torch.from_numpy(interp[None, None]).to(device)).cpu().numpy()[0, 0]
                    restored = interp + blend * (raw_restored - interp)
                for method, pred in (("bilinear", interp), ("cnn", restored)):
                    known_rows.append(dict(id=structure.id, type=structure.type, center_bp=structure.center,
                        method=method, psnr=_psnr(target, pred),
                        ssim=float(structural_similarity(target, pred, data_range=10))))
                if structure.type not in plotted_types:
                    plotted_types.add(structure.type)
                    fig, axes = plt.subplots(1, 3, figsize=(10, 3), dpi=150)
                    for ax, mat, title in zip(axes, (interp, restored, target),
                                              ("Bilinear", "CNN", "Micro-C truth")):
                        ax.imshow(mat, origin="lower", cmap="magma", vmin=-2, vmax=4)
                        ax.set_title(title)
                    fig.suptitle(f"{structure.id}: {structure.type}")
                    fig.tight_layout(); fig.savefig(out / f"known_{structure.type}.png"); plt.close(fig)
        pd.DataFrame(known_rows, columns=["id", "type", "center_bp", "method", "psnr", "ssim"]).to_csv(
            out / "known_structure_recovery.csv", index=False)
    torch.save({"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "factor": args.factor,
                "patch_bins": args.patch_bins, "seed": args.seed, "blend": blend}, out / "super_resolution.pt")
    np.savez_compressed(out / "reconstructed_test_patches.npz", center_bp=np.array(centers[test_start:]),
                        truth=truth, bilinear=baseline, cnn=output)
    for i in range(min(args.examples, len(truth))):
        fig, axes = plt.subplots(1, 3, figsize=(10, 3), dpi=150)
        for ax, mat, name in zip(axes, (baseline[i], output[i], truth[i]), ("Bilinear", "CNN", "Micro-C truth")):
            ax.imshow(mat, origin="lower", cmap="magma", vmin=-2, vmax=4)
            ax.set_title(name)
        fig.suptitle(f"{chrom}:{centers[test_start+i]} bp")
        fig.tight_layout(); fig.savefig(out / f"example_{i:02d}.png"); plt.close(fig)
    (out / "summary.json").write_text(json.dumps({"train_patches": train_end,
        "validation_patches": test_start - train_end, "test_patches": len(truth),
        "validation_blend": blend, "best_validation_mse": best_loss,
        "comparison": scores.groupby("method")[["psnr", "ssim"]].mean().to_dict(),
        "note": "Low-resolution input is simulated by bin aggregation, not independent Hi-C."}, indent=2), encoding="utf-8")


def _coarsen(c, chrom: str, target_bp: int) -> np.ndarray:
    if target_bp % c.binsize:
        raise ValueError("target bin size must be a multiple of .cool bin size")
    start, stop = map(int, c.extent(chrom))
    factor = target_bp // c.binsize
    n = int(np.ceil((stop - start) / factor))
    result = np.zeros((n, n), np.float64)
    pixels = c.pixels(join=False)
    for offset in range(0, int(c.info["nnz"]), 500000):
        chunk = pixels[offset:offset + 500000]
        left = chunk.bin1_id.to_numpy(); right = chunk.bin2_id.to_numpy()
        keep = (left >= start) & (left < stop) & (right >= start) & (right < stop)
        if not np.any(keep): continue
        a = (left[keep] - start) // factor; b = (right[keep] - start) // factor
        counts = chunk["count"].to_numpy()[keep]
        np.add.at(result, (a, b), counts)
        off = a != b
        np.add.at(result, (b[off], a[off]), counts[off])
    return result


def _contact_target(raw: np.ndarray) -> np.ndarray:
    x = np.log1p(raw)
    n = len(x)
    for offset in range(n):
        values = np.diag(x, k=offset)
        expected = np.median(values) if len(values) else 0
        i = np.arange(n - offset)
        x[i, i + offset] -= expected
        if offset: x[i + offset, i] -= expected
    x = np.maximum(x, 0)
    scale = np.quantile(x[x > 0], .95) if np.any(x > 0) else 1
    return np.clip(x / max(scale, 1e-8), 0, 1)


def _mds(distance: np.ndarray) -> np.ndarray:
    from scipy.linalg import eigh
    n = len(distance)
    sq = distance ** 2
    gram = -.5 * (sq - sq.mean(axis=0)[None, :] - sq.mean(axis=1)[:, None] + sq.mean())
    values, vectors = eigh(gram, subset_by_index=[n - 3, n - 1])
    return vectors[:, ::-1] * np.sqrt(np.maximum(values[::-1], 0))


class GenomeMLP(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, rows): return self.net(rows)


def _consistency(coords: np.ndarray, contact: np.ndarray) -> float:
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(coords))
    mask = np.triu_indices(len(coords), k=2)
    valid = contact[mask] > 0
    return float(spearmanr(contact[mask][valid], -distances[mask][valid]).statistic)


def reconstruct3d(args) -> None:
    seed_all(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open_cool(args.cool) as c:
        chrom = args.chrom or c.chromnames[0]
        raw = _coarsen(c, chrom, args.bin_bp)
    contact = _contact_target(raw)
    n = len(contact)
    distance_target = 1 / (contact + .05)
    distance_target = np.clip(distance_target, 0, 20)
    mds = _mds(distance_target)
    inputs = torch.tensor(contact, dtype=torch.float32)
    model = GenomeMLP(n)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # MDS is retained as a baseline. By default the neural model starts from
    # its own random initialization so the baseline comparison is independent.
    mds_reference = (mds - mds.mean(axis=0)) / max(float(np.std(mds)), 1e-6)
    mds_reference = torch.tensor(mds_reference, dtype=torch.float32)
    for _ in range(args.pretrain_epochs):
        fitted = model(inputs)
        loss = nn.functional.mse_loss(fitted, mds_reference)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    pair_i, pair_j = np.triu_indices(n, k=2)
    strongest = contact[pair_i, pair_j] > np.quantile(contact[pair_i, pair_j], .8)
    pair_i, pair_j = pair_i[strongest], pair_j[strongest]
    if not len(pair_i): raise ValueError("No positive contact edges")
    goal = torch.tensor(distance_target[pair_i, pair_j], dtype=torch.float32)
    scale = float(np.median(distance_target[pair_i, pair_j]))
    goal = goal / max(scale, 1e-8)
    for _ in range(args.epochs):
        model.train(); xyz = model(inputs)
        take = torch.randint(len(pair_i), (min(args.pairs_per_epoch, len(pair_i)),))
        actual = torch.linalg.vector_norm(xyz[pair_i[take]] - xyz[pair_j[take]], dim=1)
        edge_loss = nn.functional.smooth_l1_loss(actual, goal[take])
        chain = torch.linalg.vector_norm(xyz[1:] - xyz[:-1], dim=1)
        chain_loss = (chain - chain.mean()).square().mean()
        circle_loss = (xyz[0] - xyz[-1]).square().sum()
        anchor_loss = nn.functional.mse_loss(xyz, mds_reference)
        loss = edge_loss + .1 * chain_loss + .01 * circle_loss + args.anchor_weight * anchor_loss
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    coords = model(inputs).detach().numpy()
    summary = {"chrom": chrom, "bins": n, "bin_bp": args.bin_bp,
               "mlp_contact_distance_spearman": _consistency(coords, contact),
               "mds_contact_distance_spearman": _consistency(mds, contact),
               "method": "independently initialized neural coordinate refinement",
               "note": "Coordinates are determined only up to rotation, reflection, and scale; no external geometry truth is supplied."}
    pd.DataFrame({"chrom": chrom, "start": np.arange(n) * args.bin_bp,
                  "end": (np.arange(n) + 1) * args.bin_bp,
                  "x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2]}).to_csv(out / "coordinates.csv", index=False)
    from scipy.spatial.distance import pdist, squareform
    distance_map = squareform(pdist(coords))
    recovered_contact = 1 / (distance_map + .05)
    off_diagonal = np.triu_indices(n, k=2)
    display_scale = max(float(np.quantile(recovered_contact[off_diagonal], .99)), 1e-8)
    recovered_contact = np.clip(recovered_contact / display_scale, 0, 1)
    np.savez_compressed(out / "contact_and_baseline.npz", original=contact,
                        reconstructed_contact=recovered_contact, reconstructed_distance=distance_map,
                        mds_coordinates=mds, mlp_coordinates=coords)
    torch.save({"model_state_dict": model.state_dict(), "bin_bp": args.bin_bp, "seed": args.seed}, out / "genome_3d.pt")
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    for ax, mat, title in zip(axes, (contact, recovered_contact, distance_map),
                              ("Observed normalized contact", "Contact from 3D coordinates", "3D distance")):
        vmax = float(np.quantile(mat[off_diagonal], .99))
        ax.imshow(mat, origin="lower", cmap="magma", vmin=0, vmax=max(vmax, 1e-8))
        ax.set_title(title, fontsize=9)
    fig.suptitle("Each panel uses its own 99th-percentile display scale", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "contact_reconstruction.png"); plt.close(fig)
    fig = plt.figure(figsize=(7, 6), dpi=150); ax = fig.add_subplot(111, projection="3d")
    ax.plot(*coords.T, lw=.5, color="gray", alpha=.7)
    ax.scatter(*coords.T, c=np.arange(n), cmap="viridis", s=3)
    for label, idx in (("origin", 0), ("quarter", n // 4), ("half", n // 2), ("three-quarter", 3 * n // 4)):
        ax.text(*coords[idx], label, fontsize=7)
    if args.annotations:
        known = annotations(args.annotations)
        for _, row in known[known.type == "CHID"].iloc[::5].iterrows():
            idx = min(n - 1, max(0, int(row.center) // args.bin_bp))
            ax.scatter(*coords[idx], color="darkorange", s=12)
            ax.text(*coords[idx], str(row.id), fontsize=5)
    ax.set_title(f"{chrom}: learned chromosome geometry ({args.bin_bp} bp bins)")
    fig.tight_layout(); fig.savefig(out / "chromosome_3d.png"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="task", required=True)
    sr = sub.add_parser("superresolve")
    sr.add_argument("--cool", type=Path, required=True); sr.add_argument("--out", type=Path, required=True)
    sr.add_argument("--annotations", type=Path)
    sr.add_argument("--chrom"); sr.add_argument("--patch-bins", type=int, default=64)
    sr.add_argument("--factor", type=int, default=4); sr.add_argument("--max-patches", type=int, default=1200)
    sr.add_argument("--epochs", type=int, default=12); sr.add_argument("--batch-size", type=int, default=32)
    sr.add_argument("--examples", type=int, default=8); sr.add_argument("--seed", type=int, default=20260921)
    d3 = sub.add_parser("reconstruct3d")
    d3.add_argument("--cool", type=Path, required=True); d3.add_argument("--out", type=Path, required=True)
    d3.add_argument("--annotations", type=Path)
    d3.add_argument("--chrom"); d3.add_argument("--bin-bp", type=int, default=5000)
    d3.add_argument("--epochs", type=int, default=200); d3.add_argument("--pairs-per-epoch", type=int, default=20000)
    d3.add_argument("--pretrain-epochs", type=int, default=0,
                    help="optional MDS pretraining; 0 keeps the neural baseline independent")
    d3.add_argument("--anchor-weight", type=float, default=.1,
                    help="weak MDS regularization; set 0 for a fully independent model")
    d3.add_argument("--lr", type=float, default=1e-3); d3.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()
    {"superresolve": superresolve, "reconstruct3d": reconstruct3d}[args.task](args)


if __name__ == "__main__": main()
