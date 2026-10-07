"""Fixed rep1-only circular-chromosome correction with rep2 sensitivity checks.

No seed or rule is chosen using rep2. The old models are evaluated against the
same corrected targets, but the old linear-target results remain in the bundle.
This is a retrospective topology audit, not physical three-dimensional truth.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from microc.circular import circular_contact_target, circular_matched_auc, circular_pairs, cyclic_chain_variance
from microc.data import open_cool
from microc.deep import GenomeMLP, _coarsen, _contact_target, _mds, seed_all
from scripts.annotate_3d_domains import region_for_bp


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20261006, 20261007, 20261008)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def train(contact: np.ndarray, seed: int, epochs: int, pair_count: int):
    seed_all(seed)
    n = len(contact)
    i, j, _ = circular_pairs(n)
    keep = contact[i, j] > 0
    i, j = i[keep], j[keep]
    if not len(i):
        raise ValueError("No positive non-neighbor ring edges")
    inputs = torch.tensor(contact, dtype=torch.float32)
    model = GenomeMLP(n)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    target = np.clip(1 / (contact[i, j] + .05), 0, 20)
    goal = torch.tensor(target / max(float(np.median(target)), 1e-8), dtype=torch.float32)
    history = []
    for epoch in range(epochs):
        model.train()
        xyz = model(inputs)
        take = torch.randint(len(i), (min(pair_count, len(i)),))
        actual = torch.linalg.vector_norm(xyz[i[take]] - xyz[j[take]], dim=1)
        edge_loss = torch.nn.functional.smooth_l1_loss(actual, goal[take])
        loss = edge_loss + .1 * cyclic_chain_variance(xyz)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 25 == 0 or epoch == epochs - 1:
            history.append({"epoch": epoch + 1, "total_loss": float(loss.detach()),
                            "edge_loss": float(edge_loss.detach())})
    model.eval()
    with torch.no_grad():
        coordinates = model(inputs).numpy()
    return model, coordinates, history


def correlation(signal: np.ndarray, score: np.ndarray) -> float | None:
    if len(signal) < 3 or np.std(signal) < 1e-12 or np.std(score) < 1e-12:
        return None
    return float(spearmanr(signal, score).statistic)


def evaluate(contact: np.ndarray, coordinates: dict[str, np.ndarray]) -> dict:
    i, j, offsets = circular_pairs(len(contact))
    signal = contact[i, j]
    matrices = {name: squareform(pdist(xyz)) for name, xyz in coordinates.items()}
    positive = signal > 0
    cutoff = np.quantile(signal, .8)
    seam = (j - i > len(contact) // 2) & (offsets <= 20) & positive
    masks = {"all_positive": positive, "top20": signal > cutoff,
             "weaker_positive": positive & (signal <= cutoff), "origin_crossing_2_to_20": seam}
    return {
        "strata": {name: {"pairs": int(mask.sum()), **{
            method: correlation(signal[mask], -matrix[i[mask], j[mask]])
            for method, matrix in matrices.items()}} for name, mask in masks.items()},
        "exact_circular_offset_matched_auc": circular_matched_auc(contact, matrices),
    }


def figures(old: np.ndarray, corrected: np.ndarray, coordinates: dict,
            comparison: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(old)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=160)
    for ax, mat, label in zip(axes, (old, corrected), ("Legacy linear target", "Circular target")):
        im = ax.imshow(mat[:30, -30:], origin="lower", cmap="magma", vmin=0, vmax=1)
        ax.set(title=label, xlabel="Last 30 genome bins", ylabel="First 30 genome bins")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.75, label="Normalized residual contact")
    fig.suptitle("Origin-crossing contacts: fixed circular-distance correction")
    fig.subplots_adjust(top=.82, wspace=.25, right=.85)
    fig.savefig(out / "origin_contact_audit.png")
    plt.close(fig)
    methods = list(coordinates)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=160)
    for ax, metric in zip(axes, ("all_positive", "origin_crossing_2_to_20")):
        values = [comparison["strata"][metric][name] for name in methods]
        ax.bar(np.arange(len(methods)), values, color=["#999999", "#aaaaaa", "#4d8cb1"] + ["#bb743b"] * 3)
        ax.set_xticks(np.arange(len(methods)), methods, rotation=25, ha="right", fontsize=7)
        ax.set(ylabel="Rep2 contact / negative-distance Spearman", title=metric)
        ax.axhline(0, color="black", linewidth=.8)
    fig.tight_layout()
    fig.savefig(out / "rep2_circular_comparison.png")
    plt.close(fig)
    first = coordinates[f"circular_mlp_seed{SEEDS[0]}"]
    overlay = pd.read_csv(out / "coordinates.csv")
    centers = ((overlay.start.to_numpy() + overlay.end.to_numpy()) / 2).astype(int)
    labels = np.array([region_for_bp(int(bp)) for bp in centers])
    fig = plt.figure(figsize=(8, 6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    closed = np.vstack((first, first[0]))
    ax.plot(*closed.T, color="gray", linewidth=.5, alpha=.5)
    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(*first[mask].T, s=5, label=label)
    for k, label in ((0, "genome origin"), (n - 1, "last partial bin")):
        ax.text(*first[k], label, fontsize=8)
    for label, bp in (("oriC", 3920000), ("dif", 1604000)):
        k = int(np.argmin(np.abs(centers - bp)))
        ax.scatter(*first[k], marker="*", s=40, color="black")
        ax.text(*first[k], label, fontsize=8)
    ax.set_title("Circular-target MLP: fixed first seed, approximate macroregion labels")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "chromosome_circular_3d.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rep1", type=Path, required=True)
    parser.add_argument("--rep2", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/task6_circular_20261006"))
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    if args.epochs != 200:
        raise ValueError("The actual-data sensitivity protocol is fixed at 200 epochs")
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "protocol.json").exists():
        raise ValueError("Use a fresh output directory to preserve the frozen protocol")
    protocol = {
        "experiment": "task6-circular-topology-20261006", "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "run_parent_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "implementation_sha256": sha(Path(__file__)), "module_sha256": sha(ROOT / "microc/circular.py"),
        "dependency_sha256": {name: sha(ROOT / name) for name in (
            "microc/deep.py", "scripts/annotate_3d_domains.py")},
        "python_version": sys.version.split()[0], "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "rep1_sha256": sha(args.rep1), "rep2_sha256": sha(args.rep2),
        "model": "GenomeMLP, rep1 only, all positive ring-offset>=2 edges, no MDS prior",
        "seeds": list(SEEDS), "epochs": 200, "pairs_per_epoch": 20000,
        "selection": "All three seeds reported; fixed first seed for exported checkpoint and plot; same-seed linear-target training uses identical ring pair mask and cyclic chain penalty, isolating target normalization",
        "chain_penalty": "0.1 times variance of all n cyclic neighbor lengths; no separate zero-length wrap closure",
        "normalization": "log1p median at exact circular bin-index separation, clip positive, global p95",
        "evaluation": "Rep2 positive contact correlation, exact ring-offset balanced AUC, origin-crossing stratum",
        "caution": "Retrospective sensitivity; historical rep2 inspection. Final partial bin uses bin-index approximation."
    }
    write_json(args.out / "protocol.json", protocol)
    with open_cool(args.rep1) as cool:
        chrom = cool.chromnames[0]
        genome_bp = int(cool.chromsizes[chrom])
        raw1 = _coarsen(cool, chrom, 5000)
    target1 = circular_contact_target(raw1)
    legacy_target = _contact_target(raw1)
    legacy = np.load(ROOT / "results/task6_all_positive_contact_and_baseline.npz")
    if not np.allclose(legacy_target, legacy["original"]):
        raise ValueError("Legacy rep1 target does not reproduce the frozen source")
    distance = np.clip(1 / (target1 + .05), 0, 20)
    np.fill_diagonal(distance, 0)
    _, paired_linear, linear_history = train(legacy_target, SEEDS[0], 200, 20000)
    coordinates = {"legacy_mlp": legacy["mlp_coordinates"],
                   "matched_linear_mlp": paired_linear,
                   "circular_mds": _mds(distance)}
    histories = {"matched_linear_mlp": linear_history}
    for seed in SEEDS:
        model, xyz, history = train(target1, seed, 200, 20000)
        coordinates[f"circular_mlp_seed{seed}"] = xyz
        histories[str(seed)] = history
        if seed == SEEDS[0]:
            torch.save({"model_state_dict": model.state_dict(), "seed": seed, "bin_bp": 5000,
                        "protocol_sha256": sha(args.out / "protocol.json")}, args.out / "genome_circular.pt")
        print(f"finished fixed circular seed {seed}", flush=True)
    first = coordinates[f"circular_mlp_seed{SEEDS[0]}"]
    n = len(first)
    starts = np.arange(n) * 5000
    ends = np.minimum((np.arange(n) + 1) * 5000, genome_bp)
    pd.DataFrame({"chrom": chrom, "start": starts,
                  "end": np.minimum((np.arange(n) + 1) * 5000, genome_bp),
                  "published_macroregion_approx": [region_for_bp(int(bp)) for bp in (starts + ends) / 2],
                  "x": first[:, 0], "y": first[:, 1], "z": first[:, 2]}).to_csv(
                      args.out / "coordinates.csv", index=False)
    np.savez_compressed(args.out / "coordinate_models.npz", **coordinates)
    # The coordinate archive is frozen before rep2 is opened.
    freeze = {"frozen_utc": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": sha(args.out / "protocol.json"),
              "coordinate_models_sha256": sha(args.out / "coordinate_models.npz"),
              "checkpoint_sha256": sha(args.out / "genome_circular.pt")}
    write_json(args.out / "coordinate_freeze.json", freeze)
    with open_cool(args.rep2) as cool:
        raw2 = _coarsen(cool, chrom, 5000)
    target2 = circular_contact_target(raw2)
    rep1_metrics, rep2_metrics = evaluate(target1, coordinates), evaluate(target2, coordinates)
    write_json(args.out / "training_history.json", histories)
    summary = {"experiment": protocol["experiment"], "protocol_sha256": sha(args.out / "protocol.json"),
               "coordinate_freeze": freeze, "bins": n, "genome_bp": genome_bp,
               "last_bin_bp": genome_bp - (n - 1) * 5000,
               "rep1_in_sample": rep1_metrics, "rep2_cross_replicate": rep2_metrics,
               "seed_distance_spearman": {f"{a}__{b}": float(spearmanr(pdist(coordinates[a]), pdist(coordinates[b])).statistic)
                    for index, a in enumerate(coordinates) for b in list(coordinates)[index + 1:]
                    if a.startswith("circular_mlp") and b.startswith("circular_mlp")},
               "legacy_wrap_neighbor_target": float(legacy_target[0, -1]),
               "corrected_wrap_neighbor_target": float(target1[0, -1]),
               "caution": "Fixed topology sensitivity only. Corrected and legacy metrics use the same circular target here; historical linear-target numbers are a different estimand. No physical 3D truth or newly blind validation is claimed."}
    write_json(args.out / "summary.json", summary)
    figures(legacy_target, target1, coordinates, rep2_metrics, args.out)
    print(json.dumps({"bins": n, "rep2_strata": rep2_metrics["strata"],
                      "rep2_auc": rep2_metrics["exact_circular_offset_matched_auc"]["auc"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
