"""Audit random-seed stability of the fixed, MDS-free 3D reconstruction.

All seeds use the published rep1 contact target, the same 200-epoch training
rule, and no rep2 data during fitting or selection. Every seed is reported;
the best rep2 result is never selected as a replacement model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from microc.data import open_cool
from microc.deep import GenomeMLP, _coarsen, _contact_target, seed_all
from scripts.validate_3d_macroregions import core_mask, offset_matched_auc
from scripts.validate_3d_replicate import stratified_correlations


DEFAULT_SEEDS = (20260921, 20260922, 20260923, 20260924, 20260925)


def train_coordinates(contact: np.ndarray, seed: int, epochs: int = 200,
                      pairs_per_epoch: int = 20_000, lr: float = 1e-3) -> np.ndarray:
    """Match the published all-positive, zero-MDS-prior training rule."""
    if contact.ndim != 2 or contact.shape[0] != contact.shape[1]:
        raise ValueError("Contact must be a square matrix")
    if epochs < 1 or pairs_per_epoch < 1:
        raise ValueError("Epochs and pair count must be positive")
    seed_all(seed)
    n = len(contact)
    inputs = torch.as_tensor(contact, dtype=torch.float32)
    model = GenomeMLP(n)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pair_i, pair_j = np.triu_indices(n, k=2)
    selected = contact[pair_i, pair_j] > 0
    pair_i, pair_j = pair_i[selected], pair_j[selected]
    if not len(pair_i):
        raise ValueError("No positive contact edges")
    target = np.clip(1 / (contact + .05), 0, 20)[pair_i, pair_j]
    scale = max(float(np.median(target)), 1e-8)
    goal = torch.as_tensor(target / scale, dtype=torch.float32)
    for _ in range(epochs):
        model.train()
        xyz = model(inputs)
        take = torch.randint(len(pair_i), (min(pairs_per_epoch, len(pair_i)),))
        actual = torch.linalg.vector_norm(xyz[pair_i[take]] - xyz[pair_j[take]], dim=1)
        edge_loss = torch.nn.functional.smooth_l1_loss(actual, goal[take])
        chain = torch.linalg.vector_norm(xyz[1:] - xyz[:-1], dim=1)
        chain_loss = (chain - chain.mean()).square().mean()
        circle_loss = (xyz[0] - xyz[-1]).square().sum()
        loss = edge_loss + .1 * chain_loss + .01 * circle_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        return model(inputs).cpu().numpy()


def macroregion_auc(coordinates: np.ndarray, overlay: pd.DataFrame) -> dict[str, float]:
    labels = overlay.published_macroregion_approx.astype(str).to_numpy()
    centers = ((overlay.start.to_numpy() + overlay.end.to_numpy()) / 2).astype(int)
    genome_bp = int(overlay.end.max())
    distance = squareform(pdist(coordinates))
    return {
        str(trim): offset_matched_auc(labels, core_mask(centers, trim, genome_bp), distance)[
            "same_region_vs_different_auc"]
        for trim in (0, 50_000)
    }


def distance_agreement(coordinates: dict[str, np.ndarray]) -> dict[str, float]:
    vectors = {key: pdist(value) for key, value in coordinates.items()}
    return {
        f"{left}__{right}": float(spearmanr(vectors[left], vectors[right]).statistic)
        for index, left in enumerate(vectors)
        for right in list(vectors)[index + 1:]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path,
                        default=Path("results/task6_all_positive_contact_and_baseline.npz"))
    parser.add_argument("--rep2", type=Path, required=True)
    parser.add_argument("--overlay", type=Path,
                        default=Path("results/task6_domain_overlay_coordinates_with_macroregions.csv"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--pairs-per-epoch", type=int, default=20_000)
    parser.add_argument("--out", type=Path, default=Path("outputs/task6_seed_stability"))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be distinct")
    with np.load(args.bundle) as source:
        rep1 = source["original"]
        published = source["mlp_coordinates"]
        mds = source["mds_coordinates"]
    overlay = pd.read_csv(args.overlay)
    if len(overlay) != len(rep1) or published.shape != (len(rep1), 3):
        raise ValueError("Bundle and macroregion overlay dimensions differ")
    coordinates = {
        str(seed): train_coordinates(rep1, seed, args.epochs, args.pairs_per_epoch)
        for seed in args.seeds
    }
    with open_cool(args.rep2) as cooler:
        rep2 = _contact_target(_coarsen(cooler, cooler.chromnames[0], 5000))
    if rep1.shape != rep2.shape:
        raise ValueError("Rep1 and rep2 have different 5-kb bin counts")
    metrics = {}
    for seed, xyz in coordinates.items():
        test = stratified_correlations(rep2, {"mlp": xyz})
        metrics[seed] = {
            "rep2_all_positive_spearman": test["all_positive"]["mlp"],
            "rep2_top20_spearman": test["top_20_percent"]["mlp"],
            "rep2_weaker_positive_spearman": test["weaker_positive"]["mlp"],
            "rep2_exact_offset_positive_vs_zero_auc": test["exact_separation_matched_auc"]["mlp"],
            "macroregion_same_vs_different_auc": macroregion_auc(xyz, overlay),
        }
    published_metrics = stratified_correlations(rep2, {"published": published, "mds": mds})
    geometry_agreement = distance_agreement(coordinates)
    published_agreement = {
        seed: float(spearmanr(pdist(published), pdist(xyz)).statistic)
        for seed, xyz in coordinates.items()
    }
    report = {
        "protocol": "Five predeclared seeds, identical rep1-only MDS-free all-positive training; no seed selected by rep2 or macroregion labels",
        "seeds": args.seeds,
        "epochs": args.epochs,
        "pairs_per_epoch": args.pairs_per_epoch,
        "rep2_metrics_by_seed": metrics,
        "pairwise_seed_distance_spearman": geometry_agreement,
        "published_coordinate_distance_spearman_by_seed": published_agreement,
        "published_rep2_baselines": {
            "mlp_all_positive_spearman": published_metrics["all_positive"]["published"],
            "mds_all_positive_spearman": published_metrics["all_positive"]["mds"],
            "mlp_exact_offset_auc": published_metrics["exact_separation_matched_auc"]["published"],
            "mds_exact_offset_auc": published_metrics["exact_separation_matched_auc"]["mds"],
        },
        "caution": "Correlations and AUCs test contact consistency and approximate genomic labels, not true imaged 3D coordinates. Seeds are a robustness audit, not an opportunity to choose the best rep2 score.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(args.out / "coordinates.npz", **{f"seed_{k}": v for k, v in coordinates.items()})
    print(json.dumps({"seeds": args.seeds, "rep2_metrics_by_seed": metrics}, indent=2))


if __name__ == "__main__":
    main()
