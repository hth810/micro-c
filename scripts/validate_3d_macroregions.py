"""Exploratory genomic-offset-controlled check of learned 3D macroregion geometry.

Published E. coli macrodomains are *approximate genomic labels*, not imaged
3D ground truth. This script asks only whether same-label bins lie closer in
the rep1-trained embedding than different-label bins at the same circular
genomic separation, and whether that alignment survives circular label shifts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import roc_auc_score

from scripts.annotate_3d_domains import REGIONS, SOURCE


def core_mask(centers: np.ndarray, trim_bp: int, genome_bp: int) -> np.ndarray:
    boundaries = np.asarray([item[1] for item in REGIONS], dtype=int)
    distances = np.abs(centers[:, None] - boundaries[None, :])
    circular = np.minimum(distances, genome_bp - distances)
    return circular.min(axis=1) >= trim_bp


def offset_matched_auc(labels: np.ndarray, core: np.ndarray, distance: np.ndarray,
                       seed: int = 20260928) -> dict:
    n = len(labels)
    if distance.shape != (n, n) or core.shape != (n,):
        raise ValueError("labels, core mask, and distance matrix disagree")
    i, j = np.triu_indices(n, k=2)
    sep = np.minimum(j - i, n - (j - i))
    eligible = core[i] & core[j]
    same = labels[i] == labels[j]
    rng = np.random.default_rng(seed)
    chosen = []
    offsets = 0
    for k in range(2, n // 2 + 1):
        within = np.flatnonzero(eligible & (sep == k) & same)
        between = np.flatnonzero(eligible & (sep == k) & ~same)
        count = min(len(within), len(between))
        if count:
            offsets += 1
            chosen.extend((rng.choice(within, count, replace=False),
                           rng.choice(between, count, replace=False)))
    if not chosen:
        return {"not_evaluable": "No circular genomic offset has both same- and different-region pairs"}
    selected = np.concatenate(chosen)
    return {"same_region_vs_different_auc": float(roc_auc_score(same[selected], -distance[i[selected], j[selected]])),
            "matched_pairs_per_class": len(selected) // 2,
            "eligible_circular_offsets": offsets, "selection_seed": seed}


def offset_effect(labels: np.ndarray, core: np.ndarray, pairs: tuple[np.ndarray, np.ndarray],
                  sep: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Weighted mean between-minus-within distance, standardized per model.

    Within each exact circular genomic separation, both categories receive
    equal total weight. This statistic is used for the circular-shift null;
    it never selects labels or thresholds using learned coordinates.
    """
    i, j = pairs
    valid = core[i] & core[j]
    same = (labels[i] == labels[j]) & valid
    different = (labels[i] != labels[j]) & valid
    n = len(labels)
    ns = np.bincount(sep[same], minlength=n)
    nd = np.bincount(sep[different], minlength=n)
    matched = np.minimum(ns, nd)
    matched[:2] = 0
    usable = matched > 0
    if not usable.any():
        raise ValueError("No offsets contain both within- and between-region pairs")
    scores = []
    for values in distances:
        sums_same = np.bincount(sep[same], weights=values[same], minlength=n)
        sums_diff = np.bincount(sep[different], weights=values[different], minlength=n)
        delta = sums_diff[usable] / nd[usable] - sums_same[usable] / ns[usable]
        scale = max(float(np.std(values[valid])), 1e-12)
        scores.append(float(np.average(delta, weights=matched[usable]) / scale))
    return np.asarray(scores)


def validate(bundle: Path, overlay: Path, seed: int, shifts: int) -> dict:
    frame = pd.read_csv(overlay)
    if shifts < 1:
        raise ValueError("At least one circular label shift is required")
    if not {"start", "end", "published_macroregion_approx"}.issubset(frame.columns):
        raise ValueError("Macroregion overlay lacks required columns")
    starts = frame.start.to_numpy()
    if len(starts) < 10 or len(np.unique(np.diff(starts))) != 1 or np.diff(starts)[0] <= 0:
        raise ValueError("Exact genomic-offset matching requires consecutive regular bins")
    labels = frame.published_macroregion_approx.astype(str).to_numpy()
    if not set(labels).issubset({item[0] for item in REGIONS}):
        raise ValueError("Overlay contains an unknown macroregion label")
    centers = ((frame.start.to_numpy() + frame.end.to_numpy()) / 2).astype(int)
    genome_bp = int(frame.end.max())
    with np.load(bundle) as source:
        coordinates = {"all_positive_mlp": source["mlp_coordinates"],
                       "rep1_mds_baseline": source["mds_coordinates"]}
    n = len(labels)
    if any(value.shape != (n, 3) for value in coordinates.values()):
        raise ValueError("Coordinate and macroregion tables differ in length")
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    coordinates["genome_circle_baseline"] = np.column_stack(
        (np.cos(theta), np.sin(theta), np.zeros(n)))
    matrices = {name: squareform(pdist(value)) for name, value in coordinates.items()}
    i, j = np.triu_indices(n, k=2)
    sep = np.minimum(j - i, n - (j - i))
    vector = np.asarray([matrix[i, j] for matrix in matrices.values()])
    rng = np.random.default_rng(seed)
    shift_values = (np.arange(1, n) if shifts >= n - 1 else
                    np.sort(rng.choice(np.arange(1, n), shifts, replace=False)))
    result = {"bins": n, "region_source": SOURCE,
              "region_definition": "approximate published genomic macroregion intervals",
              "label_shift_null": "rotate all six region intervals together around the chromosome; excludes unshifted labels",
              "shift_seed": seed, "shift_count": len(shift_values), "trim_checks": {}}
    for trim in (0, 50_000):
        core = core_mask(centers, trim, genome_bp)
        observed = offset_effect(labels, core, (i, j), sep, vector)
        null = np.asarray([offset_effect(np.roll(labels, shift), np.roll(core, shift),
                                         (i, j), sep, vector) for shift in shift_values])
        methods = {}
        for index, (name, matrix) in enumerate(matrices.items()):
            # For a perfect genomic circle the offset-matched distance effect is
            # mathematically zero. Floating-point noise must not be assigned a p.
            empirical_p = (None if name == "genome_circle_baseline" else float(
                (1 + np.count_nonzero(null[:, index] >= observed[index])) /
                (len(shift_values) + 1)))
            methods[name] = {**offset_matched_auc(labels, core, matrix, seed),
                             "standardized_between_minus_within_distance": float(observed[index]),
                             "circular_shift_empirical_upper_p": empirical_p,
                             "shift_null_mean": float(null[:, index].mean()),
                             "shift_null_p95": float(np.quantile(null[:, index], .95))}
        result["trim_checks"][str(trim)] = {"retained_bins": int(core.sum()), "methods": methods}
    result["caution"] = ("Exploratory external genomic-label alignment, not microscopy truth; "
                          "macroregion boundaries are approximate and published in a different experimental setting. "
                          "Circular-shift p values test label alignment only, not physical 3D accuracy.")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, default=Path("results/task6_all_positive_contact_and_baseline.npz"))
    ap.add_argument("--overlay", type=Path,
                    default=Path("results/task6_domain_overlay_coordinates_with_macroregions.csv"))
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--shifts", type=int, default=500)
    ap.add_argument("--out", type=Path, default=Path("outputs/task6_macroregion_validation.json"))
    args = ap.parse_args()
    report = validate(args.bundle, args.overlay, args.seed, args.shifts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
