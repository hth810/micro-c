"""Compare rep1-trained 3D coordinates with independent WT rep2 contacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from microc.data import open_cool
from microc.deep import _coarsen, _contact_target


def separation_matched_auc(contact: np.ndarray, distances: dict[str, np.ndarray],
                           seed: int = 20260928) -> dict:
    """Compare positive and clipped-zero residuals at identical genomic offsets.

    Each offset contributes the same number of positive and zero pairs. Thus a
    predictor using only linear genomic separation has AUC 0.5 by design.
    Sampling is deterministic and uses no model score or labels from another
    replicate when selecting how many pairs to retain at an offset.
    """
    n = len(contact)
    if contact.shape != (n, n) or any(value.shape != (n, n) for value in distances.values()):
        raise ValueError("Contact and distance maps must be equally sized squares")
    rng = np.random.default_rng(seed)
    selected_i: list[np.ndarray] = []
    selected_j: list[np.ndarray] = []
    selected_y: list[np.ndarray] = []
    offsets = []
    for offset in range(2, n):
        signal = np.diag(contact, k=offset)
        positive = np.flatnonzero(signal > 0)
        zero = np.flatnonzero(signal == 0)
        count = min(len(positive), len(zero))
        if not count:
            continue
        chosen = np.concatenate((rng.choice(positive, count, replace=False),
                                 rng.choice(zero, count, replace=False)))
        selected_i.append(chosen)
        selected_j.append(chosen + offset)
        selected_y.append((signal[chosen] > 0).astype(np.int8))
        offsets.extend([offset] * len(chosen))
    if not selected_i:
        return {"matched_pairs_per_class": 0, "eligible_genomic_offsets": 0,
                "selection_seed": seed,
                "not_evaluable": "No genomic offset contains both positive and zero residuals"}
    i, j = np.concatenate(selected_i), np.concatenate(selected_j)
    labels, offsets = np.concatenate(selected_y), np.asarray(offsets)
    ranges = {"2_to_10_bins": (2, 10), "11_to_50_bins": (11, 50),
              "51_to_200_bins": (51, 200), "over_200_bins": (201, n)}
    result = {"matched_pairs_per_class": int(labels.sum()),
              "eligible_genomic_offsets": len(selected_i),
              "selection_seed": seed,
              "definition": "positive vs clipped-zero distance-normalized residual, exact genomic-offset matched"}
    for name, matrix in distances.items():
        scores = -matrix[i, j]
        result[name] = float(roc_auc_score(labels, scores))
    result["offset_ranges"] = {}
    for name, (lo, hi) in ranges.items():
        take = (offsets >= lo) & (offsets <= hi)
        if not take.any():
            continue
        block = {"matched_pairs_per_class": int(labels[take].sum())}
        for method, matrix in distances.items():
            block[method] = float(roc_auc_score(labels[take], -matrix[i[take], j[take]]))
        result["offset_ranges"][name] = block
    return result


def stratified_correlations(contact: np.ndarray, coordinates: dict[str, np.ndarray]) -> dict:
    i, j = np.triu_indices(len(contact), k=2)
    signal = contact[i, j]
    cutoff = np.quantile(signal, .8)
    masks = {"all_positive": signal > 0,
             "top_20_percent": signal > cutoff,
             "weaker_positive": (signal > 0) & (signal <= cutoff)}
    full_distances = {method: squareform(pdist(xyz)) for method, xyz in coordinates.items()}
    distances = {method: matrix[i, j]
                 for method, matrix in full_distances.items()}
    result = {}
    for name, mask in masks.items():
        result[name] = {"pairs": int(mask.sum())}
        if mask.sum() < 3:
            raise ValueError(f"Insufficient pairs for {name}")
        for method, values in distances.items():
            result[name][method] = float(spearmanr(signal[mask], -values[mask]).statistic)
    positive = signal > 0
    result["positive_vs_zero_distance_auc"] = {
        "positive_pairs": int(positive.sum()), "zero_pairs": int((~positive).sum()),
        **{method: float(roc_auc_score(positive, -values))
           for method, values in distances.items()}}
    result["exact_separation_matched_auc"] = separation_matched_auc(contact, full_distances)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rep2", type=Path, required=True)
    ap.add_argument("--original", type=Path, default=Path("outputs/task6_independent/contact_and_baseline.npz"))
    ap.add_argument("--all-positive", type=Path,
                    default=Path("outputs/task6_all_positive/contact_and_baseline.npz"))
    ap.add_argument("--out", type=Path, default=Path("outputs/task6_all_positive/replicate_validation.json"))
    args = ap.parse_args()
    old = np.load(args.original)
    improved = np.load(args.all_positive)
    rep1 = improved["original"]
    if rep1.shape != old["original"].shape or not np.allclose(rep1, old["original"]):
        raise ValueError("The two coordinate models must use the same rep1 contact target")
    coordinates = {"original_top20_mlp": old["mlp_coordinates"],
                   "all_positive_mlp": improved["mlp_coordinates"],
                   "rep1_mds_baseline": improved["mds_coordinates"]}
    theta = np.linspace(0, 2 * np.pi, len(rep1), endpoint=False)
    coordinates["genome_circle_baseline"] = np.column_stack(
        (np.cos(theta), np.sin(theta), np.zeros(len(theta))))
    if any(xyz.shape != (len(rep1), 3) for xyz in coordinates.values()):
        raise ValueError("Coordinate and contact dimensions differ")
    with open_cool(args.rep2) as cooler:
        rep2 = _contact_target(_coarsen(cooler, cooler.chromnames[0], 5000))
    if rep1.shape != rep2.shape:
        raise ValueError("Replicate matrices have different bin counts")
    report = {"rep1_in_sample": stratified_correlations(rep1, coordinates),
              "rep2_cross_replicate": stratified_correlations(rep2, coordinates),
              "note": "All coordinates were fitted to rep1 only; rep2 was not used to fit or choose the all-positive model. "
                      "Positive/zero labels refer to distance-normalized residuals, not raw observed read counts. "
                      "Exact-offset matching checks whether AUC survives control for genomic separation. "
                      "This tests contact consistency across biological replicates, not agreement with imaged 3D truth."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
