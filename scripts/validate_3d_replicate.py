"""Compare rep1-trained 3D coordinates with independent WT rep2 contacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from microc.data import open_cool
from microc.deep import _coarsen, _contact_target


def stratified_correlations(contact: np.ndarray, coordinates: dict[str, np.ndarray]) -> dict:
    i, j = np.triu_indices(len(contact), k=2)
    signal = contact[i, j]
    cutoff = np.quantile(signal, .8)
    masks = {"all_positive": signal > 0,
             "top_20_percent": signal > cutoff,
             "weaker_positive": (signal > 0) & (signal <= cutoff)}
    distances = {method: squareform(pdist(xyz))[i, j]
                 for method, xyz in coordinates.items()}
    result = {}
    for name, mask in masks.items():
        result[name] = {"pairs": int(mask.sum())}
        if mask.sum() < 3:
            raise ValueError(f"Insufficient pairs for {name}")
        for method, values in distances.items():
            result[name][method] = float(spearmanr(signal[mask], -values[mask]).statistic)
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
    if any(xyz.shape != (len(rep1), 3) for xyz in coordinates.values()):
        raise ValueError("Coordinate and contact dimensions differ")
    with open_cool(args.rep2) as cooler:
        rep2 = _contact_target(_coarsen(cooler, cooler.chromnames[0], 5000))
    if rep1.shape != rep2.shape:
        raise ValueError("Replicate matrices have different bin counts")
    report = {"rep1_in_sample": stratified_correlations(rep1, coordinates),
              "rep2_cross_replicate": stratified_correlations(rep2, coordinates),
              "note": "All coordinates were fitted to rep1 only; rep2 was not used to fit or choose the all-positive model. "
                      "This tests contact consistency across biological replicates, not agreement with imaged 3D truth."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
