"""Predict known annotated intervals with the saved contact-shape/length model."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from microc.data import annotations, chrom_name, open_cool
from scripts.validate_discovery import shape_features
from task1.evaluate_nested_threshold import apply_threshold
from task1.evaluate_interval_boundaries import interval_features
from task1.run_task1 import extract_window


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rep1", type=Path, required=True)
    ap.add_argument("--rep2", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True,
                    help="known interval CSV with chrom/start/end; not arbitrary unlabeled windows")
    ap.add_argument("--model", type=Path, default=Path("results/task1_handcrafted_final.joblib"))
    ap.add_argument("--out", type=Path, default=Path("outputs/task1_handcrafted_predictions.csv"))
    args = ap.parse_args()
    bundle = joblib.load(args.model)
    ann = annotations(args.annotations)
    feature_set = bundle.get("feature_set", "short_shape_span")
    if feature_set not in ("short_shape_span", "short_long_boundary_span"):
        raise ValueError(f"Unexpected feature set: {feature_set}")
    if feature_set == "short_shape_span" and (bundle.get("window_bp") != 5120 or
                                               bundle.get("contact_replicates") != 2):
        raise ValueError("Unexpected short-context model specification")
    if feature_set == "short_long_boundary_span" and (bundle.get("window_bins") != 128 or
                                                        bundle.get("short_aggregate") != 4 or
                                                        bundle.get("long_aggregate") != 16):
        raise ValueError("Unexpected two-scale model specification")
    features = []
    with ExitStack() as stack:
        c1, c2 = (stack.enter_context(open_cool(path)) for path in (args.rep1, args.rep2))
        if c1.binsize != c2.binsize or tuple(c1.chromnames) != tuple(c2.chromnames):
            raise ValueError("Replicates must share genome and bin resolution")
        if len(c1.chromnames) != 1:
            raise ValueError("Saved model was trained on a single-chromosome cooler")
        for _, row in ann.iterrows():
            chrom_name(c1, str(row.chrom))
            short = np.mean([shape_features(extract_window(
                cool, int(row.center), 128, 4)) for cool in (c1, c2)], axis=0)
            if feature_set == "short_long_boundary_span":
                long_matrices = [extract_window(cool, int(row.center), 128, 16) for cool in (c1, c2)]
                long = np.mean([shape_features(mat) for mat in long_matrices], axis=0)
                boundary = np.mean([interval_features(mat, int(row.start), int(row.end),
                                                      int(row.center)) for mat in long_matrices], axis=0)
                features.append(np.r_[short, long, boundary, np.log1p(row.length_bp)])
            else:
                features.append(np.r_[short, np.log1p(row.length_bp)])
    values = np.stack(features)
    if values.shape[1] != bundle["feature_count"]:
        raise ValueError("Feature count differs from saved model")
    probability = bundle["model"].predict_proba(values)
    classes = tuple(bundle["classes"])
    result = ann[["id", "chrom", "start", "end", "center", "length_bp"]].copy()
    for index, name in enumerate(classes):
        result[f"probability_{name}"] = probability[:, index]
    result["default_prediction"] = [classes[i] for i in probability.argmax(axis=1)]
    if "chid_cv_f1_threshold" in bundle:
        result["cv_f1_threshold_prediction"] = [classes[i] for i in apply_threshold(
            probability, bundle["chid_cv_f1_threshold"])]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"Saved {len(result)} known-interval predictions to {args.out}")


if __name__ == "__main__":
    main()
