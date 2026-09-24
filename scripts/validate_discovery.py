"""Independent spatial and shifted-locus checks for Task 2 discovery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from microc.analysis import _features
from microc.data import annotations, chrom_name, normalize, open_cool, upper_correlation, window


def shape_features(mat: np.ndarray) -> np.ndarray:
    upper = mat[np.triu_indices(len(mat), k=2)]
    return np.r_[_features(mat), np.quantile(upper, [.05, .25, .5, .75, .95, .99]),
                 np.mean(upper > 1), np.mean(upper < -1)].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rep1", type=Path, required=True)
    parser.add_argument("--rep2", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--candidate-file", type=Path,
                        help="optional candidate table; defaults to discovery/new_candidates.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--window-bp", type=int, default=5120)
    parser.add_argument("--null-shifts", type=int, default=100)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    known = annotations(args.annotations)
    scan = pd.read_csv(args.discovery / "all_windows.csv")
    novel = pd.read_csv(args.candidate_file or args.discovery / "new_candidates.csv")
    if novel.empty:
        raise ValueError("No proposed candidates to validate")
    with open_cool(args.rep1) as c1, open_cool(args.rep2) as c2:
        chrom = chrom_name(c1, str(known.iloc[0].chrom))
        if c1.binsize != c2.binsize or tuple(c1.chromnames) != tuple(c2.chromnames):
            raise ValueError("Replicate matrices must have matching bins")
        # Background windows do not overlap a supplied annotation and are at
        # least one full window away from its nearest boundary.
        background = scan[(~scan.is_known) & (scan.known_distance_bp >= args.window_bp)]
        background = background.sample(n=min(len(background), len(known)), random_state=args.seed)
        centers = np.r_[known.center.to_numpy(dtype=int), background.center.to_numpy(dtype=int)]
        labels = np.r_[np.ones(len(known), dtype=int), np.zeros(len(background), dtype=int)]
        groups = centers // 100_000
        features1, features2, density1, density2 = [], [], [], []
        for center in centers:
            raw1 = window(c1, chrom, int(center), args.window_bp, 64)
            raw2 = window(c2, chrom, int(center), args.window_bp, 64)
            features1.append(shape_features(normalize(raw1)))
            features2.append(shape_features(normalize(raw2)))
            density1.append(float(np.log1p(raw1).mean()))
            density2.append(float(np.log1p(raw2).mean()))
        x1, x2 = np.stack(features1), np.stack(features2)
        density_train, density_test = np.asarray(density1)[:, None], np.asarray(density2)[:, None]
        folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
        models = {
            "shape_logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(
                max_iter=2000, class_weight="balanced")),
            "shape_forest": lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                class_weight="balanced", random_state=args.seed, n_jobs=-1),
            "density_logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(
                max_iter=2000, class_weight="balanced")),
        }
        predictions = {name: np.full(len(labels), np.nan) for name in models}
        for train, test in folds.split(x1, labels, groups):
            for name, build in models.items():
                model = build()
                training = density_train if name == "density_logistic" else x1
                testing = density_test if name == "density_logistic" else x2
                model.fit(training[train], labels[train])
                predictions[name][test] = model.predict_proba(testing[test])[:, 1]
        if any(np.isnan(p).any() for p in predictions.values()):
            raise RuntimeError("Grouped cross-validation did not score every locus")
        summary = {"n_known": int(labels.sum()), "n_background": int((labels == 0).sum()),
                   "evaluation": "train on rep1; spatially held-out loci scored on rep2",
                   "auc": {name: float(roc_auc_score(labels, value)) for name, value in predictions.items()}}
        unique_groups = np.unique(groups)
        auc_intervals = {}
        for name, values in predictions.items():
            sampled_auc = []
            for _ in range(2000):
                sampled_groups = rng.choice(unique_groups, len(unique_groups), replace=True)
                sample = np.concatenate([np.flatnonzero(groups == group) for group in sampled_groups])
                if len(np.unique(labels[sample])) == 2:
                    sampled_auc.append(roc_auc_score(labels[sample], values[sample]))
            auc_intervals[name] = np.quantile(sampled_auc, [.025, .975]).tolist()
        summary["auc_spatial_group_bootstrap_95ci"] = auc_intervals
        pd.DataFrame({"center_bp": centers, "known": labels, "spatial_group": groups,
            **predictions}).to_csv(args.out / "heldout_predictions.csv", index=False)

        length = int(c1.chromsizes[chrom])
        null_rows = []
        for _, row in novel.iterrows():
            center = int(row.center)
            reference = normalize(window(c1, chrom, center, args.window_bp, 64))
            observed = upper_correlation(reference, normalize(window(c2, chrom, center, args.window_bp, 64)))
            null = []
            local_null = []
            while len(null) < args.null_shifts:
                shifted = int(rng.integers(args.window_bp // 2, length - args.window_bp // 2))
                if abs(shifted - center) < 2 * args.window_bp:
                    continue
                null.append(upper_correlation(reference, normalize(
                    window(c2, chrom, shifted, args.window_bp, 64))))
            while len(local_null) < args.null_shifts:
                offset = int(rng.integers(args.window_bp // 8, args.window_bp // 2 + 1))
                shifted = center + offset * int(rng.choice([-1, 1]))
                if shifted < args.window_bp // 2 or shifted >= length - args.window_bp // 2:
                    continue
                local_null.append(upper_correlation(reference, normalize(
                    window(c2, chrom, shifted, args.window_bp, 64))))
            null = np.asarray(null)
            local_null = np.asarray(local_null)
            p = (1 + np.sum(null >= observed)) / (len(null) + 1)
            local_p = (1 + np.sum(local_null >= observed)) / (len(local_null) + 1)
            null_rows.append({"chrom": chrom, "center_bp": center, "observed_r": observed,
                              "shifted_null_mean": float(null.mean()),
                              "shifted_null_p95": float(np.quantile(null, .95)),
                              "empirical_p": float(p),
                              "local_shift_null_p95": float(np.quantile(local_null, .95)),
                              "local_empirical_p": float(local_p)})
        null_frame = pd.DataFrame(null_rows).sort_values("empirical_p")
        ranks = np.arange(1, len(null_frame) + 1)
        null_frame["bh_q"] = np.minimum.accumulate(
            np.minimum(1, null_frame.empirical_p.to_numpy()[::-1] * len(null_frame) /
                       ranks[::-1]))[::-1]
        local_order = np.argsort(null_frame.local_empirical_p.to_numpy())
        local_p = null_frame.local_empirical_p.to_numpy()[local_order]
        local_q = np.minimum.accumulate(np.minimum(1, local_p[::-1] * len(local_p) /
                                                   np.arange(len(local_p), 0, -1)))[::-1]
        null_frame["local_bh_q"] = np.empty(len(local_q))
        null_frame.iloc[local_order, null_frame.columns.get_loc("local_bh_q")] = local_q
        null_frame.to_csv(args.out / "candidate_shift_null.csv", index=False)
        summary["candidates_shift_null_q_below_005"] = int((null_frame.bh_q < .05).sum())
        summary["candidates_local_shift_q_below_005"] = int((null_frame.local_bh_q < .05).sum())
        (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
