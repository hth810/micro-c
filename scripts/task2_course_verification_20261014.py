"""Retrospective, fixed-candidate Task 2 evidence audit on existing COOL counts.

This follow-up does not lower the historical novelty thresholds, select new
candidates, or treat a missing match as evidence of novelty. The complete
unannotated scan is score-agnostic but overlapping; only a spacing-filtered
subset is used for descriptive matched comparisons. No p-values or FDR.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from threadpoolctl import threadpool_limits

from microc.data import annotations, area_resample, normalize, open_cool, window
from scripts.task2_candidate_artifact_followup import (
    COVARIATES, native_qc, paired_r, qc_covariates,
)
from scripts.task2_residual_motif_screen_20261006 import (
    METRICS, residual_vectors, sha256, write_json,
)

SEED = 20261014
SPACING = 10240
SHAPE_FEATURES = (
    "near_diagonal_positive_mass", "midrange_positive_mass", "far_positive_mass",
    "positive_pixel_fraction", "top_one_percent_positive_mass",
    "positive_mass_centroid_x", "positive_mass_centroid_y",
    "positive_mass_spread", "row_contact_cv", "row_high_contact_fraction",
)
SCALE_BP = (3840, 5120, 7680)
SHIFTS_BP = (-320, -160, 0, 160, 320)
THIN_SEEDS = (2026101401, 2026101402, 2026101403, 2026101404, 2026101405)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def score_free_independent_indices(centers, eligible, spacing=SPACING):
    """Chromosome-order thinning: selection never uses an outcome/shape score."""
    centers = np.asarray(centers, int)
    eligible = np.asarray(eligible, bool)
    if centers.ndim != 1 or eligible.shape != centers.shape or len(set(centers)) != len(centers):
        raise ValueError("Expected unique one-dimensional centers and a matching eligibility mask")
    selected = []
    last = None
    for i in np.argsort(centers, kind="stable"):
        if eligible[i] and (last is None or centers[i] - last >= spacing):
            selected.append(int(i))
            last = centers[i]
    return np.asarray(selected, int)


def robust_covariate_scales(pool):
    """Outcome-free matching scale, including a physically meaningful floor.

    Fraction floors are one native bin/pair; other floors are predeclared small
    measurement tolerances. They do not turn absent common support into support.
    """
    pool = np.asarray(pool, float)
    if pool.ndim != 2 or pool.shape[1] != 7 or not len(pool) or not np.isfinite(pool).all():
        raise ValueError("Invalid covariate pool")
    q10, q25, q75, q90 = np.quantile(pool, [.1, .25, .75, .9], axis=0)
    floor = np.array([.05, .05, 1 / 512, 1 / (512 * 511 / 2), .005, .05, .05])
    iqr = q75 - q25
    quantile_span = (q90 - q10) / 2.563
    return np.maximum.reduce((iqr, quantile_span, floor)), iqr, quantile_span, floor


def match_controls(pool, target, centers, candidate_centers, robust=True):
    """Rep1 matching only, preserving all seven factors and common support."""
    pool, target = np.asarray(pool, float), np.asarray(target, float)
    centers, candidate_centers = np.asarray(centers, int), np.asarray(candidate_centers, int)
    if pool.shape != (len(centers), 7) or target.shape != (7,) or not np.isfinite(pool).all():
        raise ValueError("Invalid matching input")
    permitted = (np.abs(centers[:, None] - candidate_centers[None]) >= SPACING).all(axis=1)
    if not permitted.any():
        return np.empty(0, int), {"evaluable": False, "reason": "no_permitted_background"}
    scale, iqr, span, floor = robust_covariate_scales(pool[permitted])
    if not robust:
        scale = np.maximum(iqr, 1e-8)
    z = np.abs((pool - target) / scale)
    caliper = permitted & (z <= 2).all(axis=1)
    valid = np.flatnonzero(caliper)
    order = np.lexsort((centers[valid], np.linalg.norm(z[valid], axis=1)))
    selected = []
    for i in valid[order]:
        if all(abs(centers[i] - centers[j]) >= SPACING for j in selected):
            selected.append(int(i))
            if len(selected) == 80:
                break
    selected = np.asarray(selected, int)
    if len(selected):
        low, high = np.quantile(pool[selected], [.1, .9], axis=0)
        support = (target >= low) & (target <= high)
    else:
        low, high = np.full(7, np.nan), np.full(7, np.nan)
        support = np.zeros(7, bool)
    counts = [int((permitted & (z[:, j] <= 2)).sum()) for j in range(7)]
    return selected, {
        "permitted": int(permitted.sum()), "caliper_controls": len(valid),
        "independent_controls": len(selected), "scale": scale.tolist(),
        "iqr": iqr.tolist(), "quantile_span": span.tolist(), "physical_floor": floor.tolist(),
        "within_factor_caliper_counts": counts,
        "support": support.tolist(), "q10": finite_list(low), "q90": finite_list(high),
        "evaluable": bool(len(selected) >= 20 and support.all()),
        "reason": "supported" if len(selected) >= 20 and support.all() else "insufficient_common_support",
    }


def finite_list(values):
    return [float(x) if np.isfinite(x) else None for x in values]


def shape_features(maps):
    """Ten explicit shape descriptors; none is a density or sequence label."""
    maps = np.asarray(maps, float)
    if maps.ndim != 3 or maps.shape[1:] != (64, 64) or not np.isfinite(maps).all():
        raise ValueError("Expected finite N x 64 x 64 maps")
    smooth = gaussian_filter(maps, sigma=(0, 1, 1))
    x, y = np.triu_indices(64, k=8)
    sep = y - x
    positive = np.maximum(smooth[:, x, y], 0)
    mass = positive.sum(axis=1)
    weights = np.divide(positive, mass[:, None], out=np.zeros_like(positive), where=mass[:, None] > 0)
    cx, cy = weights @ (x / 63), weights @ (y / 63)
    spread = (weights * ((x[None] / 63 - cx[:, None]) ** 2 +
                         (y[None] / 63 - cy[:, None]) ** 2)).sum(axis=1) ** .5
    topn = max(1, int(np.ceil(len(x) * .01)))
    top = np.sort(weights, axis=1)[:, -topn:].sum(axis=1)
    rows = np.maximum(smooth, 0).sum(axis=2)
    rowmean = rows.mean(axis=1)
    rowcv = np.divide(rows.std(axis=1), rowmean, out=np.zeros_like(rowmean), where=rowmean > 0)
    return np.column_stack((weights[:, sep < 16].sum(axis=1),
                            weights[:, (sep >= 16) & (sep < 32)].sum(axis=1),
                            weights[:, sep >= 32].sum(axis=1),
                            (positive > .5).mean(axis=1), top, cx, cy, spread,
                            rowcv, (rows > 1.5 * rowmean[:, None]).mean(axis=1)))


def template_evidence(maps, centers, known_n, known_types):
    """Nearest known matches and honest same-type recovery diagnostics."""
    features = shape_features(maps)
    known = features[:known_n]
    median = np.median(known, axis=0)
    scale = np.maximum(np.quantile(known, .75, axis=0) - np.quantile(known, .25, axis=0), .02)
    standardized = (features - median) / scale
    dist = np.linalg.norm(standardized[:, None] - standardized[None, :known_n], axis=2)
    independent = np.abs(centers[:, None] - centers[None, :known_n]) >= SPACING
    calibration = dist[:known_n].copy()
    calibration[~independent[:known_n]] = np.inf
    nearest = np.argmin(calibration, axis=1)
    if not np.isfinite(calibration.min(axis=1)).all():
        raise ValueError("Insufficient independent known templates")
    cutoff = float(np.quantile(calibration.min(axis=1), .9))
    per_type = {}
    for kind in sorted(set(known_types)):
        mask = known_types == kind
        per_type[kind] = {"n": int(mask.sum()),
                          "same_type_nearest_fraction": float((known_types[nearest[mask]] == kind).mean())}
    residual = {metric: residual_vectors(maps, metric) for metric in METRICS}
    sims = {}
    for metric, values in residual.items():
        centered = values - values.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(centered, axis=1)
        unit = np.divide(centered, norm[:, None], out=np.zeros_like(centered), where=norm[:, None] > 0)
        sims[metric] = unit @ unit[:known_n].T
    return {
        "features": features, "distances": dist, "nearest": nearest,
        "distance_cutoff_rep1_known_q90": cutoff, "per_type_known_recovery": per_type,
        "feature_scale": scale, "feature_median": median, "similarities": sims,
    }


def thin_symmetric_counts(counts, target_upper_total, seed):
    """Binomial thinning of unique upper-triangle counts, then mirror once.

    Equal *expected* depth is explicit; sampling never increases counts and
    creates no independent biological replicate. Diagonal excluded consistently.
    """
    counts = np.asarray(counts, float)
    if counts.shape != (64, 64) or not np.isfinite(counts).all() or (counts < 0).any() or not np.allclose(counts, counts.T):
        raise ValueError("Expected symmetric nonnegative 64-bin counts")
    if not np.allclose(counts, np.rint(counts)):
        raise ValueError("Thinning requires integer count-preserving aggregation")
    if target_upper_total <= 0:
        raise ValueError("Target count depth must be positive")
    a, b = np.triu_indices(64, k=1)
    vals = np.rint(counts[a, b]).astype(np.int64)
    total = int(vals.sum())
    if not total:
        return np.zeros((64, 64)), 0., False
    p = min(1., target_upper_total / total)
    draw = np.random.default_rng(seed).binomial(vals, p)
    result = np.zeros((64, 64), float)
    result[a, b], result[b, a] = draw, draw
    return result, p, bool(total >= target_upper_total)


def extract(cool_path, manifest, size_bp=5120, progress=False):
    maps, counts, qc = [], [], []
    with open_cool(cool_path) as cool:
        for n, row in enumerate(manifest.itertuples(index=False)):
            native_bins = size_bp // int(cool.binsize)
            native = window(cool, row.chrom, int(row.center), size_bp, native_bins)
            pooled = area_resample(native, 64)
            maps.append(normalize(pooled))
            counts.append(pooled)
            qc.append({"center": int(row.center), "known_distance_bp": int(row.known_distance_bp),
                       **native_qc(native)})
            if progress and (n + 1) % 200 == 0:
                print(f"{cool_path.name}: {n + 1}/{len(manifest)}", flush=True)
    return np.asarray(maps, np.float32), np.asarray(counts), pd.DataFrame(qc)


def json_clean(value):
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def plot_evidence(out, candidates, maps1, maps2, reference1, known, shape_rows, match_audit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = pd.DataFrame(shape_rows)
    fig, axes = plt.subplots(4, 4, figsize=(15, 13), constrained_layout=True)
    for i, candidate in enumerate(candidates.itertuples(index=False)):
        nearest = rows[(rows.candidate_id == candidate.candidate_id) & (rows.replicate == "rep1") &
                       (rows.size_bp == 5120) & (rows.shift_bp == 0)].iloc[0]
        idx = int(nearest.nearest_known_index)
        for j, mat in enumerate((maps1[i], maps2[i], reference1[idx])):
            im = axes[i, j].imshow(mat, vmin=-2, vmax=4, cmap="RdBu_r", origin="lower")
            axes[i, j].set_xticks([0, 32, 63], [-2.56, 0, 2.56])
            axes[i, j].set_yticks([0, 32, 63], [-2.56, 0, 2.56])
            axes[i, j].set_xlabel("Position relative to center (kb)")
            label = ("rep1", "rep2", f"nearest known: {known.type.iloc[idx]}")[j]
            axes[i, j].set_title(f"{candidate.candidate_id} | {label}", fontsize=10)
        vals = rows[(rows.candidate_id == candidate.candidate_id) & (rows.replicate == "pair")]
        axes[i, 3].plot(vals.variant, vals.highpass_repeat_r, "o-", label="highpass")
        axes[i, 3].plot(vals.variant, vals.rowcol_repeat_r, "s-", label="row/column residual")
        axes[i, 3].set_ylim(-.1, 1)
        axes[i, 3].tick_params(axis="x", labelrotation=65, labelsize=7)
        controls = match_audit[candidate.candidate_id]["robust"]
        axes[i, 3].set_title(f"Independent controls: {controls.get('independent_controls', 0)}; "
                             f"common support: {controls['evaluable']}", fontsize=9)
        axes[i, 3].set_ylabel("Same-locus repeat correlation")
        if i == 0:
            axes[i, 3].legend(fontsize=8)
    fig.colorbar(im, ax=axes[:, :3].ravel().tolist(), label="Distance-normalized log contact")
    fig.suptitle("Fixed unannotated candidates: repeat, known-template and sensitivity evidence", fontsize=15)
    fig.savefig(out / "candidate_evidence.png", dpi=190)
    plt.close(fig)


def run(args):
    if args.out.exists():
        raise ValueError("Output already exists; never overwrite a diagnostic protocol")
    candidates = pd.read_csv(args.candidates)
    if candidates.candidate_id.tolist() != ["A1", "A2", "A3", "A4"]:
        raise ValueError("Preserve all four fixed candidates, including A3")
    scan = pd.read_csv(args.discovery / "rep1_all_windows.csv")
    known = annotations(args.annotations)
    if len(known) != 344 or len(scan) != 1812:
        raise ValueError("Unexpected frozen input sizes")
    chrom = candidates.chrom.iloc[0]
    known["chrom"] = chrom
    known["known_distance_bp"] = 0
    background = scan[(~scan.is_known) & (scan.known_distance_bp >= 2560)].copy()
    if len(background) != 1103:
        raise ValueError("Unexpected unannotated full-window pool")
    eligible = (np.abs(background.center.to_numpy()[:, None] - candidates.center.to_numpy()[None]) >= SPACING).all(axis=1)
    independent = score_free_independent_indices(background.center.to_numpy(), eligible)
    args.out.mkdir(parents=True)
    candidates.to_csv(args.out / "fixed_candidates.csv", index=False)
    background[["chrom", "center", "known_distance_bp"]].to_csv(args.out / "score_free_full_background.csv", index=False)
    background.iloc[independent][["chrom", "center", "known_distance_bp"]].to_csv(args.out / "score_free_independent_background.csv", index=False)
    input_paths = [args.rep1, args.rep2, args.candidates, args.annotations,
                   args.discovery / "rep1_all_windows.csv", args.discovery / "freeze_receipt.json"]
    source_paths = [Path(__file__), Path("tests/test_task2_course_verification_20261014.py"),
                    Path("microc/data.py"), Path("scripts/task2_candidate_artifact_followup.py"),
                    Path("scripts/task2_residual_motif_screen_20261006.py")]
    protocol = {
        "frozen_at_utc": utc_now(), "parent_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p): sha256(p) for p in source_paths},
        "input_sha256": {str(p): sha256(p) for p in input_paths},
        "dependency_versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "cooler", "scikit-learn")},
        "seed": SEED, "candidate_selection": "All four historical area candidates; no reselection",
        "previous_rep2_inspection": True, "design": "retrospective diagnostic follow-up, not prospective validation",
        "background": "All 1103 unannotated full-window scan loci, no score filtering; overlapping windows are dependent. Coordinate-order thinning is outcome-free, not a random genomic sample.",
        "independent_spacing_bp": SPACING, "matching_rep1_only": True,
        "covariates": list(COVARIATES), "matching": "All seven factors; <=2 scale caliper, <=80 independent controls, >=20 and every target inside control q10-q90. Compare legacy IQR floor 1e-8 with max(IQR, q90-q10 span/2.563, physical one-bin/pair floors). No favorable outcome filtering.",
        "scale_bp": list(SCALE_BP), "center_shifts_bp": list(SHIFTS_BP),
        "shape_features": list(SHAPE_FEATURES), "template_calibration": "Known references only, >=10240bp same-replicate exclusion; feature scaling floor .02. All-known q90 nearest-independent distance is descriptive, not type confirmation.",
        "depth_target": "Min(rep1 known q10 pooled upper-offdiagonal total, four rep1 candidate totals); freeze after rep1 before opening rep2. Independent binomial thinning of each unique pair, diagonal zero. Equal expected, not exact realized depth.",
        "depth_seeds": list(THIN_SEEDS), "new_type_rule": "No automatic new-type declaration; compute candidate-pair common morphology and known-template comparisons. Course rules are not publication-level confirmation.",
        "limits": ["No raw-read MAPQ or fragment coverage; contact-row sparsity is a proxy.",
                   "No new untouched biological sample; repeated inspection precludes blindness.",
                   "No p-values/FDR from overlapping or unsupported controls.",
                   "Missing annotation alone is not evidence of a new shape type.",
                   "Reference uniqueness does not establish actual alignment quality."],
    }
    write_json(args.out / "protocol_freeze.json", protocol)
    protocol_hash = sha256(args.out / "protocol_freeze.json")
    print(f"Protocol frozen {protocol_hash}; extracting rep1", flush=True)
    base_manifest = pd.concat([known[["chrom", "center", "known_distance_bp"]],
                               background[["chrom", "center", "known_distance_bp"]]], ignore_index=True)
    maps1, counts1, qc1 = extract(args.rep1, base_manifest, progress=True)
    nknown = len(known)
    bglookup = {int(c): nknown + i for i, c in enumerate(background.center)}
    ci = np.array([bglookup[int(c)] for c in candidates.center])
    qc1.to_csv(args.out / "rep1_complete_background_qc.csv", index=False)
    covariates = qc_covariates(qc1.iloc[nknown:])
    match_audit, matching_rows, control_ids = {}, [], {}
    for j, candidate in enumerate(candidates.itertuples(index=False)):
        match_audit[candidate.candidate_id], control_ids[candidate.candidate_id] = {}, {}
        target = qc_covariates(qc1.iloc[[ci[j]]])[0]
        for robust, arm in ((False, "legacy"), (True, "robust")):
            chosen, audit = match_controls(covariates, target, background.center.to_numpy(), candidates.center.to_numpy(), robust)
            match_audit[candidate.candidate_id][arm] = audit
            control_ids[candidate.candidate_id][arm] = [int(x) for x in chosen]
            for k, name in enumerate(COVARIATES):
                matching_rows.append({"candidate_id": candidate.candidate_id, "arm": arm, "covariate": name,
                                      "target": target[k], "scale": audit["scale"][k],
                                      "background_iqr": audit["iqr"][k], "physical_floor": audit["physical_floor"][k],
                                      "within_factor_caliper_count": audit["within_factor_caliper_counts"][k],
                                      "independent_controls": len(chosen), "common_support": audit["support"][k],
                                      "control_q10": audit["q10"][k], "control_q90": audit["q90"][k],
                                      "evaluable": audit["evaluable"]})
    pd.DataFrame(matching_rows).to_csv(args.out / "matching_support_audit.csv", index=False)
    a, b = np.triu_indices(64, k=1)
    totals1 = counts1[:, a, b].sum(axis=1)
    target_depth = int(np.floor(min(np.quantile(totals1[:nknown], .1), totals1[ci].min())))
    if target_depth < 1:
        raise ValueError("No positive rep1 depth target")
    write_json(args.out / "rep1_control_and_depth_freeze.json", json_clean({
        "frozen_at_utc": utc_now(), "protocol_sha256": protocol_hash,
        "rep2_opened_for_selection": False, "matching": match_audit, "background_indices": control_ids,
        "equal_expected_upper_count_target": target_depth, "rep1_qc_sha256": sha256(args.out / "rep1_complete_background_qc.csv"),
        "matching_audit_sha256": sha256(args.out / "matching_support_audit.csv")}))
    freeze_hash = sha256(args.out / "rep1_control_and_depth_freeze.json")
    if sha256(args.out / "protocol_freeze.json") != protocol_hash:
        raise ValueError("Protocol changed before rep2 extraction")
    print(f"Rep1 controls and depth frozen {freeze_hash}; extracting rep2", flush=True)
    maps2, counts2, qc2 = extract(args.rep2, base_manifest, progress=True)
    qc2.to_csv(args.out / "rep2_complete_background_qc.csv", index=False)
    np.savez_compressed(args.out / "base_maps_counts.npz", maps1=maps1, maps2=maps2,
                        counts1=counts1, counts2=counts2, candidate_indices=ci)
    evidence_rows, feature_rows, known_recovery_rows, comparison_rows, pair_rows = [], [], [], [], []
    variants = [(s, 0) for s in SCALE_BP] + [(5120, offset) for offset in SHIFTS_BP if offset]
    for size, shift in variants:
        print(f"Shape diagnostic size={size}, shift={shift}", flush=True)
        candidate_manifest = candidates[["chrom", "center", "known_distance_bp"]].copy()
        candidate_manifest["center"] += shift
        variant = f"{size/1000:g}kb{shift:+d}"
        if size == 5120 and shift == 0:
            current1, current2 = maps1[ci], maps2[ci]
            known1, known2 = maps1[:nknown], maps2[:nknown]
        else:
            current1, _, _ = extract(args.rep1, candidate_manifest, size)
            current2, _, _ = extract(args.rep2, candidate_manifest, size)
            # At shifted centers compare known templates at their true centers.
            if size == 5120:
                known1, known2 = maps1[:nknown], maps2[:nknown]
            else:
                known1, _, _ = extract(args.rep1, known, size)
                known2, _, _ = extract(args.rep2, known, size)
        combined1, combined2 = np.concatenate([known1, current1]), np.concatenate([known2, current2])
        centers = np.concatenate([known.center.to_numpy(), candidate_manifest.center.to_numpy()])
        diagnostic1 = template_evidence(combined1, centers, nknown, known.type.to_numpy())
        diagnostic2 = template_evidence(combined2, centers, nknown, known.type.to_numpy())
        # Scale/calibrate rep1 only; rep2 distances use its features, not its fitted scale.
        z2 = (diagnostic2["features"] - diagnostic1["feature_median"]) / diagnostic1["feature_scale"]
        dist2 = np.linalg.norm(z2[:, None] - z2[None, :nknown], axis=2)
        for rep, diag, distance in (("rep1", diagnostic1, diagnostic1["distances"]), ("rep2", diagnostic2, dist2)):
            for kind, stats in diag["per_type_known_recovery"].items():
                known_recovery_rows.append({"variant": variant, "size_bp": size, "shift_bp": shift, "replicate": rep,
                                            "type": kind, **stats})
            for j, candidate in enumerate(candidates.itertuples(index=False)):
                idx = nknown + j
                nearest = int(np.argmin(distance[idx]))
                overlap = int(((known.start < candidate_manifest.center.iloc[j] + size // 2) &
                               (known.end > candidate_manifest.center.iloc[j] - size // 2)).sum())
                evidence_rows.append({"candidate_id": candidate.candidate_id, "variant": variant,
                                      "size_bp": size, "shift_bp": shift, "replicate": rep,
                                      "nearest_known_index": nearest, "nearest_known_id": known.id.iloc[nearest],
                                      "nearest_known_type": known.type.iloc[nearest], "nearest_known_center": int(known.center.iloc[nearest]),
                                      "interpretable_feature_distance": float(distance[idx, nearest]),
                                      "rep1_known_q90_distance": diagnostic1["distance_cutoff_rep1_known_q90"],
                                      "beyond_rep1_known_q90": bool(distance[idx, nearest] > diagnostic1["distance_cutoff_rep1_known_q90"]),
                                      "best_known_highpass_r": float(diag["similarities"]["highpass"][idx].max()),
                                      "best_known_rowcol_r": float(diag["similarities"]["rowcol"][idx].max()),
                                      "known_intervals_overlapping_window": overlap})
                for k, name in enumerate(SHAPE_FEATURES):
                    feature_rows.append({"candidate_id": candidate.candidate_id, "variant": variant, "replicate": rep,
                                         "feature": name, "candidate_value": diag["features"][idx, k],
                                         "nearest_known_value": diag["features"][nearest, k],
                                         "rep1_known_iqr_scale": diagnostic1["feature_scale"][k]})
        rv1 = {m: residual_vectors(current1, m) for m in METRICS}
        rv2 = {m: residual_vectors(current2, m) for m in METRICS}
        for j, candidate in enumerate(candidates.itertuples(index=False)):
            evidence_rows.append({"candidate_id": candidate.candidate_id, "variant": variant,
                                  "size_bp": size, "shift_bp": shift, "replicate": "pair",
                                  "highpass_repeat_r": float(paired_r(rv1["highpass"], rv2["highpass"])[j]),
                                  "rowcol_repeat_r": float(paired_r(rv1["rowcol"], rv2["rowcol"])[j])})
        if size == 5120 and shift == 0:
            for rep, rv in (("rep1", rv1), ("rep2", rv2)):
                for j in range(4):
                    for k in range(j + 1, 4):
                        pair_rows.append({"candidate_a": candidates.candidate_id.iloc[j], "candidate_b": candidates.candidate_id.iloc[k],
                                          "replicate": rep, "highpass_r": float(paired_r(rv["highpass"][[j]], rv["highpass"][[k]])[0]),
                                          "rowcol_r": float(paired_r(rv["rowcol"][[j]], rv["rowcol"][[k]])[0])})
    # Descriptive comparison only when rep1 control support permits it.
    for m in METRICS:
        rv1, rv2 = residual_vectors(maps1, m), residual_vectors(maps2, m)
        repeat = paired_r(rv1, rv2)
        for j, candidate in enumerate(candidates.itertuples(index=False)):
            for arm in ("legacy", "robust"):
                ids = np.asarray(control_ids[candidate.candidate_id][arm], int) + nknown
                audit = match_audit[candidate.candidate_id][arm]
                comparison_rows.append({"candidate_id": candidate.candidate_id, "arm": arm, "metric": m,
                                        "candidate_repeat_r": repeat[ci[j]], "independent_controls": len(ids),
                                        "evaluable": audit["evaluable"], "control_median_repeat_r": float(np.median(repeat[ids])) if len(ids) else None,
                                        "fraction_controls_below_candidate": float((repeat[ids] < repeat[ci[j]]).mean()) if audit["evaluable"] else None,
                                        "interpretation": "descriptive_no_p_or_fdr" if audit["evaluable"] else "unsupported_comparator"})
    print(f"Equal expected depth sensitivity: {target_depth} unique upper contacts", flush=True)
    thinning_rows = []
    for seed in THIN_SEEDS:
        thin1, thin2, achieved1, achieved2, probs1, probs2 = [], [], [], [], [], []
        relevant = np.concatenate([np.arange(nknown), ci])
        for idx in relevant:
            t1, p1, ok1 = thin_symmetric_counts(counts1[idx], target_depth, seed + int(idx) * 2)
            t2, p2, ok2 = thin_symmetric_counts(counts2[idx], target_depth, seed + int(idx) * 2 + 1)
            thin1.append(normalize(t1)); thin2.append(normalize(t2))
            achieved1.append(ok1); achieved2.append(ok2); probs1.append(p1); probs2.append(p2)
        thin1, thin2 = np.asarray(thin1), np.asarray(thin2)
        diag1 = template_evidence(thin1, np.concatenate([known.center.to_numpy(), candidates.center.to_numpy()]), nknown, known.type.to_numpy())
        z2 = (shape_features(thin2) - diag1["feature_median"]) / diag1["feature_scale"]
        d2 = np.linalg.norm(z2[:, None] - z2[None, :nknown], axis=2)
        for m in METRICS:
            repeat = paired_r(residual_vectors(thin1, m), residual_vectors(thin2, m))
            for j, candidate in enumerate(candidates.itertuples(index=False)):
                idx = nknown + j
                thinning_rows.append({"candidate_id": candidate.candidate_id, "seed": seed, "metric": m,
                                      "target_expected_unique_count": target_depth, "rep1_thinning_probability": probs1[idx],
                                      "rep2_thinning_probability": probs2[idx], "rep1_target_supported": achieved1[idx],
                                      "rep2_target_supported": achieved2[idx], "repeat_r": repeat[idx],
                                      "rep1_nearest_known_feature_distance": diag1["distances"][idx].min(),
                                      "rep2_nearest_known_feature_distance": d2[idx].min(),
                                      "rep1_known_q90_distance": diag1["distance_cutoff_rep1_known_q90"],
                                      "rep2_beyond_known_q90": bool(d2[idx].min() > diag1["distance_cutoff_rep1_known_q90"]),
                                      "known_rep1_at_target_fraction": float(np.mean(achieved1[:nknown])),
                                      "known_rep2_at_target_fraction": float(np.mean(achieved2[:nknown]))})
    # Some references below target are not truly depth matched: mark as a limitation.
    tables = {"shape_template_and_sensitivity.csv": evidence_rows, "interpretable_shape_features.csv": feature_rows,
              "known_template_recovery.csv": known_recovery_rows, "matched_repeat_comparison.csv": comparison_rows,
              "candidate_common_morphology.csv": pair_rows, "equal_expected_depth_sensitivity.csv": thinning_rows}
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(args.out / name, index=False)
    labels = []
    shape = pd.DataFrame(evidence_rows)
    depth = pd.DataFrame(thinning_rows)
    for candidate in candidates.itertuples(index=False):
        sensitivity = shape[(shape.candidate_id == candidate.candidate_id) & (shape.replicate == "pair")]
        d = depth[(depth.candidate_id == candidate.candidate_id) & (depth.metric == "highpass")]
        default = shape[(shape.candidate_id == candidate.candidate_id) & (shape.replicate == "rep2") &
                        (shape.size_bp == 5120) & (shape.shift_bp == 0)].iloc[0]
        supported = match_audit[candidate.candidate_id]["robust"]["evaluable"]
        labels.append({"candidate_id": candidate.candidate_id, "center_bp": int(candidate.center),
                       "historical_default_dual_novelty_replication": bool(candidate.rep2_passes_both_frozen_novelty_gates),
                       "minimum_shape_repeat_r_over_scales_shifts": float(sensitivity.highpass_repeat_r.min()),
                       "maximum_shape_repeat_r_over_scales_shifts": float(sensitivity.highpass_repeat_r.max()),
                       "equal_depth_highpass_repeat_r_mean": float(d.repeat_r.mean()),
                       "equal_depth_highpass_repeat_r_min": float(d.repeat_r.min()),
                       "interpretable_feature_rep2_beyond_known_q90": bool(default.beyond_rep1_known_q90),
                       "matched_artifact_comparison_evaluable": supported,
                       "level": "reproducible_unannotated_shape_outlier_QC_unresolved" if candidate.rep2_passes_both_frozen_novelty_gates else "reproducible_unannotated_region_default_novelty_not_replicated",
                       "confirmed_new_shape_type": False,
                       "biological_confirmation": False})
    pd.DataFrame(labels).to_csv(args.out / "candidate_evidence_levels.csv", index=False)
    plot_evidence(args.out, candidates, maps1[ci], maps2[ci], maps1[:nknown], known, evidence_rows, match_audit)
    files = [p for p in args.out.iterdir() if p.is_file()]
    summary = {
        "analysis": "Retrospective fixed-candidate course evidence verification",
        "completed_at_utc": utc_now(), "protocol_sha256": protocol_hash,
        "rep1_control_and_depth_freeze_sha256": freeze_hash,
        "protocol_frozen_before_rep1_extraction": True, "rep1_control_freeze_before_new_rep2_extraction": True,
        "prospectively_blind": False, "full_score_free_unannotated_windows": len(background),
        "score_free_independent_background_count": len(independent), "fixed_candidate_count": len(candidates),
        "known_reference_count": nknown, "target_expected_upper_offdiagonal_counts": target_depth,
        "matching": match_audit, "candidate_evidence": labels,
        "confirmed_new_shape_types": 0, "p_values_or_fdr_claimed": False,
        "limitations": protocol["limits"] + ["Fixed-depth thinning compares equal expected, not equal realized totals; known references below target remain under-depth and are reported.",
                                              "Rep1-derived interpretable features are an additional descriptive view, not replacement of historical frozen novelty results."],
        "source_table_binding": {"candidate_evidence.png": ["shape_template_and_sensitivity.csv", "matching_support_audit.csv"]},
        "output_sha256": {p.name: sha256(p) for p in files},
    }
    write_json(args.out / "summary.json", json_clean(summary))
    args.results.mkdir(exist_ok=True)
    for name in ["summary.json", "protocol_freeze.json", "rep1_control_and_depth_freeze.json", "candidate_evidence_levels.csv",
                 "matching_support_audit.csv", *tables, "candidate_evidence.png"]:
        shutil.copyfile(args.out / name, args.results / ("task2_course_verification_20261014_" + name))
    print(json.dumps(json_clean({"completed": True, "protocol_sha256": protocol_hash, "candidate_evidence": labels}), indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rep1", type=Path, default=Path("data/raw/GSE272159_37C_rep1.mapq_30.10.cool"))
    p.add_argument("--rep2", type=Path, default=Path("data/raw/GSE272159_37C_rep2.mapq_30.10.cool"))
    p.add_argument("--annotations", type=Path, default=Path("data/raw/标注数据.xlsx"))
    p.add_argument("--candidates", type=Path, default=Path("results/task2_improve_20261006_individual_candidate_validation.csv"))
    p.add_argument("--discovery", type=Path, default=Path("outputs/task2_improve_20261006/area"))
    p.add_argument("--out", type=Path, default=Path("outputs/task2_course_verification_20261014"))
    p.add_argument("--results", type=Path, default=Path("results"))
    args = p.parse_args()
    with threadpool_limits(limits=2):
        run(args)


if __name__ == "__main__":
    main()
