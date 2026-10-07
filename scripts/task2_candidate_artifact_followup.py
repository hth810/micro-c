"""Fixed-candidate native-count QC, control support, and clipping sensitivity.

This retrospective follow-up never discovers candidates or fits on rep2.
All four area-screen individual loci, including failed replication A3, remain
in every table. COOL counts provide coverage proxies, not fragment-level or
sequence-alignability evidence. A diagnostic plan precedes extraction; rep1
control IDs and support decisions are frozen before new rep2 extraction.
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
from threadpoolctl import threadpool_limits

from microc.data import area_resample, normalize, open_cool, window
from scripts.task2_residual_motif_screen_20261006 import (
    METRICS, frozen_project, project_novelty, residual_vectors, sha256, write_json,
)

WINDOW_BP = 5120
INDEPENDENCE_BP = 10240
MIN_CONTROLS = 20
MAX_CONTROLS = 80
VARIANTS = ("default_clip_5", "unclipped", "explicit_clip_2_4")
COVARIATES = (
    "log_native_symmetric_total", "log_native_upper_offdiag_total",
    "native_zero_offdiag_row_fraction", "native_upper_nonzero_fraction",
    "native_top_1pct_mass_fraction", "native_offdiag_row_cv",
    "log_known_interval_distance",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def native_qc(native: np.ndarray) -> dict:
    """Contact-mass and sparsity proxies before any interpolation/normalization.

    Symmetric total counts mirrored contacts twice; upper-offdiagonal total
    counts each pair once. Diagonal self-counts do not rescue a zero row.
    Top 1% means ceil(0.01 * all upper-offdiagonal entries), including zeros.
    """
    a = np.asarray(native, dtype=np.float64)
    if (a.ndim != 2 or a.shape[0] != a.shape[1] or len(a) < 2 or
            not np.isfinite(a).all() or np.any(a < 0) or
            not np.allclose(a, a.T, rtol=1e-7, atol=1e-7)):
        raise ValueError("Expected finite, nonnegative, symmetric native counts")
    upper = a[np.triu_indices(len(a), k=1)]
    offdiag_rows = a.sum(axis=1) - np.diag(a)
    total = float(upper.sum())
    top_n = max(1, int(np.ceil(len(upper) * .01)))
    top_mass = float(np.partition(upper, len(upper) - top_n)[-top_n:].sum())
    return {
        "native_bins": len(a), "native_symmetric_total": float(a.sum()),
        "native_upper_offdiag_total": total,
        "native_zero_offdiag_rows": int((offdiag_rows == 0).sum()),
        "native_zero_offdiag_row_fraction": float((offdiag_rows == 0).mean()),
        "native_upper_nonzero_fraction": float((upper > 0).mean()),
        "native_top_1pct_mass_fraction": top_mass / total if total else 0.,
        "native_offdiag_row_cv": float(offdiag_rows.std() / offdiag_rows.mean())
        if offdiag_rows.mean() else 0.,
        "native_offdiag_row_q10": float(np.quantile(offdiag_rows, .1)),
        "native_offdiag_row_q90": float(np.quantile(offdiag_rows, .9)),
    }


def normalization_variants(counts: np.ndarray) -> tuple[dict, dict]:
    """Keep original O/E median and MAD floor; ablate only final clipping.

    Original scientific normalization clips [-5, 5]. The heatmap range [-2, 4]
    was display-only; applying it here is an explicitly new, fixed sensitivity,
    not a correction to the original experiment.
    """
    a = np.asarray(counts, dtype=np.float32)
    if a.shape != (64, 64) or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("Expected finite nonnegative 64 x 64 pooled counts")
    logged = np.log1p(a)
    expected = np.array([np.median(np.diagonal(logged, offset=d))
                         for d in range(64)], dtype=np.float32)
    offset = np.abs(np.subtract.outer(np.arange(64), np.arange(64)))
    residual = logged - expected[offset]
    upper = residual[np.triu_indices(64, k=2)]
    mad = float(np.median(np.abs(upper - np.median(upper))) * 1.4826)
    standardized = residual / max(mad, .1)
    values = standardized[np.triu_indices(64, k=8)]
    maps = {"default_clip_5": normalize(a), "unclipped": standardized.astype(np.float32),
            "explicit_clip_2_4": np.clip(standardized, -2, 4).astype(np.float32)}
    if not np.allclose(maps["default_clip_5"], np.clip(standardized, -5, 5), atol=1e-6):
        raise ValueError("Current normalization differs from frozen definition")
    return maps, {
        "pooled_total": float(a.sum()), "normalization_mad": mad,
        "normalization_scale_with_floor": max(mad, .1),
        "default_highpass_region_clipped_low_fraction": float((values < -5).mean()),
        "default_highpass_region_clipped_high_fraction": float((values > 5).mean()),
        "explicit_tight_clipped_low_fraction": float((values < -2).mean()),
        "explicit_tight_clipped_high_fraction": float((values > 4).mean()),
        "unclipped_upper_min": float(values.min()),
        "unclipped_upper_max": float(values.max()),
        "unclipped_upper_near_zero_fraction": float((np.abs(values) < 1e-6).mean()),
    }


def qc_covariates(table: pd.DataFrame) -> np.ndarray:
    return np.column_stack((
        np.log1p(table.native_symmetric_total),
        np.log1p(table.native_upper_offdiag_total),
        table.native_zero_offdiag_row_fraction, table.native_upper_nonzero_fraction,
        table.native_top_1pct_mass_fraction, table.native_offdiag_row_cv,
        np.log1p(table.known_distance_bp),
    )).astype(float)


def select_controls(pool: np.ndarray, target: np.ndarray, centers: np.ndarray,
                    candidate_centers: np.ndarray) -> tuple[np.ndarray, dict]:
    """Rep1-only bounded nearest matching, with honest common-support rejection."""
    pool = np.asarray(pool, float)
    target = np.asarray(target, float)
    centers = np.asarray(centers, int)
    candidate_centers = np.asarray(candidate_centers, int)
    if (pool.ndim != 2 or pool.shape[1] != len(COVARIATES) or
            target.shape != (len(COVARIATES),) or len(pool) != len(centers) or
            not np.isfinite(pool).all() or not np.isfinite(target).all()):
        raise ValueError("Invalid matching covariates")
    permitted = (np.abs(centers[:, None] - candidate_centers[None, :]) >= INDEPENDENCE_BP).all(axis=1)
    if not permitted.any():
        return np.empty(0, int), {"selected_controls": 0, "caliper_controls": 0,
                                 "permitted_controls": 0, "evaluable": False,
                                 "supported_covariates": [False] * len(COVARIATES),
                                 "iqr_scale": [1.] * len(COVARIATES),
                                 "control_q10": [None] * len(COVARIATES),
                                 "control_q90": [None] * len(COVARIATES)}
    scale = np.maximum(np.quantile(pool[permitted], .75, axis=0) -
                       np.quantile(pool[permitted], .25, axis=0), 1e-8)
    standardized = np.abs((pool - target) / scale)
    valid = np.flatnonzero(permitted & (standardized <= 2.).all(axis=1))
    # Center is a deterministic tie-breaker; no outcome statistic enters ranking.
    order = np.lexsort((centers[valid], np.linalg.norm(standardized[valid], axis=1)))
    selected = valid[order[:MAX_CONTROLS]]
    if len(selected):
        # Frozen pool is independent; reject malformed pools, do not silently thin.
        differences = np.abs(centers[selected, None] - centers[None, selected])
        differences[np.eye(len(selected), dtype=bool)] = INDEPENDENCE_BP
        if (differences < INDEPENDENCE_BP).any():
            raise ValueError("Selected control windows are not independent")
        low, high = np.quantile(pool[selected], [.1, .9], axis=0)
        supported = (target >= low) & (target <= high)
    else:
        low = high = np.full(len(COVARIATES), np.nan)
        supported = np.zeros(len(COVARIATES), bool)
    return selected, {
        "permitted_controls": int(permitted.sum()), "caliper_controls": len(valid),
        "selected_controls": len(selected),
        "supported_covariates": supported.tolist(),
        "evaluable": bool(len(selected) >= MIN_CONTROLS and supported.all()),
        "iqr_scale": scale.tolist(),
        "control_q10": [float(x) if np.isfinite(x) else None for x in low],
        "control_q90": [float(x) if np.isfinite(x) else None for x in high],
    }


def paired_r(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Expected paired N x feature vectors")
    a = a.astype(float) - a.mean(axis=1, keepdims=True)
    b = b.astype(float) - b.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide((a * b).sum(axis=1), denom, out=np.zeros(len(a)), where=denom > 1e-12)


def descriptive_percentile(values: np.ndarray, observed: float, evaluable: bool):
    """No p-value, FDR, pseudo-significance, or percentile without support."""
    return float((np.asarray(values) < observed).mean()) if evaluable else None


def check_discovery(folder: Path) -> dict:
    receipt = json.loads((folder / "freeze_receipt.json").read_text(encoding="utf-8"))
    for name, digest in receipt["frozen_file_sha256"].items():
        if sha256(folder / name) != digest:
            raise ValueError(f"Frozen discovery changed: {name}")
    for name, digest in receipt["implementation_sha256"].items():
        if sha256(Path(name)) != digest:
            raise ValueError(f"Frozen helper changed: {name}")
    return receipt


def extract_native(path: Path, chosen: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    maps = {name: [] for name in VARIANTS}
    rows = []
    with open_cool(path) as cool:
        if int(cool.binsize) != 10:
            raise ValueError("Locked native-count audit expects 10-bp source bins")
        for k, row in enumerate(chosen.itertuples()):
            native = window(cool, str(row.chrom), int(row.center), WINDOW_BP, 512)
            pooled = area_resample(native, 64)
            variants, clipping = normalization_variants(pooled)
            raw = native_qc(native)
            if not np.isclose(raw["native_symmetric_total"], clipping["pooled_total"], atol=1e-5):
                raise ValueError("Count conservation failed")
            for name in VARIANTS:
                maps[name].append(variants[name])
            rows.append({"center_bp": int(row.center), "role": row.role,
                         "known_distance_bp": int(row.known_distance_bp), **raw, **clipping})
            if (k + 1) % 100 == 0:
                print(f"native QC {k+1}/{len(chosen)} {path.name}", flush=True)
    return {name: np.stack(value) for name, value in maps.items()}, pd.DataFrame(rows)


def figure(path: Path, candidates: pd.DataFrame, balance: pd.DataFrame,
           sensitivity: pd.DataFrame, matches: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(candidates), 2, figsize=(13, 12), dpi=160, squeeze=False)
    for i, row in enumerate(candidates.itertuples()):
        b = balance[balance.candidate_id == row.candidate_id]
        scale = b.iqr_scale.to_numpy(float)
        target = b.target_rep1.to_numpy(float)
        lo = (b.control_q10.to_numpy(float) - target) / scale
        hi = (b.control_q90.to_numpy(float) - target) / scale
        axes[i, 0].hlines(np.arange(len(COVARIATES)), lo, hi, color="#377fa8", linewidth=4)
        axes[i, 0].scatter(np.zeros(len(COVARIATES)), np.arange(len(COVARIATES)), color="#c4463a", s=20)
        axes[i, 0].axvline(0, color="gray", linestyle="--")
        axes[i, 0].set(yticks=np.arange(len(COVARIATES)),
                       yticklabels=["log total", "log offdiag", "zero rows", "nonzero pairs",
                                    "top 1% mass", "row CV", "log known distance"],
                       xlabel="Control q10–q90 relative to candidate (rep1 IQR)",
                       title=f"{row.candidate_id}: {int(row.center):,} bp; n={matches[row.candidate_id]['selected_controls']}; "
                             + ("supported" if matches[row.candidate_id]["evaluable"] else "NOT EVALUABLE"))
        s = sensitivity[(sensitivity.candidate_id == row.candidate_id) &
                        (sensitivity.replicate == "rep2") &
                        (sensitivity.model_regime == "rep1_recalibrated_sensitivity")]
        for j, metric in enumerate(METRICS):
            values = [float(s[(s.variant == name) & (s.metric == metric)].novelty_ratio.iloc[0])
                      for name in VARIANTS]
            axes[i, 1].bar(np.arange(3) + (j-.5)*.35, values, width=.35,
                           label=metric, color=("#377fa8", "#dc9937")[j])
        axes[i, 1].axhline(1, color="gray", linestyle="--")
        axes[i, 1].set(xticks=np.arange(3), xticklabels=["Default [-5,5]", "Unclipped", "Explicit [-2,4]"],
                       ylabel="Distance / variant-specific rep1-only cutoff",
                       title="Fixed-rule sensitivity; no rep2 refitting")
        if i == 0:
            axes[i, 1].legend()
    fig.suptitle("Four fixed Task 2 candidates: native-count proxies and normalization sensitivity\n"
                 "Controls exclude novelty; unsupported comparisons have no percentile or significance claim")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, default=Path("outputs/task2_improve_20261006/area"))
    parser.add_argument("--rep1", type=Path, default=Path("data/raw/GSE272159_37C_rep1.mapq_30.10.cool"))
    parser.add_argument("--rep2", type=Path, default=Path("data/raw/GSE272159_37C_rep2.mapq_30.10.cool"))
    parser.add_argument("--out", type=Path, default=Path("outputs/task2_followup_20261006_artifact"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Use a new output directory; never overwrite a diagnostic freeze")
    discovery_receipt = check_discovery(args.discovery)
    summary = json.loads((args.discovery / "discovery_summary.json").read_text(encoding="utf-8"))
    independent = pd.read_csv(args.discovery / "rep1_independent_unannotated.csv")
    candidates = independent[independent.passes_both_novelty_gates].copy().reset_index(drop=True)
    candidates.insert(0, "candidate_id", [f"A{i+1}" for i in range(len(candidates))])
    if len(candidates) != 4 or len(independent) != 291:
        raise ValueError("Expected the original four candidates and 291 locked background loci")
    archive = np.load(args.discovery / "rep1_maps_models.npz")
    known_n = int(archive["known_n"])
    if known_n != 344:
        raise ValueError("Expected original 344 known references")
    chrom = str(independent.chrom.iloc[0])
    known = pd.DataFrame({"chrom": chrom, "center": archive["centers"][:known_n],
                          "known_distance_bp": 0, "role": "known_reference"})
    unknown = independent[["chrom", "center", "known_distance_bp"]].copy()
    unknown["role"] = "independent_unannotated"
    chosen = pd.concat([known, unknown], ignore_index=True)
    source_files = [Path(__file__).resolve().relative_to(Path.cwd()), Path("microc/data.py"),
                    Path("scripts/task2_residual_motif_screen_20261006.py"),
                    Path("scripts/task2_rep1_locked_discovery.py"),
                    Path("tests/test_task2_candidate_artifact_followup.py")]
    args.out.mkdir(parents=True)
    chosen.to_csv(args.out / "fixed_locus_manifest.csv", index=False)
    candidates.to_csv(args.out / "fixed_candidates.csv", index=False)
    plan = {
        "frozen_at_utc": utc_now(),
        "run_parent_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dependency_versions": {name: importlib.metadata.version(name) for name in
                                ("numpy", "pandas", "scipy", "scikit-learn", "cooler", "threadpoolctl")},
        "threads": 2,
        "source_sha256": {str(p).replace("\\", "/"): sha256(p) for p in source_files},
        "input_sha256": {"rep1": sha256(args.rep1), "rep2": sha256(args.rep2),
                         "original_discovery_receipt": sha256(args.discovery / "freeze_receipt.json"),
                         "original_models": sha256(args.discovery / "rep1_maps_models.npz"),
                         "original_independent_manifest": sha256(args.discovery / "rep1_independent_unannotated.csv")},
        "fixed_locus_manifest_sha256": sha256(args.out / "fixed_locus_manifest.csv"),
        "fixed_candidates_sha256": sha256(args.out / "fixed_candidates.csv"),
        "candidate_selection": "All four original rep1-frozen dual-novelty loci; no reselection",
        "native_input": "512 x 512, 10-bp counts, 5120-bp window; area sums into 64 x 64",
        "covariates": list(COVARIATES), "matching_rep1_only": True,
        "matching": {"max_iqr_caliper_all_factors": 2, "max_controls": MAX_CONTROLS,
                     "min_controls": MIN_CONTROLS, "required_q10_q90_support_all_factors": True,
                     "min_spacing_from_all_four_candidates_and_between_controls_bp": INDEPENDENCE_BP,
                     "novelty_or_candidate_score_used_in_matching": False,
                     "pool_caveat": "Original 291 representatives were rep1 score-prioritized, not an unbiased genomic null"},
        "normalization_sensitivities": list(VARIANTS),
        "frozen_model_policy": "Default: original rep1 scaler/PCA/p90 cutoff. Alternatives: two separately labelled regimes: unchanged original model input stress, and same 20-D PCA/p90 independent-known rule recalibrated on rep1 known references only. No rep2 fitting or selection.",
        "comparison": "Strict empirical fraction controls < candidate, only with all-factor support; descriptive only",
        "historical_rep2_inspection": True,
        "limits": ["COOL has no read/fragment-level coverage or alignability evidence.",
                   "Window symmetric contact mass and row sparsity are coverage proxies only.",
                   "No p-values, FDR, new class declaration, threshold lowering, or prospective blindness."],
    }
    if plan["input_sha256"]["rep1"] != discovery_receipt["input_sha256"]["rep1"]:
        raise ValueError("Native rep1 differs from original frozen discovery")
    write_json(args.out / "diagnostic_plan_freeze.json", plan)
    print("Diagnostic plan frozen; extracting rep1 only", flush=True)
    maps1, qc1 = extract_native(args.rep1, chosen)
    positions = {int(c): i for i, c in enumerate(archive["centers"])}
    original_idx = [positions[int(c)] for c in chosen.center]
    reproduction_error = float(np.max(np.abs(maps1["default_clip_5"] - archive["maps"][original_idx])))
    if reproduction_error > 1e-6:
        raise ValueError("Native area/default maps do not reproduce frozen discovery")
    qc1.to_csv(args.out / "rep1_native_qc.csv", index=False)
    # Alternative models and cutoffs are fitted before ANY new rep2 extraction.
    # The second regime is not interchangeable with unchanged-model input stress.
    alternative_models, alternative_cutoffs = {}, {}
    for variant in VARIANTS[1:]:
        alternative_models[variant], alternative_cutoffs[variant] = {}, {}
        for metric in METRICS:
            vectors = residual_vectors(maps1[variant], metric)
            _, cutoff, scaler, pca = project_novelty(vectors, known_n,
                                                   chosen.center.to_numpy(int)[:known_n])
            alternative_cutoffs[variant][metric] = float(cutoff)
            model = alternative_models[variant]
            model[metric + "_mean"] = scaler.mean_
            model[metric + "_scale"] = scaler.scale_
            model[metric + "_pca_mean"] = pca.mean_
            model[metric + "_components"] = pca.components_
    np.savez_compressed(args.out / "rep1_sensitivity_models.npz", **{
        variant + "__" + key: value for variant, model in alternative_models.items()
        for key, value in model.items()})
    pool_covariates = qc_covariates(qc1.iloc[known_n:])
    pool_centers = chosen.center.to_numpy(int)[known_n:]
    candidate_positions = {row.candidate_id: known_n + int(np.flatnonzero(pool_centers == row.center)[0])
                           for row in candidates.itertuples()}
    matches, balance_rows, control_rows, control_idx = {}, [], [], {}
    for row in candidates.itertuples():
        ci = candidate_positions[row.candidate_id]
        target = qc_covariates(qc1.iloc[[ci]])[0]
        selected, diagnostic = select_controls(pool_covariates, target, pool_centers,
                                               candidates.center.to_numpy(int))
        matches[row.candidate_id] = diagnostic
        control_idx[row.candidate_id] = selected + known_n
        for j, name in enumerate(COVARIATES):
            balance_rows.append({"candidate_id": row.candidate_id, "covariate": name,
                                 "target_rep1": float(target[j]), "iqr_scale": diagnostic["iqr_scale"][j],
                                 "control_q10": diagnostic["control_q10"][j],
                                 "control_q90": diagnostic["control_q90"][j],
                                 "supported": diagnostic["supported_covariates"][j]})
        for k in selected:
            control_rows.append({"candidate_id": row.candidate_id, "candidate_center_bp": int(row.center),
                                 "control_center_bp": int(pool_centers[k]),
                                 **dict(zip(COVARIATES, pool_covariates[k]))})
    balance = pd.DataFrame(balance_rows)
    balance.to_csv(args.out / "rep1_covariate_support.csv", index=False)
    pd.DataFrame(control_rows).to_csv(args.out / "rep1_selected_control_ids.csv", index=False)
    control_freeze = {"frozen_at_utc": utc_now(), "diagnostic_plan_sha256": sha256(args.out / "diagnostic_plan_freeze.json"),
                      "rep2_opened_in_control_selection_or_model_fitting": False, "matching": matches,
                      "rep1_only_alternative_cutoffs": alternative_cutoffs,
                      "frozen_file_sha256": {name: sha256(args.out / name) for name in
                                              ("rep1_native_qc.csv", "rep1_covariate_support.csv", "rep1_selected_control_ids.csv", "rep1_sensitivity_models.npz")}}
    write_json(args.out / "rep1_control_freeze.json", control_freeze)
    print("Rep1 control IDs/support frozen; extracting rep2 without changing rules", flush=True)
    evaluation_started_at_utc = utc_now()
    maps2, qc2 = extract_native(args.rep2, chosen)
    qc2.to_csv(args.out / "rep2_native_qc.csv", index=False)
    sensitivity_rows, matched_rows = [], []
    arms = [("frozen_input_stress", variant) for variant in VARIANTS]
    arms += [("rep1_recalibrated_sensitivity", variant) for variant in VARIANTS]
    for regime, variant in arms:
        for metric in METRICS:
            v1 = residual_vectors(maps1[variant], metric)
            v2 = residual_vectors(maps2[variant], metric)
            recalibrated = regime == "rep1_recalibrated_sensitivity" and variant != "default_clip_5"
            model = alternative_models[variant] if recalibrated else archive
            x1, x2 = frozen_project(v1, model, metric), frozen_project(v2, model, metric)
            # Preserve original reference definition separately in each replicate;
            # rep2 references are transformed, never fitted, under the rep1 model.
            novelty1 = np.linalg.norm(x1[:, None] - x1[None, :known_n], axis=2).min(axis=1)
            novelty2 = np.linalg.norm(x2[:, None] - x2[None, :known_n], axis=2).min(axis=1)
            agreement = paired_r(v1, v2)
            cutoff = (alternative_cutoffs[variant][metric] if recalibrated
                      else float(summary["novelty_thresholds"][metric]))
            for row in candidates.itertuples():
                ci = candidate_positions[row.candidate_id]
                for replicate, novelty in (("rep1", novelty1), ("rep2", novelty2)):
                    sensitivity_rows.append({"candidate_id": row.candidate_id, "center_bp": int(row.center),
                                             "variant": variant, "model_regime": regime,
                                             "replicate": replicate, "metric": metric,
                                             "novelty": float(novelty[ci]), "frozen_cutoff": cutoff,
                                             "novelty_ratio": float(novelty[ci] / cutoff),
                                             "passes_rep1_frozen_cutoff": bool(novelty[ci] > cutoff),
                                             "paired_rep1_rep2_r": float(agreement[ci])})
                ids = control_idx[row.candidate_id]
                evaluable = matches[row.candidate_id]["evaluable"]
                matched_rows.append({"candidate_id": row.candidate_id, "variant": variant,
                                     "model_regime": regime, "metric": metric,
                                     "controls": len(ids), "evaluable": evaluable,
                                     "rep2_candidate_novelty": float(novelty2[ci]),
                                     "rep2_matched_novelty_percentile": descriptive_percentile(novelty2[ids], novelty2[ci], evaluable),
                                     "paired_candidate_r": float(agreement[ci]),
                                     "paired_matched_r_percentile": descriptive_percentile(agreement[ids], agreement[ci], evaluable),
                                     "status": "descriptive_supported_comparison" if evaluable else "not_evaluable_no_percentile"})
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(args.out / "candidate_normalization_sensitivity.csv", index=False)
    pd.DataFrame(matched_rows).to_csv(args.out / "candidate_matched_descriptive_comparison.csv", index=False)
    candidate_qc = pd.concat([qc1.assign(replicate="rep1"), qc2.assign(replicate="rep2")], ignore_index=True)
    candidate_qc = candidate_qc[candidate_qc.center_bp.isin(candidates.center)].copy()
    candidate_qc["candidate_id"] = candidate_qc.center_bp.map(candidates.set_index("center").candidate_id)
    candidate_qc.to_csv(args.out / "candidate_native_qc.csv", index=False)
    figure(args.out / "candidate_artifact_followup.png", candidates, balance, sensitivity, matches)
    for name, digest in plan["source_sha256"].items():
        if sha256(Path(name)) != digest:
            raise ValueError(f"Source changed during the fixed run: {name}")
    gates = {}
    for regime, variant in arms:
        values = sensitivity[(sensitivity.variant == variant) & (sensitivity.replicate == "rep2") &
                             (sensitivity.model_regime == regime)]
        gates.setdefault(regime, {})[variant] = {
            str(key): bool(g.passes_rep1_frozen_cutoff.all()) for key, g in values.groupby("candidate_id")}
    report = {"analysis": "Fixed four-candidate native-count artifact/QC follow-up",
              "diagnostic_plan_frozen_at_utc": plan["frozen_at_utc"],
              "rep1_controls_frozen_at_utc": control_freeze["frozen_at_utc"],
              "rep2_evaluation_started_at_utc": evaluation_started_at_utc,
              "completed_at_utc": utc_now(), "known_references": known_n,
              "independent_unannotated_pool": len(independent), "fixed_candidates": len(candidates),
              "rep1_default_map_reproduction_max_abs_error": reproduction_error,
              "matching": matches, "rep2_both_rep1_frozen_novelty_gates_by_regime_and_variant": gates,
              "original_candidate_set_and_thresholds_unchanged": True,
              "new_structural_type_claim": False, "p_values_or_fdr_claimed": False,
              "limits": plan["limits"] + [plan["matching"]["pool_caveat"],
                                          "Clipping stability does not prove a functional or biological class.",
                                          "Unsupported matching is retained as a failed comparator, not evidence of exceptional novelty."],
              "reproduce": "python -m scripts.task2_candidate_artifact_followup --out outputs/task2_followup_20261006_artifact",
              "output_sha256": {p.name: sha256(p) for p in args.out.iterdir() if p.is_file()}}
    write_json(args.out / "summary.json", report)
    args.results.mkdir(parents=True, exist_ok=True)
    evidence = []
    for source in args.out.iterdir():
        if source.is_file():
            target = args.results / ("task2_followup_20261006_" + source.name)
            if target.exists():
                raise ValueError(f"Refusing to overwrite published evidence: {target}")
            shutil.copyfile(source, target)
            evidence.append({"source": str(source).replace("\\", "/"),
                             "artifact": str(target).replace("\\", "/"), "sha256": sha256(target)})
    write_json(args.results / "task2_followup_20261006_evidence_index.json", {"artifacts": evidence})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
