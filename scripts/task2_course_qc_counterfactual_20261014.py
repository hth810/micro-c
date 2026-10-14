"""Frozen counterfactual coverage stress and genuinely shared-depth sensitivity.

Artificially masking known structures is NOT a genomic null or biological
validation. This tests whether candidate-like coverage gaps can make known
structures look novel under the unchanged historical reference model. All
four candidate masks and all three depth levels are reported without choosing
the most favorable result. Existing raw count inputs only; no new training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from microc.data import annotations, area_resample, normalize, open_cool, window
from scripts.task2_course_verification_20261014 import (
    METRICS, SPACING, THIN_SEEDS, json_clean, paired_r, shape_features,
    template_evidence, thin_symmetric_counts,
)
from scripts.task2_residual_motif_screen_20261006 import (
    frozen_project, residual_vectors, sha256, write_json,
)

DEPTH_LEVELS = (10000, 20000, 30000)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def zero_row_mask(native):
    a = np.asarray(native, float)
    if a.shape != (512, 512) or not np.isfinite(a).all() or (a < 0).any() or not np.allclose(a, a.T):
        raise ValueError("Expected finite nonnegative symmetric 512-bin native counts")
    offdiag = a.sum(axis=1) - np.diag(a)
    return offdiag > 0


def apply_fixed_coverage_mask(native, keep):
    a, keep = np.asarray(native, float), np.asarray(keep, bool)
    if a.shape != (512, 512) or keep.shape != (512,) or not np.isfinite(a).all():
        raise ValueError("Invalid native matrix or mask")
    return a * keep[:, None] * keep[None, :]


def frozen_novelty(maps, known_maps1, archive, centers, known_centers):
    result = {}
    independent = np.abs(np.asarray(centers)[:, None] - np.asarray(known_centers)[None]) >= SPACING
    for metric in METRICS:
        x = frozen_project(residual_vectors(maps, metric), archive, metric)
        reference = frozen_project(residual_vectors(known_maps1, metric), archive, metric)
        distance = np.linalg.norm(x[:, None] - reference[None], axis=2)
        distance[~independent] = np.inf
        if not np.isfinite(distance.min(axis=1)).all():
            raise ValueError("No independent known reference")
        result[metric] = distance.min(axis=1)
    return result


def run(args):
    if args.out.exists():
        raise ValueError("Never overwrite a frozen diagnostic directory")
    candidates = pd.read_csv(args.candidates)
    known = annotations(args.annotations)
    if candidates.candidate_id.tolist() != ["A1", "A2", "A3", "A4"] or len(known) != 344:
        raise ValueError("Unexpected fixed candidate/reference manifest")
    base = np.load(args.base / "base_maps_counts.npz")
    base_summary = json.loads((args.base / "summary.json").read_text(encoding="utf-8"))
    for name in ("base_maps_counts.npz", "protocol_freeze.json", "rep1_control_and_depth_freeze.json"):
        if sha256(args.base / name) != base_summary["output_sha256"][name]:
            raise ValueError(f"Base receipt hash mismatch: {name}")
    # NpzFile access decompresses a whole array. Cache each once, not in a
    # per-reference/seed loop (the interrupted initial run exposed that cost).
    cached = {key: base[key] for key in ("maps1", "maps2", "counts1", "counts2", "candidate_indices")}
    base.close()
    base = cached
    archive = np.load(args.discovery / "rep1_maps_models.npz")
    discovery = json.loads((args.discovery / "discovery_summary.json").read_text(encoding="utf-8"))
    old_freeze = json.loads((args.discovery / "freeze_receipt.json").read_text(encoding="utf-8"))
    for name, digest in old_freeze["frozen_file_sha256"].items():
        if sha256(args.discovery / name) != digest:
            raise ValueError(f"Historical discovery freeze mismatch: {name}")
    args.out.mkdir(parents=True)
    source_files = [Path(__file__), Path("scripts/task2_course_verification_20261014.py"),
                    Path("tests/test_task2_course_verification_counterfactual_20261014.py"),
                    Path("microc/data.py"), Path("scripts/task2_residual_motif_screen_20261006.py")]
    paths = [args.rep1, args.rep2, args.annotations, args.candidates, args.base / "summary.json",
             args.base / "base_maps_counts.npz", args.discovery / "rep1_maps_models.npz"]
    protocol = {
        "frozen_at_utc": utc_now(), "parent_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p): sha256(p) for p in source_files}, "input_sha256": {str(p): sha256(p) for p in paths},
        "previous_rep2_inspection": True, "previous_base_diagnostic_observed": True,
        "purpose": "Counterfactual mechanism stress and complete-common-depth follow-up, not confirmation of new types",
        "fixed_masks": "Each candidate rep1 512-bin offdiagonal zero-row mask; freeze before extraction of known-template native counts. Apply identical fixed mask in both known-reference replicates; correlated artificial gaps can themselves induce replication.",
        "known_templates": "All 344 original annotations, unmasked and each of four fixed masks; do not pick exemplars based on outcomes",
        "novelty_model": "Unchanged original rep1 scaler/PCA and q90 cutoffs; references are unmasked known rep1. Exclude known references within 10240bp for all original/masked known comparisons, as in original calibration.",
        "novelty_cutoffs": discovery["novelty_thresholds"],
        "depth_levels": list(DEPTH_LEVELS), "depth_seeds": list(THIN_SEEDS),
        "depth_design": "Three predeclared equal-expected count levels after prior base counts showed all known/candidates in both replicates above 30000. Selection is retrospective based on technical count support, not shape outcomes. Every level/seed reported, no best-level selection. Exclude no candidate.",
        "limits": ["Masks are artificial counterfactual perturbations, not real background controls or new biological replicates.",
                   "No p-values, FDR or genuine genomic false-positive rate from artificial masks.",
                   "Count thinning controls observed depth only, not underlying map-quality or fragment coverage.",
                   "Known-reference independence mask is a calibration diagnostic, not a new candidate novelty threshold.",
                   "No blinded validation; historical repetition remains visible."],
    }
    write_json(args.out / "protocol_freeze.json", protocol)
    protocol_hash = sha256(args.out / "protocol_freeze.json")
    print(f"Counterfactual protocol frozen {protocol_hash}", flush=True)
    masks, rows = [], []
    chrom = candidates.chrom.iloc[0]
    with open_cool(args.rep1) as cool:
        for candidate in candidates.itertuples(index=False):
            native = window(cool, chrom, int(candidate.center), 5120, 512)
            keep = zero_row_mask(native)
            masks.append(keep)
            rows.append({"candidate_id": candidate.candidate_id, "center_bp": int(candidate.center),
                         "masked_native_rows": int((~keep).sum()), "zero_offdiag_row_fraction": float((~keep).mean())})
    masks = np.asarray(masks)
    np.savez_compressed(args.out / "rep1_fixed_masks.npz", keep_masks=masks, centers=candidates.center.to_numpy())
    pd.DataFrame(rows).to_csv(args.out / "mask_manifest.csv", index=False)
    write_json(args.out / "mask_freeze.json", {"frozen_at_utc": utc_now(), "protocol_sha256": protocol_hash,
                                               "rep2_used_for_masks": False, "mask_sha256": sha256(args.out / "rep1_fixed_masks.npz"),
                                               "mask_manifest_sha256": sha256(args.out / "mask_manifest.csv")})
    mask_freeze_hash = sha256(args.out / "mask_freeze.json")
    masked_maps = {}
    reproduction_error = {}
    for rep, cool_path in (("rep1", args.rep1), ("rep2", args.rep2)):
        all_maps = [[] for _ in range(5)]
        with open_cool(cool_path) as cool:
            for i, structure in enumerate(known.itertuples(index=False)):
                native = window(cool, chrom, int(structure.center), 5120, 512)
                all_maps[0].append(normalize(area_resample(native, 64)))
                for j, keep in enumerate(masks):
                    all_maps[j + 1].append(normalize(area_resample(apply_fixed_coverage_mask(native, keep), 64)))
                if (i + 1) % 100 == 0:
                    print(f"Coverage counterfactual {rep}: {i + 1}/344", flush=True)
        masked_maps[rep] = np.asarray(all_maps, np.float32)
        original = base["maps1" if rep == "rep1" else "maps2"][:344]
        reproduction_error[rep] = float(np.max(np.abs(masked_maps[rep][0] - original)))
        if reproduction_error[rep] > 1e-6:
            raise ValueError("Unmasked known maps fail original-count reproduction")
    known1 = base["maps1"][:344]
    novelty, repeat = {}, {}
    for rep in ("rep1", "rep2"):
        novelty[rep] = [frozen_novelty(m, known1, archive, known.center.to_numpy(), known.center.to_numpy())
                        for m in masked_maps[rep]]
    for j in range(5):
        repeat[j] = {m: paired_r(residual_vectors(masked_maps["rep1"][j], m),
                                residual_vectors(masked_maps["rep2"][j], m)) for m in METRICS}
    evidence = []
    for j, label in enumerate(["unmasked"] + candidates.candidate_id.tolist()):
        for i, structure in enumerate(known.itertuples(index=False)):
            row = {"mask_from_candidate": label, "known_id": structure.id, "known_type": structure.type,
                   "known_center_bp": int(structure.center), "highpass_repeat_r": repeat[j]["highpass"][i],
                   "rowcol_repeat_r": repeat[j]["rowcol"][i]}
            for rep in ("rep1", "rep2"):
                passes = []
                for m in METRICS:
                    row[f"{rep}_{m}_novelty"] = novelty[rep][j][m][i]
                    row[f"{rep}_{m}_passes_frozen_cutoff"] = bool(novelty[rep][j][m][i] > discovery["novelty_thresholds"][m])
                    passes.append(row[f"{rep}_{m}_passes_frozen_cutoff"])
                row[f"{rep}_dual_novelty"] = all(passes)
            row["both_replicates_dual_novelty"] = row["rep1_dual_novelty"] and row["rep2_dual_novelty"]
            evidence.append(row)
    evidence = pd.DataFrame(evidence)
    evidence.to_csv(args.out / "known_coverage_counterfactual.csv", index=False)
    summaries = []
    for label in ["unmasked"] + candidates.candidate_id.tolist():
        selected = evidence[evidence.mask_from_candidate == label]
        for kind in ["ALL", "CHID", "CHIN", "OPCID"]:
            subset = selected if kind == "ALL" else selected[selected.known_type == kind]
            summaries.append({"mask_from_candidate": label, "known_type": kind, "n": len(subset),
                              "rep1_dual_novelty_count": int(subset.rep1_dual_novelty.sum()),
                              "rep2_dual_novelty_count": int(subset.rep2_dual_novelty.sum()),
                              "both_replicates_dual_novelty_count": int(subset.both_replicates_dual_novelty.sum()),
                              "both_replicates_dual_novelty_fraction": float(subset.both_replicates_dual_novelty.mean()),
                              "median_highpass_repeat_r": float(subset.highpass_repeat_r.median()),
                              "median_rowcol_repeat_r": float(subset.rowcol_repeat_r.median()),
                              "interpretation": "artificial_counterfactual_not_genomic_null"})
    pd.DataFrame(summaries).to_csv(args.out / "counterfactual_summary.csv", index=False)
    print("Coverage masking complete; running all shared-depth levels", flush=True)
    ci = base["candidate_indices"]
    relevant = np.concatenate([np.arange(344), ci])
    centers = np.concatenate([known.center.to_numpy(), candidates.center.to_numpy()])
    a, b = np.triu_indices(64, k=1)
    t1 = base["counts1"][relevant][:, a, b].sum(axis=1)
    t2 = base["counts2"][relevant][:, a, b].sum(axis=1)
    if min(t1.min(), t2.min()) < max(DEPTH_LEVELS):
        raise ValueError("Predeclared equal-depth level exceeds common support")
    depth_rows = []
    for target in DEPTH_LEVELS:
        for seed in THIN_SEEDS:
            thin1, thin2 = [], []
            realized1, realized2 = [], []
            for idx in relevant:
                draw1, _, ok1 = thin_symmetric_counts(base["counts1"][idx], target, seed + int(idx) * 2)
                draw2, _, ok2 = thin_symmetric_counts(base["counts2"][idx], target, seed + int(idx) * 2 + 1)
                if not (ok1 and ok2):
                    raise ValueError("Incomplete technical depth support")
                thin1.append(normalize(draw1)); thin2.append(normalize(draw2))
                realized1.append(float(draw1[a, b].sum())); realized2.append(float(draw2[a, b].sum()))
            thin1, thin2 = np.asarray(thin1), np.asarray(thin2)
            diag1 = template_evidence(thin1, centers, 344, known.type.to_numpy())
            features2 = shape_features(thin2)
            z2 = (features2 - diag1["feature_median"]) / diag1["feature_scale"]
            dist2 = np.linalg.norm(z2[:, None] - z2[None, :344], axis=2)
            repetitions = {m: paired_r(residual_vectors(thin1, m), residual_vectors(thin2, m)) for m in METRICS}
            for j, candidate in enumerate(candidates.itertuples(index=False)):
                idx = 344 + j
                row = {"candidate_id": candidate.candidate_id, "target_expected_upper_count": target,
                       "seed": seed, "all_known_and_candidates_both_replicates_depth_supported": True,
                       "candidate_rep1_realized_upper_count": realized1[idx], "candidate_rep2_realized_upper_count": realized2[idx],
                       "highpass_repeat_r": repetitions["highpass"][idx], "rowcol_repeat_r": repetitions["rowcol"][idx],
                       "rep1_nearest_known_feature_distance": diag1["distances"][idx].min(),
                       "rep2_nearest_known_feature_distance": dist2[idx].min(),
                       "rep1_known_q90_distance": diag1["distance_cutoff_rep1_known_q90"],
                       "rep2_beyond_rep1_known_feature_q90": bool(dist2[idx].min() > diag1["distance_cutoff_rep1_known_q90"])}
                depth_rows.append(row)
        print(f"Shared expected depth {target} complete", flush=True)
    depth = pd.DataFrame(depth_rows)
    depth.to_csv(args.out / "complete_common_depth_sensitivity.csv", index=False)
    depth_summary = depth.groupby(["candidate_id", "target_expected_upper_count"], as_index=False).agg(
        highpass_repeat_r_mean=("highpass_repeat_r", "mean"), highpass_repeat_r_min=("highpass_repeat_r", "min"),
        highpass_repeat_r_max=("highpass_repeat_r", "max"), rowcol_repeat_r_mean=("rowcol_repeat_r", "mean"),
        rep2_beyond_rep1_known_feature_q90_fraction=("rep2_beyond_rep1_known_feature_q90", "mean"))
    depth_summary.to_csv(args.out / "complete_common_depth_summary.csv", index=False)
    plot(args.out, pd.DataFrame(summaries), depth_summary)
    summary = {
        "completed_at_utc": utc_now(), "protocol_sha256": protocol_hash, "mask_freeze_sha256": mask_freeze_hash,
        "mask_frozen_before_known_reference_counterfactual_extraction": True,
        "rep2_used_for_mask_selection": False, "prospectively_blind": False,
        "unmasked_count_reproduction_max_abs_error": reproduction_error,
        "counterfactual_all_known": [r for r in summaries if r["known_type"] == "ALL"],
        "complete_common_depth_levels": list(DEPTH_LEVELS), "minimum_observed_known_candidate_upper_counts": float(min(t1.min(), t2.min())),
        "all_reference_candidate_inputs_support_all_depth_levels": True,
        "depth_summary": json_clean(depth_summary.to_dict(orient="records")),
        "artificial_masks_are_null_or_validation": False, "p_values_or_fdr_claimed": False,
        "confirmed_new_structure_types": 0, "limitations": protocol["limits"],
        "source_table_binding": {"counterfactual_and_depth.png": ["counterfactual_summary.csv", "complete_common_depth_summary.csv"]},
        "output_sha256": {p.name: sha256(p) for p in args.out.iterdir() if p.is_file()},
    }
    write_json(args.out / "summary.json", json_clean(summary))
    args.results.mkdir(parents=True, exist_ok=True)
    for name in ["summary.json", "protocol_freeze.json", "mask_freeze.json", "mask_manifest.csv",
                 "known_coverage_counterfactual.csv", "counterfactual_summary.csv", "complete_common_depth_sensitivity.csv",
                 "complete_common_depth_summary.csv", "counterfactual_and_depth.png"]:
        shutil.copyfile(args.out / name, args.results / ("task2_course_verification_20261014_qc_" + name))
    print(json.dumps(json_clean({"completed": True, "protocol_sha256": protocol_hash,
                                "counterfactual_all_known": summary["counterfactual_all_known"],
                                "depth_summary": summary["depth_summary"]}), indent=2), flush=True)


def plot(out, counterfactual, depth):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    data = counterfactual[counterfactual.known_type == "ALL"]
    axes[0].bar(data.mask_from_candidate, data.both_replicates_dual_novelty_count, color=["#a6a6a6", "#377fa8", "#dc8c37", "#a75da1", "#46a28b"])
    axes[0].set(title="Known templates made shape-outlying by fixed masks", ylabel="Dual-novelty pass in both repeats (of 344)")
    axes[0].text(.02, .98, "Artificial counterfactual: not a genomic null", transform=axes[0].transAxes, va="top", fontsize=9)
    for candidate in ["A1", "A2", "A3", "A4"]:
        d = depth[depth.candidate_id == candidate]
        axes[1].plot(d.target_expected_upper_count / 1000, d.highpass_repeat_r_mean, "o-", label=candidate)
        axes[1].fill_between(d.target_expected_upper_count / 1000, d.highpass_repeat_r_min, d.highpass_repeat_r_max, alpha=.12)
        axes[2].plot(d.target_expected_upper_count / 1000, d.rep2_beyond_rep1_known_feature_q90_fraction, "o-", label=candidate)
    axes[1].set(title="Equal-expected-depth repeat stability", xlabel="Expected unique pair count (thousands)", ylabel="Highpass correlation", ylim=(0, 1))
    axes[2].set(title="Interpretable feature outlier sensitivity", xlabel="Expected unique pair count (thousands)", ylabel="Seeds beyond rep1 known q90", ylim=(-.05, 1.05))
    axes[1].legend(ncol=2, fontsize=9)
    fig.savefig(out / "counterfactual_and_depth.png", dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rep1", type=Path, default=Path("data/raw/GSE272159_37C_rep1.mapq_30.10.cool"))
    p.add_argument("--rep2", type=Path, default=Path("data/raw/GSE272159_37C_rep2.mapq_30.10.cool"))
    p.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    p.add_argument("--candidates", type=Path, default=Path("results/task2_improve_20261006_individual_candidate_validation.csv"))
    p.add_argument("--discovery", type=Path, default=Path("outputs/task2_improve_20261006/area"))
    p.add_argument("--base", type=Path, default=Path("outputs/task2_course_verification_20261014"))
    p.add_argument("--out", type=Path, default=Path("outputs/task2_course_verification_20261014_qc_counterfactual_v2"))
    p.add_argument("--results", type=Path, default=Path("results"))
    with threadpool_limits(limits=2):
        run(p.parse_args())


if __name__ == "__main__":
    main()
