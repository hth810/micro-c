"""Schema/figure/provenance addendum to the already frozen Task 2 audit.

Only reads published CSV/JSON, never COOL, sequences, or model fitting. Empty
controls stay empty; post-hoc factor diagnostics explain the original caliper
without changing it. All original sources and payloads remain byte-for-byte.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.task2_candidate_artifact_followup import (
    COVARIATES, INDEPENDENCE_BP, METRICS, VARIANTS, qc_covariates,
)
from scripts.task2_residual_motif_screen_20261006 import sha256, write_json

PREFIX = "task2_followup_20261006_"
NEW_PREFIX = PREFIX + "delivery_addendum_"
CONTROL_COLUMNS = ["candidate_id", "candidate_center_bp", "control_center_bp", *COVARIATES]


def schema_control_manifest(path: Path) -> pd.DataFrame:
    """Readable empty schema without overwriting the old newline-only artifact."""
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=CONTROL_COLUMNS)
    if set(frame.columns) != set(CONTROL_COLUMNS):
        raise ValueError("Unexpected original control manifest columns")
    return frame[CONTROL_COLUMNS]


def factor_diagnostics(qc: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Post-hoc explanation of fixed calipers, never candidate/control selection."""
    pool = qc[qc.role == "independent_unannotated"].copy()
    permitted = (np.abs(pool.center_bp.to_numpy(int)[:, None] -
                        candidates.center.to_numpy(int)[None, :]) >= INDEPENDENCE_BP).all(axis=1)
    x = qc_covariates(pool)[permitted]
    if not len(x):
        raise ValueError("No permitted pool for descriptive factor diagnostics")
    scale = np.maximum(np.quantile(x, .75, axis=0) - np.quantile(x, .25, axis=0), 1e-8)
    rows = []
    for row in candidates.itertuples():
        candidate_qc = qc[qc.center_bp == row.center]
        if len(candidate_qc) != 1:
            raise ValueError("Candidate QC must map to exactly one fixed locus")
        target = qc_covariates(candidate_qc)[0]
        within = np.abs(x - target) <= 2 * scale
        for j, name in enumerate(COVARIATES):
            rows.append({"candidate_id": row.candidate_id, "covariate": name,
                         "target_rep1": float(target[j]), "permitted_controls": len(x),
                         "pool_min": float(x[:, j].min()), "pool_q10": float(np.quantile(x[:, j], .1)),
                         "pool_q90": float(np.quantile(x[:, j], .9)), "pool_max": float(x[:, j].max()),
                         "within_2iqr_controls": int(within[:, j].sum()), "iqr_scale": float(scale[j]),
                         "joint_original_caliper_controls": int(within.all(axis=1).sum())})
    return pd.DataFrame(rows)


def plot(path: Path, candidates: pd.DataFrame, sensitivity: pd.DataFrame,
         factors: pd.DataFrame, matching: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(candidates), 2, figsize=(11.5, 11.5), dpi=160, squeeze=False)
    for i, candidate in enumerate(candidates.itertuples()):
        left, right = axes[i]
        audit = matching[candidate.candidate_id]
        rows = factors[(factors.candidate_id == candidate.candidate_id) &
                       (factors.covariate == "native_zero_offdiag_row_fraction")]
        if len(rows) != 1:
            raise ValueError("Missing zero-row factor diagnostic")
        zero = rows.iloc[0]
        left.set_axis_off()
        left.text(.5, .94, f"{candidate.candidate_id}: {int(candidate.center):,} bp", ha="center", va="top", fontsize=13)
        left.text(.5, .67, "No matched controls / not evaluable" if not audit["evaluable"]
                  else "Supported descriptive comparison", ha="center", va="center", fontsize=12, color="#963e36")
        left.text(.5, .34, f"Frozen 7-factor controls: {audit['selected_controls']} / {audit['permitted_controls']}\n"
                  f"Rep1 zero native contact rows: {zero.target_rep1:.1%}\n"
                  f"Permitted pool q10–q90: {zero.pool_q10:.1%}–{zero.pool_q90:.1%}",
                  ha="center", va="center", fontsize=11, linespacing=1.6)
        subset = sensitivity[(sensitivity.candidate_id == candidate.candidate_id) &
                             (sensitivity.replicate == "rep2") &
                             (sensitivity.model_regime == "rep1_recalibrated_sensitivity")]
        for j, metric in enumerate(METRICS):
            heights = []
            for variant in VARIANTS:
                values = subset[(subset.variant == variant) & (subset.metric == metric)]
                if len(values) != 1:
                    raise ValueError("Missing/duplicate fixed sensitivity row")
                heights.append(float(values.novelty_ratio.iloc[0]))
            right.bar(np.arange(3) + (j-.5)*.35, heights, width=.35,
                      color=("#377fa8", "#dc9937")[j], label=("High-pass", "Row-col")[j])
        right.axhline(1, color="gray", linestyle="--")
        right.set(xticks=np.arange(3), xticklabels=["Default\n[-5,5]", "No final\nclip", "Explicit\n[-2,4]"],
                  ylabel="Novelty / rep1 cutoff", title="Rep2; same rep1-known calibration rule")
        if i == 0:
            right.legend(loc="upper right", fontsize=9)
    fig.suptitle("Task 2: all four original candidates, no new selection\n"
                 "Coverage proxies lack matched support; sensitivity does not establish a new biological class", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .95), h_pad=2.0, w_pad=3.0)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    root = args.results
    output_names = ("empty_control_manifest.csv", "caliper_factor_diagnostic.csv", "qc_sensitivity.png",
                    "environment_receipt.json", "evidence_index.json")
    if any((root / (NEW_PREFIX + name)).exists() for name in output_names):
        raise ValueError("Do not overwrite an existing delivery addendum")
    old_index = root / (PREFIX + "evidence_index.json")
    index = json.loads(old_index.read_text(encoding="utf-8"))
    for item in index["artifacts"]:
        artifact = Path(item["artifact"])
        if sha256(artifact) != item["sha256"]:
            raise ValueError(f"Original frozen payload changed: {artifact}")
    candidates = pd.read_csv(root / (PREFIX + "fixed_candidates.csv"))
    qc = pd.read_csv(root / (PREFIX + "rep1_native_qc.csv"))
    sensitivity = pd.read_csv(root / (PREFIX + "candidate_normalization_sensitivity.csv"))
    control_freeze = json.loads((root / (PREFIX + "rep1_control_freeze.json")).read_text(encoding="utf-8"))
    plan = json.loads((root / (PREFIX + "diagnostic_plan_freeze.json")).read_text(encoding="utf-8"))
    controls = schema_control_manifest(root / (PREFIX + "rep1_selected_control_ids.csv"))
    if len(controls) != sum(x["selected_controls"] for x in control_freeze["matching"].values()):
        raise ValueError("Control manifest and freeze disagree")
    factors = factor_diagnostics(qc, candidates)
    for key, group in factors.groupby("candidate_id"):
        if not (group.joint_original_caliper_controls == control_freeze["matching"][key]["caliper_controls"]).all():
            raise ValueError("Post-hoc diagnostics do not reproduce the original caliper")
    controls.to_csv(root / (NEW_PREFIX + "empty_control_manifest.csv"), index=False)
    factors.to_csv(root / (NEW_PREFIX + "caliper_factor_diagnostic.csv"), index=False)
    plot(root / (NEW_PREFIX + "qc_sensitivity.png"), candidates, sensitivity, factors, control_freeze["matching"])
    source_paths = [Path(__file__).resolve().relative_to(Path.cwd()),
                    Path("tests/test_task2_artifact_addendum_followup.py"),
                    Path("scripts/task2_candidate_artifact_followup.py"),
                    Path("scripts/task2_residual_motif_screen_20261006.py")]
    environment = {
        "addendum_created_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_kind": "Post-run delivery/provenance clarification; not a new experiment",
        "original_device": "CPU, NumPy/SciPy/scikit-learn; no GPU path",
        "original_requested_numerical_threads": plan["threads"],
        "original_dependency_versions": plan["dependency_versions"],
        "source_sha256": {str(p).replace("\\", "/"): sha256(p) for p in source_paths},
        "original_evidence_index_sha256": sha256(old_index),
        "original_freeze_or_source_files_modified": False,
        "scientific_values_or_matching_rules_changed": False,
        "empty_manifest": "Original newline-only CSV remains frozen. New header-only manifest has zero rows and is readable by pandas.",
        "factor_diagnostics": "Post-hoc marginal count abs(pool-target)<=2*max(IQR,1e-8); excludes all four candidate neighborhoods, uses rep1 native-QC CSV only, and independently reproduces zero joint controls. This does not relax matching or constitute significance.",
        "limitations": ["Coverage proxies are not actual fragment coverage or aligner mappability.",
                        "All four comparisons remain not evaluable; percentile fields remain empty.",
                        "Alternative calibration used rep1 knowns only; original candidate set and failed A3 remain unchanged."],
        "reproduce": "python -m scripts.task2_artifact_addendum_followup",
    }
    write_json(root / (NEW_PREFIX + "environment_receipt.json"), environment)
    write_json(root / (NEW_PREFIX + "evidence_index.json"), {
        "files": {NEW_PREFIX + name: sha256(root / (NEW_PREFIX + name)) for name in output_names[:-1]},
        "source_sha256": environment["source_sha256"],
        "original_evidence_index_sha256": sha256(old_index),
    })
    print(json.dumps({"files": [NEW_PREFIX + name for name in output_names],
                      "matched_controls": len(controls), "original_artifacts_unchanged": True}, indent=2), flush=True)


if __name__ == "__main__":
    main()
