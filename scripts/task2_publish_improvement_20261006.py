"""Publish compact, hash-bound Task 2 improvement evidence without raw maps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from microc.data import area_window, normalize, open_cool
from scripts.task2_residual_motif_screen_20261006 import sha256, write_json
from scripts.task2_resampling_comparison_20261006 import check_freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("outputs/task2_improve_20261006"))
    parser.add_argument("--rep2", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    prefix = "task2_improve_20261006_"
    index = []
    arms = {}
    for arm in ("legacy", "area"):
        folder = args.base / arm
        freeze = check_freeze(folder)
        discovery = json.loads((folder / "discovery_summary.json").read_text(encoding="utf-8"))
        validation = json.loads((folder / "validation/validation_summary.json").read_text(encoding="utf-8"))
        if validation["freeze_receipt_sha256"] != sha256(folder / "freeze_receipt.json"):
            raise ValueError("Validation and discovery freezes do not correspond")
        if validation["rep2_sha256"] != sha256(args.rep2):
            raise ValueError("Current replicate differs from validated raw input")
        for name, digest in validation["output_sha256"].items():
            if sha256(folder / "validation" / name) != digest:
                raise ValueError("Validation output changed since receipt")
        sources = [folder / name for name in (
            "protocol.json", "freeze_receipt.json", "discovery_summary.json", "candidate_freeze.csv",
            "historical_diagnostic_freeze.csv", "rep1_all_windows.csv", "rep1_independent_unannotated.csv",
            "rep1_all_eligible_triads.csv")]
        sources += list((folder / "validation").glob("*.csv"))
        sources += list((folder / "validation").glob("*.png"))
        sources += [folder / "validation/validation_summary.json"]
        for source in sources:
            target = args.results / (prefix + arm + "_" + source.name)
            shutil.copyfile(source, target)
            index.append({"source": str(source).replace("\\", "/"),
                          "artifact": str(target).replace("\\", "/"), "sha256": sha256(target)})
        edges = pd.read_csv(folder / "validation/candidate_edge_validation.csv")
        independent = pd.read_csv(folder / "rep1_independent_unannotated.csv")
        genome = pd.read_csv(folder / "validation/genome_window_replication.csv")
        frozen_loci = independent[independent.passes_both_novelty_gates].center
        evidence = genome[genome.center.isin(frozen_loci)].copy()
        evidence["rep2_passes_both_frozen_novelty_gates"] = (
            (evidence.rep2_highpass_novelty > evidence.frozen_highpass_novelty_threshold) &
            (evidence.rep2_rowcol_novelty > evidence.frozen_rowcol_novelty_threshold))
        arms[arm] = {
            "frozen_at_utc": freeze["frozen_at_utc"],
            "genome_windows": discovery["genome_windows"],
            "full_window_unannotated_background_loci": discovery["full_window_unannotated_background_loci"],
            "independent_background_pairs": discovery["background_independent_pairs"],
            "independent_unannotated_representatives": discovery["independent_unannotated_representatives"],
            "rep1_frozen_individual_outlying_loci": len(evidence),
            "individual_loci_passing_both_frozen_rep2_novelty_gates": int(evidence.rep2_passes_both_frozen_novelty_gates.sum()),
            "rep1_frozen_three_locus_motifs": validation["frozen_disjoint_motifs"],
            "motifs_passing_exploratory_replication_rule": validation["motifs_passing_rule"],
            "known_rep2_reference_agreement": validation["known_rep2_recovery"],
            "historical_edge_controls_minimum": int(edges.matched_controls.min()),
            "historical_edges_with_full_covariate_support": int(edges.drop_duplicates(["center_a_bp", "center_b_bp"]).balanced_all_covariates.sum()),
            "historical_total_edges": len(edges.drop_duplicates(["center_a_bp", "center_b_bp"])),
            "historical_edge_min_rep2_matched_percentile": float(edges.rep2_matched_percentile.min()),
            "historical_edge_max_rep2_matched_percentile": float(edges.rep2_matched_percentile.max()),
            "historical_diagnostic_passes": validation["historical_diagnostics_passing_rule"],
            "genome_wide_context": validation["genome_wide_novelty_context"],
        }
        if arm == "area":
            area_evidence = evidence
    source = args.base / "resampling_comparison"
    comparison = json.loads((source / "comparison_summary.json").read_text(encoding="utf-8"))
    for name, digest in comparison["output_sha256"].items():
        if sha256(source / name) != digest:
            raise ValueError("Paired ablation output changed")
    for original in source.glob("*"):
        if original.is_file():
            target = args.results / (prefix + original.name)
            shutil.copyfile(original, target)
            index.append({"source": str(original).replace("\\", "/"),
                          "artifact": str(target).replace("\\", "/"), "sha256": sha256(target)})
    # These IDs follow the prevalidation rep1 manifest order, not a rep2 ranking.
    manifest = pd.read_csv(args.base / "area/rep1_independent_unannotated.csv")
    order = manifest[manifest.passes_both_novelty_gates].center.to_list()
    evidence = area_evidence.set_index("center").loc[order].reset_index()
    evidence.insert(0, "candidate_id", [f"A{i+1}" for i in range(len(evidence))])
    evidence.insert(2, "start_bp", evidence.center - 2560)
    evidence.insert(3, "end_bp", evidence.center + 2560)
    paired = pd.read_csv(source / "paired_reproducibility.csv")
    for metric in ("upper", "highpass", "rowcol"):
        values = paired[(paired.arm == "area") & (paired.metric == metric)].set_index("center_bp").replicate_r
        evidence[f"paired_rep1_rep2_{metric}_r"] = evidence.center.map(values)
    evidence["interpretation"] = "Independent shape-outlying locus; no validated new type or genomic FDR"
    candidate_csv = args.results / (prefix + "individual_candidate_validation.csv")
    evidence.to_csv(candidate_csv, index=False)
    candidate_figure = args.results / (prefix + "individual_candidates.png")
    plot(evidence, args.base / "area/rep1_maps_models.npz", args.rep2, candidate_figure)
    summary = {
        "analysis": "Count-preserving measurement, expanded background and frozen residual novelty audit",
        "arms": arms,
        "paired_measurement_ablation": comparison["reports"],
        "count_conservation_max_absolute_error": comparison["source_counts_conserved_max_absolute_error"],
        "individual_area_candidates": evidence[["candidate_id", "center", "start_bp", "end_bp",
                                                "rep2_passes_both_frozen_novelty_gates"]].to_dict("records"),
        "validated_new_biological_types": 0,
        "historical_rep2_inspection": True,
        "rep2_used_in_current_candidate_or_threshold_selection": False,
        "improvement": "All source contact counts enter area pooling, known-reference shape agreement improves and three rep1-frozen independent loci retain residual shape novelty in rep2. Broader genome controls now expose the old clique's failure instead of relying on an empty comparator set.",
        "limits": ["Three residual-novel loci do not form the required coherent three-locus class.",
                   "Individual locus novelty is not adjusted for mappability, fragment coverage or normalization clipping; the seven-factor matched-control audit applies to the historical clique, not a significance test for the four individual loci.",
                   "Area pooling does not improve every metric in unannotated windows; row/column-residual reproducibility declines there.",
                   "Nearest annotated-type agreement is a descriptive morphology check, not classifier holdout accuracy; CHID agreement remains low.",
                   "Part of the old clique's control sets still lack full covariate support. No p-value or FDR is claimed.",
                   "Rep1 pooling ablation and thresholds were frozen before this run's rep2 reads, but past project analyses already inspected rep2; no prospective blindness is claimed."],
        "reproduce": [
            "python -m scripts.task2_residual_motif_screen_20261006 discover --rep1 data/raw/GSE272159_37C_rep1.mapq_30.10.cool --resampling legacy --out outputs/task2_improve_20261006/legacy",
            "python -m scripts.task2_residual_motif_screen_20261006 discover --rep1 data/raw/GSE272159_37C_rep1.mapq_30.10.cool --resampling area --out outputs/task2_improve_20261006/area",
            "python -m scripts.task2_residual_motif_screen_20261006 validate --rep2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool --discovery outputs/task2_improve_20261006/legacy --out outputs/task2_improve_20261006/legacy/validation",
            "python -m scripts.task2_residual_motif_screen_20261006 validate --rep2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool --discovery outputs/task2_improve_20261006/area --out outputs/task2_improve_20261006/area/validation",
            "python -m scripts.task2_resampling_comparison_20261006 --rep2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool",
            "python -m scripts.task2_publish_improvement_20261006 --rep2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool"],
        "publish_script_sha256": sha256(Path(__file__)),
    }
    summary_path = args.results / (prefix + "summary.json")
    write_json(summary_path, summary)
    for p in (candidate_csv, candidate_figure, summary_path):
        index.append({"artifact": str(p).replace("\\", "/"), "sha256": sha256(p)})
    write_json(args.results / (prefix + "evidence_index.json"), {"artifacts": index})
    print(json.dumps(summary, indent=2), flush=True)


def plot(evidence, archive_path, rep2, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    archive = np.load(archive_path)
    pos = {int(c): i for i, c in enumerate(archive["centers"])}
    fig, axes = plt.subplots(len(evidence), 3, figsize=(12, 3 * len(evidence)), dpi=160, squeeze=False)
    with open_cool(rep2) as cool:
        for i, row in enumerate(evidence.itertuples()):
            m1 = archive["maps"][pos[int(row.center)]]
            m2 = normalize(area_window(cool, str(row.chrom), int(row.center), 5120, 64))
            for col, (label, mat) in enumerate((("rep1", m1), ("rep2", m2))):
                axes[i, col].imshow(mat, origin="lower", cmap="magma", vmin=-2, vmax=4)
                axes[i, col].set_title(f"{row.candidate_id}: {int(row.center):,} bp / {label}")
            vals = [row.rep1_highpass_novelty / row.frozen_highpass_novelty_threshold,
                    row.rep2_highpass_novelty / row.frozen_highpass_novelty_threshold,
                    row.rep1_rowcol_novelty / row.frozen_rowcol_novelty_threshold,
                    row.rep2_rowcol_novelty / row.frozen_rowcol_novelty_threshold]
            axes[i, 2].bar(["HP rep1", "HP rep2", "RC rep1", "RC rep2"], vals,
                           color=["#216fa1", "#d98729", "#216fa1", "#d98729"])
            axes[i, 2].axhline(1, linestyle="--", color="gray")
            axes[i, 2].set(ylabel="Distance / frozen rep1 cutoff", title="Both rep2 gates pass" if row.rep2_passes_both_frozen_novelty_gates else "Rep2 novelty failed")
            axes[i, 2].tick_params(axis="x", labelrotation=18)
    fig.suptitle("Rep1-frozen area-pooled individual loci; no shared new type established")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
