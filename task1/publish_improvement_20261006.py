"""Publish compact task 1 evidence, unchanged frozen protocols and OOF plots."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from task1.evaluate_nested_neural_20261006 import sha256
from task1.run_task1 import CLASSES


def run(args: argparse.Namespace) -> dict:
    root = args.inputs
    result = args.results
    result.mkdir(parents=True, exist_ok=True)
    prefix = "task1_improve_20261006_"
    copied = []
    for run in ("nested_neural_seed21", "roi_cnn_seed21", "roi_cnn_seed22", "hybrid_seed21", "hybrid_seed22", "roi_cnn_ablation"):
        for name in ("protocol.json", "summary.json", "outer_predictions.csv", "fold_choices.csv", "inner_options.csv"):
            source = root / run / name
            if source.is_file():
                target = result / f"{prefix}{run}_{name}"
                shutil.copy2(source, target)
                copied.append(target.name)
    for name in ("audit_protocol.json", "audit_summary.json", "oof_occlusion_samples.csv", "occlusion_matching.csv",
                 "oof_saliency_by_class.png", "final_interval_roi_cnn.pt", "final_interval_roi_cnn.json"):
        target = result / f"{prefix}{name}"
        shutil.copy2(root / "roi_cnn_audit" / name, target)
        copied.append(target.name)
    summaries = {run: json.loads((root / run / "summary.json").read_text(encoding="utf-8"))
                 for run in ("nested_neural_seed21", "roi_cnn_seed21", "roi_cnn_seed22", "hybrid_seed21", "hybrid_seed22", "roi_cnn_ablation")}
    score_rows = [{"run": run, "model": model, **{key: value for key, value in score.items()
                                               if key in ("accuracy", "macro_f1", "chid_precision", "chid_recall")}}
                  for run, summary in summaries.items() for model, score in summary["scores"].items()]
    score_path = result / f"{prefix}score_table.csv"
    pd.DataFrame(score_rows).to_csv(score_path, index=False)
    copied.append(score_path.name)
    original = summaries["roi_cnn_seed21"]["scores"]["published_interval_cnn"]
    new = summaries["roi_cnn_seed21"]["scores"]["interval_roi64"]
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for axis, title, score in zip(axes, ("Previous interval CNN", "CNN + ROI + boundary descriptors"), (original, new)):
        cm = np.asarray(score["confusion_matrix"])
        norm = cm / cm.sum(axis=1, keepdims=True)
        axis.imshow(norm, vmin=0, vmax=1, cmap="Blues")
        for i in range(3):
            for j in range(3):
                axis.text(j, i, f"{cm[i,j]}\n{norm[i,j]:.1%}", ha="center", va="center",
                          color="white" if norm[i,j] > 0.5 else "black")
        axis.set_xticks(range(3), CLASSES)
        axis.set_yticks(range(3), CLASSES)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Original annotation class")
        axis.set_title(f"{title}\nAccuracy {score['accuracy']:.3f}; macro F1 {score['macro_f1']:.3f}", fontsize=10)
    figure.suptitle("Same five spatial folds: primary retrospective OOF comparison")
    plot = result / f"{prefix}confusion_comparison.png"
    figure.savefig(plot, dpi=170)
    plt.close(figure)
    copied.append(plot.name)
    occlusion = pd.read_csv(root / "roi_cnn_audit" / "oof_occlusion_samples.csv")
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.boxplot([occlusion.loc[occlusion.type.eq(label), "inside_minus_outside_drop"] for label in CLASSES],
                 tick_labels=CLASSES, showfliers=False)
    axis.axhline(0, color="black", linewidth=1)
    axis.set_ylabel("True-class probability drop: inside minus outside")
    axis.set_title("OOF contact occlusion: exact global diagonal-distance matching")
    plot = result / f"{prefix}occlusion_matched.png"
    figure.savefig(plot, dpi=170)
    plt.close(figure)
    copied.append(plot.name)
    # Preserve historical frozen protocols; make the timing correction explicit.
    cnn_completed = summaries["roi_cnn_seed21"]["completed_at_utc"]
    hybrid_protocol = json.loads((root / "hybrid_seed21" / "protocol.json").read_text(encoding="utf-8"))
    note = {"primary_model_name": "CNN + interval ROI + explicit contact-boundary descriptors",
            "primary_result": new,
            "scope": "Original single-label three-class classification of supplied known annotated intervals; not a genome-wide unknown-interval classifier",
            "sample_counts": {"CHIN": 250, "OPCID": 68, "CHID": 26},
            "historical_data_reuse": True,
            "second_seed_interpretation": "Same annotations, alternative spatial fold grouping and model initializations; sensitivity analysis, not new blind data",
            "hybrid_timing_addendum": {"cnn_summary_completed_at_utc": cnn_completed,
                                       "hybrid_protocol_frozen_at_utc": hybrid_protocol["frozen_at_utc"],
                                       "meaning": "Hybrid freeze was after the primary CNN computation finished. It is a retrospective fixed equal-weight auxiliary comparison; the freeze cannot establish it preceded CNN outer results or inspection."},
            "original_cnn_protocol_addendum": {"actual_initializations": 2,
                                                 "scoring_text_correction": "Inherited field says 3; actual source, offsets0/31 and computation use 2",
                                                 "original_frozen_protocols_retained": True},
            "contact_evidence": "OOF matched inside/outside occlusion excludes first two diagonals and preserves the interval mask/length; contact-derived side summaries update; effect is perturbation evidence, not functional proof",
            "inference_checkpoint": "Fits all supplied annotations; no in-sample accuracy is reported",
            "failed_model_retained": "Nested descriptor MLP macro F1 .719 versus previous CNN .756 on matching folds; no improvement claim",
            "occlusion_unmatched_records_definition": "audit_summary.unmatched_samples counts scale x replicate records, not unique structures. All structures retain long-scale matches; some short windows fully contained in annotated intervals cannot supply outside contacts.",
            "published_artifact_sha256": {name: sha256(result / name) for name in copied}}
    (result / f"{prefix}evidence_index.json").write_text(json.dumps(note, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Published {len(copied) + 1} small Task 1 evidence artifacts.")
    return note


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("outputs/task1_improve_20261006"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
