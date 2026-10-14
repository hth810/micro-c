"""Auxiliary fixed-weight, closed-set classifier context for frozen candidates.

The candidate windows are provisional ROIs, not independently identified
biological boundaries. Same-width known controls expose this mismatch. The
Task 1 model was trained on all known annotations, so its reference inference
is NOT held-out performance. Softmax is not an OOD test or proof of novelty.
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
import torch
from threadpoolctl import threadpool_limits

from microc.data import annotations, open_cool
from scripts.task2_course_verification_20261014 import json_clean
from scripts.task2_residual_motif_screen_20261006 import sha256, write_json
from task1.audit_roi_cnn_20261006 import predict_states, side_features
from task1.evaluate_roi_cnn_20261006 import make_images
from task1.run_task1 import CLASSES, extract_window


def provisional_roi(table, width_bp=5120):
    result = table.copy().reset_index(drop=True)
    if width_bp != 5120 or "center" not in result or result.id.duplicated().any():
        raise ValueError("Use unique IDs and the original frozen 5120-bp windows")
    result["start"], result["end"] = result.center - 2560, result.center + 2560
    result["length_bp"] = 5120
    return result


def infer(images, table, models, role):
    image = make_images(images, table)["interval_roi64"]
    side = side_features(images, table)
    probabilities = predict_states(image, side, models).reshape(2, len(table), 3)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=2), 1, atol=1e-5):
        raise ValueError("Invalid fixed-model probabilities")
    rows = []
    for i, annotation in enumerate(table.itertuples(index=False)):
        mean = probabilities[:, i].mean(axis=0)
        row = {"id": annotation.id, "role": role, "roi_start_bp": int(annotation.start),
               "roi_end_bp": int(annotation.end), "roi_length_bp": int(annotation.length_bp),
               "known_truth": annotation.type if role.startswith("known_") else None,
               "most_similar_closed_set_label": CLASSES[int(mean.argmax())],
               "replicate_label_agreement": bool(probabilities[0, i].argmax() == probabilities[1, i].argmax()),
               "maximum_mean_softmax": float(mean.max()), "softmax_is_ood_probability": False}
        for k, name in enumerate(CLASSES):
            row[f"mean_softmax_{name}"] = float(mean[k])
            row[f"rep1_softmax_{name}"] = float(probabilities[0, i, k])
            row[f"rep2_softmax_{name}"] = float(probabilities[1, i, k])
        rows.append(row)
    return rows


def run(args):
    if args.out.exists():
        raise ValueError("Never overwrite classifier-context protocol")
    candidates = pd.read_csv(args.candidates).rename(columns={"candidate_id": "id", "start_bp": "start", "end_bp": "end"})
    known = annotations(args.annotations)
    if candidates.id.tolist() != ["A1", "A2", "A3", "A4"] or len(known) != 344:
        raise ValueError("Preserve the four fixed candidates and complete known references")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if sha256(args.checkpoint) != config["checkpoint_sha256"] or sha256(args.known_windows) != config["input_sha256"]["windows"]:
        raise ValueError("Fixed Task 1 checkpoint/windows hashes disagree with config")
    if sha256(args.annotations) != config["input_sha256"]["annotations"]:
        raise ValueError("Known annotations differ from model-training source")
    args.out.mkdir(parents=True)
    candidate_roi, known_fixed_roi = provisional_roi(candidates), provisional_roi(known)
    if not np.array_equal(candidate_roi[["start", "end"]].to_numpy(), candidates[["start", "end"]].to_numpy()):
        raise ValueError("Candidate interval differs from frozen Task 2 bounds")
    candidate_roi[["id", "center", "start", "end", "length_bp"]].to_csv(args.out / "fixed_candidate_provisional_rois.csv", index=False)
    known_fixed_roi.to_csv(args.out / "known_same_width_provisional_rois.csv", index=False)
    source = [Path(__file__), Path("tests/test_task2_course_verification_classifier_20261014.py"),
              Path("task1/predict_roi_cnn_20261006.py"), Path("task1/evaluate_roi_cnn_20261006.py"),
              Path("task1/evaluate_interval_conditioned_cnn.py"), Path("task1/audit_roi_cnn_20261006.py"), Path("task1/run_task1.py")]
    inputs = [args.candidates, args.annotations, args.known_windows, args.checkpoint, args.config, args.rep1, args.rep2]
    protocol = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p): sha256(p) for p in source}, "input_sha256": {str(p): sha256(p) for p in inputs},
        "candidate_selection": "All four pre-existing fixed windows; no candidate/ROI/type threshold selection",
        "previous_rep2_and_qc_results_observed": True, "prospectively_blind": False,
        "model": "Unchanged trusted final Task 1 two-initialization interval-ROI CNN ensemble; no fitting",
        "model_requires_true_annotated_intervals": True,
        "candidate_roi_policy": "Use original 5120bp Task 2 windows as provisional ROIs only. No claim these are biological boundaries.",
        "known_controls": "All 344 known references: true supplied ROIs and matched-width center +/-2560bp provisional ROIs, same contact inputs",
        "contact_inputs": "Exact Task 1 extract_window: 128 bins average-pooling 4/16 native 10bp bins into 5.12/20.48kb inputs, independently normalized. Known cache verified by hash and one real extraction per type/replicate.",
        "interpretation": "Auxiliary known-class similarity context only. Closed-set softmax never rejects all known classes, is not calibrated OOD probability, and cannot prove a new structure.",
        "limits": ["Known model-fit examples are not independent tests or new accuracy estimates.",
                   "Fixed-length provisional ROIs shift the model's boundary and length features away from annotated training inputs.",
                   "No classifier confidence threshold, novelty threshold, candidate selection, or biological label is changed.",
                   "This is not an annotation-free scan or validated unknown-type classifier."],
    }
    write_json(args.out / "protocol_freeze.json", protocol)
    protocol_hash = sha256(args.out / "protocol_freeze.json")
    print(f"Classifier-context protocol frozen {protocol_hash}", flush=True)
    cached = np.load(args.known_windows)
    known_images = cached["X"].astype(np.float32)
    if not np.array_equal(cached["group"].astype(str), np.tile(known.id.to_numpy(), 2)):
        raise ValueError("Known cache IDs do not align with annotations")
    candidate_images, extraction_errors = [], []
    known_sample_indices = [int(np.flatnonzero(known.type.to_numpy() == kind)[0]) for kind in CLASSES]
    for rep, cool_path in enumerate((args.rep1, args.rep2)):
        with open_cool(cool_path) as cool:
            for candidate in candidates.itertuples(index=False):
                candidate_images.append(np.stack([extract_window(cool, int(candidate.center), 128, aggregate=4),
                                                 extract_window(cool, int(candidate.center), 128, aggregate=16)]))
            for i in known_sample_indices:
                source = np.stack([extract_window(cool, int(known.center.iloc[i]), 128, aggregate=4),
                                   extract_window(cool, int(known.center.iloc[i]), 128, aggregate=16)])
                extraction_errors.append(float(np.max(np.abs(source - known_images[rep * 344 + i]))))
    if max(extraction_errors) > 1e-6:
        raise ValueError("Exact Task 1 extraction does not reproduce known cache")
    candidate_images = np.asarray(candidate_images, np.float32)
    np.savez_compressed(args.out / "candidate_task1_contact_inputs.npz", X=candidate_images,
                        group=np.tile(candidates.id.to_numpy(), 2))
    # Only load the trusted locally exported checkpoint after hash verification.
    bundle = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if tuple(bundle["classes"]) != tuple(CLASSES) or len(bundle["models"]) != 2:
        raise ValueError("Unexpected trusted model classes/ensemble size")
    rows = infer(candidate_images, candidate_roi, bundle["models"], "candidate_provisional_roi")
    rows += infer(known_images, known, bundle["models"], "known_true_roi_model_fit_reference")
    rows += infer(known_images, known_fixed_roi, bundle["models"], "known_same_width_provisional_roi")
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "fixed_classifier_context.csv", index=False)
    candidates_out = table[table.role == "candidate_provisional_roi"]
    candidates_out.to_csv(args.out / "candidate_classifier_context.csv", index=False)
    summaries = []
    for role in ("known_true_roi_model_fit_reference", "known_same_width_provisional_roi"):
        selected = table[table.role == role]
        for kind in CLASSES:
            subset = selected[selected.known_truth == kind]
            summaries.append({"role": role, "known_truth": kind, "n": len(subset),
                              "same_label_fraction_descriptive_only": float((subset.most_similar_closed_set_label == kind).mean()),
                              "median_maximum_mean_softmax": float(subset.maximum_mean_softmax.median()),
                              "not_held_out_performance": True})
    pd.DataFrame(summaries).to_csv(args.out / "known_roi_context_summary.csv", index=False)
    summary = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": protocol_hash,
               "checkpoint_unchanged": True, "trained_new_models": False, "candidate_count": 4,
               "known_cache_reproduction_max_abs_error": max(extraction_errors),
               "candidate_closed_set_context": candidates_out.to_dict(orient="records"), "known_roi_context": summaries,
               "supports_new_type_proof": False, "supports_ood_detection": False,
               "limitations": protocol["limits"],
               "output_sha256": {p.name: sha256(p) for p in args.out.iterdir() if p.is_file()}}
    write_json(args.out / "summary.json", json_clean(summary))
    args.results.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "protocol_freeze.json", "candidate_classifier_context.csv", "known_roi_context_summary.csv", "fixed_classifier_context.csv"):
        shutil.copyfile(args.out / name, args.results / ("task2_course_verification_20261014_classifier_" + name))
    print(json.dumps(json_clean({"completed": True, "candidate_context": summary["candidate_closed_set_context"],
                                "known_context": summaries}), indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rep1", type=Path, default=Path("data/raw/GSE272159_37C_rep1.mapq_30.10.cool"))
    p.add_argument("--rep2", type=Path, default=Path("data/raw/GSE272159_37C_rep2.mapq_30.10.cool"))
    p.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    p.add_argument("--candidates", type=Path, default=Path("results/task2_improve_20261006_individual_candidate_validation.csv"))
    p.add_argument("--known-windows", type=Path, default=Path("outputs/task1_multiscale/windows.npz"))
    p.add_argument("--checkpoint", type=Path, default=Path("results/task1_improve_20261006_final_interval_roi_cnn.pt"))
    p.add_argument("--config", type=Path, default=Path("results/task1_improve_20261006_final_interval_roi_cnn.json"))
    p.add_argument("--out", type=Path, default=Path("outputs/task2_course_verification_20261014_classifier_context"))
    p.add_argument("--results", type=Path, default=Path("results"))
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        run(p.parse_args())


if __name__ == "__main__":
    main()
