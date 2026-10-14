"""Train-only full-window intensity calibration of frozen SR inference.

Whole-window normalization differs from the original 640-bp training scale.
We learn two affine intensity parameters on complete TRAIN intervals only,
for CNN and BOTH interpolation controls with the same budget. This is an
inference/scale adaptation, not a peak detector. Validation freezes the CNN
residual mixture; complete test maps and rep2 are evaluated afterwards.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from microc.data import annotations, open_cool
from scripts.evaluate_sr_ensemble import load_model
from scripts.task5_full_structure_verification_20261014 import (
    ROOT, SEED, PRIMARY_TYPES, cohort_manifest, corr, criterion, digest,
    image_metrics, morphology, prepare_full_pair, summarize, utc, whole_cnn,
    write_json,
)


FRACTIONS = (.25, .5, .75, 1.)


def affine_fit(sample: np.ndarray, target: np.ndarray) -> dict:
    x, y = np.asarray(sample, dtype=np.float64), np.asarray(target, dtype=np.float64)
    matrix = np.column_stack([x, np.ones(len(x))])
    slope, intercept = np.linalg.lstsq(matrix, y, rcond=None)[0]
    # A positive scale preserves contact ordering and is fixed as a constraint.
    if slope < 0:
        slope, intercept = 0., float(y.mean())
    return {"slope": float(slope), "intercept": float(intercept), "training_pixels": len(x)}


def calibrate(image: np.ndarray, parameters: dict) -> np.ndarray:
    return np.clip(parameters["slope"] * image + parameters["intercept"], -5, 5).astype(np.float32)


def train_calibration(cool_path: Path, cohort: pd.DataFrame, model, checkpoint) -> dict:
    source, targets = {name: [] for name in ("bilinear", "bicubic", "cnn")}, []
    rng = np.random.default_rng(SEED)
    with open_cool(cool_path) as cooler:
        chrom = cooler.chromnames[0]
        for number, record in enumerate(cohort.sort_values("center").itertuples(), 1):
            raw = cooler.matrix(balance=False, sparse=True).fetch((chrom, int(record.window_start_bp), int(record.window_end_bp))).toarray().astype(np.float32)
            base, cubic, truth = prepare_full_pair(raw)
            cnn = base + float(checkpoint["blend"]) * (whole_cnn(model, base) - base)
            i, j = np.triu_indices(len(raw), k=2)
            picked = rng.choice(len(i), min(5000, len(i)), replace=False)
            i, j = i[picked], j[picked]
            targets.append(truth[i, j])
            for name, image in (("bilinear", base), ("bicubic", cubic), ("cnn", cnn)):
                source[name].append(image[i, j])
            print(f"Train full-window scale: {number}/{len(cohort)} {record.id}", flush=True)
    target = np.concatenate(targets)
    return {name: affine_fit(np.concatenate(values), target) for name, values in source.items()}


def transformed_images(base: np.ndarray, cubic: np.ndarray, cnn: np.ndarray, weights: dict) -> dict:
    calibrated_base = calibrate(base, weights["bilinear"])
    calibrated_cnn = calibrate(cnn, weights["cnn"])
    result = {"bilinear": base, "bicubic": cubic, "frozen_main_whole": cnn,
              "train_calibrated_bilinear": calibrated_base,
              "train_calibrated_bicubic": calibrate(cubic, weights["bicubic"]),
              "train_calibrated_cnn": calibrated_cnn}
    result.update({f"calibrated_cnn_fraction_{a:g}": a * calibrated_cnn + (1-a) * calibrated_base for a in FRACTIONS})
    return result


def evaluate(cool_path: Path, cohort: pd.DataFrame, model, checkpoint: dict,
             calibration: dict, out: Path, methods: list[str] | None = None,
             save_maps: bool = False) -> pd.DataFrame:
    rows, shown = [], set()
    out.mkdir(parents=True, exist_ok=True)
    with open_cool(cool_path) as cooler:
        chrom = cooler.chromnames[0]
        for ordinal, record in enumerate(cohort.sort_values("center").itertuples(), 1):
            raw = cooler.matrix(balance=False, sparse=True).fetch((chrom, int(record.window_start_bp), int(record.window_end_bp))).toarray().astype(np.float32)
            base, cubic, target = prepare_full_pair(raw)
            cnn = base + float(checkpoint["blend"]) * (whole_cnn(model, base) - base)
            images = transformed_images(base, cubic, cnn, calibration)
            target_parts = morphology(target)
            for method in (methods or list(images)):
                rows.append({"id": record.id, "type": record.type, "center_bp": record.center,
                             "start_bp": record.start, "end_bp": record.end,
                             "window_start_bp": record.window_start_bp, "window_end_bp": record.window_end_bp,
                             "native_bins": record.native_bins, "overlap_component": record.overlap_component,
                             "method": method, **image_metrics(target, images[method], target_parts)})
            if save_maps:
                selected = images[methods[-1]]
                np.savez_compressed(out / f"{record.id}_full_maps.npz", truth=target, bilinear=base,
                                    bicubic=cubic, prediction=selected,
                                    calibrated_bilinear=images["train_calibrated_bilinear"],
                                    calibrated_bicubic=images["train_calibrated_bicubic"],
                                    start_bp=record.window_start_bp, end_bp=record.window_end_bp)
                fig, axes = plt.subplots(1, 4, figsize=(12, 3), dpi=140)
                for ax, image, title in zip(axes, (base, images["train_calibrated_bilinear"], selected, target),
                                            ("Bilinear", "Train-calibrated bilinear", "Final CNN scale-adapted", "Micro-C target")):
                    ax.imshow(image, origin="lower", cmap="magma", vmin=-2, vmax=4,
                              extent=[record.window_start_bp, record.window_end_bp]*2)
                    ax.set_title(title, fontsize=8)
                    ax.set(xlabel="position bp", ylabel="position bp")
                    ax.tick_params(labelsize=6)
                    ax.axvline(record.start, color="cyan", lw=.35, alpha=.7)
                    ax.axvline(record.end, color="cyan", lw=.35, alpha=.7)
                fig.suptitle(f"{record.id} full annotated span {record.length_bp} bp + fixed flanks", fontsize=10)
                fig.tight_layout()
                fig.savefig(out / f"{record.id}_full.png")
                plt.close(fig)
                if record.type not in shown:
                    shutil.copyfile(out / f"{record.id}_full.png", out / f"example_{record.type}.png")
                    shown.add(record.type)
            print(f"{out.name}: {ordinal}/{len(cohort)} {record.id} full {record.native_bins} bins", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "per_structure.csv", index=False)
    return frame


def choose(frame: pd.DataFrame) -> tuple[str, list[dict]]:
    grid = []
    for method in ("train_calibrated_cnn", *[f"calibrated_cnn_fraction_{a:g}" for a in FRACTIONS]):
        report = summarize(frame, method)
        feasible = report["all"]["psnr"]["mean_gain"] > 0 and report["all"]["ssim"]["mean_gain"] > 0
        feasible = feasible and all(report[kind]["shape_r"]["mean_gain"] >= -.03 and
                                    report[kind]["profile_r"]["mean_gain"] >= -.03 for kind in PRIMARY_TYPES)
        grid.append({"method": method, "feasible": bool(feasible), "psnr_gain": report["all"]["psnr"]["mean_gain"],
                     "ssim_gain": report["all"]["ssim"]["mean_gain"], "shape_gain": report["all"]["shape_r"]["mean_gain"],
                     "profile_gain": report["all"]["profile_r"]["mean_gain"]})
    usable = [row for row in grid if row["feasible"]]
    chosen = max(usable, key=lambda row: (row["ssim_gain"], row["psnr_gain"]))["method"] if usable else "train_calibrated_cnn"
    return chosen, grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("select", "test"), required=True)
    parser.add_argument("--rep1", type=Path, default=ROOT / "data/raw/GSE272159_37C_rep1.mapq_30.10.cool")
    parser.add_argument("--rep2", type=Path, default=ROOT / "data/raw/GSE272159_37C_rep2.mapq_30.10.cool")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results/task5_super_resolution.pt")
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/structures.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/task5_full_structure_20261014/scale_repair")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(args.checkpoint)
    with open_cool(args.rep1) as cooler:
        genome_bp = int(cooler.chromsizes.iloc[0])
    cohort, split = cohort_manifest(annotations(args.annotations), genome_bp)
    sources = {"repair_script": Path(__file__), "full_evaluator": ROOT / "scripts/task5_full_structure_verification_20261014.py",
               "deep": ROOT / "microc/deep.py", "data": ROOT / "microc/data.py", "checkpoint": args.checkpoint,
               "annotations": args.annotations, "rep1": args.rep1, "rep2": args.rep2}
    hashes = {key: digest(path) for key, path in sources.items()}
    if args.phase == "select":
        write_json(args.out / "protocol.json", {"experiment": "task5-full-structure-scale-repair-20261014", "frozen_at_utc": utc(),
                   "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   "source_and_input_sha256": hashes, "device": "cpu", "threads": args.threads, "split": split,
                   "training_rule": "Two nonnegative-affine intensity parameters per method, fitted by least squares on <=5000 offdiagonal pixels per complete TRAIN structure; identical budget for CNN/bilinear/bicubic. No type label enters fit or image generation.",
                   "inference_rule": "Frozen CNN halo-tiled whole-map inference and global symmetrization; train-only scalar affine intensity mapping then [-5,5] clipping; validation may blend calibrated CNN and calibrated bilinear at .25/.5/.75/1.",
                   "input_rule": "40-bp raw mean aggregation and input-only normalization; separately normalized HR used only as training target/evaluation truth; no HR map statistic is an inference feature.",
                   "selection_rule": "Only validation complete windows choose CNN fraction; require PSNR/SSIM gains>0 and CHIN/OPCID shape/profile gains>=-.03; maximize SSIM then PSNR. All matched train-calibrated controls retained.",
                   "success_rule": "Same preregistered operational full-shape/image criteria as initial full-window verification; requires both repetitions, no mandatory single-peak superiority and no claim of de novo structure location.",
                   "uncertainty": "2000 paired bootstrap resamples of complete overlap-connected windows; type stratification; repeat2 same coordinates.",
                   "caution": "Initial full-window validation failure motivated train-only scale adaptation before any current test/rep2 matrix evaluation; historically inspected regions remain retrospective."})
        cohort.to_csv(args.out / "cohort_manifest.csv", index=False)
        print("Repair protocol frozen; train complete windows only", flush=True)
        calibration = train_calibration(args.rep1, cohort[cohort.split == "train"], model, checkpoint)
        write_json(args.out / "trained_scale_parameters.json", {"trained_at_utc": utc(), "parameters": calibration,
                   "source_and_input_sha256": hashes, "training_ids": cohort[cohort.split == "train"].id.tolist()})
        validation = evaluate(args.rep1, cohort[cohort.split == "validation"], model, checkpoint, calibration, args.out / "validation")
        chosen, grid = choose(validation)
        write_json(args.out / "selection_freeze.json", {"frozen_at_utc": utc(), "selected_method": chosen, "grid": grid,
                   "protocol_sha256": digest(args.out / "protocol.json"), "source_and_input_sha256": hashes,
                   "calibration_sha256": digest(args.out / "trained_scale_parameters.json"),
                   "cohort_manifest_sha256": digest(args.out / "cohort_manifest.csv"),
                   "validation_sha256": digest(args.out / "validation/per_structure.csv"),
                   "test_matrix_region_read": False, "rep2_matrix_read": False,
                   "validation_selected": summarize(validation, chosen)})
        print(json.dumps({"calibration": calibration, "selected_method": chosen, "grid": grid}, indent=2), flush=True)
    else:
        freeze = json.loads((args.out / "selection_freeze.json").read_text(encoding="utf-8"))
        if hashes != freeze["source_and_input_sha256"] or digest(args.out / "protocol.json") != freeze["protocol_sha256"]:
            raise ValueError("Code/input/protocol drift after freeze")
        if digest(args.out / "trained_scale_parameters.json") != freeze["calibration_sha256"] or digest(args.out / "cohort_manifest.csv") != freeze["cohort_manifest_sha256"]:
            raise ValueError("Training weights/cohort drift after freeze")
        calibration = json.loads((args.out / "trained_scale_parameters.json").read_text(encoding="utf-8"))["parameters"]
        chosen = freeze["selected_method"]
        methods = list(dict.fromkeys(("bilinear", "bicubic", "frozen_main_whole", "train_calibrated_bilinear", "train_calibrated_bicubic", chosen)))
        result = {"selected_method": chosen, "calibration": calibration, "selection_freeze_sha256": digest(args.out / "selection_freeze.json"),
                  "replicates": {}, "scope": "Complete annotated intervals plus160bp flanks,10bp native output, simulated40bp input"}
        for name, path in (("rep1_test", args.rep1), ("rep2_fixed_test", args.rep2)):
            frame = evaluate(path, cohort[cohort.split == "test"], model, checkpoint, calibration, args.out / name, methods, True)
            report = summarize(frame, chosen)
            result["replicates"][name] = {"selected": report, "success": criterion(report),
                                          "matched_controls": {method: summarize(frame, method) for method in methods if method != chosen}}
        result["overall_operational_success"] = all(item["success"]["operational_course_success"] for item in result["replicates"].values())
        result["wall_seconds"] = time.perf_counter() - started
        write_json(args.out / "summary.json", result)
        artifacts = {str(path.relative_to(args.out)): digest(path) for path in args.out.rglob("*") if path.is_file() and path.name != "receipt.json"}
        write_json(args.out / "receipt.json", {"completed_at_utc": utc(), "source_and_input_sha256": hashes, "output_sha256": artifacts,
                                              "validation_only_selection": True, "test_opened_after_freeze": True, "unchanged_old_checkpoint": True})
        for name in ("summary.json", "protocol.json", "selection_freeze.json", "trained_scale_parameters.json", "receipt.json", "cohort_manifest.csv"):
            shutil.copyfile(args.out / name, ROOT / "results" / f"task5_full_structure_20261014_scale_repair_{name}")
        for name in result["replicates"]:
            shutil.copyfile(args.out / name / "per_structure.csv", ROOT / "results" / f"task5_full_structure_20261014_scale_repair_{name}.csv")
            for kind in ("CHIN", "OPCID", "CHID"):
                shutil.copyfile(args.out / name / f"example_{kind}.png", ROOT / "results" / f"task5_full_structure_20261014_scale_repair_{name}_{kind}.png")
        print(json.dumps({"selected_method": chosen, "overall_operational_success": result["overall_operational_success"],
                          "success": {name: item["success"] for name, item in result["replicates"].items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
