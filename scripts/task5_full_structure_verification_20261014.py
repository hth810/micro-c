"""Full annotated-interval SR verification with frozen validation-only selection.

Inputs are simulated from native counts before independent input/target
normalization. Whole-map CNN inference is halo tiled, without incorrectly
symmetrizing off-diagonal tiles. Complete intervals and a fixed flank are
evaluated, including type stratification and overlap-component uncertainty.
The old checkpoint/results remain untouched. This is retrospective spatial
validation because the project previously inspected the test genomic region.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, maximum_filter, zoom
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity
import torch

from microc.data import annotations, area_resample, normalize, open_cool
from microc.deep import SuperResolutionCNN, _psnr
from scripts.evaluate_sr_ensemble import load_model, split_centers


ROOT = Path(__file__).resolve().parents[1]
DATE = "20261014"
FLANK_BP = 160
SEED = 20261014
METRICS = ("psnr", "ssim", "shape_r", "profile_r", "peak_error_bp", "top5_f1")
ALPHAS = (.25, .5, .75, 1.)
PRIMARY_TYPES = ("CHIN", "OPCID")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def full_interval(record, genome_bp: int) -> tuple[int, int]:
    """Every annotation residue is included, with equal fixed flanks."""
    left = max(0, ((int(record.start) - FLANK_BP) // 40) * 40)
    right = min((genome_bp // 40) * 40, ((int(record.end) + FLANK_BP + 39) // 40) * 40)
    if right < int(record.end) or left > int(record.start):
        raise ValueError("Full annotation crosses the regular-bin chromosome edge")
    return left, right


def cohort_manifest(ann: pd.DataFrame, genome_bp: int) -> tuple[pd.DataFrame, dict]:
    centers, train_end, test_start = split_centers(genome_bp, 640, 1200)
    validation_start, test_boundary = centers[train_end] - 320, centers[test_start] - 320
    rows = []
    for record in ann.itertuples():
        left, right = full_interval(record, genome_bp)
        split = ("train" if right <= validation_start else
                 "validation" if left >= validation_start and right <= test_boundary else
                 "test" if left >= test_boundary else "excluded_boundary_crossing")
        rows.append({**record._asdict(), "window_start_bp": left, "window_end_bp": right,
                     "native_bins": (right - left) // 10, "split": split})
    frame = pd.DataFrame(rows)
    group = -1
    previous_end = -1
    groups = {}
    for record in frame.sort_values(["window_start_bp", "window_end_bp"]).itertuples():
        if int(record.window_start_bp) >= previous_end:
            group += 1
        previous_end = max(previous_end, int(record.window_end_bp))
        groups[record.id] = group
    frame["overlap_component"] = frame.id.map(groups)
    spec = {"validation_start_bp": validation_start, "test_start_bp": test_boundary,
            "legacy_tile_counts": [train_end, test_start - train_end, len(centers) - test_start],
            "complete_intervals_only": True, "flank_bp": FLANK_BP, "grid_alignment_bp": 40,
            "counts": {str(split): block.type.value_counts().to_dict() for split, block in frame.groupby("split")},
            "dependence": "Whole overlapping windows form connected components for paired resampling. Rep2 has the same loci, not new independent annotations."}
    return frame, spec


def prepare_full_pair(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Both interpolation controls use only aggregated raw input statistics."""
    n = len(raw)
    if raw.shape != (n, n) or n % 4:
        raise ValueError("Native square size must be a multiple of four")
    low = raw.reshape(n // 4, 4, n // 4, 4).mean(axis=(1, 3))
    bilinear = normalize(zoom(low, 4, order=1).astype(np.float32))
    bicubic = normalize(zoom(low, 4, order=3).astype(np.float32))
    return bilinear, bicubic, normalize(raw)


def whole_cnn(model: SuperResolutionCNN, image: np.ndarray, core: int = 96,
              halo: int = 16, batch_size: int = 4) -> np.ndarray:
    """Equivalent to whole-map fully convolutional inference, within FP error.

    The maximum convolution receptive field is 25 pixels; halo >=12 supplies
    complete context. At genuine map boundaries each layer retains its native
    zero padding. Off-diagonal tile transposes are NOT independent contacts.
    """
    n = len(image)
    if image.shape != (n, n) or halo < 12 or core < 1:
        raise ValueError("Invalid image/core/halo")
    tiles = defaultdict(list)
    for row in range(0, n, core):
        for col in range(0, n, core):
            hi_row, hi_col = min(n, row + core), min(n, col + core)
            a, b = max(0, row - halo), min(n, hi_row + halo)
            c, d = max(0, col - halo), min(n, hi_col + halo)
            tiles[(b - a, d - c)].append((row, hi_row, col, hi_col, a, b, c, d))
    result = np.empty_like(image, dtype=np.float32)
    gate = float(np.any(image != 0))
    with torch.inference_mode():
        for shape in sorted(tiles):
            entries = tiles[shape]
            for first in range(0, len(entries), batch_size):
                batch = entries[first:first + batch_size]
                x = torch.from_numpy(np.stack([image[a:b, c:d] for _, _, _, _, a, b, c, d in batch])[:, None].astype(np.float32))
                output = x + gate * model.tail(model.body(model.head(x)))
                output = output.numpy()[:, 0]
                for prediction, (r, rr, c0, cc, a, _, c, _) in zip(output, batch):
                    result[r:rr, c0:cc] = prediction[r-a:rr-a, c0-c:cc-c]
    return (result + result.T) / 2


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def peaks(image: np.ndarray, count: int = 5) -> np.ndarray:
    smooth = gaussian_filter(image, sigma=4)
    mask = np.triu(np.ones(image.shape, bool), k=8)
    maxima = (smooth == maximum_filter(smooth, size=9)) & mask
    rows, cols = np.where(maxima)
    order = np.argsort(-smooth[rows, cols], kind="stable")
    chosen = []
    for index in order:
        point = np.array([rows[index], cols[index]])
        if all(np.linalg.norm(point - other) >= 8 for other in chosen):
            chosen.append(point)
        if len(chosen) == count:
            break
    return np.asarray(chosen, dtype=float).reshape(-1, 2)


def morphology(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    smooth = gaussian_filter(image, sigma=4)
    coarse = area_resample(smooth, 64) / (len(image) / 64) ** 2
    shape = coarse[np.triu_indices(64, k=2)]
    n = len(image)
    distance = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    profile = np.maximum(smooth, 0)
    profile[distance < 8] = 0
    profile = gaussian_filter(profile.sum(axis=1), sigma=4)
    return shape, profile, peaks(image)


def image_metrics(target: np.ndarray, prediction: np.ndarray,
                  target_parts: tuple | None = None) -> dict:
    truth_shape, truth_profile, truth_peaks = target_parts or morphology(target)
    shape, profile, predicted_peaks = morphology(prediction)
    top1 = float(np.linalg.norm(truth_peaks[0] - predicted_peaks[0]) * 10)
    from scipy.optimize import linear_sum_assignment
    distances = np.linalg.norm(truth_peaks[:, None] - predicted_peaks[None], axis=2) * 10
    i, j = linear_sum_assignment(distances)
    matches = int((distances[i, j] <= 80).sum())
    return {"psnr": _psnr(target, prediction),
            "ssim": float(structural_similarity(target, prediction, data_range=10)),
            "shape_r": corr(truth_shape, shape), "profile_r": corr(truth_profile, profile),
            "peak_error_bp": top1, "top5_f1": float(2 * matches / (len(truth_peaks) + len(predicted_peaks)))}


def evaluate(cool_path: Path, subset: pd.DataFrame, model: SuperResolutionCNN,
             blend: float, out: Path, methods: list[str] | None = None,
             save_maps: bool = False) -> pd.DataFrame:
    rows = []
    out.mkdir(parents=True, exist_ok=True)
    shown = set()
    with open_cool(cool_path) as cooler:
        chrom = cooler.chromnames[0]
        for ordinal, record in enumerate(subset.sort_values("center").itertuples(), 1):
            raw = cooler.matrix(balance=False, sparse=True).fetch((chrom, int(record.window_start_bp), int(record.window_end_bp))).toarray().astype(np.float32)
            base, cubic, target = prepare_full_pair(raw)
            restored = base + blend * (whole_cnn(model, base) - base)
            images = {"bilinear": base, "bicubic": cubic, "frozen_main_whole": restored}
            images.update({f"bounded_residual_{alpha:g}": np.clip(base + alpha * (restored - base), -5, 5) for alpha in ALPHAS})
            target_parts = morphology(target)
            for method in (methods or list(images)):
                image = images[method]
                rows.append({"id": record.id, "type": record.type, "center_bp": record.center,
                             "start_bp": record.start, "end_bp": record.end,
                             "window_start_bp": record.window_start_bp, "window_end_bp": record.window_end_bp,
                             "native_bins": record.native_bins, "overlap_component": record.overlap_component,
                             "method": method, **image_metrics(target, image, target_parts)})
            if save_maps:
                selected_image = images[methods[-1]]
                np.savez_compressed(out / f"{record.id}_full_maps.npz", truth=target, bilinear=base, bicubic=cubic,
                                    prediction=selected_image, start_bp=record.window_start_bp, end_bp=record.window_end_bp)
                # Every structure gets a thumbnail; first-by-position examples are not best-case selection.
                fig, axes = plt.subplots(1, 3, figsize=(10, 3), dpi=140)
                for ax, matrix, title in zip(axes, (base, selected_image, target), ("Bilinear", methods[-1], "Micro-C target")):
                    ax.imshow(matrix, origin="lower", cmap="magma", vmin=-2, vmax=4,
                              extent=[record.window_start_bp, record.window_end_bp]*2)
                    ax.set_title(title, fontsize=9)
                    ax.set(xlabel="position bp", ylabel="position bp")
                    ax.tick_params(labelsize=6)
                fig.suptitle(f"{record.id} complete {record.length_bp} bp interval + fixed flanks", fontsize=10)
                fig.tight_layout()
                fig.savefig(out / f"{record.id}_full.png")
                plt.close(fig)
                if record.type not in shown:
                    import shutil
                    shutil.copyfile(out / f"{record.id}_full.png", out / f"example_{record.type}.png")
                    shown.add(record.type)
            print(f"{out.name}: {ordinal}/{len(subset)} {record.id} full {record.native_bins} bins", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(out / "per_structure.csv", index=False)
    return result


def mean_interval(values: np.ndarray, groups: np.ndarray) -> list[float]:
    unique = np.unique(groups)
    blocks = [values[groups == group] for group in unique]
    rng = np.random.default_rng(SEED)
    samples = [np.concatenate([blocks[index] for index in rng.integers(len(blocks), size=len(blocks))]).mean() for _ in range(2000)]
    return [float(value) for value in np.quantile(samples, [.025, .975])]


def summarize(frame: pd.DataFrame, method: str) -> dict:
    source = frame[frame.method == method].sort_values("id")
    baseline = frame[frame.method == "bilinear"].sort_values("id")
    if source.id.tolist() != baseline.id.tolist():
        raise ValueError("Methods differ in complete structure IDs")
    output = {}
    for name, mask in (("all", np.ones(len(source), bool)), *[(kind, source.type.to_numpy() == kind) for kind in ("CHIN", "OPCID", "CHID")]):
        if not mask.any():
            continue
        a, b = source.iloc[np.flatnonzero(mask)], baseline.iloc[np.flatnonzero(mask)]
        report = {"n": len(a), "overlap_components": int(a.overlap_component.nunique())}
        for metric in METRICS:
            x, y = a[metric].to_numpy(), b[metric].to_numpy()
            delta = (y - x) if metric == "peak_error_bp" else (x - y)
            report[metric] = {"bilinear_mean": float(y.mean()), "model_mean": float(x.mean()),
                              "model_median": float(np.median(x)), "mean_gain": float(delta.mean()),
                              "paired_component_95ci_gain": mean_interval(delta, a.overlap_component.to_numpy()),
                              "improved_fraction": float((delta > 0).mean())}
        report["shape_r_at_least_0_5_fraction"] = float((a.shape_r >= .5).mean())
        output[name] = report
    return output


def selection(frame: pd.DataFrame) -> tuple[str, list[dict]]:
    """Only full validation windows select residual amount; test is unopened."""
    grid = []
    for method in ["frozen_main_whole", *[f"bounded_residual_{alpha:g}" for alpha in ALPHAS]]:
        summary = summarize(frame, method)
        whole = summary["all"]
        shape_ok = all(summary[kind]["shape_r"]["mean_gain"] >= -.03 and
                       summary[kind]["profile_r"]["mean_gain"] >= -.03 for kind in PRIMARY_TYPES)
        feasible = whole["psnr"]["mean_gain"] > 0 and whole["ssim"]["mean_gain"] > 0 and shape_ok
        grid.append({"method": method, "feasible": bool(feasible),
                     "psnr_gain": whole["psnr"]["mean_gain"], "ssim_gain": whole["ssim"]["mean_gain"],
                     "shape_gain": whole["shape_r"]["mean_gain"], "profile_gain": whole["profile_r"]["mean_gain"]})
    feasible = [row for row in grid if row["feasible"]]
    chosen = max(feasible, key=lambda row: (row["ssim_gain"], row["psnr_gain"]))["method"] if feasible else "frozen_main_whole"
    return chosen, grid


def criterion(summary: dict) -> dict:
    """Predeclared operational morphology evidence, not a teacher grade guarantee."""
    pixels = all(summary["all"][metric]["paired_component_95ci_gain"][0] > 0 for metric in ("psnr", "ssim"))
    shape = all(summary[kind]["shape_r"]["model_median"] >= .7 and
                summary[kind]["profile_r"]["model_median"] >= .7 and
                summary[kind]["shape_r_at_least_0_5_fraction"] >= .8 and
                summary[kind]["shape_r"]["mean_gain"] >= -.03 and
                summary[kind]["profile_r"]["mean_gain"] >= -.03 for kind in PRIMARY_TYPES)
    return {"image_gain_supported": pixels, "complete_morphology_supported": shape,
            "operational_course_success": bool(pixels and shape),
            "interpretation": "Fixed full-window shape/profile criteria for CHIN and OPCID; all classes and actual image peak metrics are retained. Original 640-bp results and this evaluation are different scales. No physical/function/real-HiC or blind-validation claim."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("select", "test"), required=True)
    parser.add_argument("--rep1", type=Path, default=ROOT / "data/raw/GSE272159_37C_rep1.mapq_30.10.cool")
    parser.add_argument("--rep2", type=Path, default=ROOT / "data/raw/GSE272159_37C_rep2.mapq_30.10.cool")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results/task5_super_resolution.pt")
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/structures.csv")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/task5_full_structure_20261014")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(args.checkpoint)
    with open_cool(args.rep1) as cooler:
        if cooler.binsize != 10:
            raise ValueError("Full structure protocol requires 10-bp native input")
        genome_bp = int(cooler.chromsizes.iloc[0])
    cohort, split = cohort_manifest(annotations(args.annotations), genome_bp)
    source_paths = {"script": Path(__file__), "deep": ROOT / "microc/deep.py", "data": ROOT / "microc/data.py",
                    "checkpoint": args.checkpoint, "annotations": args.annotations, "rep1": args.rep1, "rep2": args.rep2}
    hashes = {key: digest(path) for key, path in source_paths.items()}
    if args.phase == "select":
        protocol = {"experiment": "task5-full-structure-20261014", "frozen_at_utc": utc(),
                    "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "source_and_input_sha256": hashes, "split": split,
                    "device": "cpu", "threads": args.threads,
                    "dependencies": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "pandas", "scipy", "scikit-learn", "scikit-image", "cooler")},
                    "input_rule": "Mean 4x4 native raw aggregation, bilinear/cubic upsampling, input-only normalization; separate target normalization. No annotation type or HR summary generates model input.",
                    "inference": "Frozen main 48-channel 4-residual-block checkpoint; whole-map convolution by 96px cores +16px halo, global transpose symmetrization; residual fractions .25/.5/.75/1 clipped to known [-5,5] target range.",
                    "morphology_metrics": "Whole-window Gaussian sigma4 native bins, count-area average to64 shape correlation excluding first2 coarse diagonals; positive-row profile excludes native offsets<8; actual image top5 local maxima matched within80bp, no detector/ranker.",
                    "selection": "Validation only: PSNR/SSIM gains positive, CHIN/OPCID shape/profile mean gains>=-.03; maximize SSIM then PSNR. Retain all methods and bicubic control. Fallback fixed whole CNN if none.",
                    "success_rule": "Each replicate: overlap-component 95% gain CI>0 for PSNR and SSIM; CHIN/OPCID median full shape and profile r>=.7, >=80% shape r>=.5, mean shape/profile gain>=-.03. Operational evidence, not prescribed grading thresholds.",
                    "uncertainty": "2000 paired resamples of complete overlap-connected annotation windows; type-wise strata; repeats reported separately.",
                    "caution": "Historically inspected test region and rep2. This run freezes before current test and uses only validation for selection; not prospective blind discovery."}
        write_json(args.out / "protocol.json", protocol)
        cohort.to_csv(args.out / "cohort_manifest.csv", index=False)
        print("Protocol frozen; extracting validation complete windows only", flush=True)
        validation = evaluate(args.rep1, cohort[cohort.split == "validation"], model, float(checkpoint["blend"]), args.out / "validation")
        chosen, grid = selection(validation)
        write_json(args.out / "selection_freeze.json", {"frozen_at_utc": utc(), "selected_method": chosen, "grid": grid,
                   "protocol_sha256": digest(args.out / "protocol.json"), "source_and_input_sha256": hashes,
                   "cohort_manifest_sha256": digest(args.out / "cohort_manifest.csv"),
                   "validation_sha256": digest(args.out / "validation/per_structure.csv"), "rep2_matrix_opened": False,
                   "test_matrix_region_read": False, "validation_selected_summary": summarize(validation, chosen)})
        print(json.dumps({"selected_method": chosen, "grid": grid}, indent=2), flush=True)
    else:
        freeze = json.loads((args.out / "selection_freeze.json").read_text(encoding="utf-8"))
        if hashes != freeze["source_and_input_sha256"] or digest(args.out / "protocol.json") != freeze["protocol_sha256"]:
            raise ValueError("Source/protocol drift after validation freeze")
        if digest(args.out / "cohort_manifest.csv") != freeze["cohort_manifest_sha256"]:
            raise ValueError("Frozen cohort changed")
        chosen = freeze["selected_method"]
        methods = list(dict.fromkeys(("bilinear", "bicubic", "frozen_main_whole", chosen)))
        summary = {"selection_freeze_sha256": digest(args.out / "selection_freeze.json"), "selected_method": chosen,
                   "scope": "Complete known intervals +160bp flanks at native10bp output; simulated40bp input", "replicates": {}}
        for name, path in (("rep1_test", args.rep1), ("rep2_fixed_test", args.rep2)):
            frame = evaluate(path, cohort[cohort.split == "test"], model, float(checkpoint["blend"]), args.out / name, methods, True)
            selected_summary = summarize(frame, chosen)
            summary["replicates"][name] = {"selected": selected_summary, "frozen_main_whole": summarize(frame, "frozen_main_whole"),
                                              "bicubic": summarize(frame, "bicubic"), "success": criterion(selected_summary)}
        summary["overall_operational_success"] = all(item["success"]["operational_course_success"] for item in summary["replicates"].values())
        summary["wall_seconds"] = time.perf_counter() - started
        write_json(args.out / "summary.json", summary)
        output_hashes = {str(path.relative_to(args.out)): digest(path) for path in args.out.rglob("*") if path.is_file() and path.name != "receipt.json"}
        write_json(args.out / "receipt.json", {"completed_at_utc": utc(), "source_and_input_sha256": hashes, "outputs_sha256": output_hashes,
                    "rep1_test_and_rep2_only_after_selection_freeze": True, "frozen_old_files_modified": False,
                    "validation_only_selection": True, "seed": SEED})
        import shutil
        for name in ("summary.json", "protocol.json", "selection_freeze.json", "cohort_manifest.csv", "receipt.json"):
            shutil.copyfile(args.out / name, ROOT / "results" / f"task5_full_structure_20261014_{name}")
        for name in summary["replicates"]:
            shutil.copyfile(args.out / name / "per_structure.csv", ROOT / "results" / f"task5_full_structure_20261014_{name}.csv")
            for kind in ("CHIN", "OPCID", "CHID"):
                shutil.copyfile(args.out / name / f"example_{kind}.png", ROOT / "results" / f"task5_full_structure_20261014_{name}_{kind}.png")
        print(json.dumps({"selected_method": chosen, "overall_operational_success": summary["overall_operational_success"],
                          "success": {name: item["success"] for name, item in summary["replicates"].items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
