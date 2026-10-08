"""Non-training performance review on real contacts and archived held-out outputs.

Run with python -m scripts.benchmark_performance --data-dir PATH --out PATH.
No frozen experiment, model, or result is overwritten. GPU timings synchronize
CUDA and include input transfer and output retrieval for each ensemble pass.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from skimage.metrics import structural_similarity

from microc.circular import circular_contact_target
from microc.data import area_resample, normalize, open_cool
from microc.deep import _coarsen, _psnr
from scripts.task6_circular_sensitivity import evaluate
from scripts.verify_improvements_20261006 import digest
from task1.audit_roi_cnn_20261006 import predict_states, side_features
from task1.evaluate_roi_cnn_20261006 import CompactIntervalCNN, make_images
from task1.run_task1 import CLASSES, extract_window


ROOT = Path(__file__).resolve().parents[1]


def timed(call, repeats=5, cuda=False):
    values = []
    result = None
    for _ in range(repeats):
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = call()
        if cuda:
            torch.cuda.synchronize()
        values.append(time.perf_counter() - start)
    return result, {"median_seconds": float(np.median(values)),
                    "min_seconds": min(values), "max_seconds": max(values),
                    "repeats": repeats, "samples_seconds": values}


def classification_audit():
    outputs = {}
    for seed in (21, 22):
        prefix = f"task1_improve_20261006_roi_cnn_seed{seed}_"
        frame = pd.read_csv(ROOT / "results" / (prefix + "outer_predictions.csv"))
        archived = json.loads((ROOT / "results" / (prefix + "summary.json")).read_text(encoding="utf-8"))
        ann = pd.read_csv(ROOT / "data/structures.csv").set_index("id").loc[frame.id]
        assert frame.id.nunique() == len(frame) == 344
        assert np.array_equal(frame.type, ann.type)
        assert frame.groupby("spatial_group").outer_fold.nunique().max() == 1
        scores = {}
        for mode in ("contact64", "interval_roi64"):
            p = np.stack([frame[[f"{mode}_rep{rep}_p_{label}" for label in CLASSES]].to_numpy()
                          for rep in (1, 2)])
            assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
            assert np.allclose(p.sum(axis=2), 1, atol=1e-6)
            predicted = np.asarray(CLASSES)[p.mean(axis=0).argmax(axis=1)]
            assert np.array_equal(predicted, frame[f"{mode}_prediction"])
            report = classification_report(frame.type, predicted, labels=CLASSES,
                                           output_dict=True, zero_division=0)
            score = {"accuracy": accuracy_score(frame.type, predicted),
                     "macro_f1": f1_score(frame.type, predicted, labels=CLASSES, average="macro"),
                     "CHID": report["CHID"]}
            for key in ("accuracy", "macro_f1"):
                assert np.isclose(score[key], archived["scores"][mode][key], atol=1e-12, rtol=0)
            scores[mode] = score
        outputs[f"seed{seed}"] = scores
    return {"recomputed_archived_oof_scores": outputs,
            "new_training": False, "caution": "Retrospective spatial OOF with known intervals; not new blind validation"}


def sr_audit():
    archived = json.loads((ROOT / "results/task5_summary.json").read_text(encoding="utf-8"))
    with np.load(ROOT / "results/task5_reconstructed_test_patches.npz") as bundle:
        truth = bundle["truth"]
        scores = {}
        for method in ("bilinear", "cnn"):
            predicted = bundle[method]
            assert predicted.shape == truth.shape == (208, 64, 64)
            assert np.isfinite(truth).all() and np.isfinite(predicted).all()
            scores[method] = {
                "psnr": float(np.mean([_psnr(a, b) for a, b in zip(truth, predicted)])),
                "ssim": float(np.mean([structural_similarity(a, b, data_range=10)
                                        for a, b in zip(truth, predicted)]))}
            for key, value in scores[method].items():
                assert np.isclose(value, archived["comparison"][key][method], atol=1e-7, rtol=0)
    return {"test_patches": len(truth), "recomputed_scores": scores,
            "caution": "Saved held-out outputs, simulated low-resolution data; no retraining"}


def inference_benchmark(data_dir, repeats):
    ann = pd.read_csv(ROOT / "data/structures.csv")
    ann = pd.concat([ann[ann.type == label].sample(n=16, random_state=20261007)
                     for label in CLASSES], ignore_index=True)
    x = []
    started = time.perf_counter()
    for rep in (1, 2):
        path = data_dir / f"GSE272159_37C_rep{rep}.mapq_30.10.cool"
        with open_cool(path) as cool:
            for row in ann.itertuples():
                x.append(np.stack([extract_window(cool, int(row.center), 128, aggregate=factor)
                                   for factor in (4, 16)]))
        print(f"Extracted real contacts: replicate {rep}, {len(ann)} intervals", flush=True)
    extraction_seconds = time.perf_counter() - started
    x = np.stack(x).astype(np.float32)
    (image, side), preprocessing = timed(
        lambda: (make_images(x, ann)["interval_roi64"], side_features(x, ann)), repeats)
    checkpoint = ROOT / "results/task1_improve_20261006_final_interval_roi_cnn.pt"
    config = json.loads((ROOT / "results/task1_improve_20261006_final_interval_roi_cnn.json").read_text(encoding="utf-8"))
    assert digest(checkpoint) == config["checkpoint_sha256"]
    states = torch.load(checkpoint, map_location="cpu", weights_only=False)["models"]
    predict_states(image, side, states)  # warm-up outside timing
    reference, production = timed(lambda: predict_states(image, side, states), repeats)
    production["windows_per_second"] = len(x) / production["median_seconds"]
    runs = {}
    for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        models = []
        for state in states:
            model = CompactIntervalCNN(image.shape[1], side.shape[1]).to(device)
            model.load_state_dict(state["state_dict"])
            model.eval()
            models.append(model)
        for batch_size in (32, 64, 128):
            def infer():
                probabilities = np.zeros((len(image), 3), np.float32)
                with torch.inference_mode():
                    for state, model in zip(states, models):
                        for start in range(0, len(image), batch_size):
                            stop = start + batch_size
                            xb = torch.from_numpy(image[start:stop]).to(device)
                            sb = torch.from_numpy(((side[start:stop] - state["side_mean"]) /
                                                    state["side_scale"]).astype(np.float32)).to(device)
                            probabilities[start:stop] += model(xb, sb).softmax(dim=1).cpu().numpy() / len(states)
                return probabilities
            infer()
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            actual, timing = timed(infer, repeats, cuda=device == "cuda")
            difference = float(np.max(np.abs(actual - reference)))
            assert np.allclose(actual, reference, atol=1e-5, rtol=1e-5), difference
            assert np.array_equal(actual.argmax(axis=1), reference.argmax(axis=1))
            timing.update({"windows_per_second": len(x) / timing["median_seconds"],
                           "ms_per_interval_two_replicates": 1000 * timing["median_seconds"] / len(ann),
                           "max_probability_difference_from_cpu": difference,
                           "label_agreement_with_cpu": True})
            if device == "cuda":
                timing["peak_gpu_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024 ** 2
            runs[f"{device}_batch{batch_size}"] = timing
        del models
    return {"structures": len(ann), "replicates": 2, "windows": len(x),
            "selected_ids": ann.id.tolist(), "extraction_seconds": extraction_seconds,
            "preprocessing": preprocessing, "existing_cpu_api": production,
            "resident_model_cpu_gpu_comparison": runs,
            "timing_scope": "48 real known intervals; 2 scales/replicate. Extraction includes native reads/normalization; hot inference includes host/device transfers, excludes model loading.",
            "caution": "Final checkpoint was trained on these annotations; predictions are used only to benchmark runtime, never to estimate accuracy"}


def task6_audit(data_dir):
    path = data_dir / "GSE272159_37C_rep2.mapq_30.10.cool"
    start = time.perf_counter()
    with open_cool(path) as cool:
        raw = _coarsen(cool, cool.chromnames[0], 5000)
    coarsen_seconds = time.perf_counter() - start
    target, normalization = timed(lambda: circular_contact_target(raw), 1)
    print("Task 6: raw rep2 coarsened and circular target rebuilt", flush=True)
    with np.load(ROOT / "results/task6_controlled_followup_20261006_coordinate_models.npz") as bundle:
        coords = {name: bundle[name] for name in bundle.files}
    current, metric_timing = timed(lambda: evaluate(target, coords), 1)
    archived = json.loads((ROOT / "results/task6_controlled_followup_20261006_summary.json").read_text(encoding="utf-8"))["rep2_cross_replicate"]
    differences = []
    for stratum, values in current["strata"].items():
        for name, value in values.items():
            differences.append(abs(value - archived["strata"][stratum][name]))
    for name, value in current["exact_circular_offset_matched_auc"]["auc"].items():
        differences.append(abs(value - archived["exact_circular_offset_matched_auc"]["auc"][name]))
    assert max(differences) < 1e-10, max(differences)
    return {"raw_rep2_file": str(path), "bins": len(target), "coarsen_seconds": coarsen_seconds,
            "normalization": normalization, "evaluation": metric_timing,
            "max_metric_difference_from_published": max(differences),
            "rep2_scores": current,
            "caution": "Re-evaluation of frozen coordinates, not new independent physical-geometry validation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "micro-c数据(1)")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/performance_review_20261007/report.json")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("At least three timing repetitions are required")
    if args.out.exists():
        raise FileExistsError(f"Keep previous benchmark report; choose a new --out: {args.out}")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    report = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "numpy": np.__version__, "torch_cpu_threads": torch.get_num_threads(),
                              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
    report["task1_quality"] = classification_audit()
    report["task5_quality"], report["task5_metric_recompute_timing"] = timed(sr_audit, 1)
    print("Task 1 archived OOF and Task 5 held-out metrics verified", flush=True)
    report["task1_runtime"] = inference_benchmark(args.data_dir.resolve(), args.repeats)
    print("CPU/GPU inference benchmark complete", flush=True)
    rng = np.random.default_rng(20261007)
    mat = rng.poisson(2, (512, 512)).astype(float)
    mat = (mat + mat.T) / 2
    matrix_tests = {}
    for size in (128, 127):
        result, timing = timed(lambda: area_resample(mat, size), args.repeats)
        assert np.isclose(result.sum(), mat.sum(), rtol=1e-12)
        matrix_tests[f"area_512_to_{size}"] = timing
    _, matrix_tests["normalize_128"] = timed(lambda: normalize(mat[:128, :128]), args.repeats)
    report["matrix_primitives"] = matrix_tests
    report["task6_quality_and_runtime"] = task6_audit(args.data_dir.resolve())
    def check():
        completed = subprocess.run([str(Path(__import__("sys").executable)), "-m", "scripts.check_deliverables"],
                                   cwd=ROOT, check=True, capture_output=True, text=True)
        return json.loads(completed.stdout)
    report["deliverable_audit"], report["deliverable_audit_timing"] = timed(check, 3)
    try:
        import psutil
        report["ending_process_rss_mib"] = psutil.Process().memory_info().rss / 1024 ** 2
    except ImportError:
        pass
    report["wall_seconds"] = time.perf_counter() - started
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"All metric consistency checks passed. Benchmark report: {args.out}", flush=True)
    print(json.dumps({"wall_seconds": report["wall_seconds"],
                      "task1_runtime": report["task1_runtime"]["resident_model_cpu_gpu_comparison"]}, indent=2))


if __name__ == "__main__":
    main()
