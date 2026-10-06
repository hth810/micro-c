"""A fixed equal-weight CNN/HGB probability ensemble, never outer tuned.

This is an explicitly hybrid exploratory comparator. Its HGB component uses
fixed contact/interval descriptors; it must not be described as a pure CNN.
The initial protocol can be frozen before CNN results are completed.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold

from task1.audit_interval_conditioned_cnn import paired_bootstrap
from task1.evaluate_chid_overlap import spatial_groups
from task1.evaluate_nested_neural_20261006 import load_features, sha256
from task1.evaluate_nested_threshold import fit_model
from task1.evaluate_pairwise import evaluate
from task1.run_task1 import CLASSES


def freeze(args: argparse.Namespace) -> dict:
    protocol = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "caution": "Retrospective follow-up, these annotation labels and historical baseline scores were inspected; this is not independent confirmation",
                "model": "Fixed equal arithmetic mean of compact interval CNN and HGB class probabilities",
                "cnn_weight": 0.5, "hgb_weight": 0.5, "threshold": "3-class argmax",
                "hgb": "max_iter100/max_leaf_nodes7/l2=1; inverse class frequencies from fold training only",
                "source_sha256": {"task1/evaluate_hybrid_20261006.py": sha256(Path(__file__)),
                                    "task1/evaluate_roi_cnn_20261006.py": sha256(Path("task1/evaluate_roi_cnn_20261006.py")),
                                    "task1/evaluate_nested_neural_20261006.py": sha256(Path("task1/evaluate_nested_neural_20261006.py")),
                                    "task1/evaluate_nested_threshold.py": sha256(Path("task1/evaluate_nested_threshold.py"))},
                "input_sha256": {"annotations": sha256(args.annotations), "windows": sha256(args.windows)},
                "outer_seed": args.seed, "spatial_gap_bp": 40000,
                "selection": "Equal weights chosen without inspecting this CNN experiment's outer scores; no search over fusion weights",
                "cnn_run": args.cnn.name,
                "comparison": "same folds and labels; bootstrap samples spatial components, not biological repeats as independent samples"}
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "protocol.json"
    if target.exists():
        raise ValueError("Hybrid protocol already frozen")
    target.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    return protocol


def score(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    protocol_path = args.out / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["outer_seed"] != args.seed or protocol["cnn_run"] != args.cnn.name:
        raise ValueError("Frozen protocol does not match run")
    for name, digest in protocol["source_sha256"].items():
        if sha256(Path(name)) != digest:
            raise ValueError(f"Source changed after freezing: {name}")
    if protocol["input_sha256"] != {"annotations": sha256(args.annotations), "windows": sha256(args.windows)}:
        raise ValueError("Input changed after freeze")
    ann, descriptors = load_features(args.windows, args.annotations)
    n = len(ann)
    y = ann.type.map({label: i for i, label in enumerate(CLASSES)}).to_numpy(dtype=int)
    groups = spatial_groups(ann, 40000)
    saved = pd.read_csv(args.cnn / "outer_predictions.csv").set_index("id").loc[ann.id]
    cnn_summary = json.loads((args.cnn / "summary.json").read_text(encoding="utf-8"))
    cnn_protocol = json.loads((args.cnn / "protocol.json").read_text(encoding="utf-8"))
    if cnn_summary["seed"] != args.seed or cnn_protocol["source_sha256"]["task1/evaluate_roi_cnn_20261006.py"] != protocol["source_sha256"]["task1/evaluate_roi_cnn_20261006.py"]:
        raise ValueError("CNN seed/source differs from frozen hybrid protocol")
    if not np.array_equal(saved.type, ann.type) or not np.array_equal(saved.spatial_group, groups):
        raise ValueError("CNN IDs, labels or spatial groups do not align")
    cnn = np.stack([saved[[f"interval_roi64_rep{rep}_p_{label}" for label in CLASSES]].to_numpy()
                    for rep in (1, 2)]).mean(axis=0)
    values = descriptors["contact_interval"].reshape(2, n, -1).mean(axis=0)
    hgb = np.full((n, 3), np.nan)
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (train, test) in enumerate(folds.split(ann, y, groups)):
        if not np.all(saved.outer_fold.iloc[test].to_numpy() == fold):
            raise ValueError("CNN outer test folds differ")
        hgb[test] = fit_model(values, y, train, args.seed + fold).predict_proba(values[test])
    if not np.isfinite(hgb).all() or not np.isfinite(cnn).all():
        raise ValueError("Incomplete probabilities")
    fused = 0.5 * cnn + 0.5 * hgb
    predictions = {"fixed_hybrid": fused.argmax(axis=1), "compact_interval_cnn": cnn.argmax(axis=1),
                   "same_fold_hgb": hgb.argmax(axis=1)}
    if "published_interval_cnn_prediction" in saved:
        predictions["published_interval_cnn"] = saved.published_interval_cnn_prediction.map(
            {label: i for i, label in enumerate(CLASSES)}).to_numpy()
    scores = {mode: {**evaluate(y, p), "classification_report": classification_report(y, p, labels=range(3),
                         target_names=CLASSES, output_dict=True, zero_division=0)} for mode, p in predictions.items()}
    uncertainty = paired_bootstrap(y, groups, predictions, "fixed_hybrid", tuple(
                                  mode for mode in predictions if mode != "fixed_hybrid"), 4000, args.seed + 97)
    rows = {"id": ann.id, "type": ann.type, "chrom": ann.chrom, "center": ann.center,
            "spatial_group": groups, "outer_fold": saved.outer_fold.to_numpy(),
            **{f"{mode}_prediction": [CLASSES[i] for i in p] for mode, p in predictions.items()}}
    for label, index in zip(CLASSES, range(3)):
        rows[f"cnn_p_{label}"], rows[f"hgb_p_{label}"], rows[f"hybrid_p_{label}"] = cnn[:, index], hgb[:, index], fused[:, index]
    pd.DataFrame(rows).to_csv(args.out / "outer_predictions.csv", index=False)
    summary = {"protocol_sha256": sha256(protocol_path), "frozen_at_utc": protocol["frozen_at_utc"],
               "caution": protocol["caution"], "seed": args.seed, "scores": scores,
               "paired_spatial_uncertainty": uncertainty,
               "source_cnn_protocol_sha256": sha256(args.cnn / "protocol.json"),
               "output_sha256": {"outer_predictions.csv": sha256(args.out / "outer_predictions.csv")},
               "wall_seconds": time.perf_counter() - started,
               "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"scores": {mode: {k: v for k, v in score.items() if k != "classification_report"}
                                 for mode, score in scores.items()}}, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=Path("outputs/task1_multiscale/windows.npz"))
    parser.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    parser.add_argument("--cnn", type=Path, default=Path("outputs/task1_improve_20261006/roi_cnn_seed21"))
    parser.add_argument("--out", type=Path, default=Path("outputs/task1_improve_20261006/hybrid_seed21"))
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--phase", choices=("freeze", "score"), required=True)
    args = parser.parse_args()
    freeze(args) if args.phase == "freeze" else score(args)


if __name__ == "__main__":
    main()
