"""Infer annotated interval classes from the exported compact CNN ensemble.

The checkpoint is a local artifact produced by audit_roi_cnn_20261006 and
contains NumPy scaling arrays. Load only that trusted exported checkpoint.
This interface accepts the same two independently normalized contact scales
and requires known interval boundaries; it is not an annotation-free scan.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from task1.audit_roi_cnn_20261006 import predict_states, side_features
from task1.evaluate_nested_neural_20261006 import sha256
from task1.evaluate_roi_cnn_20261006 import make_images
from task1.run_task1 import CLASSES


def load_inputs(windows_path: Path, annotations_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Reject malformed coordinates and caches before running the ensemble."""
    ann = pd.read_csv(annotations_path, dtype={"id": "string"}).reset_index(drop=True)
    required = {"id", "start", "end", "center"}
    if not required.issubset(ann) or ann.empty:
        raise ValueError("Inference annotations need IDs, start/end/center and at least one interval")
    if ann.id.isna().any() or ann.id.astype(str).str.strip().eq("").any():
        raise ValueError("Inference annotation IDs must be nonempty")
    ann["id"] = ann.id.astype(str)
    if ann.id.duplicated().any():
        raise ValueError("Inference annotation IDs must be unique")
    for column in ("start", "end", "center"):
        ann[column] = pd.to_numeric(ann[column], errors="raise")
    coords = ann[["start", "end", "center"]].to_numpy(dtype=np.float64)
    if (not np.isfinite(coords).all() or (coords != np.floor(coords)).any() or
            (ann.start < 0).any() or (ann.end <= ann.start).any() or
            (ann.center < ann.start).any() or (ann.center >= ann.end).any()):
        raise ValueError("Inference intervals need finite integer coordinates and an interior center")
    length = ann.end - ann.start
    if "length_bp" in ann:
        supplied = pd.to_numeric(ann.length_bp, errors="raise").to_numpy()
        if not np.array_equal(supplied, length.to_numpy()):
            raise ValueError("Inference length_bp disagrees with interval boundaries")
    ann["length_bp"] = length
    with np.load(windows_path, allow_pickle=False) as windows:
        if not {"X", "group"}.issubset(windows.files):
            raise ValueError("Inference cache requires X and group arrays")
        x = windows["X"].astype(np.float32)
        if x.shape != (2 * len(ann), 2, 128, 128):
            raise ValueError("Inference cache needs two replicates and two 128-bin contact scales")
        if not np.array_equal(windows["group"].astype(str), np.tile(ann.id.to_numpy(), 2)):
            raise ValueError("Inference windows do not align with both annotation replicates")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite inference contact matrix")
    return x, ann


def predict(args: argparse.Namespace) -> pd.DataFrame:
    if args.out.resolve() in {path.resolve() for path in
                              (args.windows, args.annotations, args.checkpoint, args.config)}:
        raise ValueError("Prediction output must not overwrite an input or checkpoint")
    torch.set_num_threads(2)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if sha256(args.checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("Inference checkpoint hash differs from its config")
    bundle = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if tuple(bundle["classes"]) != tuple(CLASSES) or len(bundle["models"]) != 2:
        raise ValueError("Unexpected model classes or ensemble size")
    x, ann = load_inputs(args.windows, args.annotations)
    image = make_images(x, ann)["interval_roi64"]
    side = side_features(x, ann)
    probabilities = predict_states(image, side, bundle["models"]).reshape(2, len(ann), 3)
    mean = probabilities.mean(axis=0)
    if not np.allclose(mean.sum(axis=1), 1, atol=1e-5) or not np.isfinite(mean).all():
        raise ValueError("Invalid model probabilities")
    output = pd.DataFrame({"id": ann.id, "predicted_type": [CLASSES[i] for i in mean.argmax(axis=1)],
                           "replicate_prediction_agreement": probabilities[0].argmax(axis=1) == probabilities[1].argmax(axis=1)})
    for cls, label in enumerate(CLASSES):
        output[f"p_{label}"] = mean[:, cls]
        for rep in (0, 1):
            output[f"rep{rep + 1}_p_{label}"] = probabilities[rep, :, cls]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(f"Saved {len(output)} annotated-interval predictions. This is inference, not an OOF performance estimate.")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=Path("outputs/task1_multiscale/windows.npz"))
    parser.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/task1_improve_20261006_final_interval_roi_cnn.pt"))
    parser.add_argument("--config", type=Path, default=Path("results/task1_improve_20261006_final_interval_roi_cnn.json"))
    parser.add_argument("--out", type=Path, default=Path("outputs/task1_improve_20261006/inference_predictions.csv"))
    predict(parser.parse_args())


if __name__ == "__main__":
    main()
