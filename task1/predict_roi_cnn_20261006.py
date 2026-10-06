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


def predict(args: argparse.Namespace) -> pd.DataFrame:
    torch.set_num_threads(2)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if sha256(args.checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("Inference checkpoint hash differs from its config")
    bundle = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if tuple(bundle["classes"]) != tuple(CLASSES) or len(bundle["models"]) != 2:
        raise ValueError("Unexpected model classes or ensemble size")
    ann = pd.read_csv(args.annotations).reset_index(drop=True)
    if "length_bp" not in ann:
        ann["length_bp"] = ann.end - ann.start
    required = {"id", "start", "end", "center", "length_bp"}
    if not required.issubset(ann) or ann.id.duplicated().any() or (ann.end <= ann.start).any():
        raise ValueError("Inference annotations need unique IDs and valid known intervals")
    with np.load(args.windows) as windows:
        x = windows["X"].astype(np.float32)
        if not np.array_equal(windows["group"].astype(str), np.tile(ann.id.to_numpy(), 2)):
            raise ValueError("Inference windows do not align with both annotation replicates")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite inference contact matrix")
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
