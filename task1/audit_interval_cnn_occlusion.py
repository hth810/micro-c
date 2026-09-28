"""Held-out contact/interval occlusion for the Task 1 interval CNN.

Re-fits exactly the predeclared five outer-fold models; perturbations are
applied only to each model's held-out structures. Zero denotes the
distance-normalized median contact, not a missing-data token. These are
sensitivity diagnostics, not causal or independent biological validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from task1.evaluate_chid_overlap import spatial_groups
from task1.evaluate_interval_conditioned_cnn import make_modes
from task1.evaluate_pairwise import evaluate, fit_fold, nearby_chid_chin_pairs
from task1.run_task1 import CLASSES
from task1.saliency_audit import distance_matched_enrichment


def matched_long_contact_occlusions(sample: np.ndarray,
                                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int]:
    """Zero equal pixel counts inside/outside interval at each diagonal offset."""
    inside = sample.copy()
    outside = sample.copy()
    square = sample[2].astype(bool)
    n = square.shape[0]
    matched = 0
    for offset in range(2, n):
        ii = np.arange(n - offset)
        jj = ii + offset
        a = np.flatnonzero(square[ii, jj])
        b = np.flatnonzero(~square[ii, jj])
        count = min(len(a), len(b))
        if not count:
            continue
        a = rng.choice(a, count, replace=False)
        b = rng.choice(b, count, replace=False)
        inside[1, ii[a], jj[a]] = inside[1, jj[a], ii[a]] = 0
        outside[1, ii[b], jj[b]] = outside[1, jj[b], ii[b]] = 0
        matched += count
    return inside, outside, matched


def margins(logits: np.ndarray, truth: np.ndarray) -> np.ndarray:
    rows = np.arange(len(truth))
    other = logits.copy()
    other[rows, truth] = -np.inf
    return logits[rows, truth] - other.max(axis=1)


def predict(model: torch.nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(chunk).to(device)).cpu().numpy()
                               for chunk in np.array_split(x, max(1, int(np.ceil(len(x) / 32))))])


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    ann = pd.read_csv(args.annotations).reset_index(drop=True)
    n = len(ann)
    with np.load(args.windows) as bundle:
        x = bundle["X"].astype(np.float32)
        ids = bundle["group"].astype(str)
    if not np.array_equal(ids, np.tile(ann.id.to_numpy(), 2)):
        raise ValueError("Cached replicate IDs do not match annotations")
    images = make_modes(x, ann)["contact_interval"]
    y = ann.type.map({name: i for i, name in enumerate(CLASSES)}).to_numpy(dtype=int)
    groups = spatial_groups(ann, 40000)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260921)
                 .split(ann, y, groups))
    modes = ("base", "no_short_contact", "no_long_contact", "no_contact",
             "no_interval", "inside_long_contact", "matched_outside_long_contact")
    logits = {name: np.full((2, n, 3), np.nan, np.float32) for name in modes}
    matched_pixels = np.zeros((2, n), dtype=int)
    # All rare CHID and their nearest CHIN within 2 kb are a fixed,
    # biologically relevant saliency audit set, not a success-only gallery.
    pairs = nearby_chid_chin_pairs(ann, 2000)
    saliency_indices = set(np.flatnonzero(y == 2)) | set(pairs[:, 1])
    saliency_rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fold, (train, test) in enumerate(folds):
        model = fit_fold(images, y, train, np.empty((0, 2), dtype=int), 12,
                         20260921 + fold, 0.0, device)
        for rep in (0, 1):
            samples = images[test + rep * n]
            variants = {"base": samples}
            for name, channels in (("no_short_contact", (0,)),
                                   ("no_long_contact", (1,)),
                                   ("no_contact", (0, 1)),
                                   ("no_interval", (2, 3))):
                modified = samples.copy()
                modified[:, channels] = 0
                variants[name] = modified
            within = np.empty_like(samples)
            background = np.empty_like(samples)
            for j, structure_index in enumerate(test):
                rng = np.random.default_rng(args.seed + 2 * structure_index + rep)
                within[j], background[j], count = matched_long_contact_occlusions(samples[j], rng)
                matched_pixels[rep, structure_index] = count
            variants["inside_long_contact"] = within
            variants["matched_outside_long_contact"] = background
            for name, values in variants.items():
                logits[name][rep, test] = predict(model, values, device)
            for j, structure_index in enumerate(test):
                if structure_index not in saliency_indices:
                    continue
                tensor = torch.from_numpy(samples[j:j + 1]).to(device).requires_grad_(True)
                target_logit = model(tensor)[0, int(y[structure_index])]
                gradient = torch.autograd.grad(target_logit, tensor)[0][0, 1].abs().detach().cpu().numpy()
                metrics = distance_matched_enrichment(
                    gradient, samples[j, 1], np.diagonal(samples[j, 2]).astype(bool))
                saliency_rows.append({"id": ann.iloc[structure_index].id,
                                      "true_type": ann.iloc[structure_index].type,
                                      "replicate": rep + 1,
                                      "predicted_type": CLASSES[logits["base"][rep, structure_index].argmax()],
                                      "spatial_group": groups[structure_index], **metrics})
        print(f"occlusion fold {fold + 1}/5", flush=True)
    if any(not np.isfinite(values).all() for values in logits.values()):
        raise RuntimeError("Incomplete held-out occlusion predictions")
    expected = pd.read_csv(args.predictions)
    expected = expected[expected["mode"] == "contact_interval"].set_index("id").loc[ann.id]
    for k, name in enumerate(CLASSES):
        if not np.allclose(logits["base"][:, :, k].mean(axis=0),
                           expected[f"logit_{name}"].to_numpy(), atol=1e-4):
            raise ValueError("Refitted model does not match saved out-of-fold predictions")
    rows = []
    for rep in (0, 1):
        for i, row in ann.iterrows():
            record = {"id": row.id, "true_type": row.type, "replicate": rep + 1,
                      "spatial_group": groups[i], "matched_offset_pixel_pairs": int(matched_pixels[rep, i]),
                      "base_prediction": CLASSES[logits["base"][rep, i].argmax()]}
            for name in modes:
                record[f"margin_{name}"] = float(margins(logits[name][rep, i:i + 1], y[i:i + 1])[0])
                record[f"prediction_{name}"] = CLASSES[logits[name][rep, i].argmax()]
            rows.append(record)
    samples = pd.DataFrame(rows)
    effects = {}
    for name in modes[1:]:
        samples[f"margin_drop_{name}"] = samples.margin_base - samples[f"margin_{name}"]
        effects[name] = {}
        for class_name, subset in samples.groupby("true_type"):
            per_structure = subset.groupby("id")[f"margin_drop_{name}"].mean()
            effects[name][class_name] = {
                "structures": int(len(per_structure)),
                "median_true_class_margin_drop": float(per_structure.median()),
                "mean_true_class_margin_drop": float(per_structure.mean()),
                "positive_drop_structures": int((per_structure > 0).sum())}
    matched = samples.groupby("id", sort=False).agg({
        "true_type": "first", "margin_drop_inside_long_contact": "mean",
        "margin_drop_matched_outside_long_contact": "mean"})
    matched["inside_minus_matched_outside"] = (
        matched.margin_drop_inside_long_contact - matched.margin_drop_matched_outside_long_contact)
    saliency = pd.DataFrame(saliency_rows)
    saliency_summary = {}
    for class_name, subset in saliency.groupby("true_type"):
        usable = subset[subset.evaluable]
        per_structure = usable.groupby("id").attribution_enrichment.mean()
        saliency_summary[class_name] = {
            "selected_structures": int(subset.id.nunique()),
            "evaluable_structures": int(len(per_structure)),
            "median_distance_matched_gradient_enrichment":
                float(per_structure.median()) if len(per_structure) else None,
            "structures_gradient_enrichment_gt_1": int((per_structure > 1).sum()),
            "correct_evaluable_structures": int(usable[usable.true_type == usable.predicted_type].id.nunique()),
        }
    summary = {
        "protocol": "refit fixed 12-epoch five-fold CNN; seven held-out perturbations; both replicates grouped by annotation",
        "caution": "zeroing channels is a distribution shift; attribution is sensitivity, not proof of morphology or novel structure; masks encode annotation span",
        "zero_value": "distance-normalized median contact",
        "matched_offset_control": "for every genomic diagonal offset >=2, zero equal random counts inside and outside the interval square in the 20.48 kb contact channel",
        "random_seed": args.seed,
        "structure_scores": {name: evaluate(y, values.mean(axis=0).argmax(axis=1))
                             for name, values in logits.items()},
        "true_class_margin_effects": effects,
        "matched_contact_region_effect": {name: {
            "structures": int(len(subset)),
            "median_inside_minus_outside_margin_drop": float(subset.inside_minus_matched_outside.median()),
            "positive_structures": int((subset.inside_minus_matched_outside > 0).sum())}
            for name, subset in matched.groupby("true_type")},
        "saliency": {"method": "absolute true-class gradient in 20.48 kb contact channel; within annotated square vs same genomic-offset background",
                     "selection": "all CHID and nearest CHIN within 2 kb, including misclassified samples",
                     "paired_chid_chin": int(len(pairs)),
                     "classes": saliency_summary},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    samples.to_csv(args.out / "occlusion_samples.csv", index=False)
    saliency.to_csv(args.out / "saliency_samples.csv", index=False)
    (args.out / "occlusion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", type=Path, default=Path("outputs/task1_multiscale/windows.npz"))
    ap.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    ap.add_argument("--predictions", type=Path,
                    default=Path("outputs/task1_interval_conditioned_cnn/crossfit_predictions.csv"))
    ap.add_argument("--out", type=Path, default=Path("outputs/task1_interval_conditioned_cnn"))
    ap.add_argument("--seed", type=int, default=20260928)
    summary = run(ap.parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
