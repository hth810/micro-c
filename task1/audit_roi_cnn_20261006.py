"""Reproduce the fixed OOF CNN, audit matched occlusion, and export inference.

Contacts inside/outside the annotated interval are matched exactly for the
global |i-j| distribution separately at both contact scales. The interval
mask and length remain fixed. Contact-derived boundary summaries are updated
after replacing contacts by normalized zero. Predictions remain OOF.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from task1.evaluate_chid_overlap import spatial_groups
from task1.evaluate_interval_boundaries import interval_features
from task1.evaluate_nested_neural_20261006 import sha256
from task1.evaluate_roi_cnn_20261006 import CompactIntervalCNN, EPOCHS, fit_predict, make_images
from task1.run_task1 import CLASSES


def matched_contact_masks(start: float, end: float, center: float, bin_bp: int,
                          rng: np.random.Generator, bins: int = 128) -> tuple[np.ndarray, np.ndarray, list[int]]:
    coordinates = center + (np.arange(bins) - bins / 2 + 0.5) * bin_bp
    inside = (coordinates >= start) & (coordinates < end)
    inner_mask = np.zeros((bins, bins), dtype=bool)
    outer_mask = np.zeros_like(inner_mask)
    matched_offsets = []
    # Exclude first two diagonals; compare identical global distance offsets.
    # Capping at 32 bins limits the audit to contact distances represented in
    # both regions rather than matching a diagonal-heavy inner region against
    # only remote off-diagonal contacts.
    for distance in range(2, 33):
        a = np.arange(bins - distance)
        b = a + distance
        within = a[inside[a] & inside[b]]
        outside = a[~inside[a] & ~inside[b]]
        available = min(len(within), len(outside))
        if available == 0:
            continue
        count = max(1, int(np.floor(available * 0.25)))
        selected_inner = rng.choice(within, count, replace=False)
        selected_outer = rng.choice(outside, count, replace=False)
        inner_mask[selected_inner, selected_inner + distance] = True
        outer_mask[selected_outer, selected_outer + distance] = True
        matched_offsets.extend([distance] * count)
    inner_mask |= inner_mask.T
    outer_mask |= outer_mask.T
    return inner_mask, outer_mask, matched_offsets


def side_features(x: np.ndarray, ann: pd.DataFrame) -> np.ndarray:
    n = len(ann)
    return np.stack([np.r_[interval_features(x[rep * n + i, 1], int(row.start),
                                           int(row.end), int(row.center)), np.log1p(row.length_bp)]
                     for rep in (0, 1) for i, row in ann.iterrows()]).astype(np.float32)


def predict_states(image: np.ndarray, side: np.ndarray, states: list[dict]) -> np.ndarray:
    output = np.zeros((len(image), 3), dtype=np.float32)
    for state in states:
        model = CompactIntervalCNN(image.shape[1], side.shape[1])
        model.load_state_dict(state["state_dict"])
        model.eval()
        with torch.no_grad():
            pieces = []
            for indices in np.array_split(np.arange(len(image)), max(1, int(np.ceil(len(image) / 64)))):
                xb = torch.from_numpy(image[indices])
                sb = torch.from_numpy(((side[indices] - state["side_mean"]) / state["side_scale"]).astype(np.float32))
                pieces.append(model(xb, sb).softmax(dim=1).numpy())
            output += np.concatenate(pieces) / len(states)
    return output


def group_mean_interval(values: np.ndarray, groups: np.ndarray, rng: np.random.Generator,
                        resamples: int = 4000) -> list[float]:
    unique = np.unique(groups)
    numerator = np.asarray([values[groups == group].sum() for group in unique])
    denominator = np.asarray([(groups == group).sum() for group in unique])
    sampled = rng.integers(0, len(unique), (resamples, len(unique)))
    mean = numerator[sampled].sum(axis=1) / denominator[sampled].sum(axis=1)
    return np.quantile(mean, [0.025, 0.975]).tolist()


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(2)
    ann = pd.read_csv(args.annotations).reset_index(drop=True)
    n = len(ann)
    with np.load(args.windows) as bundle:
        x = bundle["X"].astype(np.float32)
        if not np.array_equal(bundle["group"].astype(str), np.tile(ann.id.to_numpy(), 2)):
            raise ValueError("Misaligned cached contact IDs")
    y = ann.type.map({label: i for i, label in enumerate(CLASSES)}).to_numpy(dtype=np.int64)
    groups = spatial_groups(ann, 40000)
    saved = pd.read_csv(args.cnn / "outer_predictions.csv").set_index("id").loc[ann.id]
    original_protocol = json.loads((args.cnn / "protocol.json").read_text(encoding="utf-8"))
    if original_protocol["outer_seed"] != args.seed:
        raise ValueError("CNN seed differs from requested audit")
    if original_protocol["source_sha256"]["task1/evaluate_roi_cnn_20261006.py"] != sha256(Path("task1/evaluate_roi_cnn_20261006.py")):
        raise ValueError("CNN source differs from frozen primary run")
    if sha256(args.windows) != original_protocol["input_sha256"]["windows"] or sha256(args.annotations) != original_protocol["input_sha256"]["annotations"]:
        raise ValueError("Audit inputs changed from evaluated model")
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": sha256(Path(__file__)),
                "cnn_protocol_sha256": sha256(args.cnn / "protocol.json"), "seed": args.seed,
                "masking_seed": 20261006,
                "occlusion": "25% of min(inner,outside) available contacts per exact |i-j| band 2..32, separately at each 40/160 bp scale and replicate; replace selected symmetric contacts by normalized zero",
                "geometry": "within both endpoints inside interval versus both endpoints outside; contact pixels and derived boundary statistics updated, masks and length unchanged",
                "matching": "Identical number of upper triangle contact pairs at every global diagonal distance in inner/outside masks; absent matchable bands excluded, recorded per sample",
                "caution": "Exploratory OOF perturbation evidence for learned contact dependence; not functional/causal proof; interval geometry and clipping remain directly available",
                "saliency": "OOF class-logit gradient times image; side branch fixed; representative is correct OOF sample nearest class median interval length",
                "original_protocol_addendum": {"scoring": "CNN uses 2 initialization models and 2 biological repeats, not the 3 initializations inherited in the protocol scoring text",
                                                "preprocessing": "15 side features standardized with training replicates only; contact images already independently normalized per cached input window",
                                                "architecture_note": "ROI input and two mask channels make structure width directly available; side branch contains contact-derived summaries and log length",
                                                "fixed_protocol_retained": True}}
    audit_protocol = args.out / "audit_protocol.json"
    if audit_protocol.exists():
        raise ValueError("Audit protocol already exists")
    audit_protocol.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    original_images = make_images(x, ann)["interval_roi64"]
    original_sides = side_features(x, ann)
    perturbed = {"inside": x.copy(), "outside": x.copy()}
    matching_rows = []
    for rep in (0, 1):
        for i, row in ann.iterrows():
            for scale, bin_bp in enumerate((40, 160)):
                rng = np.random.default_rng(20261006 + rep * 100000 + i * 10 + scale)
                inner, outside, offsets = matched_contact_masks(row.start, row.end, row.center, bin_bp, rng)
                if inner.sum() != outside.sum():
                    raise ValueError("Unequal contact occlusion count")
                a1, b1 = np.where(np.triu(inner))
                a2, b2 = np.where(np.triu(outside))
                if not np.array_equal(np.sort(b1 - a1), np.sort(b2 - a2)):
                    raise ValueError("Occlusion global diagonal distributions differ")
                perturbed["inside"][rep * n + i, scale][inner] = 0
                perturbed["outside"][rep * n + i, scale][outside] = 0
                matching_rows.append({"id": row.id, "replicate": rep + 1, "scale_bp_per_bin": bin_bp,
                                      "matched_pairs_per_region": len(offsets),
                                      "min_offset_bins": min(offsets) if offsets else None,
                                      "max_offset_bins": max(offsets) if offsets else None,
                                      "inner_outside_offset_counts_identical": True,
                                      "outside_available": bool(offsets)})
    perturbed_images = {mode: make_images(value, ann)["interval_roi64"] for mode, value in perturbed.items()}
    perturbed_sides = {mode: side_features(value, ann) for mode, value in perturbed.items()}
    for images in perturbed_images.values():
        if not np.array_equal(images[:, 2:4], original_images[:, 2:4]):
            raise ValueError("Perturbation changed known interval masks")
    del perturbed
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed).split(ann, y, groups))
    probabilities = {mode: np.full((2, n, 3), np.nan, dtype=np.float32)
                     for mode in ("original", "inside", "outside")}
    figure, axes = plt.subplots(3, 4, figsize=(13, 10), constrained_layout=True)
    representatives = {}
    for cls, label in enumerate(CLASSES):
        correct = ann.type.eq(label) & saved.interval_roi64_prediction.to_numpy().__eq__(label)
        choices = ann.index[correct]
        median_length = ann.loc[ann.type.eq(label), "length_bp"].median()
        representative = min(choices, key=lambda i: (abs(ann.length_bp.iloc[i] - median_length), i))
        representatives[cls] = int(representative)
    for fold, (train, test) in enumerate(folds):
        if not np.all(saved.outer_fold.iloc[test].to_numpy() == fold):
            raise ValueError("OOF audit fold does not match source predictions")
        probabilities["original"][:, test], states = fit_predict(original_images, original_sides, y,
                                                                 train, test, args.seed + fold * 100)
        expected = np.stack([saved[[f"interval_roi64_rep{rep}_p_{label}" for label in CLASSES]].to_numpy()[test]
                             for rep in (1, 2)])
        if not np.allclose(probabilities["original"][:, test], expected, rtol=1e-5, atol=2e-6):
            raise ValueError("Reproduced OOF CNN predictions differ from frozen primary run")
        torch.save({"models": states, "fold": fold, "train_ids": ann.id.iloc[train].tolist(),
                    "test_ids": ann.id.iloc[test].tolist()}, args.out / f"fold_{fold}_cnn.pt")
        sample_ids = np.r_[test, test + n]
        for mode in ("inside", "outside"):
            probabilities[mode][:, test] = predict_states(perturbed_images[mode][sample_ids],
                                                         perturbed_sides[mode][sample_ids], states).reshape(2, len(test), 3)
        for cls, representative in representatives.items():
            if representative not in test:
                continue
            gradients = []
            image = original_images[representative:representative + 1]
            side = original_sides[representative:representative + 1]
            for state in states:
                model = CompactIntervalCNN(5, 15)
                model.load_state_dict(state["state_dict"])
                model.eval()
                image_tensor = torch.from_numpy(image).requires_grad_(True)
                side_tensor = torch.from_numpy(((side - state["side_mean"]) / state["side_scale"]).astype(np.float32))
                model(image_tensor, side_tensor)[0, cls].backward()
                gradients.append((image_tensor.grad.detach().numpy()[0] * image[0]))
            saliency = np.abs(np.mean(gradients, axis=0))
            panels = (image[0, 1], image[0, 4], saliency[1], saliency[4])
            titles = ("Long contact / 64 bins", "Annotated interval ROI", "Long contact |grad x input|", "ROI |grad x input|")
            for column, (panel, title) in enumerate(zip(panels, titles)):
                axis = axes[cls, column]
                lim = np.quantile(np.abs(panel), 0.99) or 1
                if column < 2:
                    axis.imshow(panel, cmap="coolwarm", vmin=-lim, vmax=lim)
                else:
                    axis.imshow(panel, cmap="magma", vmin=0, vmax=lim)
                axis.set_title(title, fontsize=9)
                axis.set_xticks([]); axis.set_yticks([])
                if column == 0:
                    axis.set_ylabel(f"{CLASSES[cls]}\n{ann.id.iloc[representative]}\nOOF fold {fold}")
        print(f"OOF occlusion completed fold {fold + 1}/5; exact prediction reproduction passed", flush=True)
    if any(not np.isfinite(value).all() for value in probabilities.values()):
        raise ValueError("Incomplete OOF occlusion predictions")
    figure.suptitle("OOF compact CNN: image-branch saliency; fixed interval/side metadata")
    figure.savefig(args.out / "oof_saliency_by_class.png", dpi=160)
    plt.close(figure)
    rows = []
    true_probabilities = {mode: p[:, np.arange(n), y].mean(axis=0) for mode, p in probabilities.items()}
    for i, row in ann.iterrows():
        rows.append({"id": row.id, "type": row.type, "spatial_group": groups[i],
                     "outer_fold": int(saved.outer_fold.iloc[i]), "length_bp": row.length_bp,
                     "original_true_probability": true_probabilities["original"][i],
                     "inside_true_probability": true_probabilities["inside"][i],
                     "outside_true_probability": true_probabilities["outside"][i],
                     "inside_probability_drop": true_probabilities["original"][i] - true_probabilities["inside"][i],
                     "outside_probability_drop": true_probabilities["original"][i] - true_probabilities["outside"][i],
                     "inside_minus_outside_drop": true_probabilities["outside"][i] - true_probabilities["inside"][i],
                     **{f"{mode}_prediction": CLASSES[p[:, i].mean(axis=0).argmax()] for mode, p in probabilities.items()}})
    occlusion = pd.DataFrame(rows)
    occlusion.to_csv(args.out / "oof_occlusion_samples.csv", index=False)
    pd.DataFrame(matching_rows).to_csv(args.out / "occlusion_matching.csv", index=False)
    summary = {"audit_protocol_sha256": sha256(audit_protocol), "source_sha256": protocol["source_sha256"],
               "caution": protocol["caution"], "reproduced_predictions_match": True,
               "unchanged_interval_masks": True, "unchanged_interval_lengths": True,
               "exact_global_diagonal_matching": True,
               "unmatched_samples": int(sum(not row["outside_available"] for row in matching_rows)),
               "representative_ids": {CLASSES[cls]: ann.id.iloc[i] for cls, i in representatives.items()},
               "occlusion_summary": {}, "original_protocol_addendum": protocol["original_protocol_addendum"]}
    for label in ("all", *CLASSES):
        subset = occlusion if label == "all" else occlusion[occlusion.type == label]
        effect = subset.inside_minus_outside_drop.to_numpy()
        summary["occlusion_summary"][label] = {"structures": len(subset),
            "inside_mean_true_class_probability_drop": float(subset.inside_probability_drop.mean()),
            "outside_mean_true_class_probability_drop": float(subset.outside_probability_drop.mean()),
            "inside_minus_outside_mean_drop": float(effect.mean()),
            "paired_spatial_95ci": group_mean_interval(effect, subset.spatial_group.to_numpy(), np.random.default_rng(20261006))}
    # Final training is for inference only; no in-sample performance is scored.
    _, final_states = fit_predict(original_images, original_sides, y, np.arange(n), np.array([], dtype=int), args.seed)
    torch.save({"models": final_states, "classes": CLASSES, "channels": 5, "side_features": 15,
                "architecture": "CompactIntervalCNN", "epochs": EPOCHS,
                "scope": "Given known interval boundaries and both cached 128-bin contact scales",
                "source_sha256": original_protocol["source_sha256"],
                "input_sha256": original_protocol["input_sha256"],
                "evaluation": "Performance is the separately saved OOF result; this checkpoint fits all annotations"},
               args.out / "final_interval_roi_cnn.pt")
    final_config = {"checkpoint_sha256": sha256(args.out / "final_interval_roi_cnn.pt"),
                    "classes": CLASSES, "epochs": EPOCHS, "initializations": 2,
                    "source_cnn_protocol_sha256": sha256(args.cnn / "protocol.json"),
                    "input_sha256": original_protocol["input_sha256"],
                    "evaluation": "Inference checkpoint trained on all 344 annotations; only OOF performance is reported",
                    "requires_annotated_interval": True, "not_for_unannotated_genome_scanning": True}
    (args.out / "final_interval_roi_cnn.json").write_text(json.dumps(final_config, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["output_sha256"] = {name: sha256(args.out / name) for name in
        ("oof_occlusion_samples.csv", "occlusion_matching.csv", "oof_saliency_by_class.png", "final_interval_roi_cnn.pt", "final_interval_roi_cnn.json")}
    (args.out / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["occlusion_summary"], indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=Path("outputs/task1_multiscale/windows.npz"))
    parser.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    parser.add_argument("--cnn", type=Path, default=Path("outputs/task1_improve_20261006/roi_cnn_seed21"))
    parser.add_argument("--out", type=Path, default=Path("outputs/task1_improve_20261006/roi_cnn_audit"))
    parser.add_argument("--seed", type=int, default=20260921)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
