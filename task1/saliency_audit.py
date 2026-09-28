"""Distance-matched attribution audit for spatially held-out Task 1 samples.

Gradients target the *true-class* logit.  They describe local sensitivity, not
causal evidence or proof that a class was recognized.  Attribution inside an
annotated square is compared with outside pixels at the same genomic offset
from the diagonal, so the strong Micro-C distance trend is not the control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


CLASSES = ("CHIN", "OPCID", "CHID")


def interval_bin_mask(*, center_bp: int, start_bp: int, end_bp: int,
                      n_bins: int, bin_bp: int, genome_bins: int | None = None,
                      raw_bin_bp: int | None = None) -> np.ndarray:
    """Mark pooled input bins whose genomic midpoints lie in an annotation."""
    if n_bins <= 0 or bin_bp <= 0 or start_bp >= end_bp:
        raise ValueError("Invalid input geometry or annotation interval")
    if raw_bin_bp is not None:
        if bin_bp % raw_bin_bp:
            raise ValueError("Pooled bin size must be a multiple of the raw bin size")
        aggregate = bin_bp // raw_bin_bp
        raw_window_bins = n_bins * aggregate
        raw_start = max(0, center_bp // raw_bin_bp - raw_window_bins // 2)
        if genome_bins is not None:
            raw_end = min(genome_bins, raw_start + raw_window_bins)
            raw_start = max(0, raw_end - raw_window_bins)
        midpoints = (raw_start + (np.arange(n_bins) + .5) * aggregate) * raw_bin_bp
    else:
        center_bin = center_bp // bin_bp
        window_start = max(0, center_bin - n_bins // 2)
        if genome_bins is not None:
            window_start = max(0, min(window_start, genome_bins - n_bins))
        midpoints = (window_start + np.arange(n_bins) + .5) * bin_bp
    return (midpoints >= start_bp) & (midpoints < end_bp)


def distance_matched_enrichment(saliency: np.ndarray, image: np.ndarray,
                                annotated_bins: np.ndarray) -> dict:
    """Compare inside-square gradient mass with same-diagonal-offset background.

    Only the upper triangle at offsets >= 2 is used.  For each usable offset,
    outside-annotation pixels provide the expected per-pixel gradient.  Large
    annotations that fill the input window are explicitly non-evaluable.
    """
    saliency = np.asarray(saliency, dtype=float)
    image = np.asarray(image, dtype=float)
    annotated_bins = np.asarray(annotated_bins, dtype=bool)
    n = len(annotated_bins)
    if saliency.shape != (n, n) or image.shape != (n, n):
        raise ValueError("Saliency, image and annotation mask dimensions differ")
    if not np.isfinite(saliency).all() or not np.isfinite(image).all() or (saliency < 0).any():
        raise ValueError("Expected finite nonnegative absolute gradients and finite image")
    observed = expected = 0.0
    input_observed = input_expected = 0.0
    matched_inside = matched_background = matched_offsets = 0
    valid = np.triu(np.ones((n, n), dtype=bool), k=2)
    square = np.outer(annotated_bins, annotated_bins)
    raw_inside = float(saliency[valid & square].sum())
    raw_total = float(saliency[valid].sum())
    area_inside = int((valid & square).sum())
    area_total = int(valid.sum())
    positive_input = np.maximum(image, 0)
    for offset in range(2, n):
        ii = np.arange(n - offset)
        jj = ii + offset
        inside = annotated_bins[ii] & annotated_bins[jj]
        outside = ~inside
        if not inside.any() or not outside.any():
            continue
        count = int(inside.sum())
        observed += float(saliency[ii[inside], jj[inside]].sum())
        expected += float(saliency[ii[outside], jj[outside]].mean()) * count
        input_observed += float(positive_input[ii[inside], jj[inside]].sum())
        input_expected += float(positive_input[ii[outside], jj[outside]].mean()) * count
        matched_inside += count
        matched_background += int(outside.sum())
        matched_offsets += 1
    evaluable = matched_inside >= 4 and matched_background >= 4 and expected > 0
    return {
        "evaluable": bool(evaluable),
        "annotated_bins": int(annotated_bins.sum()),
        "matched_offsets": matched_offsets,
        "matched_inside_pixels": matched_inside,
        "matched_background_pixels": matched_background,
        "inside_attribution_mass": observed,
        "expected_background_attribution_mass": expected,
        "attribution_enrichment": float(observed / expected) if evaluable else np.nan,
        "positive_input_enrichment": (float(input_observed / input_expected)
                                      if evaluable and input_expected > 0 else np.nan),
        "raw_inside_attribution_fraction": float(raw_inside / raw_total) if raw_total > 0 else np.nan,
        "raw_inside_area_fraction": float(area_inside / area_total) if area_total else np.nan,
    }


def _figure_selection(correct: np.ndarray) -> tuple[np.ndarray, str]:
    hits = np.flatnonzero(correct)
    if len(hits):
        return hits[:20], "correctly classified"
    return np.arange(min(len(correct), 20)), "MISCLASSIFIED fallback"


def _save_class_figure(out: Path, class_name: str, maps: list[np.ndarray],
                       masks: list[np.ndarray], predictions: list[str],
                       correct: list[bool]) -> dict:
    chosen, status = _figure_selection(np.asarray(correct, dtype=bool))
    if not len(chosen):
        return {"selection": "no held-out samples", "shown": 0}
    sal = np.mean([maps[i] for i in chosen], axis=0)
    coverage = np.mean([np.outer(masks[i], masks[i]) for i in chosen], axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=180, sharex=True, sharey=True)
    im = axes[0].imshow(sal, cmap="magma", origin="lower")
    fig.colorbar(im, ax=axes[0], fraction=.046, label="|d true-class logit / d input|")
    axes[0].set_title("Absolute input gradient")
    axes[1].imshow(coverage, cmap="Blues", origin="lower", vmin=0, vmax=1)
    axes[1].set_title("Fraction inside annotated square")
    for ax in axes:
        ax.set(xlabel="window bin", ylabel="window bin")
    prediction_counts = pd.Series([predictions[i] for i in chosen]).value_counts().to_dict()
    fig.suptitle(f"True {class_name}: {status}; {len(chosen)} shown / {len(maps)} held out\n"
                 f"Predictions in shown samples: {prediction_counts}; gradient is not recognition proof",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"saliency_{class_name}.png")
    plt.close(fig)
    return {"selection": status, "shown": int(len(chosen)),
            "heldout_samples": len(maps), "correct_samples": int(np.sum(correct)),
            "shown_predicted_types": prediction_counts}


def run_saliency_audit(*, model: torch.nn.Module, x_test: torch.Tensor,
                       y_test: torch.Tensor, sample_metadata: pd.DataFrame,
                       out: Path, device: torch.device, pooled_bin_bp: int,
                       genome_bins: int | None = None,
                       raw_bin_bp: int | None = None,
                       logit_bias: np.ndarray | None = None,
                       split_mode: str = "spatial") -> dict:
    """Write per-sample audit, class summary, and explicitly labeled figures."""
    if len(x_test) != len(y_test) or len(x_test) != len(sample_metadata):
        raise ValueError("Held-out tensors and metadata must have identical order and length")
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    bias = np.asarray(logit_bias if logit_bias is not None else np.zeros(len(CLASSES)), dtype=float)
    rows: list[dict] = []
    figure_maps: dict[str, list[np.ndarray]] = {name: [] for name in CLASSES}
    figure_masks: dict[str, list[np.ndarray]] = {name: [] for name in CLASSES}
    figure_predictions: dict[str, list[str]] = {name: [] for name in CLASSES}
    figure_correct: dict[str, list[bool]] = {name: [] for name in CLASSES}
    for position, (_, metadata) in enumerate(sample_metadata.iterrows()):
        target = int(y_test[position].item())
        if str(metadata.type) != CLASSES[target]:
            raise ValueError(f"Label mismatch for {metadata.id}")
        xx = x_test[position:position + 1].to(device).detach().clone().requires_grad_(True)
        logits = model(xx)[0]
        predicted = int(np.argmax(logits.detach().cpu().numpy() + bias))
        gradient = torch.autograd.grad(logits[target], xx)[0][0, 0].abs().detach().cpu().numpy()
        image = xx[0, 0].detach().cpu().numpy()
        n = image.shape[0]
        interval_mask = interval_bin_mask(center_bp=int(metadata.center),
                                          start_bp=int(metadata.start), end_bp=int(metadata.end),
                                          n_bins=n, bin_bp=pooled_bin_bp,
                                          genome_bins=genome_bins, raw_bin_bp=raw_bin_bp)
        metrics = distance_matched_enrichment(gradient, image, interval_mask)
        class_name = CLASSES[target]
        rows.append({"structure_id": metadata.id, "replicate": int(metadata.replicate),
                     "true_type": class_name, "predicted_type": CLASSES[predicted],
                     "correct": bool(predicted == target),
                     "annotation_start_bp": int(metadata.start),
                     "annotation_end_bp": int(metadata.end), **metrics})
        figure_maps[class_name].append(gradient)
        figure_masks[class_name].append(interval_mask)
        figure_predictions[class_name].append(CLASSES[predicted])
        figure_correct[class_name].append(predicted == target)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "saliency_audit_samples.csv", index=False)
    summary: dict = {"method": "absolute true-class input gradient; annotated square vs outside square at same genomic diagonal offset >=2",
                     "split_mode": split_mode,
                     "pooled_bin_bp": pooled_bin_bp,
                     "raw_bin_bp": raw_bin_bp,
                     "caution": "Local sensitivity is not causal structure evidence; misclassified-class figures are explicitly labeled. Large annotations filling the window lack a matched background."
                     + (" This split does not guarantee spatial independence." if split_mode != "spatial" else ""),
                     "classes": {}, "figures": {}}
    for class_name in CLASSES:
        class_rows = frame[frame.true_type == class_name]
        summary["figures"][class_name] = _save_class_figure(
            out, class_name, figure_maps[class_name], figure_masks[class_name],
            figure_predictions[class_name], figure_correct[class_name])
        class_summary = {}
        for scope, subset in (("all", class_rows), ("correct_only", class_rows[class_rows.correct])):
            evaluable = subset[subset.evaluable]
            # Replicates of the same annotation are not independent units.
            per_structure = evaluable.groupby("structure_id").attribution_enrichment.mean()
            class_summary[scope] = {
                "heldout_samples": int(len(subset)),
                "evaluable_samples": int(len(evaluable)),
                "evaluable_structures": int(len(per_structure)),
                "median_structure_enrichment": (float(per_structure.median())
                                                if len(per_structure) else None),
                "structures_enrichment_gt_1": int((per_structure > 1).sum()),
                "median_positive_input_enrichment": (float(evaluable.positive_input_enrichment.median())
                                                     if evaluable.positive_input_enrichment.notna().any() else None),
            }
        summary["classes"][class_name] = class_summary
    (out / "saliency_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/task1_spatial"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cool", type=Path,
                        help="original replicate .cool file, used to reproduce edge-clamped windows exactly")
    args = parser.parse_args()
    run_dir = args.run_dir
    out = args.out or run_dir
    from task1.run_task1 import StructureCNN

    checkpoint = torch.load(run_dir / "task1_cnn.pt", map_location="cpu", weights_only=True)
    model = StructureCNN(len(CLASSES), int(checkpoint.get("spatial_grid", 1)))
    model.load_state_dict(checkpoint["model_state_dict"])
    manifest = pd.read_csv(run_dir / "annotation_manifest.csv").set_index("id")
    predictions = pd.read_csv(run_dir / "test_predictions.csv")
    with np.load(run_dir / "extracted_windows.npz") as windows:
        is_test = windows["is_test"]
        ids = windows["group"][is_test]
        x_test = torch.from_numpy(windows["X"][is_test])
        y_test = torch.from_numpy(windows["y"][is_test])
    if ids.tolist() != predictions.structure_id.tolist():
        raise ValueError("Saved test predictions and windows are not aligned")
    metadata = manifest.loc[ids].reset_index()
    metadata["replicate"] = predictions.replicate.to_numpy()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if metrics["split_mode"] != "spatial":
        raise ValueError("This audit requires a spatially held-out split")
    pooled_bin_bp = int(metrics["window_bp"] // metrics["window_bins"])
    raw_bin_bp = int(metrics.get("raw_bin_bp", pooled_bin_bp // int(metrics["aggregate"])))
    cool_path = args.cool or Path("data/raw") / metrics["replicates"][0]
    genome_bins = None
    if cool_path.exists():
        from microc.data import open_cool
        with open_cool(cool_path) as cooler:
            genome_bins = int(cooler.shape[0])
            if int(cooler.binsize) != raw_bin_bp:
                raise ValueError("Replicate bin size differs from saved model geometry")
    summary = run_saliency_audit(model=model, x_test=x_test, y_test=y_test,
                                 sample_metadata=metadata, out=out, device=torch.device("cpu"),
                                 pooled_bin_bp=pooled_bin_bp,
                                 raw_bin_bp=raw_bin_bp, genome_bins=genome_bins,
                                 logit_bias=np.asarray(checkpoint.get("logit_bias", [0, 0, 0])))
    audited = pd.read_csv(out / "saliency_audit_samples.csv")
    if audited.predicted_type.tolist() != predictions.predicted_type.tolist():
        raise ValueError("Audited predictions differ from saved model predictions")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
