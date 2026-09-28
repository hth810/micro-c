"""Fail fast if any of the six course-task result bundles is incomplete."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def required(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise AssertionError(f"Missing or empty deliverable: {path}")
    return path


def read_json(path: Path) -> dict:
    return json.loads(required(path).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    root = args.root.resolve()
    bundle, generated = root / "results", root / "outputs"
    annotation = pd.read_csv(required(root / "data/structures.csv"))
    assert len(annotation) == 344
    assert annotation.type.value_counts().to_dict() == {"CHIN": 250, "OPCID": 68, "CHID": 26}

    # Task 1: original CNN deliverables and the separately identified best
    # annotation-aligned classifier. Its hybrid score must not be called CNN-only.
    boundary = read_json(bundle / "task1_interval_boundaries_summary.json")
    for name in ("task1_spatial_task1_cnn.pt", "task1_spatial_confusion_matrix.csv",
                 "task1_spatial_confusion_matrix.png", "task1_spatial_saliency_CHIN.png",
                 "task1_spatial_saliency_OPCID.png", "task1_spatial_saliency_CHID.png",
                 "task1_spatial_saliency_audit_summary.json",
                 "task1_spatial_saliency_audit_samples.csv",
                 "task1_boundary_final.joblib", "task1_interval_boundaries_crossfit_predictions.csv"):
        required(bundle / name)
    saliency = read_json(bundle / "task1_spatial_saliency_audit_summary.json")
    assert saliency["figures"]["CHID"]["selection"] == "MISCLASSIFIED fallback"
    predictions = pd.read_csv(bundle / "task1_interval_boundaries_crossfit_predictions.csv")
    assert predictions.id.nunique() == 344
    assert all(group.id.nunique() == 344 for _, group in predictions.groupby("mode"))
    balanced = read_json(bundle / "task1_balanced_cnn_summary.json")
    assert balanced["class_weight_power"] == 1.0
    assert len(pd.read_csv(required(bundle / "task1_balanced_cnn_crossfit_predictions.csv"))) == 344

    # Task 2: a candidate cluster means independent loci, not duplicate slide windows.
    discovery = read_json(bundle / "task2_summary.json")
    strict = read_json(bundle / "task2_strict_revised_summary.json")
    novel = read_json(bundle / "task2_novel_cluster_summary.json")
    loci = pd.read_csv(required(bundle / "task2_novel_cluster_putative_novel_clusters.csv"))
    assert discovery["windows"] == 1812 and discovery["known_structure_recall_after_reproducibility"] > .9
    assert strict["cluster_supported_windows"] == 0 and strict["new_candidates"] == 9
    assert novel["putative_novel_clusters"] == loci.putative_cluster.nunique() == 1
    assert len(loci) == loci.center.nunique() == 2
    assert abs(int(loci.center.iloc[0]) - int(loci.center.iloc[1])) >= 10240
    assert all(not ((annotation.start < row.end) & (annotation.end > row.start)).any()
               for row in loci.itertuples())
    assert len(pd.read_csv(required(bundle / "task2_novel_cluster_all_candidate_pairs.csv"))) == 21
    review_suggestions = pd.read_csv(required(bundle / "task2_annotation_review_model_suggestions.csv"))
    assert len(review_suggestions) == 7 and review_suggestions.center_bp.nunique() == 7
    review_page = required(bundle / "task2_annotation_review.html").read_text(encoding="utf-8")
    assert "导出人工复核 JSON" in review_page
    assert review_page.count('"suggestion": {') == len(review_suggestions)
    for center in loci.center:
        for replicate in (1, 2):
            required(bundle / f"task2_novel_cluster_N1_{center}_rep{replicate}.png")
    for condition in ("44c", "rif25", "hns"):
        transfer = read_json(bundle / f"task2_novel_cluster_transfer_{condition}.json")
        assert transfer["pair_centers_bp"] == loci.center.astype(int).tolist()
        assert transfer["independent_background_pairs"] == 7456

    # Task 3: the full PNG corpus is ignored in Git; validate the committed
    # manifest in a clean checkout, and every local PNG when present.
    starts = sorted(int(re.search(r"_(\d+)\.png$", path.name).group(1))
                    for path in (generated / "task3_labeled").glob("*.png"))
    expected_starts = list(range(0, 4_641_652, 10_000))
    if starts:
        assert starts == expected_starts
    manifest = pd.read_csv(required(bundle / "task3_track_manifest.csv"))
    assert manifest.start_bp.tolist() == expected_starts
    assert manifest.end_bp.iloc[-1] == 4_641_652
    if starts:
        for row in manifest.itertuples():
            image = generated / "task3_labeled" / row.filename
            assert image.stat().st_size == row.bytes
    required(bundle / "task3_example_005.png")

    # Task 4: paired-repeat table and a heatmap for every selected annotation.
    changes = pd.read_csv(required(bundle / "task4_reproducible_changes.csv"))
    assert len(changes) == 21 and changes.id.nunique() == 21
    assert changes.repeat_direction_agreement.all() and changes.exceeds_repeat_variation.all()
    task4 = read_json(bundle / "task4_order_invariant_sensitivity.json")
    stable = pd.read_csv(required(bundle / "task4_heat_stable_changes.csv"))
    assert len(stable) == task4["heat"]["all_seed_intersection_count"] == 17
    assert len(pd.read_csv(required(bundle / "task4_rif_stable_changes.csv"))) == 98
    for record in changes.itertuples():
        required(bundle / f"task4_{record.id}_wt.png")
        required(bundle / f"task4_{record.id}_heat_44c.png")
    required(bundle / "task4_rif_class_summary.csv")

    # Task 5: reconstruction data, interpolated baseline, and shape audit.
    sr = read_json(bundle / "task5_summary.json")
    shape = read_json(bundle / "task5_shape_validation.json")
    with np.load(required(bundle / "task5_reconstructed_test_patches.npz")) as patches:
        assert all(patches[name].shape == (208, 64, 64)
                   for name in ("truth", "bilinear", "cnn"))
    assert sr["test_patches"] == 208
    assert shape["informative_test_windows"] + shape["zero_signal_windows_excluded"] == 208
    required(bundle / "task5_super_resolution.pt")
    required(bundle / "task5_shape_validation.csv")
    known_shape = read_json(bundle / "task5_known_shape_summary.json")
    assert known_shape["known_centers"] == 68
    assert len(pd.read_csv(required(bundle / "task5_known_shape_per_structure.csv"))) == 136
    peak_trial = read_json(bundle / "task5_peak_weighted_summary.json")
    peak_shape = read_json(bundle / "task5_peak_known_shape_summary.json")
    assert peak_trial["peak_weight"] == .5 and peak_shape["known_centers"] == 68
    required(bundle / "task5_peak_weighted_super_resolution.pt")
    assert len(pd.read_csv(required(bundle / "task5_peak_known_shape_per_structure.csv"))) == 136
    sr_ensemble = read_json(bundle / "task5_ensemble_summary.json")
    assert sr_ensemble["validation_known_centers"] == 28
    assert sr_ensemble["test_known_centers"] == 68
    required(bundle / "task5_ensemble_validation_tradeoff.png")
    sr_rerun = read_json(bundle / "task5_reproducibility_audit.json")
    assert sr_rerun["matched_test_data"] and sr_rerun["test_predictions"]["patches"] == 208
    assert abs(sr_rerun["test_psnr_cnn"]["rerun"] - sr["comparison"]["psnr"]["cnn"]) < 1e-8
    assert sr_rerun["checkpoint_sha256"]["archived"] != sr_rerun["checkpoint_sha256"]["rerun"]

    # Task 6: coordinates, independent baseline, and WT-replicate check.
    rep = read_json(bundle / "task6_all_positive_replicate_validation.json")
    metric = read_json(bundle / "task6_all_positive_metrics.json")
    coords = pd.read_csv(required(bundle / "task6_all_positive_coordinates.csv"))
    assert len(coords) == metric["bins"] == 929
    assert {"x", "y", "z"}.issubset(coords.columns)
    assert np.isfinite(coords[["x", "y", "z"]].to_numpy()).all()
    with np.load(required(bundle / "task6_all_positive_contact_and_baseline.npz")) as maps:
        assert maps["original"].shape == maps["reconstructed_contact"].shape == (929, 929)
    required(bundle / "task6_all_positive_genome_3d.pt")
    required(bundle / "task6_all_positive_chromosome_3d.png")
    required(bundle / "task6_all_positive_contact_reconstruction.png")
    assert rep["rep2_cross_replicate"]["all_positive"]["pairs"] > 200_000
    assert 0 <= rep["rep2_cross_replicate"]["positive_vs_zero_distance_auc"]["all_positive_mlp"] <= 1
    matched = rep["rep2_cross_replicate"]["exact_separation_matched_auc"]
    assert matched["matched_pairs_per_class"] > 200_000
    assert abs(matched["genome_circle_baseline"] - .5) < .01
    overlay = read_json(bundle / "task6_domain_overlay_summary.json")
    assert sum(overlay["region_bin_counts"].values()) == len(coords)
    required(bundle / "task6_domain_overlay_chromosome_3d_macroregions.png")

    # Every local Markdown evidence link must resolve in a clean checkout.
    for document in ("README.md", "REPRODUCE.md", "METHODS.md", "REPORT.md",
                     "AUDIT.md", "results/README.md", "submission/PRESENTATION_OUTLINE.md"):
        path = root / document
        for destination in re.findall(r"\[[^]]+\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if not destination.startswith(("https://", "http://", "#")):
                assert (path.parent / destination).exists(), f"Broken link in {document}: {destination}"

    report = {"checks_passed": 6, "known_annotations": len(annotation),
              "task2_putative_clusters": novel["putative_novel_clusters"],
              "task3_track_pngs_local": len(starts), "task3_manifest_pngs": len(manifest),
              "task4_primary_selected_annotations": len(changes),
              "task4_stable_annotations": len(stable),
              "task5_test_patches": sr["test_patches"], "task6_bins": len(coords),
              "caution": "Completeness/integrity checks only; not proof of biological novelty or physical 3D truth."}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
