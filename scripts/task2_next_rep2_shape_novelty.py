"""Rep2-only shape-novelty audit of the previously frozen three WT loci.

The loci are read from the rep1-locked discovery; neither candidate coordinates
nor decision thresholds are chosen using their rep2 shapes. This remains a
retrospective audit because historical analyses had already viewed rep2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from microc.analysis import _features
from microc.data import annotations
from scripts.task2_rep1_locked_discovery import SEED, extract_maps, nearest_other_known


def novelty_distances(known: np.ndarray, candidates: np.ndarray,
                      known_centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Known leave-one-remote-out and candidate nearest-known distances."""
    known_calibration = nearest_other_known(known, known_centers)
    candidate_distance = np.linalg.norm(candidates[:, None] - known[None, :], axis=2)
    return known_calibration, candidate_distance


def project(known_fit: np.ndarray, known_eval: np.ndarray, candidates: np.ndarray,
            known_centers: np.ndarray, candidate_centers: list[int], mode: str,
            ann: pd.DataFrame, rep1_novel: list[bool],
            frozen_centers: set[int], frozen_cutoff: bool) -> tuple[pd.DataFrame, dict]:
    scaler = StandardScaler().fit(known_fit)
    fit_scaled = scaler.transform(known_fit)
    pca = PCA(n_components=min(20, len(known_fit) - 1, known_fit.shape[1]),
              random_state=SEED).fit(fit_scaled)
    fit_known_x = pca.transform(fit_scaled)
    known_x = pca.transform(scaler.transform(known_eval))
    candidate_x = pca.transform(scaler.transform(candidates))
    calibration, distances = novelty_distances(known_x, candidate_x, known_centers)
    reference_threshold = float(np.quantile(calibration, .9))
    threshold = (float(np.quantile(nearest_other_known(fit_known_x, known_centers), .9))
                 if frozen_cutoff else reference_threshold)
    nearest = distances.argmin(axis=1)
    rows = []
    for i, center in enumerate(candidate_centers):
        nearest_ann = ann.iloc[int(nearest[i])]
        rows.append({"projection": mode, "center_bp": center,
                     "rep1_shape_novel": bool(rep1_novel[i]),
                     "frozen_clique_member": center in frozen_centers,
                     "rep2_known_shape_distance": float(distances[i, nearest[i]]),
                     "decision_threshold": threshold,
                     "threshold_source": ("rep1_known_frozen" if frozen_cutoff else "rep2_known_reference"),
                     "rep2_known_reference_p90": reference_threshold,
                     "rep2_known_calibration_percentile": float(
                         np.mean(calibration < distances[i, nearest[i]])),
                     "passes_rep2_shape_novelty_p90": bool(distances[i, nearest[i]] > threshold),
                     "nearest_known_id": str(nearest_ann.id),
                     "nearest_known_type": str(nearest_ann.type),
                     "nearest_known_center_bp": int(nearest_ann.center),
                     "nearest_known_genomic_distance_bp": abs(center - int(nearest_ann.center))})
    all_frame = pd.DataFrame(rows)
    summary = {"projection": mode,
               "decision_threshold": threshold,
               "threshold_source": ("rep1_known_frozen" if frozen_cutoff else "rep2_known_reference"),
               "rep2_known_reference_p90": reference_threshold,
               "rep1_score_gated_unannotated_loci": len(rows),
               "rep1_shape_novel_loci": int(all_frame.rep1_shape_novel.sum()),
               "all_loci_passing_rep2_p90": int(all_frame.passes_rep2_shape_novelty_p90.sum()),
               "non_frozen_loci_passing_rep2_p90": int(all_frame.loc[
                   ~all_frame.frozen_clique_member, "passes_rep2_shape_novelty_p90"].sum()),
               "non_frozen_loci_denominator": int((~all_frame.frozen_clique_member).sum()),
               "rep1_novel_loci_passing_rep2_p90": int(all_frame.loc[
                   all_frame.rep1_shape_novel, "passes_rep2_shape_novelty_p90"].sum()),
               "rep1_novel_non_frozen_loci_passing_rep2_p90": int(all_frame.loc[
                   all_frame.rep1_shape_novel & ~all_frame.frozen_clique_member,
                   "passes_rep2_shape_novelty_p90"].sum()),
               "rep1_novel_non_frozen_loci_denominator": int((
                   all_frame.rep1_shape_novel & ~all_frame.frozen_clique_member).sum()),
               "rep1_non_novel_loci_passing_rep2_p90": int(all_frame.loc[
                   ~all_frame.rep1_shape_novel, "passes_rep2_shape_novelty_p90"].sum()),
               "frozen_clique_loci_passing_rep2_p90": int(all_frame.loc[
                   all_frame.frozen_clique_member, "passes_rep2_shape_novelty_p90"].sum()),
               "known_reference_count": len(known_centers)}
    return all_frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, default=Path("outputs/task2_rep1_locked"))
    parser.add_argument("--rep2", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/task2_next_rep2_shape_novelty"))
    args = parser.parse_args()
    summary = json.loads((args.discovery / "discovery_summary.json").read_text(encoding="utf-8"))
    if summary["rep2_used_in_selection"] is not False:
        raise ValueError("The discovery was not rep1-only")
    frozen_centers = [int(x) for x in summary["rep1_top_three_locus_clique_bp"]]
    if len(frozen_centers) != 3 or len(set(frozen_centers)) != 3:
        raise ValueError("Expected three frozen distinct candidate loci")
    eligible = pd.read_csv(args.discovery / "rep1_eligible_loci.csv")
    centers = eligible.center.astype(int).tolist()
    if not set(frozen_centers).issubset(set(centers)):
        raise ValueError("Frozen candidate loci are absent from rep1 eligibility table")
    ann = annotations(args.annotations)
    locked = np.load(args.discovery / "rep1_locked_maps.npz")
    known_centers = locked["known_centers"].astype(int)
    if len(ann) != len(known_centers) or not np.array_equal(ann.center.to_numpy(dtype=int), known_centers):
        raise ValueError("Known annotation manifest differs from frozen discovery")
    rep1_known = np.stack([_features(mat) for mat in locked["maps"][:len(ann)]])
    rep2_maps = extract_maps(args.rep2, str(ann.chrom.iloc[0]), np.r_[known_centers, centers])
    rep2_known = np.stack([_features(mat) for mat in rep2_maps[:len(ann)]])
    rep2_candidates = np.stack([_features(mat) for mat in rep2_maps[len(ann):]])
    frames, reports = [], []
    for mode, fit, frozen_cutoff in (
        ("rep1_known_frozen_projection_and_cutoff", rep1_known, True),
        ("rep1_projection_rep2_reference_recalibration_sensitivity", rep1_known, False),
        ("rep2_known_refit_sensitivity", rep2_known, False),
    ):
        frame, report = project(fit, rep2_known, rep2_candidates,
                                known_centers, centers, mode, ann,
                                eligible.passes_rep1_novelty.astype(bool).tolist(),
                                set(frozen_centers), frozen_cutoff)
        frames.append(frame)
        reports.append(report)
    result = pd.concat(frames, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out / "all_rep1_eligible_loci_rep2_novelty.csv", index=False)
    result[result.frozen_clique_member].to_csv(
        args.out / "frozen_three_loci_rep2_novelty.csv", index=False)
    report = {"frozen_centers_bp": frozen_centers,
              "rep2_path": str(args.rep2),
              "projections": reports,
              "first_projection_coordinates_model_and_decision_cutoff_frozen_on_rep1": True,
              "rep2_known_recalibrations_are_sensitivity_only": True,
              "rep1_novel_pair_hypotheses_examined": int(summary["novel_independent_pairs"]),
              "candidate_rep2_shapes_used_in_selection_or_cutoff": False,
              "caution": ("Retrospective candidate-shape audit, not prospective blinded "
                          "replication or formal new-type significance. Rep2 known "
                          "reference calibration/refit is sensitivity only; the first "
                          "projection and cutoff are fully frozen from rep1 known loci. "
                          "Candidate rep2 shapes are not used in fitting or cutoff selection. "
                          "The rep2 pass rate among all rep1 score-gated unannotated loci "
                          "is a necessary specificity context for the frozen three. "
                          "The three were selected from 120 rep1 pair hypotheses, so "
                          "three-of-three is not a single pre-specified test.")}
    (args.out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(result[result.frozen_clique_member].to_string(index=False))


if __name__ == "__main__":
    main()
