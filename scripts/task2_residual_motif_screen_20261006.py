"""Rep1-frozen, genome-wide residual morphology screen for Task 2.

The historical project has already inspected rep2. This experiment is therefore
retrospective, not a prospective blind replication. Discovery reads only rep1
and rep1-derived scan fields. A byte-hashed freeze is written before validate
can open rep2. Rep1-selected genome-wide controls remove the former restriction
to 16 novelty-screened loci; an unsupported comparator is still a failed test.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from microc.data import annotations, chrom_name, normalize, open_cool, window
from scripts.task2_rep1_locked_discovery import corr_matrix, independent_greedy


WINDOW_BP = 5120
INDEPENDENCE_BP = 10240
SEED = 20261006
METRICS = ("highpass", "rowcol")
PROTOCOL = {
    "window_bp": WINDOW_BP, "independence_bp": INDEPENDENCE_BP,
    "smoothing_sigma_bins": 1.0, "highpass_sigma_bins": 4.0,
    "upper_triangle_min_separation_bins": 8,
    "known_reference_pca_dimensions": 20,
    "novelty_cutoff_known_independent_nearest_quantile": .9,
    "edge_cutoff_all_independent_unannotated_pairs_quantile": .99,
    "minimum_edge_correlation": .2, "minimum_independent_motif_loci": 3,
    "maximum_disjoint_triads": 10,
    "control_two_iqr_caliper": 2., "maximum_controls_per_edge": 200,
    "minimum_controls_per_edge": 20,
    "required_control_q10_q90_balance_all_covariates": True,
    "rep2_edge_control_percentile": .99,
    "rep2_required_each_locus_frozen_novelty": True,
    "rep2_pair_exceeds_member_best_independent_known_similarity": True,
    "seed": SEED,
    "caution": "Retrospective exploratory screen. Previous rep2 inspection and multiple historical screens preclude prospective blindness, genomic FDR, or proof of a new biological type.",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def residual_vectors(maps: np.ndarray, metric: str) -> np.ndarray:
    """Fixed smoothing followed by removal of broad or additive artifacts.

    Row/column subtraction is an exact additive-effect residual; a symmetric
    matrix containing only a+b row/column effects gives zero (up to precision).
    No operation pools across sequences, windows, or replicates.
    """
    if maps.ndim != 3 or maps.shape[1:] != (64, 64):
        raise ValueError("Expected N x 64 x 64 maps")
    if metric not in METRICS:
        raise ValueError(metric)
    smooth = gaussian_filter(maps.astype(np.float64), sigma=(0, 1, 1))
    if metric == "highpass":
        residual = smooth - gaussian_filter(smooth, sigma=(0, 4, 4))
    else:
        residual = (smooth - smooth.mean(axis=1, keepdims=True)
                    - smooth.mean(axis=2, keepdims=True)
                    + smooth.mean(axis=(1, 2), keepdims=True))
    a, b = np.triu_indices(64, k=8)
    return residual[:, a, b].astype(np.float32)


def independent_nearest(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    dist = np.linalg.norm(x[:, None] - x[None, :], axis=2)
    dist[np.abs(centers[:, None] - centers[None, :]) < INDEPENDENCE_BP] = np.inf
    if not np.isfinite(dist.min(axis=1)).all():
        raise ValueError("Insufficient independent annotated references")
    return dist.min(axis=1)


def project_novelty(vectors: np.ndarray, known_n: int, known_centers: np.ndarray):
    scaler = StandardScaler().fit(vectors[:known_n])
    pca = PCA(n_components=min(20, known_n - 1), random_state=SEED).fit(
        scaler.transform(vectors[:known_n]))
    x = pca.transform(scaler.transform(vectors))
    calibration = independent_nearest(x[:known_n], known_centers)
    cutoff = float(np.quantile(calibration, .9))
    novelty = np.linalg.norm(x[:, None] - x[None, :known_n], axis=2).min(axis=1)
    return novelty, cutoff, scaler, pca


def enumerate_triads(similarities: dict, novel: np.ndarray, centers: np.ndarray,
                     cutoffs: dict) -> pd.DataFrame:
    """Every edge must pass both fixed views and spacing, without rep2 fields."""
    allowed = novel[:, None] & novel[None, :]
    allowed &= np.abs(centers[:, None] - centers[None, :]) >= INDEPENDENCE_BP
    for metric in METRICS:
        allowed &= similarities[metric] >= cutoffs[metric]
    np.fill_diagonal(allowed, False)
    rows = []
    for a in np.flatnonzero(novel):
        later = np.flatnonzero(allowed[a] & (np.arange(len(centers)) > a))
        for b, c in itertools.combinations(later, 2):
            if not allowed[b, c]:
                continue
            edges = [(a, b), (a, c), (b, c)]
            row = {"a": int(a), "b": int(b), "c": int(c),
                   "centers_bp": ",".join(str(int(centers[i])) for i in (a, b, c))}
            for metric in METRICS:
                row[f"rep1_{metric}_min_r"] = min(float(similarities[metric][i, j]) for i, j in edges)
            row["weakest_edge_margin"] = min(row[f"rep1_{m}_min_r"] - cutoffs[m] for m in METRICS)
            rows.append(row)
    columns = ["a", "b", "c", "centers_bp", "rep1_highpass_min_r", "rep1_rowcol_min_r", "weakest_edge_margin"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values(["weakest_edge_margin", "centers_bp"],
                                         ascending=[False, True]).reset_index(drop=True)


def pair_covariates(a: np.ndarray, b: np.ndarray, centers: np.ndarray,
                    density: np.ndarray, novelty: np.ndarray,
                    sims: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack((np.minimum(density[a], density[b]),
                            np.maximum(density[a], density[b]),
                            np.log1p(np.abs(centers[a] - centers[b])),
                            np.minimum(novelty[a], novelty[b]),
                            np.maximum(novelty[a], novelty[b]),
                            sims["highpass"][a, b], sims["rowcol"][a, b]))


def match_covariates(pool: np.ndarray, target: np.ndarray, permitted: np.ndarray,
                     limit: int = 200) -> tuple[np.ndarray, dict]:
    """Select controls using rep1 only; enforce support, rather than manufacture it."""
    if not len(pool):
        return np.empty(0, int), {"permitted_pairs": 0, "caliper_pairs": 0,
                                 "selected_pairs": 0, "balanced_all_covariates": False}
    scale = np.maximum(np.quantile(pool, .75, axis=0) - np.quantile(pool, .25, axis=0), 1e-6)
    standardized = np.abs((pool - target) / scale)
    valid = np.flatnonzero(permitted & (standardized <= 2).all(axis=1))
    distance = np.linalg.norm(standardized[valid], axis=1)
    selected = valid[np.argsort(distance, kind="stable")[:limit]]
    if len(selected):
        lo, hi = np.quantile(pool[selected], [.1, .9], axis=0)
        supported = (target >= lo) & (target <= hi)
    else:
        supported = np.zeros(pool.shape[1], dtype=bool)
    return selected, {
        "permitted_pairs": int(permitted.sum()), "caliper_pairs": len(valid),
        "selected_pairs": len(selected), "balanced_all_covariates": bool(supported.all()),
        "supported_covariates": supported.tolist(),
        "covariate_names": ["coverage_low", "coverage_high", "log_separation",
                            "novelty_low", "novelty_high", "rep1_highpass_r", "rep1_rowcol_r"],
        "nearest_permitted_max_iqr_gap": (float(standardized[permitted].max(axis=1).min())
                                         if permitted.any() else None),
    }


def extract(cool_path: Path, centers: np.ndarray, chrom: str, coverage_needed=True, resampling="legacy"):
    maps, coverage = [], []
    with open_cool(cool_path) as cool:
        chrom = chrom_name(cool, chrom)
        if resampling == "area":
            from microc.data import area_window
            reader = area_window
        elif resampling == "legacy":
            reader = window
        else:
            raise ValueError(resampling)
        for i, center in enumerate(centers):
            mat = normalize(reader(cool, chrom, int(center), WINDOW_BP, 64))
            maps.append(mat)
            # Density is the existing rep1 scan statistic, except known references.
            if coverage_needed:
                coverage.append(float(np.log1p(np.maximum(window(cool, chrom, int(center), WINDOW_BP, 32), 0)).mean()))
            if (i + 1) % 100 == 0:
                print(f"extracted {i + 1}/{len(centers)} {cool_path.name}", flush=True)
    return np.stack(maps).astype(np.float32), np.asarray(coverage)


def known_recovery(sim: np.ndarray, centers: np.ndarray, kinds: np.ndarray) -> dict:
    sim = sim.copy()
    sim[np.abs(centers[:, None] - centers[None, :]) < INDEPENDENCE_BP] = -np.inf
    best = sim.argmax(axis=1)
    predictions = kinds[best]
    per_type = {str(k): float((predictions[kinds == k] == k).mean()) for k in np.unique(kinds)}
    return {"nearest_independent_same_type_fraction": float((predictions == kinds).mean()),
            "macro_same_type_fraction": float(np.mean(list(per_type.values()))),
            "majority_type_fraction": float(max((kinds == k).mean() for k in np.unique(kinds))),
            "per_type_same_type_fraction": per_type,
            "per_type_n": {str(k): int((kinds == k).sum()) for k in np.unique(kinds)}}


def discover(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    cols = ["chrom", "center", "shape_score", "density_score", "reconstruction_error", "is_known", "known_distance_bp"]
    scan = pd.read_csv(args.scan, usecols=cols)
    if scan.chrom.nunique() != 1 or scan.center.duplicated().any():
        raise ValueError("Expected unique centers in a single-chromosome scan")
    for field in ("shape_score", "density_score", "reconstruction_error"):
        scan[field + "_pct"] = scan[field].rank(pct=True)
    scan["candidate_score"] = (.4 * scan.reconstruction_error_pct + .4 * scan.shape_score_pct + .2 * scan.density_score_pct)
    ann = annotations(args.annotations)
    known_n = len(ann)
    centers = np.r_[ann.center.to_numpy(int), scan.center.to_numpy(int)]
    # Extract every map from the byte-hashed source, avoiding unverified caches.
    maps, coverage = extract(args.rep1, centers, str(scan.chrom.iloc[0]), resampling=args.resampling)
    if not np.allclose(coverage[known_n:], scan.density_score.to_numpy(), rtol=1e-6, atol=1e-8):
        raise ValueError("Existing scan coverage is inconsistent with current rep1 extraction")
    novelty, cutoffs, sims, models, views = {}, {}, {}, {}, {}
    for metric in METRICS:
        vectors = residual_vectors(maps, metric)
        views[metric] = vectors
        sims[metric] = corr_matrix(vectors)
        novelty[metric], cutoffs[metric], scaler, pca = project_novelty(vectors, known_n, centers[:known_n])
        models[metric + "_mean"] = scaler.mean_
        models[metric + "_scale"] = scaler.scale_
        models[metric + "_pca_mean"] = pca.mean_
        models[metric + "_components"] = pca.components_
        scan[metric + "_novelty"] = novelty[metric][known_n:]
    # All full genome windows form the control landscape, not just score-gated outliers.
    unknown = np.flatnonzero((~scan.is_known) & (scan.known_distance_bp >= WINDOW_BP // 2))
    pa, pb = np.triu_indices(len(unknown), k=1)
    pa, pb = unknown[pa] + known_n, unknown[pb] + known_n
    independent = np.abs(centers[pa] - centers[pb]) >= INDEPENDENCE_BP
    pa, pb = pa[independent], pb[independent]
    edge_cutoffs = {m: max(.2, float(np.quantile(sims[m][pa, pb], .99))) for m in METRICS}
    selected = independent_greedy(scan.iloc[unknown])
    pos = {int(c): known_n + i for i, c in enumerate(scan.center)}
    idx = np.array([pos[int(c)] for c in selected.center])
    robust_novel = np.ones(len(idx), dtype=bool)
    for metric in METRICS:
        selected[metric + "_novelty"] = novelty[metric][idx]
        robust_novel &= novelty[metric][idx] > cutoffs[metric]
    selected["passes_both_novelty_gates"] = robust_novel
    small_sims = {m: sims[m][np.ix_(idx, idx)] for m in METRICS}
    triads = enumerate_triads(small_sims, robust_novel, centers[idx], edge_cutoffs)
    locked, used = [], []
    for row in triads.itertuples():
        members = [int(c) for c in row.centers_bp.split(",")]
        if all(abs(c - u) >= INDEPENDENCE_BP for c in members for u in used):
            locked.append(row._asdict())
            used.extend(members)
        if len(locked) >= 10:
            break
    frozen = pd.DataFrame(locked, columns=list(triads.columns))
    if "Index" in frozen:
        frozen = frozen.drop(columns="Index")
    frozen.insert(0, "candidate_id", [f"R{i+1}" for i in range(len(frozen))])
    frozen.insert(1, "candidate_role", "new_rep1_candidate")
    old = json.loads(args.historical_discovery.read_text(encoding="utf-8"))
    if old["rep2_used_in_selection"] is not False:
        raise ValueError("Historical diagnostic source must itself be rep1-only")
    historical_centers = old.get("rep1_top_three_locus_clique_bp")
    historical = pd.DataFrame(columns=["candidate_id", "candidate_role", "centers_bp"])
    if historical_centers is not None:
        if len(historical_centers) != 3 or not all(int(c) in pos for c in historical_centers):
            raise ValueError("Historical diagnostic does not map to current full-genome scan")
        historical = pd.DataFrame([{"candidate_id": "HISTORICAL", "candidate_role": "historical_diagnostic",
                                    "centers_bp": ",".join(map(str, historical_centers))}])
    scan.to_csv(args.out / "rep1_all_windows.csv", index=False)
    selected.to_csv(args.out / "rep1_independent_unannotated.csv", index=False)
    triads.to_csv(args.out / "rep1_all_eligible_triads.csv", index=False)
    frozen.to_csv(args.out / "candidate_freeze.csv", index=False)
    historical.to_csv(args.out / "historical_diagnostic_freeze.csv", index=False)
    np.savez_compressed(args.out / "rep1_maps_models.npz", maps=maps, centers=centers,
                        known_n=known_n, density=coverage, known_types=ann.type.to_numpy(str),
                        pool_a=pa, pool_b=pb, **models)
    protocol = {**PROTOCOL, "resampling": args.resampling,
                "resampling_definition": ("count-preserving native-bin area aggregation" if args.resampling == "area"
                                          else "historical scipy zoom(order=1) linear interpolation"),
                "selection_priority": "Historical rep1 AE/shape score controls representative priority only; thresholds and control rules are identical between resampling arms."}
    write_json(args.out / "protocol.json", protocol)
    known_stats = {m: known_recovery(sims[m][:known_n, :known_n], centers[:known_n], ann.type.to_numpy()) for m in METRICS}
    summary = {
        "analysis": "rep1-only retrospective residual morphology discovery",
        "protocol": protocol, "rep2_used_in_selection": False,
        "genome_windows": len(scan), "known_references": known_n,
        "full_window_unannotated_background_loci": len(unknown), "background_independent_pairs": len(pa),
        "independent_unannotated_representatives": len(selected),
        "representatives_passing_both_novelty_gates": int(robust_novel.sum()),
        "novelty_thresholds": cutoffs, "edge_thresholds": edge_cutoffs,
        "eligible_three_locus_motifs": len(triads), "frozen_disjoint_motifs": len(frozen),
        "frozen_historical_diagnostics": len(historical),
        "known_rep1_recovery": known_stats,
        "selection_used_scan_fields": cols,
        "covariate_matching": "Seven rep1 factors: both locus coverages, log separation, minimum/maximum joint novelty ratio, highpass and rowcol edge correlation. Candidate proximity excluded. No rep2 matching.",
    }
    write_json(args.out / "discovery_summary.json", summary)
    frozen_files = ["protocol.json", "candidate_freeze.csv", "historical_diagnostic_freeze.csv",
                    "rep1_all_windows.csv", "rep1_independent_unannotated.csv", "rep1_maps_models.npz", "discovery_summary.json"]
    freeze = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
              "run_code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "script_sha256": sha256(Path(__file__)),
              "implementation_sha256": {str(p).replace("\\", "/"): sha256(p) for p in
                                        (Path(__file__).resolve().relative_to(Path.cwd()),
                                         Path("microc/data.py"), Path("scripts/task2_rep1_locked_discovery.py"))},
              "input_sha256": {"rep1": sha256(args.rep1), "scan": sha256(args.scan), "annotations": sha256(args.annotations),
                               "historical_rep1_discovery": sha256(args.historical_discovery)},
              "frozen_file_sha256": {f: sha256(args.out / f) for f in frozen_files},
              "rep2_opened_in_discovery": False,
              "historical_rep2_inspection": True}
    write_json(args.out / "freeze_receipt.json", freeze)
    print(json.dumps(summary, indent=2), flush=True)


def frozen_project(vectors: np.ndarray, manifest: dict, metric: str) -> np.ndarray:
    scaled = (vectors - manifest[metric + "_mean"]) / manifest[metric + "_scale"]
    return (scaled - manifest[metric + "_pca_mean"]) @ manifest[metric + "_components"].T


def validate(args) -> None:
    freeze = json.loads((args.discovery / "freeze_receipt.json").read_text(encoding="utf-8"))
    for name, digest in freeze["frozen_file_sha256"].items():
        if sha256(args.discovery / name) != digest:
            raise ValueError(f"Freeze hash mismatch: {name}")
    if freeze["script_sha256"] != sha256(Path(__file__)):
        raise ValueError("Discovery implementation changed since candidate freeze")
    for name, digest in freeze["implementation_sha256"].items():
        if sha256(Path(name)) != digest:
            raise ValueError(f"Implementation dependency changed since candidate freeze: {name}")
    summary = json.loads((args.discovery / "discovery_summary.json").read_text(encoding="utf-8"))
    candidates = pd.read_csv(args.discovery / "candidate_freeze.csv")
    historical = pd.read_csv(args.discovery / "historical_diagnostic_freeze.csv")
    frozen = pd.concat([candidates, historical], ignore_index=True)
    archive = np.load(args.discovery / "rep1_maps_models.npz")
    centers, known_n = archive["centers"], int(archive["known_n"])
    maps1, density = archive["maps"], archive["density"]
    pa, pb = archive["pool_a"], archive["pool_b"]
    maps2, _ = extract(args.rep2, centers, str(pd.read_csv(args.discovery / "rep1_all_windows.csv").chrom.iloc[0]),
                       coverage_needed=False, resampling=summary["protocol"]["resampling"])
    sims1, sims2, novel1, novel2 = {}, {}, {}, {}
    for metric in METRICS:
        v1, v2 = residual_vectors(maps1, metric), residual_vectors(maps2, metric)
        sims1[metric], sims2[metric] = corr_matrix(v1), corr_matrix(v2)
        x1, x2 = frozen_project(v1, archive, metric), frozen_project(v2, archive, metric)
        novel1[metric] = np.linalg.norm(x1[:, None] - x1[None, :known_n], axis=2).min(axis=1)
        # Known rep2 references projected with rep1-fixed coordinates; no refitting.
        novel2[metric] = np.linalg.norm(x2[:, None] - x2[None, :known_n], axis=2).min(axis=1)
    joint1 = np.minimum(*(novel1[m] / summary["novelty_thresholds"][m] for m in METRICS))
    pool = pair_covariates(pa, pb, centers, density, joint1, sims1)
    position = {int(c): i for i, c in enumerate(centers)}
    rows, controls, motif_rows = [], [], []
    for row in frozen.itertuples():
        member_centers = [int(c) for c in row.centers_bp.split(",")]
        members = [position[c] for c in member_centers]
        away = np.array([all(abs(int(c) - t) >= INDEPENDENCE_BP for t in member_centers) for c in centers])
        allowed = away[pa] & away[pb]
        locus_pass = all(novel2[m][i] > summary["novelty_thresholds"][m] for m in METRICS for i in members)
        edge_passes = []
        for a, b in itertools.combinations(members, 2):
            target = pair_covariates(np.array([a]), np.array([b]), centers, density, joint1, sims1)[0]
            selected, diagnostic = match_covariates(pool, target, allowed)
            edges_pass = len(selected) >= 20 and diagnostic["balanced_all_covariates"]
            for k in selected:
                controls.append({"candidate_id": row.candidate_id, "candidate_role": row.candidate_role,
                                 "target_a_bp": int(centers[a]),
                                 "target_b_bp": int(centers[b]), "control_a_bp": int(centers[pa[k]]),
                                 "control_b_bp": int(centers[pb[k]]),
                                 **{f"control_{n}": float(v) for n, v in zip(diagnostic["covariate_names"], pool[k])},
                                 **{f"target_{n}": float(v) for n, v in zip(diagnostic["covariate_names"], target)}})
            for m in METRICS:
                values = sims2[m][pa[selected], pb[selected]]
                observed = float(sims2[m][a, b])
                percentile = float(np.mean(values < observed)) if len(values) else None
                known_a = np.abs(centers[:known_n] - centers[a]) >= INDEPENDENCE_BP
                known_b = np.abs(centers[:known_n] - centers[b]) >= INDEPENDENCE_BP
                best_known = float(max(sims2[m][a, :known_n][known_a].max(), sims2[m][b, :known_n][known_b].max()))
                pass_view = percentile is not None and percentile >= .99 and observed > best_known and observed > 0
                edges_pass &= pass_view
                rows.append({"candidate_id": row.candidate_id, "candidate_role": row.candidate_role,
                             "center_a_bp": int(centers[a]),
                             "center_b_bp": int(centers[b]), "metric": m,
                             "rep1_r": float(sims1[m][a, b]), "rep2_r": observed,
                             "rep2_matched_percentile": percentile,
                             "rep2_best_independent_known_r": best_known,
                             "rep2_exceeds_both_best_known": observed > best_known,
                             "rep2_locus_novelty_a": float(novel2[m][a]),
                             "rep2_locus_novelty_b": float(novel2[m][b]),
                             "frozen_novelty_threshold": summary["novelty_thresholds"][m],
                             "matched_controls": len(selected), "caliper_pairs": diagnostic["caliper_pairs"],
                             "balanced_all_covariates": diagnostic["balanced_all_covariates"],
                             "supported_covariates": ",".join(str(x) for x in diagnostic["supported_covariates"]),
                             "passes_view_rule": bool(pass_view)})
            edge_passes.append(bool(edges_pass))
        motif_rows.append({"candidate_id": row.candidate_id, "candidate_role": row.candidate_role,
                           "centers_bp": row.centers_bp,
                           "all_loci_novel_in_both_frozen_views_rep2": bool(locus_pass),
                           "all_edges_replicate_with_supported_controls": all(edge_passes),
                           "passes_exploratory_replication_rule": bool(locus_pass and all(edge_passes))})
    args.out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "candidate_edge_validation.csv", index=False)
    pd.DataFrame(controls).to_csv(args.out / "rep1_selected_controls.csv", index=False)
    pd.DataFrame(motif_rows).to_csv(args.out / "candidate_replication.csv", index=False)
    known_stats = {m: known_recovery(sims2[m][:known_n, :known_n], centers[:known_n], archive["known_types"]) for m in METRICS}
    # Specificity context across every unannotated genome window, independent of nomination.
    scan = pd.read_csv(args.discovery / "rep1_all_windows.csv")
    unknown = np.flatnonzero((~scan.is_known) & (scan.known_distance_bp >= WINDOW_BP // 2)) + known_n
    pass1 = np.logical_and.reduce([novel1[m][unknown] > summary["novelty_thresholds"][m] for m in METRICS])
    pass2 = np.logical_and.reduce([novel2[m][unknown] > summary["novelty_thresholds"][m] for m in METRICS])
    locus_evidence = scan[["chrom", "center", "is_known", "known_distance_bp", "candidate_score"]].copy()
    for m in METRICS:
        locus_evidence[f"rep1_{m}_novelty"] = novel1[m][known_n:]
        locus_evidence[f"rep2_{m}_novelty"] = novel2[m][known_n:]
        locus_evidence[f"frozen_{m}_novelty_threshold"] = summary["novelty_thresholds"][m]
    locus_evidence.to_csv(args.out / "genome_window_replication.csv", index=False)
    plot(args.out / "residual_screen.png", maps1, maps2, centers, frozen, frame, known_stats)
    report = {"analysis": "retrospective fixed-rule rep2 residual motif replication",
              "frozen_at_utc": freeze["frozen_at_utc"],
              "validation_started_after_hash_verification": True,
              "freeze_receipt_sha256": sha256(args.discovery / "freeze_receipt.json"),
              "rep2_sha256": sha256(args.rep2),
              "rep2_used_for_candidate_selection_or_matching": False,
              "frozen_disjoint_motifs": len(candidates), "historical_diagnostic_motifs": len(historical),
              "motifs_passing_rule": sum(r["passes_exploratory_replication_rule"] for r in motif_rows if r["candidate_role"] == "new_rep1_candidate"),
              "historical_diagnostics_passing_rule": sum(r["passes_exploratory_replication_rule"] for r in motif_rows if r["candidate_role"] == "historical_diagnostic"),
              "known_rep2_recovery": known_stats,
              "genome_wide_novelty_context": {"unannotated_windows": len(unknown),
                                             "rep1_both_novelty_pass": int(pass1.sum()),
                                             "rep2_both_novelty_pass": int(pass2.sum()),
                                             "pass_in_both_replicates": int((pass1 & pass2).sum())},
              "minimum_control_count": int(frame.matched_controls.min()) if len(frame) else None,
              "all_edge_control_sets_balanced": bool(frame.balanced_all_covariates.all()) if len(frame) else None,
              "protocol": summary["protocol"],
              "claim": "No validated new biological structure type. Passing this exploratory rule would still require independent samples and biological interpretation.",
              "output_sha256": {p.name: sha256(p) for p in args.out.glob("*") if p.is_file() and p.suffix in (".csv", ".png")}}
    write_json(args.out / "validation_summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


def plot(path: Path, maps1, maps2, centers, frozen, frame, known_stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = min(len(frozen), 2)
    fig, axes = plt.subplots(1 + 3 * n, 2, figsize=(9, 3 + 7 * n), dpi=150, squeeze=False)
    for col, m in enumerate(METRICS):
        kinds = list(known_stats[m]["per_type_same_type_fraction"])
        values = [known_stats[m]["per_type_same_type_fraction"][k] for k in kinds]
        axes[0, col].bar(kinds, values, color=["#296f9b", "#d28d2d", "#68679b"])
        axes[0, col].set(ylim=(0, 1), ylabel="Same-type nearest-reference fraction", title=f"Rep2 fixed {m} known-reference retrieval")
    pos = {int(c): i for i, c in enumerate(centers)}
    for group, row in enumerate(frozen.head(n).itertuples()):
        for loc, center in enumerate(map(int, row.centers_bp.split(","))):
            i = pos[center]
            for col, (source, label) in enumerate(((maps1, "rep1"), (maps2, "rep2"))):
                ax = axes[1 + 3 * group + loc, col]
                ax.imshow(source[i], origin="lower", cmap="magma", vmin=-2, vmax=4)
                ax.set_title(f"{row.candidate_id}: {center:,} bp / {label}")
    fig.suptitle("Residual artifact controls and locked motifs (retrospective)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    first = sub.add_parser("discover")
    first.add_argument("--rep1", type=Path, required=True)
    first.add_argument("--scan", type=Path, default=Path("results/task2_all_windows.csv"))
    first.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    first.add_argument("--historical-discovery", type=Path, default=Path("results/task2_rep1_locked_discovery_summary.json"))
    first.add_argument("--resampling", choices=("legacy", "area"), default="legacy")
    first.add_argument("--out", type=Path, default=Path("outputs/task2_improve_20261006"))
    second = sub.add_parser("validate")
    second.add_argument("--rep2", type=Path, required=True)
    second.add_argument("--discovery", type=Path, default=Path("outputs/task2_improve_20261006"))
    second.add_argument("--out", type=Path, default=Path("outputs/task2_improve_20261006/validation"))
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        {"discover": discover, "validate": validate}[args.command](args)


if __name__ == "__main__":
    main()
