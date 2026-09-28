"""Rep1-only motif discovery followed by a locked-locus rep2 audit.

This is a retrospective split: rep2 was inspected in earlier analyses, so it
must not be described as a prospective blinded experiment. The two subcommands
make the data boundary explicit and independently executable. No statistic
from rep2 influences candidate selection, shape threshold, or pair selection.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from microc.analysis import _features
from microc.data import annotations, chrom_name, normalize, open_cool, window


WINDOW_BP = 5120
INDEPENDENCE_BP = 10240
SEED = 20260928
UPPER = np.triu_indices(64, k=2)


def corr_matrix(vectors: np.ndarray) -> np.ndarray:
    x = np.asarray(vectors, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    length = np.linalg.norm(x, axis=1, keepdims=True)
    x /= np.maximum(length, 1e-12)
    return np.clip(x @ x.T, -1, 1)


def independent_greedy(table: pd.DataFrame) -> pd.DataFrame:
    """One top-scoring representative per 10.24-kb neighborhood."""
    ranked = table.sort_values(["candidate_score", "center"], ascending=[False, True])
    chosen: list[int] = []
    rows = []
    for _, row in ranked.iterrows():
        center = int(row.center)
        if all(abs(center - other) >= INDEPENDENCE_BP for other in chosen):
            chosen.append(center)
            rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def nearest_other_known(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    dist = np.linalg.norm(x[:, None] - x[None, :], axis=2)
    dist[np.abs(centers[:, None] - centers[None, :]) < INDEPENDENCE_BP] = np.inf
    nearest = dist.min(axis=1)
    if not np.isfinite(nearest).all():
        raise ValueError("Known annotations lack independent shape references")
    return nearest


def pair_rows(centers: np.ndarray, score: np.ndarray, novelty: np.ndarray,
              upper_r: np.ndarray, coarse_r: np.ndarray) -> pd.DataFrame:
    rows = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            if abs(int(centers[i]) - int(centers[j])) < INDEPENDENCE_BP:
                continue
            rows.append({"i": i, "j": j, "center_a_bp": int(centers[i]),
                         "center_b_bp": int(centers[j]),
                         "distance_bp": abs(int(centers[i]) - int(centers[j])),
                         "min_score": float(min(score[i], score[j])),
                         "mean_score": float((score[i] + score[j]) / 2),
                         "min_novelty": float(min(novelty[i], novelty[j])),
                         "mean_novelty": float((novelty[i] + novelty[j]) / 2),
                         "rep1_upper_r": float(upper_r[i, j]),
                         "rep1_coarse_r": float(coarse_r[i, j])})
    return pd.DataFrame(rows)


def rank_morphology(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for col in ("rep1_upper_r", "rep1_coarse_r"):
        result[col + "_pct"] = result[col].rank(method="min", pct=True)
    result["rep1_joint_percentile_floor"] = result[
        ["rep1_upper_r_pct", "rep1_coarse_r_pct"]].min(axis=1)
    return result.sort_values(["rep1_joint_percentile_floor", "rep1_upper_r",
                               "center_a_bp", "center_b_bp"],
                              ascending=[False, False, True, True]).reset_index(drop=True)


def rep1_cliques(pairs: pd.DataFrame) -> pd.DataFrame:
    """Enumerate three-node cliques using only locked rep1 pair ranks."""
    if pairs.empty:
        return pd.DataFrame()
    edge = {(min(int(r.i), int(r.j)), max(int(r.i), int(r.j))): r
            for r in pairs.itertuples() if r.rep1_upper_r_pct >= .9 and
            r.rep1_coarse_r_pct >= .9}
    nodes = sorted(set(pairs.i.astype(int)) | set(pairs.j.astype(int)))
    index = {(min(int(r.i), int(r.j)), max(int(r.i), int(r.j))): r
             for r in pairs.itertuples()}
    rows = []
    for triple in itertools.combinations(nodes, 3):
        keys = list(itertools.combinations(triple, 2))
        if not all(key in edge for key in keys):
            continue
        members = sorted(set([int(index[key].center_a_bp) for key in keys] +
                             [int(index[key].center_b_bp) for key in keys]))
        floors = [float(index[key].rep1_joint_percentile_floor) for key in keys]
        rows.append({"node_indices": ",".join(map(str, triple)),
                     "centers_bp": ",".join(map(str, members)),
                     "rep1_min_joint_percentile_floor": min(floors),
                     "rep1_mean_joint_percentile_floor": float(np.mean(floors))})
    if not rows:
        return pd.DataFrame(columns=["node_indices", "centers_bp",
                                     "rep1_min_joint_percentile_floor",
                                     "rep1_mean_joint_percentile_floor"])
    return pd.DataFrame(rows).sort_values(
        ["rep1_min_joint_percentile_floor", "rep1_mean_joint_percentile_floor", "centers_bp"],
        ascending=[False, False, True]).reset_index(drop=True)


def extract_maps(cool_path: Path, chrom: str, centers: np.ndarray) -> np.ndarray:
    with open_cool(cool_path) as cool:
        chrom = chrom_name(cool, chrom)
        return np.stack([normalize(window(cool, chrom, int(center), WINDOW_BP, 64))
                         for center in centers]).astype(np.float32)


def discover(args: argparse.Namespace) -> None:
    # Only rep1-derived fields are read. In the historical scan, the AE was
    # trained and evaluated on rep1. replicate_oe_r is deliberately excluded.
    # Recompute the published score from source components rather than trusting
    # a cached candidate-score field in a table that also contains rep2 data.
    cols = ["chrom", "center", "shape_score", "density_score",
            "reconstruction_error", "is_known", "known_distance_bp"]
    scan = pd.read_csv(args.scan, usecols=cols)
    for field in ("shape_score", "density_score", "reconstruction_error"):
        scan[field + "_pct"] = scan[field].rank(pct=True)
    scan["candidate_score"] = (.4 * scan.reconstruction_error_pct +
                               .4 * scan.shape_score_pct + .2 * scan.density_score_pct)
    if scan.chrom.nunique() != 1:
        raise ValueError("Expected the one-chromosome E. coli genome")
    ann = annotations(args.annotations)
    if len(ann) < 20:
        raise ValueError("Too few known structures for novelty calibration")
    cutoff = float(scan.candidate_score.quantile(.8))
    eligible = scan[(scan.candidate_score >= cutoff) & (~scan.is_known) &
                    (scan.known_distance_bp >= WINDOW_BP // 2)]
    eligible = independent_greedy(eligible)
    if len(eligible) < 5:
        raise ValueError("Too few independent rep1-only score-gated windows")
    centers = eligible.center.to_numpy(dtype=int)
    known_centers = ann.center.to_numpy(dtype=int)
    all_centers = np.r_[known_centers, centers]
    maps = extract_maps(args.rep1, str(scan.chrom.iloc[0]), all_centers)
    coarse = np.stack([_features(mat) for mat in maps])
    # Fit the reference metric on known structures only, before scoring unknowns.
    scaler = StandardScaler().fit(coarse[:len(ann)])
    known_scaled = scaler.transform(coarse[:len(ann)])
    latent_dim = min(20, len(ann) - 1, coarse.shape[1])
    pca = PCA(n_components=latent_dim, random_state=SEED).fit(known_scaled)
    embedded = pca.transform(scaler.transform(coarse))
    known_x, candidate_x = embedded[:len(ann)], embedded[len(ann):]
    calibration = nearest_other_known(known_x, known_centers)
    novelty_cutoff = float(np.quantile(calibration, .9))
    candidate_novelty = np.linalg.norm(candidate_x[:, None] - known_x[None, :], axis=2).min(axis=1)
    eligible["known_shape_distance"] = candidate_novelty
    eligible["passes_rep1_novelty"] = candidate_novelty > novelty_cutoff
    novel_idx = np.flatnonzero(eligible.passes_rep1_novelty.to_numpy())
    n_known = len(ann)
    upper_sim = corr_matrix(maps[:, UPPER[0], UPPER[1]])
    coarse_sim = corr_matrix(coarse)
    if len(novel_idx) >= 2:
        novel = eligible.iloc[novel_idx].reset_index(drop=True)
        original_idx = n_known + novel_idx
        pairs = pair_rows(novel.center.to_numpy(dtype=int),
                          novel.candidate_score.to_numpy(dtype=float),
                          novel.known_shape_distance.to_numpy(dtype=float),
                          upper_sim[np.ix_(original_idx, original_idx)],
                          coarse_sim[np.ix_(original_idx, original_idx)])
        pairs = rank_morphology(pairs) if len(pairs) else pairs
    else:
        pairs = pd.DataFrame()
    cliques = rep1_cliques(pairs)
    args.out.mkdir(parents=True, exist_ok=True)
    eligible.to_csv(args.out / "rep1_eligible_loci.csv", index=False)
    pairs.to_csv(args.out / "rep1_novel_pairs.csv", index=False)
    cliques.to_csv(args.out / "rep1_three_locus_cliques.csv", index=False)
    np.savez_compressed(args.out / "rep1_locked_maps.npz", centers=all_centers,
                        maps=maps, coarse=coarse, eligible_centers=centers,
                        known_centers=known_centers, novel_indices=novel_idx)
    selected = (None if pairs.empty else
                [int(pairs.iloc[0].center_a_bp), int(pairs.iloc[0].center_b_bp)])
    summary = {
        "analysis": "retrospective rep1-only discovery; earlier rep2 inspection prevents prospective blindness",
        "rep1_path": str(args.rep1), "scan_path": str(args.scan),
        "scan_fields_used": cols, "score_gate_quantile": .8,
        "score_gate": cutoff, "independence_bp": INDEPENDENCE_BP,
        "known_annotations": len(ann), "known_shape_distance_p90": novelty_cutoff,
        "score_gated_independent_loci": len(eligible),
        "shape_novel_loci": len(novel_idx), "novel_independent_pairs": len(pairs),
        "rep1_top_pair_bp": selected,
        "rep1_three_locus_cliques": len(cliques),
        "rep1_top_three_locus_clique_bp": (
            [int(value) for value in cliques.iloc[0].centers_bp.split(",")]
            if len(cliques) else None),
        "three_locus_rule": "Every edge at or above the rep1 90th percentile in both upper and coarse similarity among independently spaced rep1-novel pairs; select clique maximizing its weakest joint rank",
        "rep2_used_in_selection": False,
        "caution": "Unannotated, rep1 shape-outlying positions are candidates, not validated new types."}
    (args.out / "discovery_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def nearest_matched_controls(pairs: pd.DataFrame, selected_index: int,
                             max_controls: int = 100) -> tuple[np.ndarray, np.ndarray, dict]:
    """Require independent pairs and broad two-IQR balance in all covariates.

    If no such comparator exists, return an empty set; ranking against highly
    dissimilar pairs would manufacture an apparent validation signal.
    """
    if len(pairs) < 3:
        return np.empty(0, dtype=int), np.empty(0), {"independent_pairs": 0,
                                                       "two_iqr_caliper_pairs": 0}
    covariates = pairs[["min_score", "mean_score", "min_novelty", "mean_novelty",
                        "rep1_upper_r", "rep1_coarse_r"]].to_numpy(dtype=float)
    covariates = np.c_[covariates, np.log1p(pairs.distance_bp.to_numpy(dtype=float))]
    scale = np.maximum(np.subtract(*np.quantile(covariates, [.75, .25], axis=0)), 1e-6)
    standardized = np.abs((covariates - covariates[selected_index]) / scale)
    distance = np.linalg.norm(standardized, axis=1)
    row = pairs.iloc[selected_index]
    independent = ~((pairs.i == row.i) | (pairs.i == row.j) |
                    (pairs.j == row.i) | (pairs.j == row.j))
    balanced = (standardized <= 2).all(axis=1)
    valid = np.flatnonzero(independent & balanced)
    chosen = valid[np.argsort(distance[valid], kind="stable")[:max_controls]]
    diagnostic = {"independent_pairs": int(independent.sum()),
                  "two_iqr_caliper_pairs": len(valid),
                  "covariate_only_caliper_sensitivity": {
                      str(bound): int((independent & (standardized <= bound).all(axis=1)).sum())
                      for bound in (1, 2, 4, 8, 10)
                  },
                  "nearest_independent_standardized_distance": (
                      float(distance[independent].min()) if independent.any() else None),
                  "nearest_independent_largest_covariate_gap_iqr": (
                      float(standardized[independent].max(axis=1).min())
                      if independent.any() else None)}
    return chosen, distance[chosen], diagnostic


def plot_locked_pair(rep1: np.ndarray, rep2: np.ndarray, pairs: pd.DataFrame,
                     eligible_index: dict[int, int], out: Path) -> None:
    """Show frozen loci and the full, *unmatched* pair landscape."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = pairs.iloc[0]
    loci = [int(selected.center_a_bp), int(selected.center_b_bp)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=160)
    for row, center in enumerate(loci):
        idx = eligible_index[center]
        for col, (label, source) in enumerate((("rep1", rep1), ("rep2", rep2))):
            axes[row, col].imshow(source[idx], origin="lower", cmap="magma", vmin=-2, vmax=4)
            axes[row, col].set_title(f"{center:,} bp — {label}")
            axes[row, col].set(xlabel="Genomic bin", ylabel="Genomic bin")
    for row, metric in enumerate(("upper", "coarse")):
        ax = axes[row, 2]
        ax.scatter(pairs[f"rep1_{metric}_r"], pairs[f"rep2_{metric}_r"],
                   s=20, color="gray", alpha=.45, label="all rep1-novel pairs")
        ax.scatter([selected[f"rep1_{metric}_r"]], [selected[f"rep2_{metric}_r"]],
                   marker="*", s=170, color="crimson", label="locked pair")
        ax.set(xlabel=f"rep1 {metric} shape r", ylabel=f"rep2 {metric} shape r")
        ax.legend(fontsize=7)
    fig.suptitle("Rep1-locked candidate: visually replicated, but no covariate-matched independent controls")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_locked_clique(rep1: np.ndarray, rep2: np.ndarray, centers: list[int],
                       eligible_index: dict[int, int], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(centers), 2, figsize=(7, 3 * len(centers)), dpi=160)
    for row, center in enumerate(centers):
        idx = eligible_index[center]
        for col, (label, source) in enumerate((("rep1", rep1), ("rep2", rep2))):
            axes[row, col].imshow(source[idx], origin="lower", cmap="magma", vmin=-2, vmax=4)
            axes[row, col].set_title(f"{center:,} bp — {label}")
            axes[row, col].set(xlabel="Genomic bin", ylabel="Genomic bin")
    fig.suptitle("Rep1-locked three-locus motif: exploratory, not a validated new type")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def validate(args: argparse.Namespace) -> None:
    discovery = json.loads((args.discovery / "discovery_summary.json").read_text(encoding="utf-8"))
    if discovery["rep2_used_in_selection"] is not False:
        raise ValueError("Discovery was not rep1-only")
    eligible = pd.read_csv(args.discovery / "rep1_eligible_loci.csv")
    locked_pair_file = args.discovery / "rep1_novel_pairs.csv"
    locked_pair_sha256 = hashlib.sha256(locked_pair_file.read_bytes()).hexdigest()
    pairs = pd.read_csv(locked_pair_file)
    if pairs.empty:
        report = {"rep1_pair_nominated": False,
                  "reason": "Fewer than two independent shape-novel loci under rep1-only gates"}
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "validation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    locked = np.load(args.discovery / "rep1_locked_maps.npz")
    all_centers = locked["centers"]
    n_known = len(locked["known_centers"])
    n_eligible = len(eligible)
    if not np.array_equal(all_centers[n_known:], eligible.center.to_numpy(dtype=int)):
        raise ValueError("Discovery manifest and saved maps differ")
    rep2 = extract_maps(args.rep2, str(eligible.chrom.iloc[0]), all_centers)
    sim_upper = corr_matrix(rep2[:, UPPER[0], UPPER[1]])
    sim_coarse = corr_matrix(np.stack([_features(mat) for mat in rep2]))
    eligible_index = {int(center): n_known + i for i, center in enumerate(eligible.center)}
    pairs = pairs.copy()
    idx_a = pairs.center_a_bp.map(eligible_index).to_numpy(dtype=int)
    idx_b = pairs.center_b_bp.map(eligible_index).to_numpy(dtype=int)
    pairs["rep2_upper_r"] = sim_upper[idx_a, idx_b]
    pairs["rep2_coarse_r"] = sim_coarse[idx_a, idx_b]
    for metric in ("upper", "coarse"):
        pairs[f"rep2_{metric}_r_pct"] = pairs[f"rep2_{metric}_r"].rank(method="min", pct=True)
    pairs["rep2_pair_better_than_member_best_known_upper"] = [
        bool(sim_upper[a, b] > max(sim_upper[a, :n_known].max(), sim_upper[b, :n_known].max()))
        for a, b in zip(idx_a, idx_b)]
    pairs["rep2_pair_better_than_member_best_known_coarse"] = [
        bool(sim_coarse[a, b] > max(sim_coarse[a, :n_known].max(), sim_coarse[b, :n_known].max()))
        for a, b in zip(idx_a, idx_b)]
    best = pairs.iloc[0]
    match_idx, match_dist, matching = nearest_matched_controls(pairs, 0)
    # Controls are rep1-novel pairs, matched to the locked pair on both member
    # scores, novelty, rep1 morphology and genomic separation. This is a
    # descriptive rank, not a formal p-value: rep2 was seen in earlier
    # exploratory work, and even matched pairs need not be exchangeable.
    ranks = {}
    for metric in ("upper", "coarse"):
        vals = pairs.iloc[match_idx][f"rep2_{metric}_r"].to_numpy(dtype=float)
        ranks[metric] = (float(np.mean(vals < best[f"rep2_{metric}_r"]))
                         if len(vals) else None)
    independent = ~((pairs.i == best.i) | (pairs.i == best.j) |
                    (pairs.j == best.i) | (pairs.j == best.j))
    unadjusted = {metric: float(np.mean(pairs.loc[independent, f"rep2_{metric}_r"] <
                                        best[f"rep2_{metric}_r"]))
                  for metric in ("upper", "coarse")}
    passes = (len(match_idx) >= 20 and all(value is not None and value >= .99 for value in ranks.values())
              and bool(best.rep2_pair_better_than_member_best_known_upper)
              and bool(best.rep2_pair_better_than_member_best_known_coarse))
    args.out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.out / "rep2_all_locked_pairs.csv", index=False)
    pairs.iloc[match_idx].assign(match_distance=match_dist).to_csv(
        args.out / "rep2_matched_controls.csv", index=False)
    plot_locked_pair(locked["maps"], rep2, pairs, eligible_index,
                     args.out / "rep1_locked_pair_replication.png")
    cliques = pd.read_csv(args.discovery / "rep1_three_locus_cliques.csv")
    clique_report = {"rep1_three_locus_cliques": len(cliques),
                     "rep1_locked_clique_bp": None,
                     "rep2_all_edges_top_decile_both_views": False,
                     "rep2_all_edges_better_than_member_best_known_both_views": False}
    if len(cliques):
        clique = [int(value) for value in str(cliques.iloc[0].centers_bp).split(",")]
        clique_edges = pairs[
            pairs.center_a_bp.isin(clique) & pairs.center_b_bp.isin(clique)].copy()
        if len(clique_edges) != 3:
            raise ValueError("Rep1 locked clique is not a three-edge clique in rep2 table")
        clique_edges.to_csv(args.out / "rep2_locked_clique_edges.csv", index=False)
        clique_report = {
            "rep1_three_locus_cliques": len(cliques),
            "rep1_locked_clique_bp": clique,
            "rep2_min_upper_r": float(clique_edges.rep2_upper_r.min()),
            "rep2_min_coarse_r": float(clique_edges.rep2_coarse_r.min()),
            "rep2_min_upper_percentile_unmatched": float(clique_edges.rep2_upper_r_pct.min()),
            "rep2_min_coarse_percentile_unmatched": float(clique_edges.rep2_coarse_r_pct.min()),
            "rep2_all_edges_top_decile_both_views": bool((
                (clique_edges.rep2_upper_r_pct >= .9) &
                (clique_edges.rep2_coarse_r_pct >= .9)).all()),
            "rep2_all_edges_better_than_member_best_known_both_views": bool((
                clique_edges.rep2_pair_better_than_member_best_known_upper &
                clique_edges.rep2_pair_better_than_member_best_known_coarse).all()),
        }
        plot_locked_clique(locked["maps"], rep2, clique, eligible_index,
                           args.out / "rep1_locked_clique_replication.png")
    report = {"analysis": "retrospective rep1-only selection; rep2 held out within this run only",
              "rep1_locked_pair_table_sha256": locked_pair_sha256,
              "rep1_pair_nominated": True,
              "selected_pair_bp": [int(best.center_a_bp), int(best.center_b_bp)],
              "n_rep1_novel_pairs": len(pairs), "matched_control_pairs": len(match_idx),
              "matching_diagnostics": matching,
              "matched_distance_median": float(np.median(match_dist)) if len(match_dist) else None,
              "rep1_upper_r": float(best.rep1_upper_r),
              "rep1_coarse_r": float(best.rep1_coarse_r),
              "rep2_upper_r": float(best.rep2_upper_r),
              "rep2_coarse_r": float(best.rep2_coarse_r),
              "rep2_matched_percentiles_strict": ranks,
              "rep2_unadjusted_independent_pair_percentiles_descriptive": unadjusted,
              "rep2_better_than_known_upper": bool(best.rep2_pair_better_than_member_best_known_upper),
              "rep2_better_than_known_coarse": bool(best.rep2_pair_better_than_member_best_known_coarse),
              "three_locus_audit": clique_report,
              "passes_exploratory_rep2_rule": bool(passes),
              "rule": "At least 20 controls sharing no selected locus and within two IQR of all seven score, novelty, rep1 morphology and separation covariates; rep2 upper and coarse above 99% of controls; pair exceeds each member's best annotated-reference similarity in both views",
              "caution": "No formal discovery p/FDR. An unmatched descriptive percentile cannot validate a new class. Historical analyses inspected rep2; this is not prospective blindness or biological proof."}
    (args.out / "validation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    first = subs.add_parser("discover", help="select candidates from rep1 only")
    first.add_argument("--rep1", type=Path, required=True)
    first.add_argument("--scan", type=Path, default=Path("results/task2_all_windows.csv"))
    first.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    first.add_argument("--out", type=Path, default=Path("outputs/task2_rep1_locked"))
    second = subs.add_parser("validate", help="audit frozen loci using rep2")
    second.add_argument("--rep2", type=Path, required=True)
    second.add_argument("--discovery", type=Path, default=Path("outputs/task2_rep1_locked"))
    second.add_argument("--out", type=Path, default=Path("outputs/task2_rep1_locked"))
    args = parser.parse_args()
    {"discover": discover, "validate": validate}[args.command](args)


if __name__ == "__main__":
    main()
