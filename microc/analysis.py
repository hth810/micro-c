"""Tasks 2-4: discovery, genome tracks, and condition differences."""
from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .data import annotation_chrom_mask, annotations, chrom_name, normalize, open_cool, save_heatmap, upper_correlation, window


class WindowAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, 2, 1), nn.ReLU(), nn.Flatten(), nn.Linear(16 * 8 * 8, 32))
        self.decoder = nn.Sequential(nn.Linear(32, 16 * 8 * 8), nn.ReLU(), nn.Unflatten(1, (16, 8, 8)),
            nn.ConvTranspose2d(16, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 4, 2, 1))

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


def _learned_features(matrices, epochs: int, seed: int, out: Path):
    """Learn on rep1 only; rep2 is reserved for independent repeatability checks."""
    if epochs <= 0:
        return np.zeros((len(matrices), 0), np.float32), np.zeros(len(matrices))
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WindowAutoencoder().to(device)
    x = torch.from_numpy(np.stack([pair[0] for pair in matrices])[:, None])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        for indices in torch.randperm(len(x)).split(64):
            batch = x[indices].to(device)
            _, reconstruction = model(batch)
            loss = nn.functional.mse_loss(reconstruction, batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval(); latents, errors = [], []
    with torch.no_grad():
        for batch in x.split(64):
            batch = batch.to(device)
            embedding, reconstruction = model(batch)
            latents.append(embedding.cpu().numpy())
            errors.append(((reconstruction - batch) ** 2).mean(dim=(1, 2, 3)).cpu().numpy())
    torch.save({"model_state_dict": model.cpu().state_dict(), "seed": seed, "epochs": epochs}, out / "autoencoder.pt")
    return np.concatenate(latents), np.concatenate(errors)


def _features(mat: np.ndarray) -> np.ndarray:
    """Coarse shape plus strand/corner profiles; independent of mean coverage."""
    n = len(mat)
    coarse = mat.reshape(16, n // 16, 16, n // 16).mean(axis=(1, 3))
    coarse = (coarse + coarse.T) / 2
    band = np.array([np.mean(np.diagonal(mat, offset=d)) for d in (2, 4, 8, 16, 32)])
    return np.r_[coarse[np.triu_indices(16, 2)], band].astype(np.float32)


def _shape_score(mat: np.ndarray) -> float:
    smoothed = gaussian_filter(mat, 2)
    n = len(mat)
    mask = np.triu(np.ones((n, n), bool), k=3)
    return float(np.quantile(smoothed[mask], .98) - np.median(smoothed[mask]))


def discover(args) -> None:
    ann = annotations(args.annotations)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open_cool(args.rep1) as c1, open_cool(args.rep2) as c2:
        if c1.binsize != c2.binsize or tuple(c1.chromnames) != tuple(c2.chromnames):
            raise ValueError("Replicates must have identical chromosomes and bin size")
        records, features, matrices = [], [], []
        for chrom in c1.chromnames:
            size = int(c1.chromsizes[chrom])
            centers = range(args.window_bp // 2, size - args.window_bp // 2, args.stride_bp)
            local_ann = ann[annotation_chrom_mask(ann, chrom)]
            for center in centers:
                m1 = normalize(window(c1, chrom, center, args.window_bp, 64))
                m2 = normalize(window(c2, chrom, center, args.window_bp, 64))
                if len(local_ann):
                    gaps = np.maximum(np.maximum(local_ann.start.to_numpy() - center,
                                                   center - local_ann.end.to_numpy()), 0)
                    nearest = int(np.min(gaps))
                    overlap = bool(((local_ann.start < center + args.window_bp // 2) &
                                    (local_ann.end > center - args.window_bp // 2)).any())
                else:
                    nearest, overlap = -1, False
                shape = _shape_score(m1)
                density = float(np.log1p(np.maximum(window(c1, chrom, center, args.window_bp, 32), 0)).mean())
                records.append(dict(chrom=chrom, start=center - args.window_bp // 2,
                                    end=center + args.window_bp // 2, center=center,
                                    length_bp=args.window_bp, shape_score=shape, density_score=density,
                                    replicate_oe_r=upper_correlation(m1, m2),
                                    known_distance_bp=nearest, is_known=overlap))
                features.append(_features(m1)); matrices.append((m1, m2))
        table = pd.DataFrame(records)
        if table.empty:
            raise ValueError("No full genome windows were produced")
        latent, reconstruction_error = _learned_features(matrices, args.ae_epochs, args.seed, out)
        table["reconstruction_error"] = reconstruction_error
        # Calibrate shape and density separately; detection emphasizes shape.
        for field in ("shape_score", "density_score"):
            table[field + "_pct"] = table[field].rank(pct=True)
        if args.ae_epochs:
            table["reconstruction_error_pct"] = table.reconstruction_error.rank(pct=True)
            table["candidate_score"] = (.4 * table.reconstruction_error_pct +
                                        .4 * table.shape_score_pct + .2 * table.density_score_pct)
        else:
            table["candidate_score"] = .75 * table.shape_score_pct + .25 * table.density_score_pct
        table.to_csv(out / "all_windows.csv", index=False)
        threshold = table.candidate_score.quantile(1 - args.top_fraction)
        selected = table[(table.candidate_score >= threshold) &
                         (table.replicate_oe_r >= args.min_replicate_r)].copy()
        if len(selected) < args.clusters:
            raise ValueError("Too few reproducible candidates; relax thresholds")
        all_features = np.concatenate([np.stack(features), latent], axis=1)
        x = all_features[selected.index]
        x = StandardScaler().fit_transform(x)
        x = PCA(n_components=min(20, x.shape[0] - 1, x.shape[1]), random_state=args.seed).fit_transform(x)
        cluster_model = KMeans(n_clusters=args.clusters, random_state=args.seed, n_init=10).fit(x)
        selected["cluster"] = cluster_model.labels_
        counts = selected.groupby("cluster").size()
        known_counts = selected.groupby("cluster").is_known.sum()
        selected["cluster_size"] = selected.cluster.map(counts)
        selected["cluster_known_count"] = selected.cluster.map(known_counts)
        selected["known_feature_distance"] = np.nan
        selected["novel_cluster"] = -1
        selected["novel_cluster_size"] = 0
        selected["candidate_individual"] = False
        novelty_threshold = None
        known_mask = selected.is_known.to_numpy()
        if known_mask.sum() >= 3 and (~known_mask).sum() >= args.clusters:
            known_neighbors = NearestNeighbors(n_neighbors=2).fit(x[known_mask])
            known_nearest = known_neighbors.kneighbors(x[known_mask])[0][:, 1]
            novelty_threshold = float(np.quantile(known_nearest, .9))
            distances = known_neighbors.kneighbors(x[~known_mask], n_neighbors=1)[0][:, 0]
            selected.loc[~selected.is_known, "known_feature_distance"] = distances
            distinct = (~known_mask) & (selected.known_feature_distance.to_numpy() > novelty_threshold)
            selected.loc[distinct, "candidate_individual"] = True
            if distinct.sum() >= args.clusters:
                novelty_model = KMeans(n_clusters=args.clusters, random_state=args.seed,
                                       n_init=10).fit(x[distinct])
                selected.loc[distinct, "novel_cluster"] = novelty_model.labels_
                novel_counts = selected[selected.novel_cluster >= 0].groupby("novel_cluster").size()
                selected.loc[distinct, "novel_cluster_size"] = selected.loc[distinct, "novel_cluster"].map(novel_counts)
        selected["candidate_cluster_supported"] = selected.candidate_individual & (selected.novel_cluster >= 0) & (
            selected.novel_cluster_size >= args.min_cluster)
        candidate_rule = getattr(args, "candidate_rule", "cluster")
        selected["candidate_new"] = (selected.candidate_individual if candidate_rule == "individual"
                                     else selected.candidate_cluster_supported)
        selected["known_distinction"] = np.where(selected.is_known, "overlaps a known structure",
            np.where(selected.candidate_new, "non-overlapping, distant from known feature manifold, replicated",
                     "unannotated but feature novelty or cluster-size criterion not met"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        for group, color in ((False, "steelblue"), (True, "darkorange")):
            subset = selected.is_known.to_numpy() == group
            ax.scatter(x[subset, 0], x[subset, 1], s=10, alpha=.6,
                       label="known overlap" if group else "unannotated", c=color)
        ax.set(xlabel="PC1", ylabel="PC2", title="Candidate window shape features")
        ax.legend(); fig.tight_layout(); fig.savefig(out / "embedding.png"); plt.close(fig)
        auc = None
        labels = table.is_known.to_numpy(dtype=int)
        if len(np.unique(labels)) == 2:
            groups = (table.center // 100000).to_numpy()
            if len(np.unique(groups)) >= 2:
                train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=.25,
                    random_state=args.seed).split(features, labels, groups))
                if len(np.unique(labels[train_idx])) == 2 and len(np.unique(labels[test_idx])) == 2:
                    scaler = StandardScaler().fit(all_features[train_idx])
                    clf = LogisticRegression(max_iter=500).fit(scaler.transform(all_features[train_idx]), labels[train_idx])
                    auc = float(roc_auc_score(labels[test_idx], clf.predict_proba(
                        scaler.transform(all_features[test_idx]))[:, 1]))
        selected.to_csv(out / "all_candidates.csv", index=False)
        selected[selected.candidate_new].to_csv(out / "new_candidates.csv", index=False)
        selected[selected.candidate_individual].to_csv(out / "individual_novelty_candidates.csv", index=False)
        medoids = []
        for cluster in range(args.clusters):
            distances = np.linalg.norm(x - cluster_model.cluster_centers_[cluster], axis=1)
            distances[cluster_model.labels_ != cluster] = np.inf
            medoids.append(selected.index[int(np.argmin(distances))])
        selected.loc[medoids].to_csv(out / "cluster_centers.csv", index=False)
        for rank, (idx, row) in enumerate(selected[selected.candidate_new].sort_values(
                "candidate_score", ascending=False).head(args.max_heatmaps).iterrows(), 1):
            save_heatmap(matrices[idx][0], out / "heatmaps" / f"candidate_{rank:03d}_rep1.png",
                         f"{row.chrom}:{row.start}-{row.end} rep1")
            save_heatmap(matrices[idx][1], out / "heatmaps" / f"candidate_{rank:03d}_rep2.png",
                         f"{row.chrom}:{row.start}-{row.end} rep2")
        recall = float((table[table.is_known].candidate_score >= threshold).mean()) if table.is_known.any() else None
        structure_hits = []
        for _, structure in ann.iterrows():
            matched = selected[annotation_chrom_mask(selected, str(structure.chrom)) &
                (selected.start < structure.end) & (selected.end > structure.start)]
            structure_hits.append(bool(len(matched)))
        summary = {"windows": len(table), "selected": len(selected), "new_candidates": int(selected.candidate_new.sum()),
                   "individual_novelty_windows": int(selected.candidate_individual.sum()),
                   "cluster_supported_windows": int(selected.candidate_cluster_supported.sum()),
                   "known_window_recall_before_reproducibility": recall,
                   "known_structure_recall_after_reproducibility": float(np.mean(structure_hits)),
                   "shape_representation_known_vs_background_auc": auc,
                   "known_feature_distance_threshold_p90": novelty_threshold,
                   "threshold": float(threshold), "parameters": vars(args),
                   "caution": "Novel means outside supplied annotations under this explicit rule; it is not biological proof."}
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _genes(gff: Path | None) -> pd.DataFrame:
    if gff is None:
        return pd.DataFrame(columns=["chrom", "start", "end", "name"])
    rows = []
    import gzip
    opener = gzip.open if str(gff).endswith(".gz") else open
    with opener(gff, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            attrs = dict(part.split("=", 1) for part in fields[8].split(";") if "=" in part)
            rows.append((fields[0], int(fields[3]) - 1, int(fields[4]), attrs.get("Name", "gene")))
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "name"])


def tracks(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ann, genes = annotations(args.annotations), _genes(args.gff)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        names = getattr(args, "labels", None) or [Path(p).stem for p in args.cool]
        if len(names) != len(args.cool):
            raise ValueError("--labels must have one name per contact matrix")
        conditions = [(name, stack.enter_context(open_cool(p))) for name, p in zip(names, args.cool)]
        if len(conditions) < 3:
            raise ValueError("Task 3 requires at least three conditions")
        bin_size = conditions[0][1].binsize
        if any(c.binsize != bin_size for _, c in conditions):
            raise ValueError("All conditions must have equal resolution")
        for chrom in conditions[0][1].chromnames:
            length = int(conditions[0][1].chromsizes[chrom])
            # One complete-genome set of fixed-width intervals.
            for start in range(0, length, args.interval_bp):
                end = min(length, start + args.interval_bp)
                curves = []
                for name, c in conditions:
                    block = c.matrix(balance=False, sparse=True).fetch((chrom, start, end))
                    depth = max(float(c.info.get("sum", 1)), 1)
                    curves.append((name, np.sqrt(np.asarray(block.sum(axis=1)).ravel().astype(float)
                                                  * 1e6 / depth)))
                n = min(len(values) for _, values in curves)
                x = start + np.arange(n) * bin_size
                fig, axes = plt.subplots(4, 1, figsize=(12, 6), dpi=120, sharex=True,
                                         gridspec_kw={"height_ratios": [3, 1, 1, 1]})
                baseline = np.mean([v[:n] for _, v in curves], axis=0)
                axes[0].fill_between(x, baseline, color="lightgray", alpha=.6, label="condition mean")
                for name, values in curves:
                    axes[0].plot(x, values[:n], lw=.7, label=name)
                axes[0].legend(fontsize=7, ncol=2); axes[0].set_ylabel("sqrt CPM")
                axes[1].plot(x, baseline, color="black", lw=.8)
                axes[1].set_ylabel("mean")
                gene_chrom = chrom if chrom in set(genes.chrom) else "NC_000913.3"
                local_genes = genes[(genes.chrom == gene_chrom) & (genes.start < end) & (genes.end > start)]
                for _, gene in local_genes.iterrows():
                    axes[2].broken_barh([(max(start, gene.start), min(end, gene.end) - max(start, gene.start))],
                                         (0, 1), facecolors="steelblue")
                axes[2].set(yticks=[], ylabel="genes")
                local_ann = ann[annotation_chrom_mask(ann, chrom) & (ann.start < end) &
                                (ann.end > start) & ann.type.isin(["CHIN", "OPCID"])]
                for _, structure in local_ann.iterrows():
                    axes[3].broken_barh([(max(start, structure.start), min(end, structure.end) - max(start, structure.start))],
                                         (0, 1), facecolors="darkorange")
                    axes[3].text(structure.center, .5, str(structure.id), fontsize=6, ha="center")
                axes[3].set(yticks=[], ylabel="structures", xlabel=f"{chrom} position (bp)", xlim=(start, end))
                fig.suptitle(f"{chrom}:{start:,}-{end:,}")
                fig.tight_layout(); fig.savefig(out / f"{chrom}_{start:07d}.png"); plt.close(fig)


def _replicate_direction_agreement(wt_scores, condition_scores) -> bool:
    """Require every condition-vs-WT contrast to have the same strict sign.

    Biological repeats are not presumed to be paired by their file numbers.
    This is a conservative, order-invariant heuristic, not a significance test.
    """
    wt = np.asarray(wt_scores, dtype=float)
    condition = np.asarray(condition_scores, dtype=float)
    if len(wt) < 2 or len(condition) < 2 or not np.isfinite(wt).all() or not np.isfinite(condition).all():
        raise ValueError("Two finite repeats per condition are required")
    return bool(condition.min() > wt.max() or condition.max() < wt.min())


def differential(args) -> None:
    ann = annotations(args.annotations)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    displayed_matrices = {}
    seed = int(getattr(args, "seed", 20260921))
    if bool(getattr(args, "wt2", None)) != bool(getattr(args, "condition2", None)):
        raise ValueError("Provide both --wt2 and --condition2 for a paired repeat analysis")
    with ExitStack() as stack:
        wt_files = [args.wt] + ([args.wt2] if getattr(args, "wt2", None) else [])
        condition_files = [args.condition] + ([args.condition2] if getattr(args, "condition2", None) else [])
        wt = [stack.enter_context(open_cool(path)) for path in wt_files]
        treatment = [stack.enter_context(open_cool(path)) for path in condition_files]
        if any(c.binsize != wt[0].binsize or tuple(c.chromnames) != tuple(wt[0].chromnames)
               for c in wt + treatment):
            raise ValueError("WT and condition require matching bins")
        target_depth = min(float(c.info["sum"]) for c in wt + treatment)

        def matched(c, source_path, chrom, center):
            # Aggregate the original 10 bp integer counts before independent
            # binomial thinning. This preserves the count distribution better
            # than multiplying an already normalized image by a depth ratio.
            bins = int(round(args.window_bp / c.binsize))
            if bins % 64:
                raise ValueError("Differential window must have a multiple of 64 bins")
            raw = window(c, chrom, center, args.window_bp, bins)
            factor = bins // 64
            counts = np.rint(raw.reshape(64, factor, 64, factor).sum(axis=(1, 3))).astype(np.int64)
            probability = min(1.0, target_depth / float(c.info["sum"]))
            triangle = np.triu(np.maximum(counts, 0))
            # Assign the thinning draw to the sample and locus, not its place
            # in the iteration. Swapping independent replicate file order must
            # not alter the matched matrices or downstream hit selection.
            draw_key = f"{seed}|{Path(source_path).name}|{int(c.info['sum'])}|{chrom}|{center}"
            draw_seed = int.from_bytes(hashlib.blake2b(draw_key.encode("utf-8"), digest_size=8).digest(), "little")
            thinned = np.random.default_rng(draw_seed).binomial(triangle, probability).astype(np.float32)
            thinned = thinned + np.triu(thinned, 1).T
            return normalize(thinned)

        for _, record in ann.iterrows():
            chrom = chrom_name(wt[0], record.chrom)
            center = int(record.center)
            wt_mats = [matched(c, path, chrom, center) for c, path in zip(wt, wt_files)]
            condition_mats = [matched(c, path, chrom, center) for c, path in zip(treatment, condition_files)]
            a, b = np.mean(wt_mats, axis=0), np.mean(condition_mats, axis=0)
            displayed_matrices[str(record.id)] = (a, b)
            region = np.triu_indices(64, k=3)
            wt_scores = [float(np.maximum(m[region], 0).mean()) for m in wt_mats]
            condition_scores = [float(np.maximum(m[region], 0).mean()) for m in condition_mats]
            strength_wt = float(np.mean(wt_scores))
            strength_condition = float(np.mean(condition_scores))
            delta = strength_condition - strength_wt
            wt_repeat_gap = abs(wt_scores[0] - wt_scores[-1]) if len(wt_scores) > 1 else np.nan
            condition_repeat_gap = (abs(condition_scores[0] - condition_scores[-1])
                                    if len(condition_scores) > 1 else np.nan)
            delta_rep1 = condition_scores[0] - wt_scores[0]
            delta_rep2 = (condition_scores[1] - wt_scores[1]) if len(wt_scores) > 1 else np.nan
            rows.append(dict(id=record.id, type=record.type, chrom=chrom, start=record.start,
                             end=record.end, condition=args.name, wt_strength=strength_wt,
                             condition_strength=strength_condition, delta=delta,
                             delta_rep1=delta_rep1, delta_rep2=delta_rep2,
                             repeat_direction_agreement=(_replicate_direction_agreement(wt_scores, condition_scores)
                                 if len(wt_mats) > 1 else np.nan),
                             exceeds_repeat_variation=(bool(abs(delta) > max(wt_repeat_gap, condition_repeat_gap))
                                 if len(wt_mats) > 1 else np.nan),
                             wt_repeat_difference=wt_repeat_gap,
                             condition_repeat_difference=condition_repeat_gap,
                             direction="weakened" if delta < 0 else "strengthened",
                             oe_correlation=upper_correlation(a, b)))
        result = pd.DataFrame(rows).sort_values("delta")
        repeat_gaps = result[["wt_repeat_difference", "condition_repeat_difference"]].to_numpy()
        if np.isfinite(repeat_gaps).any():
            repeat_p95 = float(np.nanquantile(repeat_gaps, .95))
            result["exceeds_global_repeat_p95"] = result.delta.abs() > repeat_p95
            result["reproducible_change"] = (result.repeat_direction_agreement &
                result.exceeds_repeat_variation & result.exceeds_global_repeat_p95)
        else:
            repeat_p95 = np.nan
            result["exceeds_global_repeat_p95"] = False
            result["reproducible_change"] = False
        result.to_csv(out / "structure_changes.csv", index=False)
        result[result.reproducible_change].to_csv(out / "reproducible_changes.csv", index=False)
        class_summary = result.groupby("type").agg(count=("delta", "size"), mean=("delta", "mean"),
            median=("delta", "median"),
            repeat_direction_agreement_rate=("repeat_direction_agreement", "mean"),
            exceeds_repeat_variation_rate=("exceeds_repeat_variation", "mean"),
            reproducible_changes=("reproducible_change", "sum"))
        class_summary.to_csv(out / "class_summary.csv")
        # Reuse the already depth-matched matrices for the displayed examples.
        source = result[result.reproducible_change]
        if source.empty:
            source = result
        chosen = pd.concat([source.head(args.heatmaps // 2), source.tail(args.heatmaps // 2)]).drop_duplicates("id")
        for _, row in chosen.iterrows():
            a, b = displayed_matrices[str(row.id)]
            save_heatmap(a, out / "heatmaps" / f"{row.id}_wt.png", f"{row.id} WT")
            save_heatmap(b, out / "heatmaps" / f"{row.id}_{args.name}.png", f"{row.id} {args.name}")
    (out / "README.txt").write_text(
        f"Raw counts were aggregated to 64 bins and then independently binomial-thinned to the lowest "
        f"library depth ({target_depth:.0f} contacts) before distance normalization. "
        "Delta is the treatment-minus-WT difference "
        "in positive normalized off-diagonal contact strength. Biological repeats are averaged when supplied; "
        f"the exploratory reproducible-change cutoff is the pooled 95th percentile of within-condition repeat "
        f"differences ({repeat_p95:.6g}), plus all treatment-vs-WT replicate contrasts sharing a direction "
        "and an effect larger than its own "
        "repeat differences. This is not a formal false-discovery rate or causal test.\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="task", required=True)
    d = sub.add_parser("discover", help="Task 2: genome-wide novel-candidate discovery")
    d.add_argument("--rep1", type=Path, required=True); d.add_argument("--rep2", type=Path, required=True)
    d.add_argument("--annotations", type=Path, required=True); d.add_argument("--out", type=Path, required=True)
    d.add_argument("--window-bp", type=int, default=5120); d.add_argument("--stride-bp", type=int, default=2560)
    d.add_argument("--top-fraction", type=float, default=.5); d.add_argument("--min-replicate-r", type=float, default=.2)
    d.add_argument("--clusters", type=int, default=8); d.add_argument("--min-cluster", type=int, default=5)
    d.add_argument("--candidate-rule", choices=("cluster", "individual"), default="cluster",
                   help="cluster is the legacy rule; individual retains rare shape-novel windows")
    d.add_argument("--max-heatmaps", type=int, default=10000); d.add_argument("--seed", type=int, default=20260921)
    d.add_argument("--ae-epochs", type=int, default=8, help="autoencoder training epochs; 0 disables it")
    t = sub.add_parser("tracks", help="Task 3: whole-genome multi-condition tracks")
    t.add_argument("--cool", type=Path, required=True, nargs="+"); t.add_argument("--annotations", type=Path, required=True)
    t.add_argument("--labels", nargs="+", help="short legend names matching --cool order")
    t.add_argument("--gff", type=Path); t.add_argument("--out", type=Path, required=True)
    t.add_argument("--interval-bp", type=int, default=10000)
    c = sub.add_parser("differential", help="Task 4: condition-specific structure changes")
    c.add_argument("--wt", type=Path, required=True); c.add_argument("--condition", type=Path, required=True)
    c.add_argument("--wt2", type=Path); c.add_argument("--condition2", type=Path)
    c.add_argument("--seed", type=int, default=20260921)
    c.add_argument("--name", required=True); c.add_argument("--annotations", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True); c.add_argument("--window-bp", type=int, default=5120)
    c.add_argument("--heatmaps", type=int, default=20)
    args = ap.parse_args()
    {"discover": discover, "tracks": tracks, "differential": differential}[args.task](args)


if __name__ == "__main__":
    main()
