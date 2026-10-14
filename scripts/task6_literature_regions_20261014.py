"""Append a sourced, circular macroregion-coordinate sensitivity evaluation.

The 2013 RSC SI Table 1 defines the whole six-region version; it is not used to
train, choose seeds, or select boundaries. Older approximate-region results stay
unchanged. Published rounded intervals are a literature overlay, not same-sample
base-resolution segmentation. In particular Ori continues across genome zero.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from scripts.task6_full_geometry_verification_20261014 import (
    ROOT, MODEL_KEYS, MACRODOMAINS, TRIMS, SEED, bh, bin_table,
    circular_pair_index, dump, load_models, matched_selection, pair_auc, sha,
)
from scripts.validate_3d_macroregions import offset_effect


SOURCE = "https://www.rsc.org/suppdata/mb/c3/c3mb25364c/c3mb25364c.pdf"
ARTICLE = "https://doi.org/10.1039/C3MB25364C"
VERSION = "Zarei2013_SI_Table1"
PREFIX = "task6_literature_regions_20261014_"
# All six source intervals are retained together, not a results-chosen hybrid.
# Tuple: name, start bp, exclusive end bp; start > end means crossing zero.
REGIONS = (("Ori", 3760000, 50000), ("NSR", 50000, 600000),
           ("Right", 600000, 1210000), ("Ter", 1210000, 2180000),
           ("Left", 2180000, 2880000), ("NSL", 2880000, 3760000))


def labels_and_core(centers: np.ndarray, genome_bp: int, trim_bp: int,
                    regions: tuple = REGIONS) -> tuple[np.ndarray, np.ndarray]:
    """Assign half-open circular intervals and trim only real domain boundaries."""
    centers = np.asarray(centers, dtype=float)
    if genome_bp <= 0 or trim_bp < 0 or np.any(centers < 0) or np.any(centers >= genome_bp):
        raise ValueError("Expected valid circular chromosome positions and nonnegative trim")
    labels = np.full(len(centers), "", dtype=object)
    assigned = np.zeros(len(centers), dtype=int)
    boundaries = []
    for name, start, end in regions:
        if not (0 <= start < genome_bp and 0 <= end < genome_bp) or start == end:
            raise ValueError("Invalid or full-circle region")
        pick = ((centers >= start) & (centers < end)) if start < end else ((centers >= start) | (centers < end))
        labels[pick] = name
        assigned[pick] += 1
        boundaries.extend((start, end))
    if np.any(assigned != 1):
        raise ValueError("Source intervals must partition every input position exactly once")
    boundaries = np.unique(boundaries)
    # Validate the complete circle, including gaps/overlaps that contain no
    # sampled bin centers. Interval membership only changes at boundaries.
    edges = np.unique(np.r_[0, boundaries, genome_bp])
    probes = (edges[:-1] + edges[1:]) / 2
    coverage = np.zeros(len(probes), dtype=int)
    for _, start, end in regions:
        coverage += ((probes >= start) & (probes < end)) if start < end else ((probes >= start) | (probes < end))
    if np.any(coverage != 1):
        raise ValueError("Source intervals must partition the complete chromosome")
    delta = np.abs(centers[:, None] - boundaries[None, :])
    gap = np.minimum(delta, genome_bp - delta).min(axis=1)
    return labels.astype(str), gap >= trim_bp


def evaluate_regions(models: dict, frame: pd.DataFrame, shifts: np.ndarray):
    n, genome = len(frame), int(frame.end.max())
    centers = frame.center_bp.to_numpy(float)
    labels, _ = labels_and_core(centers, genome, 0)
    i, j, offset = circular_pair_index(n)
    names = list(models) + ["exact_genome_circle"]
    vectors = np.array([squareform(pdist(x))[i, j] for x in models.values()] + [2 * np.sin(np.pi * offset / n)])
    rows, null_rows, domain_rows = [], [], []
    for scope in ("four_macrodomains", "all_six_regions"):
        for trim in TRIMS:
            _, core = labels_and_core(centers, genome, trim)
            if scope == "four_macrodomains":
                core &= np.isin(labels, MACRODOMAINS)
            eligible = core[i] & core[j]
            a, b = matched_selection(eligible & (labels[i] == labels[j]), eligible & (labels[i] != labels[j]), offset, n)
            observed = offset_effect(labels, core, (i, j), offset, vectors)
            null = []
            for count, shift in enumerate(shifts):
                null.append(offset_effect(np.roll(labels, shift), np.roll(core, shift), (i, j), offset, vectors))
                if count and count % 200 == 0:
                    print(f"{VERSION} {scope} trim={trim}: {count}/{len(shifts)} controls", flush=True)
            null = np.array(null)
            for column, model in enumerate(names):
                p = None if model == "exact_genome_circle" else (1 + np.count_nonzero(null[:, column] >= observed[column])) / (len(shifts) + 1)
                rows.append({"coordinate_version": VERSION, "scope": scope, "trim_bp": trim, "model": model,
                             "retained_bins": int(core.sum()), "matched_pairs_per_class": len(a),
                             "same_domain_near_auc": pair_auc(vectors[column], a, b),
                             "standardized_between_minus_within": float(observed[column]),
                             "cyclic_shift_upper_p": p, "shift_null_mean": float(null[:, column].mean()),
                             "shift_null_p95": float(np.quantile(null[:, column], .95))})
                for shift, value in zip(shifts, null[:, column]):
                    null_rows.append({"coordinate_version": VERSION, "scope": scope, "trim_bp": trim,
                                      "model": model, "shift_bins": int(shift), "effect": float(value)})
            if scope == "four_macrodomains":
                for region in MACRODOMAINS:
                    positive = eligible & (labels[i] == region) & (labels[j] == region)
                    negative = eligible & ((labels[i] == region) ^ (labels[j] == region))
                    pa, pb = matched_selection(positive, negative, offset, n)
                    for column, model in enumerate(names):
                        domain_rows.append({"coordinate_version": VERSION, "trim_bp": trim, "region": region,
                                            "model": model, "matched_pairs_per_class": len(pa),
                                            "within_domain_near_auc": pair_auc(vectors[column], pa, pb)})
    macro = pd.DataFrame(rows)
    use = macro.cyclic_shift_upper_p.notna()
    macro.loc[use, "cyclic_shift_bh_q_all_16_checks"] = bh(macro.loc[use, "cyclic_shift_upper_p"].tolist())
    return macro, pd.DataFrame(null_rows), pd.DataFrame(domain_rows), labels


def make_figure(macro, domains, old_macro, old_domains, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    for table, name, marker in ((old_macro, "Prior approximate overlay", "o"), (macro, "2013 SI Table 1", "s")):
        use = table[(table.scope == "four_macrodomains") & (table.trim_bp == 50000)].set_index("model")
        axes[0].plot(range(4), [use.loc[k, "same_domain_near_auc"] for k in MODEL_KEYS], marker=marker, label=name)
    use = domains[domains.trim_bp == 50000].pivot(index="region", columns="model", values="within_domain_near_auc")
    for k, model in enumerate(MODEL_KEYS):
        axes[1].plot(range(4), [use.loc[r, model] for r in MACRODOMAINS], marker="o", label=model.replace("rank_mlp_seed", "Rank "))
    for ax in axes:
        ax.axhline(.5, color="gray", ls="--", lw=.8)
        ax.set(ylabel="Exact ring-offset matched AUC")
        ax.legend(fontsize=8)
    axes[0].set(xticks=range(4), xticklabels=[x.replace("rank_mlp_seed", "Rank ") for x in MODEL_KEYS], title="Four macrodomains; 50 kb boundary trim")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].set(xticks=range(4), xticklabels=MACRODOMAINS, title="Individual domains; complete literature version")
    fig.savefig(out / "coordinate_version_comparison.png", dpi=180)
    plt.close(fig)


def run(args):
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Use a fresh output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if sha(args.bundle) != freeze["output_sha256"]["coordinate_models.npz"] or sha(args.coordinates) != freeze["output_sha256"]["coordinates.csv"]:
        raise ValueError("Inputs differ from the final model freeze")
    frame = bin_table(args.coordinates)
    if len(frame) != 929 or int(frame.end.max()) != 4641652:
        raise ValueError("Expected full 929-bin MG1655 chromosome")
    models = load_models(args.bundle, MODEL_KEYS, len(frame))
    sources = [Path(__file__), ROOT / "scripts/task6_full_geometry_verification_20261014.py",
               ROOT / "scripts/validate_3d_macroregions.py", ROOT / "tests/test_task6_literature_regions_20261014.py"]
    inputs = {"bundle": args.bundle, "coordinates": args.coordinates, "original_freeze": args.freeze,
              "prior_macrogeometry": args.prior_macro, "prior_individual_domains": args.prior_domains}
    protocol = {"experiment": "task6-literature-coordinate-version-20261014", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
                "input_sha256": {k: sha(p) for k, p in inputs.items()}, "input_paths": {k: str(p) for k, p in inputs.items()},
                "literature_version": VERSION, "article": ARTICLE, "source": SOURCE,
                "source_location": "Supplementary Table 1, printed page 7, PDF zero-index page 6",
                "source_access": "Full publisher SI text parsed via web retrieval; direct local PDF download returned HTTP404. Coordinate transcription is frozen; no original PDF SHA is invented.",
                "region_coordinates_bp": [list(r) for r in REGIONS], "coordinate_convention": "Half-open interval; Ori start>end wraps across genomic zero",
                "source_precision": "Source uses two decimal places in Mb; rounded literature regions, not base-resolution same-sample truth",
                "coordinate_frame": "E. coli K-12 MG1655 genome frame; historical assembly differences not separately lifted, 50kb trim provides coarse boundary sensitivity, not proof of equivalence",
                "primary_scope": "four_macrodomains", "primary_trim_bp": 50000, "trim_bp": list(TRIMS),
                "scopes": ["four_macrodomains", "all_six_regions"], "model_keys": list(MODEL_KEYS),
                "selection_seed": SEED, "cyclic_shifts": list(range(1, len(frame))),
                "distance_matching": "Exactly equal circular bin offsets>=2; final short bin retained by bin-index approximation",
                "boundary_core": "Trim circular distance to each actual interval boundary; genome zero is NOT a boundary inside Ori",
                "null": "Rotate labels AND core together through all 928 nonzero shifts; BH across 16 version-specific model/scope/trim checks",
                "no_boundary_or_seed_selection": True, "no_label_training": True, "prior_version_retained": True,
                "interpretation_plan": "Report both versions and every domain. Positive overall AUC alone is not evidence that every domain is recovered; cyclic-shift specificity is a separate assessment.",
                "historical_evaluation": "Additive retrospective coordinate-source sensitivity after old geometry was examined; no new blind experiment"}
    dump(args.out / "protocol.json", protocol)
    print("Whole literature coordinate version frozen before metric evaluation", flush=True)
    macro, null, domains, labels = evaluate_regions(models, frame, np.arange(1, len(frame)))
    macro.to_csv(args.out / "macrogeometry.csv", index=False)
    null.to_csv(args.out / "cyclic_shift_effects.csv", index=False)
    domains.to_csv(args.out / "individual_domains.csv", index=False)
    frame["literature_region"] = labels
    frame[["chrom", "start", "end", "center_bp", "published_macroregion_approx", "literature_region"]].to_csv(args.out / "coordinate_labels.csv", index=False)
    boundary = pd.DataFrame(REGIONS, columns=["region", "start_bp", "end_exclusive_bp"])
    boundary["wraps_zero"] = boundary.start_bp > boundary.end_exclusive_bp
    boundary.to_csv(args.out / "literature_boundaries.csv", index=False)
    old_macro, old_domains = pd.read_csv(args.prior_macro), pd.read_csv(args.prior_domains)
    make_figure(macro, domains, old_macro, old_domains, args.out)
    primary = macro[(macro.scope == "four_macrodomains") & (macro.trim_bp == 50000)]
    summary = {"protocol_sha256": sha(args.out / "protocol.json"), "literature_version": VERSION,
               "four_macrodomains_50kb": primary.to_dict("records"),
               "individual_domains_50kb": domains[domains.trim_bp == 50000].to_dict("records"),
               "changed_label_bins": int(np.count_nonzero(labels != frame.published_macroregion_approx)),
               "Ori4_9900bp_literature_label": labels_and_core(np.array([9900]), 4641652, 0)[0][0],
               "no_retraining": True, "prior_version_retained": True, "interpretation_plan": protocol["interpretation_plan"]}
    dump(args.out / "summary.json", summary)
    dump(args.out / "receipt.json", {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
         "protocol_sha256": sha(args.out / "protocol.json"), "source_sha256": protocol["source_sha256"],
         "input_sha256": protocol["input_sha256"], "outputs_sha256": {p.name: sha(p) for p in args.out.iterdir() if p.is_file()},
         "old_frozen_evidence_overwritten": False})
    if args.publish:
        for p in args.out.iterdir():
            if (ROOT / "results" / (PREFIX + p.name)).exists():
                raise FileExistsError("Keep previous additive publication")
        for p in args.out.iterdir():
            shutil.copyfile(p, ROOT / "results" / (PREFIX + p.name))
    return json.loads((args.out / "summary.json").read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("results/task6_controlled_followup_20261006_coordinate_models.npz"))
    parser.add_argument("--coordinates", type=Path, default=Path("results/task6_controlled_followup_20261006_coordinates.csv"))
    parser.add_argument("--freeze", type=Path, default=Path("results/task6_controlled_followup_20261006_coordinate_freeze.json"))
    parser.add_argument("--prior-macro", type=Path, default=Path("results/task6_full_geometry_20261014_macrogeometry.csv"))
    parser.add_argument("--prior-domains", type=Path, default=Path("results/task6_full_geometry_20261014_individual_domains.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/task6_full_geometry_20261014/literature_regions"))
    parser.add_argument("--publish", action="store_true")
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
