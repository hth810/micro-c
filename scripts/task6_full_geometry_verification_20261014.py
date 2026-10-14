"""Additive final-model geometry evaluation; never fit to macroregion labels.

All declared seeds, approximate external domains, boundary trims and complete
cyclic-shift controls are evaluated. Existing frozen scripts/results are not
modified. This is a retrospective course-level structural check, not imaging
ground truth or a new blind experiment.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import itertools
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from scripts.annotate_3d_domains import REGIONS, SOURCE, region_for_bp
from scripts.validate_3d_macroregions import core_mask, offset_effect


ROOT = Path(__file__).resolve().parents[1]
MODEL_KEYS = tuple(f"rank_mlp_seed{s}" for s in (20261006, 20261007, 20261008)) + ("circular_mds",)
MACRODOMAINS = ("Ori", "Ter", "Right", "Left")
SEED = 20261014
TRIMS = (0, 50_000)
LOCUS_SOURCE = "https://doi.org/10.1371/journal.pgen.1006758"
LOCUS_BP = {"Ori4": 9900, "NSR5": 515000, "Right2": 738000, "dif_approx": 1604000}
PREFIX = "task6_full_geometry_20261014_"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path: Path, value: dict) -> None:
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        if isinstance(item, np.generic):
            return clean(item.item())
        if isinstance(item, float) and np.isnan(item):
            # Pandas promotes intentionally absent baseline p / diagnostic
            # fields to NaN. Preserve their meaning as JSON null, not a score.
            return None
        if isinstance(item, float) and not np.isfinite(item):
            raise ValueError("Nonfinite evaluation score cannot be serialized")
        return item
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_models(bundle: Path, keys: tuple[str, ...], n: int) -> dict[str, np.ndarray]:
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Specify distinct explicit coordinate keys")
    with np.load(bundle, allow_pickle=False) as data:
        missing = set(keys) - set(data.files)
        if missing:
            raise ValueError(f"Missing requested final-model keys: {sorted(missing)}")
        models = {key: np.asarray(data[key], dtype=float) for key in keys}
    if any(x.shape != (n, 3) or not np.isfinite(x).all() for x in models.values()):
        raise ValueError("Coordinate keys must contain finite n by 3 arrays")
    return models


def bin_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if not {"chrom", "start", "end"}.issubset(frame):
        raise ValueError("Coordinate table lacks genomic bin columns")
    start, end = frame.start.to_numpy(int), frame.end.to_numpy(int)
    if len(frame) < 10 or start[0] != 0 or np.any(end <= start):
        raise ValueError("Expected positive whole-chromosome bins from zero")
    if np.any(end[:-1] != start[1:]) or np.unique(np.diff(start)).size != 1:
        raise ValueError("Bins must be contiguous and regularly indexed")
    frame["center_bp"] = (start + end) / 2
    frame["published_macroregion_approx"] = [region_for_bp(int(v)) for v in frame.center_bp]
    if (frame.published_macroregion_approx == "Unassigned").any():
        raise ValueError("Approximate domain labels do not cover the chromosome")
    return frame


def circular_pair_index(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n < 6:
        raise ValueError("Too few bins")
    i, j = np.triu_indices(n, 1)
    offset = np.minimum(j - i, n - (j - i))
    keep = offset >= 2
    return i[keep], j[keep], offset[keep]


def matched_selection(positive: np.ndarray, negative: np.ndarray, offset: np.ndarray,
                      n: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    if any(a.shape != offset.shape for a in (positive, negative)) or np.any(positive & negative):
        raise ValueError("Pair categories must be disjoint and aligned")
    rng = np.random.default_rng(seed)
    p, q = [], []
    for step in range(2, n // 2 + 1):
        a = np.flatnonzero(positive & (offset == step))
        b = np.flatnonzero(negative & (offset == step))
        size = min(len(a), len(b))
        if size:
            p.append(rng.choice(a, size, replace=False))
            q.append(rng.choice(b, size, replace=False))
    return (np.concatenate(p), np.concatenate(q)) if p else (np.array([], int), np.array([], int))


def pair_auc(values: np.ndarray, a: np.ndarray, b: np.ndarray) -> float | None:
    if not len(a):
        return None
    return float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], -np.r_[values[a], values[b]]))


def bh(values: list[float]) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    order = np.argsort(a, kind="stable")
    sorted_q = np.minimum.accumulate((a[order] * len(a) / np.arange(1, len(a) + 1))[::-1])[::-1]
    q = np.empty(len(a))
    q[order] = np.clip(sorted_q, 0, 1)
    return q


def geometric_checks(models: dict[str, np.ndarray], frame: pd.DataFrame,
                     shifts: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = frame.published_macroregion_approx.to_numpy(str)
    n, genome = len(frame), int(frame.end.max())
    i, j, offset = circular_pair_index(n)
    names = list(models) + ["exact_genome_circle"]
    matrices = {key: squareform(pdist(x)) for key, x in models.items()}
    # Equal ring offsets have bit-identical chord distances, avoiding spurious
    # discrimination by floating point noise in a plotted perfect circle.
    vectors = np.array([matrices[key][i, j] for key in models] + [2 * np.sin(np.pi * offset / n)])
    rows, null_rows, domain_rows = [], [], []
    for scope in ("four_macrodomains", "all_six_regions"):
        for trim in TRIMS:
            core = core_mask(frame.center_bp.to_numpy(), trim, genome)
            if scope == "four_macrodomains":
                core &= np.isin(labels, MACRODOMAINS)
            eligible = core[i] & core[j]
            a, b = matched_selection(eligible & (labels[i] == labels[j]),
                                     eligible & (labels[i] != labels[j]), offset, n)
            observed = offset_effect(labels, core, (i, j), offset, vectors)
            null = []
            for count, shift in enumerate(shifts):
                null.append(offset_effect(np.roll(labels, shift), np.roll(core, shift), (i, j), offset, vectors))
                if count and count % 200 == 0:
                    print(f"{scope} trim={trim}: {count}/{len(shifts)} cyclic controls", flush=True)
            null = np.asarray(null)
            for column, key in enumerate(names):
                p = None if key == "exact_genome_circle" else float((1 + np.count_nonzero(null[:, column] >= observed[column])) / (len(shifts) + 1))
                rows.append({"scope": scope, "trim_bp": trim, "model": key, "retained_bins": int(core.sum()),
                             "matched_pairs_per_class": len(a), "same_domain_near_auc": pair_auc(vectors[column], a, b),
                             "standardized_between_minus_within": float(observed[column]),
                             "cyclic_shift_upper_p": p, "shift_null_mean": float(null[:, column].mean()),
                             "shift_null_p95": float(np.quantile(null[:, column], .95))})
                for shift, score in zip(shifts, null[:, column]):
                    null_rows.append({"scope": scope, "trim_bp": trim, "model": key,
                                      "shift_bins": int(shift), "effect": float(score)})
            if scope == "four_macrodomains":
                for region in MACRODOMAINS:
                    positive = eligible & (labels[i] == region) & (labels[j] == region)
                    negative = eligible & ((labels[i] == region) ^ (labels[j] == region))
                    pa, pb = matched_selection(positive, negative, offset, n)
                    for column, key in enumerate(names):
                        domain_rows.append({"trim_bp": trim, "region": region, "model": key,
                                            "matched_pairs_per_class": len(pa),
                                            "within_domain_near_auc": pair_auc(vectors[column], pa, pb)})
            print(f"Finished {scope} trim={trim}", flush=True)
    result = pd.DataFrame(rows)
    use = result.cyclic_shift_upper_p.notna()
    result.loc[use, "cyclic_shift_bh_q_all_16_checks"] = bh(result.loc[use, "cyclic_shift_upper_p"].tolist())
    return result, pd.DataFrame(null_rows), pd.DataFrame(domain_rows)


def index_for_bp(frame: pd.DataFrame, bp: int) -> int:
    if not 0 <= bp < int(frame.end.max()):
        raise ValueError("Landmark outside chromosome")
    selected = np.flatnonzero((frame.start <= bp) & (frame.end > bp))
    if len(selected) != 1:
        raise ValueError("Landmark is not assigned to exactly one bin")
    return int(selected[0])


def ori_from_gff(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        hits = [row.rstrip().split("\t") for row in stream if "\torigin_of_replication\t" in row and "Note=oriC" in row]
    if len(hits) != 1:
        raise ValueError("Expected one explicitly annotated oriC")
    row = hits[0]
    start, end = int(row[3]) - 1, int(row[4])
    return {"name": "oriC", "start_0based": start, "end_exclusive": end, "bp": (start + end) // 2,
            "source": "NC_000913.3 RefSeq origin_of_replication feature; converted from 1-based inclusive GFF"}


def landmark_checks(models: dict[str, np.ndarray], frame: pd.DataFrame,
                    ori: dict, shifts: np.ndarray) -> tuple[pd.DataFrame, dict]:
    n = len(frame)
    loci = dict(LOCUS_BP, oriC=int(ori["bp"]))
    ids = {key: index_for_bp(frame, bp) for key, bp in loci.items()}
    i, j, offset = circular_pair_index(n)
    rows = []
    for name, xyz in models.items():
        d = squareform(pdist(xyz))
        med = np.array([np.median(d[i[offset == step], j[offset == step]]) if step >= 2 else 0 for step in range(n // 2 + 1)])
        norm = max(float(np.std(d[i, j])), 1e-12)
        def adjusted(a, b):
            k = min(abs(a - b), n - abs(a - b))
            return float((d[a, b] - med[k]) / norm)
        a, b, c = ids["Ori4"], ids["NSR5"], ids["Right2"]
        score = adjusted(a, c) - adjusted(a, b)
        null = np.array([adjusted((a + shift) % n, (c + shift) % n) -
                         adjusted((a + shift) % n, (b + shift) % n) for shift in shifts])
        rows.append({"model": name, "check": "Ori4_NSR5_closer_than_Right2", "first_bin": a,
                     "second_bin": b, "third_bin": c,
                     "distance_Ori4_NSR5": float(d[a, b]), "distance_Ori4_Right2": float(d[a, c]),
                     "raw_published_order_reproduced": bool(d[a, b] < d[a, c]),
                     "offset_adjusted_right_minus_nsr": score,
                     "cyclic_shift_upper_p": float((1 + np.count_nonzero(null >= score)) / (len(shifts) + 1)),
                     "interpretation": "Published three-locus ordering; offset-adjusted positive effect is additional to genomic proximity"})
        a, b = ids["oriC"], ids["dif_approx"]
        k = min(abs(a - b), n - abs(a - b))
        background = d[i[offset == k], j[offset == k]]
        rows.append({"model": name, "check": "oriC_dif_offset_matched_diagnostic", "first_bin": a,
                     "second_bin": b, "third_bin": None, "distance_Ori4_NSR5": None,
                     "distance_Ori4_Right2": None, "raw_published_order_reproduced": None,
                     "offset_adjusted_right_minus_nsr": None, "cyclic_shift_upper_p": None,
                     "oriC_dif_distance": float(d[a, b]), "ring_offset": k,
                     "same_offset_distance_percentile": float(np.mean(background <= d[a, b])),
                     "same_offset_standardized_distance": adjusted(a, b),
                     "interpretation": "No predeclared stronger-than-background separation hypothesis; descriptive only"})
    return pd.DataFrame(rows), {"loci": {key: {"bp": bp, "bin": ids[key]} for key, bp in loci.items()},
                                "oriC_reference": ori, "dif_is_approximate": True,
                                "literature_source": LOCUS_SOURCE,
                                "published_order": "Ori4 (9.9kb) is closer to NSR5 (515kb) than Right2 (738kb); no microscope-distance calibration",
                                "coarse_macroregion_labels_are_not_used_to_assign_these_three_loci": True}


def rep2_check(path: Path, models: dict[str, np.ndarray], comparison: Path) -> tuple[dict, pd.DataFrame]:
    from microc.circular import circular_contact_target
    from microc.data import open_cool
    from microc.deep import _coarsen
    from scripts.task6_circular_sensitivity import evaluate
    with open_cool(path) as cool:
        target = circular_contact_target(_coarsen(cool, cool.chromnames[0], 5000))
    values = evaluate(target, models)
    frozen = pd.read_csv(comparison).set_index("model")
    rows = []
    for name in models:
        row = {"model": name, "matched_auc": values["exact_circular_offset_matched_auc"]["auc"][name]}
        for label in values["strata"]:
            row[label + "_rho"] = values["strata"][label][name]
        pairs = (("matched_auc", "rep2_exact_offset_matched_auc"), ("all_positive_rho", "rep2_all_positive_rho"),
                 ("top20_rho", "rep2_top20_rho"), ("weaker_positive_rho", "rep2_weaker_positive_rho"),
                 ("origin_crossing_2_to_20_rho", "rep2_origin_crossing_2_to_20_rho"))
        difference = max(abs(float(row[new]) - float(frozen.loc[name, old])) for new, old in pairs)
        if difference > 1e-10:
            raise ValueError(f"Fresh rep2 evaluation disagrees with frozen result: {name}, {difference}")
        row["max_abs_difference_from_frozen"] = difference
        rows.append(row)
    return values, pd.DataFrame(rows)


def figures(models: dict[str, np.ndarray], frame: pd.DataFrame, macro: pd.DataFrame,
            null: pd.DataFrame, landmarks: pd.DataFrame, out: Path, loci: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {name: color for name, _, _, color in REGIONS}
    labels = frame.published_macroregion_approx.to_numpy(str)
    short = {key: key.replace("rank_mlp_seed", "Rank ") for key in models}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), layout="constrained")
    selected = macro[macro.scope == "four_macrodomains"]
    for trim, marker in ((0, "o"), (50000, "s")):
        values = selected[selected.trim_bp == trim].set_index("model")
        for ax, metric in zip(axes, ("same_domain_near_auc", "standardized_between_minus_within")):
            ax.plot(range(len(models)), [values.loc[k, metric] for k in models], marker=marker, label=f"trim {trim // 1000} kb")
    axes[0].axhline(.5, color="gray", ls="--", lw=.8)
    axes[1].axhline(0, color="gray", ls="--", lw=.8)
    for ax, title in zip(axes, ("Four macrodomains: exact-offset AUC", "Between minus within standardized distance")):
        ax.set(xticks=range(len(models)), xticklabels=[short[k] for k in models], title=title)
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=8)
    fig.savefig(out / "macrogeometry.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4), layout="constrained")
    for ax, key in zip(axes, models):
        values = null[(null.scope == "four_macrodomains") & (null.trim_bp == 50000) & (null.model == key)].effect
        actual = float(selected[(selected.trim_bp == 50000) & (selected.model == key)].standardized_between_minus_within.iloc[0])
        ax.hist(values, bins=30, color="#b8c9d5")
        ax.axvline(actual, color="#a85225", label="observed")
        ax.set(title=short[key], xlabel="offset-matched distance effect", ylabel="cyclic shifts")
        ax.legend(fontsize=7)
    fig.savefig(out / "cyclic_null.png", dpi=180)
    plt.close(fig)
    fig = plt.figure(figsize=(13, 10))
    for panel, (key, xyz) in enumerate(models.items(), 1):
        ax = fig.add_subplot(2, 2, panel, projection="3d")
        closed = np.vstack((xyz, xyz[0]))
        ax.plot(*closed.T, lw=.35, color="#aaaaaa")
        for label in colors:
            pick = labels == label
            ax.scatter(*xyz[pick].T, s=3, color=colors[label], label=label)
        for label in ("oriC", "dif_approx"):
            k = loci["loci"][label]["bin"]
            ax.scatter(*xyz[k], color="black", marker="*", s=50)
            ax.text(*xyz[k], label, fontsize=8)
        ax.set(title=short[key], xlabel="x", ylabel="y", zlabel="z")
        ax.set_box_aspect((1, 1, 1)); ax.view_init(25, -55)
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Frozen final models; quantified macrogeometry accompanies the external label overlay")
    fig.tight_layout()
    fig.savefig(out / "all_model_3d.png", dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Use a fresh output path; existing evaluation is not overwritten")
    args.out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if sha(args.bundle) != freeze["output_sha256"]["coordinate_models.npz"]:
        raise ValueError("Final coordinate archive does not match its original freeze")
    if sha(args.coordinates) != freeze["output_sha256"]["coordinates.csv"]:
        raise ValueError("Bin coordinate table does not match final freeze")
    frame = bin_table(args.coordinates)
    n = len(frame)
    if set(args.model_keys) != set(MODEL_KEYS) or len(args.model_keys) != len(MODEL_KEYS):
        raise ValueError("Final verification retains all three declared final seeds and circular MDS")
    models = load_models(args.bundle, tuple(args.model_keys), n)
    if n != 929 or int(frame.end.max()) != 4641652:
        raise ValueError("Published final experiment is the complete 929-bin MG1655 chromosome")
    if not np.allclose(frame[["x", "y", "z"]].to_numpy(), models[MODEL_KEYS[0]], atol=1e-6, rtol=1e-6):
        raise ValueError("Exported coordinates do not correspond to the declared first final seed")
    if args.shifts != n - 1:
        raise ValueError("Actual-data final verification requires every nonzero cyclic shift")
    ori = ori_from_gff(args.gff)
    sources = [Path(__file__), ROOT / "scripts/annotate_3d_domains.py", ROOT / "scripts/validate_3d_macroregions.py",
               ROOT / "scripts/task6_circular_sensitivity.py", ROOT / "microc/circular.py", ROOT / "microc/deep.py",
               ROOT / "tests/test_task6_full_geometry_verification_20261014.py"]
    inputs = {"bundle": args.bundle, "coordinates": args.coordinates, "original_freeze": args.freeze,
              "original_comparison": args.comparison, "rep2": args.rep2, "gff": args.gff}
    protocol = {"experiment": "task6-full-final-geometry-20261014", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
                "input_sha256": {k: sha(p) for k, p in inputs.items()}, "input_paths": {k: str(p) for k, p in inputs.items()},
                "python": sys.version, "environment": {k: importlib.metadata.version(k) for k in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "cooler", "matplotlib")},
                "device": "CPU evaluation only; no fitting or gradient updates", "model_keys": list(models),
                "scopes": ["four_macrodomains", "all_six_regions"], "primary_scope": "four_macrodomains",
                "trim_bp": list(TRIMS), "primary_trim_bp": 50000, "cyclic_shifts": list(range(1, n)),
                "selection_seed": SEED, "region_source": SOURCE, "approximate_regions": [list(r[:3]) for r in REGIONS],
                "distance_matching": "Exactly equal circular BIN offsets >=2; both categories balanced within offsets; final short bin retained with index-offset approximation",
                "effect": "Weighted between-minus-within Euclidean distance standardized by eligible pair SD; higher means domain compactness",
                "null": "Rotate labels AND trimmed/core membership together; exclude zero shift; exact p=(1+ge)/(1+928); BH across 16 declared model/scope/trim checks",
                "landmarks": dict(LOCUS_BP, oriC=ori["bp"]), "landmark_source": LOCUS_SOURCE,
                "published_locus_order": "Ori4 closer to NSR5 than Right2; same-offset-median-adjusted contrast and all cyclic shifts are additional diagnostics",
                "oriC_dif": "Descriptive same-offset distance percentile; no independent quantified separation target",
                "no_label_training": True, "no_seed_or_threshold_selection": True,
                "historical_evaluation": "Coordinates and rep2 were previously inspected; these are additional retrospective evaluations, not a new blind experiment",
                "scientific_boundary": "External approximate genomic domains and selected imaging ordering test course-level global organization; no assertion of calibrated cell coordinates or full imaging truth"}
    dump(args.out / "protocol.json", protocol)
    print("Protocol frozen; evaluating all final models", flush=True)
    macro, null, domains = geometric_checks(models, frame, np.arange(1, n))
    landmarks, landmark_metadata = landmark_checks(models, frame, ori, np.arange(1, n))
    rep2, replication = rep2_check(args.rep2, models, args.comparison)
    macro.to_csv(args.out / "macrogeometry.csv", index=False)
    null.to_csv(args.out / "cyclic_shift_effects.csv", index=False)
    domains.to_csv(args.out / "individual_domains.csv", index=False)
    landmarks.to_csv(args.out / "landmarks.csv", index=False)
    replication.to_csv(args.out / "rep2_reverification.csv", index=False)
    frame[["chrom", "start", "end", "center_bp", "published_macroregion_approx"]].to_csv(args.out / "coordinate_labels.csv", index=False)
    dump(args.out / "landmark_metadata.json", landmark_metadata)
    figures(models, frame, macro, null, landmarks, args.out, landmark_metadata)
    primary = macro[(macro.scope == "four_macrodomains") & (macro.trim_bp == 50000)]
    summary = {"analysis": "Frozen final model whole-chromosome geometry, complete cyclic controls and rep2 consistency",
               "protocol_sha256": sha(args.out / "protocol.json"), "bins": n, "genome_bp": int(frame.end.max()),
               "all_seed_primary_macrogeometry": primary.to_dict("records"),
               "published_three_locus_order": landmarks[landmarks.check == "Ori4_NSR5_closer_than_Right2"].to_dict("records"),
               "rep2_reverification": replication.to_dict("records"), "landmark_metadata": landmark_metadata,
               "not_a_training_experiment": True, "no_model_replacement_or_seed_selection": True,
               "interpretation": "Positive domain effect and AUC above .5 support the declared global feature; individual domain and locus failures stay visible. Course completion is not contingent on every feature beating MDS.",
               "caution": protocol["scientific_boundary"]}
    dump(args.out / "summary.json", summary)
    outputs = {p.name: sha(p) for p in args.out.iterdir() if p.is_file()}
    receipt = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": sha(args.out / "protocol.json"),
               "input_sha256": protocol["input_sha256"], "source_sha256": protocol["source_sha256"],
               "outputs_sha256": outputs, "all_declared_models_evaluated": list(models),
               "old_frozen_evidence_overwritten": False}
    dump(args.out / "receipt.json", receipt)
    if args.publish:
        for source in args.out.iterdir():
            if source.is_file():
                target = ROOT / "results" / (PREFIX + source.name)
                if target.exists():
                    raise FileExistsError(f"Keep prior additive publication: {target}")
        for source in args.out.iterdir():
            if source.is_file():
                shutil.copyfile(source, ROOT / "results" / (PREFIX + source.name))
    return json.loads((args.out / "summary.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("results/task6_controlled_followup_20261006_coordinate_models.npz"))
    parser.add_argument("--coordinates", type=Path, default=Path("results/task6_controlled_followup_20261006_coordinates.csv"))
    parser.add_argument("--freeze", type=Path, default=Path("results/task6_controlled_followup_20261006_coordinate_freeze.json"))
    parser.add_argument("--comparison", type=Path, default=Path("results/task6_controlled_followup_20261006_comparison.csv"))
    parser.add_argument("--rep2", type=Path, default=Path("data/raw/GSE272159_37C_rep2.mapq_30.10.cool"))
    parser.add_argument("--gff", type=Path, default=Path("data/raw/GCF_000005845.2_ASM584v2_genomic.gff.gz"))
    parser.add_argument("--model-keys", nargs="+", default=list(MODEL_KEYS))
    parser.add_argument("--shifts", type=int, default=928)
    parser.add_argument("--out", type=Path, default=Path("outputs/task6_full_geometry_20261014"))
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
