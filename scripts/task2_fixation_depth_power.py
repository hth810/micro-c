"""Fixed-locus depth-sensitivity check for the 37 C fixation Micro-C sample.

The three loci and four contact-map views are frozen by the earlier rep1-only
Task 2 audit. Standard WT rep1/rep2 native 10-bp contact matrices are thinned
at each locus to match the 37 C fixation sample's >=640-bp cis count ratio.
No locus, preprocessing view, or threshold is chosen using the fixation map.
This tests only whether uniform read loss plausibly explains an observed weak
motif; it is not new-type significance or a biological replicate test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

from microc.data import chrom_name, normalize, open_cool, window
from scripts.audit_task2_shape_artifacts import METRICS, masked_vectors
from scripts.task2_rep1_locked_discovery import corr_matrix


WINDOW_BP = 5120
ANALYSIS_BINS = 64
SEED = 20260928
EDGES = ((0, 1), (0, 2), (1, 2))


def thin_symmetric_counts(raw: np.ndarray, fraction: float,
                          rng: np.random.Generator) -> np.ndarray:
    """Binomial-thin each unique contact once and retain matrix symmetry."""
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("Expected a square cis matrix")
    if not 0 <= fraction <= 1:
        raise ValueError("Thinning fraction outside [0, 1]")
    if not np.allclose(raw, raw.T) or not np.allclose(raw, np.rint(raw)) or np.any(raw < 0):
        raise ValueError("Expected symmetric nonnegative integer raw counts")
    upper = np.triu_indices(len(raw))
    counts = np.rint(raw[upper]).astype(np.int64)
    sampled = rng.binomial(counts, fraction).astype(np.float32)
    thinned = np.zeros_like(raw, dtype=np.float32)
    thinned[upper] = sampled
    thinned[(upper[1], upper[0])] = sampled
    return thinned


def analysis_map(raw: np.ndarray) -> np.ndarray:
    """Use exactly the 512-to-64 interpolation and normalization of window()."""
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("Expected a square cis matrix")
    resized = zoom(raw, ANALYSIS_BINS / len(raw), order=1)
    resized = resized[:ANALYSIS_BINS, :ANALYSIS_BINS]
    if resized.shape != (ANALYSIS_BINS, ANALYSIS_BINS):
        raise ValueError(f"Unexpected interpolated shape: {resized.shape}")
    return normalize((resized + resized.T) / 2)


def metric_edges(native_maps: list[np.ndarray]) -> dict[str, list[float]]:
    maps = np.stack([analysis_map(raw) for raw in native_maps])
    result = {}
    for metric in METRICS:
        similarity = corr_matrix(masked_vectors(maps, metric))
        result[metric] = [float(similarity[a, b]) for a, b in EDGES]
    return result


def load_native(path: Path, centers: list[int]) -> list[np.ndarray]:
    with open_cool(path) as cool:
        chrom = chrom_name(cool, "MG1655")
        if int(cool.binsize) != 10:
            raise ValueError(f"Expected native 10-bp Micro-C map: {path}")
        native_bins = WINDOW_BP // 10
        return [window(cool, chrom, center, WINDOW_BP, native_bins)
                for center in centers]


def summarize_draws(draws: list[dict[str, list[float]]], full: dict[str, list[float]],
                    observed: dict[str, list[float]], centers: list[int]) -> dict:
    metrics = {}
    for metric in METRICS:
        values = np.asarray([draw[metric] for draw in draws])
        edges = []
        for index, (a, b) in enumerate(EDGES):
            series = values[:, index]
            edges.append({
                "centers_bp": [centers[a], centers[b]],
                "full_depth_r": full[metric][index],
                "fix37_observed_r": observed[metric][index],
                "thinned_median_r": float(np.median(series)),
                "thinned_q05_q95_r": [float(x) for x in np.quantile(series, [.05, .95])],
                "fraction_thinned_r_le_fix37": float(np.mean(series <= observed[metric][index])),
            })
        min_edge = values.min(axis=1)
        metrics[metric] = {
            "edges": edges,
            "thinned_all_three_edges_positive_fraction": float(np.mean((values > 0).all(axis=1))),
            "thinned_min_edge_median_r": float(np.median(min_edge)),
            "thinned_min_edge_q05_q95_r": [float(x) for x in np.quantile(min_edge, [.05, .95])],
            "fix37_min_edge_r": float(min(observed[metric])),
        }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--qc", type=Path, required=True)
    parser.add_argument("--rep1", type=Path, required=True)
    parser.add_argument("--rep2", type=Path, required=True)
    parser.add_argument("--fix37", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1:
        raise ValueError("Need at least one thinning iteration")
    discovery = json.loads((args.discovery / "discovery_summary.json").read_text(encoding="utf-8"))
    centers = [int(x) for x in discovery["rep1_top_three_locus_clique_bp"]]
    qc = json.loads(args.qc.read_text(encoding="utf-8"))
    if qc["frozen_centers_bp"] != centers:
        raise ValueError("QC and discovery loci differ")
    natives = {"standard_rep1": load_native(args.rep1, centers),
               "standard_rep2": load_native(args.rep2, centers),
               "wt_37c_growth_37c_fix": load_native(args.fix37, centers)}
    observed = metric_edges(natives["wt_37c_growth_37c_fix"])
    rng = np.random.default_rng(SEED)
    reports = {}
    for name in ("standard_rep1", "standard_rep2"):
        original = natives[name]
        original_qc = qc["samples"][name]["loci"]
        fixation_qc = qc["samples"]["wt_37c_growth_37c_fix"]["loci"]
        fractions = []
        for expected_center, original_locus, fixation_locus in zip(
                centers, original_qc, fixation_qc):
            if original_locus["center_bp"] != expected_center or fixation_locus["center_bp"] != expected_center:
                raise ValueError("QC locus ordering differs")
            numerator = fixation_locus["raw_far_640bp_upper_count"]
            denominator = original_locus["raw_far_640bp_upper_count"]
            fractions.append(min(1.0, float(numerator / denominator)))
        full = metric_edges(original)
        draws = [metric_edges([thin_symmetric_counts(raw, fraction, rng)
                               for raw, fraction in zip(original, fractions)])
                 for _ in range(args.iterations)]
        reports[name] = {
            "far_count_thinning_fraction_by_locus": dict(zip(map(str, centers), fractions)),
            "metrics": summarize_draws(draws, full, observed, centers),
        }
    report = {
        "frozen_centers_bp": centers,
        "iterations_per_standard_replicate": args.iterations,
        "seed": SEED,
        "views": list(METRICS),
        "simulation": "Each unique native 10-bp cis count is independently binomial-thinned at a locus-specific fraction equal to fix37/standard >=640-bp cis count ratio; then the existing 512-to-64 interpolation and map-wise distance normalization are applied.",
        "caution": "A fixed-locus coverage-power sensitivity check, not a new-type significance test. Uniform count thinning cannot reproduce fixation-dependent distance decay, locus biology, or technical biases; the fix37 sample is not an equal-protocol biological replicate.",
        "sources": {"standard_rep1": str(args.rep1), "standard_rep2": str(args.rep2),
                    "fix37": str(args.fix37), "qc": str(args.qc)},
        "standard_replicates": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} ({args.iterations} iterations per standard replicate)", flush=True)


if __name__ == "__main__":
    main()
