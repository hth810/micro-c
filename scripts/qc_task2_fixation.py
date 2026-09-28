"""Depth and map-usability QC for three frozen Task 2 candidate loci.

The loci come exclusively from the previously frozen rep1 discovery summary.
This script does not search, score, or reselect candidate positions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from microc.data import chrom_name, normalize, open_cool, window


WINDOW_BP = 5120
ANALYSIS_BINS = 64
ANALYSIS_FAR_BINS = 8


def summary_for_sample(path: Path, centers: list[int], chrom: str) -> dict:
    loci = []
    with open_cool(path) as cool:
        sample_chrom = chrom_name(cool, chrom)
        bin_size = int(cool.binsize)
        if WINDOW_BP % bin_size:
            raise ValueError(f"Window size not divisible by bin size: {path}: {bin_size}")
        native_bins = WINDOW_BP // bin_size
        far_native = int(ANALYSIS_FAR_BINS * WINDOW_BP / ANALYSIS_BINS / bin_size)
        native_mask = np.triu_indices(native_bins)
        native_far_mask = np.triu_indices(native_bins, k=far_native)
        analysis_mask = np.triu_indices(ANALYSIS_BINS, k=ANALYSIS_FAR_BINS)
        for center in centers:
            raw_native = window(cool, sample_chrom, center, WINDOW_BP, native_bins)
            raw_analysis = window(cool, sample_chrom, center, WINDOW_BP, ANALYSIS_BINS)
            normalized = normalize(raw_analysis)
            unique_counts = raw_native[native_mask]
            far_counts = raw_native[native_far_mask]
            far_analysis = raw_analysis[analysis_mask]
            far_norm = normalized[analysis_mask]
            loci.append({
                "center_bp": center,
                "raw_cis_upper_count": float(unique_counts.sum()),
                "raw_cis_upper_nonzero_fraction": float(np.mean(unique_counts > 0)),
                "raw_far_640bp_upper_count": float(far_counts.sum()),
                "raw_far_640bp_upper_nonzero_fraction": float(np.mean(far_counts > 0)),
                "analysis_64bin_far8_nonzero_fraction": float(np.mean(far_analysis > 0)),
                "normalized_64bin_far8_finite_fraction": float(np.mean(np.isfinite(far_norm))),
                "normalized_64bin_far8_std": float(np.std(far_norm)),
                "normalized_64bin_far8_q05_q95": [float(x) for x in np.quantile(far_norm, [.05, .95])],
            })
    return {"path": str(path), "chrom": sample_chrom, "native_bin_bp": bin_size,
            "native_window_bins": native_bins, "loci": loci,
            "median_raw_cis_upper_count": float(np.median([r["raw_cis_upper_count"] for r in loci])),
            "median_raw_far_640bp_upper_nonzero_fraction": float(np.median([
                r["raw_far_640bp_upper_nonzero_fraction"] for r in loci]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--rep1", type=Path, required=True)
    parser.add_argument("--rep2", type=Path, required=True)
    parser.add_argument("--fix37", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    discovery = json.loads((args.discovery / "discovery_summary.json").read_text(encoding="utf-8"))
    centers = [int(x) for x in discovery["rep1_top_three_locus_clique_bp"]]
    if len(centers) != 3 or len(set(centers)) != 3:
        raise ValueError("Expected three distinct frozen loci")
    chrom = "MG1655"
    samples = {
        name: summary_for_sample(path, centers, chrom)
        for name, path in (("standard_rep1", args.rep1), ("standard_rep2", args.rep2),
                           ("wt_37c_growth_37c_fix", args.fix37))
    }
    report = {
        "frozen_centers_bp": centers,
        "window_bp": WINDOW_BP,
        "count_definition": "unbalanced native-bin cis matrix, upper triangle counted once including diagonal",
        "far_definition": "upper triangle at >=640 bp genomic bin separation; analysis maps use 64 bins and k>=8",
        "normalization": "microc.data.normalize independently for each 64x64 local map",
        "caution": "Three preselected loci only. Depth/occupancy and fixation protocol differ; a negative shape result is not proof of biological absence.",
        "samples": samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
