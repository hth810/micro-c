"""Render both WT replicate heatmaps for strict individual-novelty loci."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from microc.data import normalize, open_cool, save_heatmap, window


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loci", type=Path, required=True)
    parser.add_argument("--rep1", type=Path, required=True)
    parser.add_argument("--rep2", type=Path, required=True)
    parser.add_argument("--window-bp", type=int, default=5120)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    loci = pd.read_csv(args.loci)
    for replicate, path in ((1, args.rep1), (2, args.rep2)):
        with open_cool(path) as cool:
            for _, row in loci.iterrows():
                center = int(row.representative_center_bp)
                mat = normalize(window(cool, row.chrom, center, args.window_bp, 64))
                save_heatmap(mat, args.out / f"{center}_rep{replicate}.png",
                             f"{row.chrom}:{center} WT rep{replicate}")


if __name__ == "__main__":
    main()
