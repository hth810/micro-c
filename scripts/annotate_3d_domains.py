"""Label a learned 3D chromosome trace with published MG1655 macroregions.

The approximate genomic bands follow the WT schematic in
https://doi.org/10.1371/journal.pgen.1006758 (Fig. 1). Coloring the learned
trace by external coordinates is an interpretive overlay, not a validation
that these macrodomains were recovered as distinct spatial territories.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REGIONS = (
    ("NSR", 0, 600_000, "#8f8f8f"),
    ("Right", 600_000, 1_139_000, "#df6f43"),
    ("Ter", 1_139_000, 2_000_000, "#3379b7"),
    ("Left", 2_000_000, 2_803_000, "#7756a6"),
    ("NSL", 2_803_000, 3_740_000, "#b8a876"),
    ("Ori", 3_740_000, 4_700_000, "#35a66a"),
)
SOURCE = "https://doi.org/10.1371/journal.pgen.1006758"


def region_for_bp(bp: int) -> str:
    return next((name for name, start, end, _ in REGIONS if start <= bp < end), "Unassigned")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coordinates", type=Path,
                    default=Path("results/task6_all_positive_coordinates.csv"))
    ap.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    ap.add_argument("--out", type=Path, default=Path("outputs/task6_domain_overlay"))
    args = ap.parse_args()
    frame = pd.read_csv(args.coordinates)
    if not {"start", "end", "x", "y", "z"}.issubset(frame.columns):
        raise ValueError("Coordinate table lacks required fields")
    positions = frame[["x", "y", "z"]].to_numpy(dtype=float)
    if len(positions) < 10 or not np.isfinite(positions).all():
        raise ValueError("Expected finite whole-chromosome coordinates")
    centers = ((frame.start.to_numpy() + frame.end.to_numpy()) / 2).astype(int)
    frame["published_macroregion_approx"] = [region_for_bp(bp) for bp in centers]
    if (frame.published_macroregion_approx == "Unassigned").any():
        raise ValueError("Published region overlay does not cover the modeled chromosome")
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "coordinates_with_macroregions.csv", index=False)
    fig = plt.figure(figsize=(12, 5.5), dpi=180)
    for panel, (elevation, azimuth) in enumerate(((22, -55), (34, 120)), 1):
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        ax.plot(*positions.T, color="#aab0b5", linewidth=.35, alpha=.45)
        for name, _, _, color in REGIONS:
            indices = np.flatnonzero(frame.published_macroregion_approx.to_numpy() == name)
            ax.plot(*positions[indices].T, linewidth=1.1, color=color, label=name)
            ax.scatter(*positions[indices[::max(1, len(indices) // 80)]].T,
                       s=4, color=color, alpha=.85)
        for label, bp in (("oriC", 3_920_000), ("dif", 1_604_000)):
            idx = int(np.argmin(np.abs(centers - bp)))
            ax.scatter(*positions[idx], s=32, marker="*", color="black")
            ax.text(*positions[idx], label, fontsize=7, color="black")
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set(xlabel="x", ylabel="y", zlabel="z", title=f"View {panel}")
        if panel == 2:
            ax.legend(loc="upper left", bbox_to_anchor=(.97, 1.0), fontsize=7)
    fig.suptitle("Rep1-learned 3D trace; approximate published genomic macroregions", fontsize=11)
    fig.text(.5, .015, "Color follows genomic position; spatial domains were not independently validated.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, .9, .96))
    fig.savefig(args.out / "chromosome_3d_macroregions.png")
    plt.close(fig)
    summary = {"bins": len(frame), "source": SOURCE, "approximate_region_boundaries_bp": [
        {"name": name, "start": start, "end": end} for name, start, end, _ in REGIONS],
        "region_bin_counts": frame.published_macroregion_approx.value_counts().to_dict(),
        "caution": "External published genomic labels overlaid on learned coordinates; not evidence that the predicted geometry physically reproduces macrodomains."}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
