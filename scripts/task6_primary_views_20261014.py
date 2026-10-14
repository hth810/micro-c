"""First declared final seed, two 3-D views with the sourced circular overlay."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from scripts.task6_full_geometry_verification_20261014 import (
    ROOT, MODEL_KEYS, dump, load_models, ori_from_gff, index_for_bp, sha,
)
from scripts.task6_literature_regions_20261014 import VERSION, labels_and_core


COLORS = {"Ori": "#269b68", "NSR": "#8f969a", "Right": "#dd7c42", "Ter": "#368ac0", "Left": "#9163ac", "NSL": "#bcaa58"}


def draw_primary_views(xyz, frame, ori_bp, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n, genome = len(frame), int(frame.end.max())
    if xyz.shape != (n, 3) or not np.isfinite(xyz).all():
        raise ValueError("Expected finite full-chromosome three-dimensional coordinates")
    centers = (frame.start.to_numpy() + frame.end.to_numpy()) / 2
    labels, _ = labels_and_core(centers, genome, 0)
    first = xyz.mean(axis=0)
    radius = max(float(np.ptp(xyz, axis=0).max()) / 2, 1e-8) * 1.10
    fig = plt.figure(figsize=(11.8, 5.4), layout="constrained")
    for panel, azimuth in enumerate((-55, 125), 1):
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        closed = np.vstack((xyz, xyz[0]))
        ax.plot(*closed.T, color="#9c9c9c", lw=.3, alpha=.55)
        for label, color in COLORS.items():
            ax.scatter(*xyz[labels == label].T, s=6, color=color, label=label, alpha=.85, depthshade=False)
        for name, bp, marker in (("oriC", ori_bp, "*"), ("dif (approx.)", 1604000, "D")):
            k = index_for_bp(frame, bp)
            ax.scatter(*xyz[k], s=60 if marker == "*" else 25, marker=marker,
                       color="black", edgecolors="white", linewidths=.35, depthshade=False, zorder=10)
            # Fixed label offset is visual only; marker remains at exact bin coordinate.
            point = xyz[k] + np.array([.025, .025, .075]) * radius
            ax.text(*point, name, fontsize=9, color="black", zorder=12,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .7, "pad": .5})
        ax.set(xlim=(first[0] - radius, first[0] + radius), ylim=(first[1] - radius, first[1] + radius),
               zlim=(first[2] - radius, first[2] + radius), xlabel="x", ylabel="y", zlabel="z",
               title=f"View {panel}   azimuth {azimuth} degrees")
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=25, azim=azimuth)
        ax.tick_params(labelsize=8)
    handles, names = ax.get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(.5, -.005), ncol=6, frameon=False, fontsize=10, markerscale=1.5)
    fig.suptitle("Rank MLP seed 20261006 | whole-chromosome two-view reconstruction\n2013 literature macroregions; relative model coordinates", fontsize=12)
    fig.savefig(target, dpi=200, bbox_inches="tight", pad_inches=.10)
    plt.close(fig)
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("results/task6_controlled_followup_20261006_coordinate_models.npz"))
    parser.add_argument("--coordinates", type=Path, default=Path("results/task6_controlled_followup_20261006_coordinates.csv"))
    parser.add_argument("--geometry-protocol", type=Path, default=Path("results/task6_literature_regions_20261014_protocol.json"))
    parser.add_argument("--gff", type=Path, default=Path("data/raw/GCF_000005845.2_ASM584v2_genomic.gff.gz"))
    parser.add_argument("--out", type=Path, default=Path("outputs/task6_full_geometry_20261014/primary_views"))
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Use a fresh figure output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    geometry = json.loads(args.geometry_protocol.read_text(encoding="utf-8"))
    if geometry["literature_version"] != VERSION or sha(args.bundle) != geometry["input_sha256"]["bundle"] or sha(args.coordinates) != geometry["input_sha256"]["coordinates"]:
        raise ValueError("Figure inputs disagree with the declared complete coordinate-source version")
    frame = pd.read_csv(args.coordinates)
    xyz = load_models(args.bundle, (MODEL_KEYS[0],), len(frame))[MODEL_KEYS[0]]
    ori = ori_from_gff(args.gff)
    target = args.out / "primary_two_views.png"
    labels = draw_primary_views(xyz, frame, ori["bp"], target)
    receipt = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "model": MODEL_KEYS[0],
               "selection": "First originally declared seed, not best geometry seed", "coordinate_version": VERSION,
               "source_sha256": {"scripts/task6_primary_views_20261014.py": sha(Path(__file__)),
                                  "scripts/task6_literature_regions_20261014.py": sha(ROOT / "scripts/task6_literature_regions_20261014.py")},
               "input_sha256": {"bundle": sha(args.bundle), "coordinates": sha(args.coordinates),
                                "geometry_protocol": sha(args.geometry_protocol), "gff": sha(args.gff)},
               "outputs_sha256": {target.name: sha(target)}, "bins": len(frame),
               "label_counts": pd.Series(labels).value_counts().to_dict(), "genome_bp": int(frame.end.max()),
               "oriC": ori, "dif_approximate_bp": 1604000,
               "views": [{"elevation": 25, "azimuth": -55}, {"elevation": 25, "azimuth": 125}],
               "equal_axis_and_view_scales": True, "labels_do_not_enter_training": True,
               "distance_units": "Relative model units, no cell or microscopy calibration"}
    dump(args.out / "primary_views_receipt.json", receipt)
    if args.publish:
        for p in args.out.iterdir():
            destination = ROOT / "results" / ("task6_literature_regions_20261014_" + p.name)
            if destination.exists():
                raise FileExistsError("Do not overwrite an existing published figure")
            shutil.copyfile(p, destination)
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
