"""Render Task 1 confusion, saliency, and occlusion diagnostics from saved OOF results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from task1.run_task1 import CLASSES


MODES = ("contact_short", "contact_dual", "contact_interval", "interval_only")


def render_confusions(summary: dict, path: Path) -> None:
    vmax = max(np.max(summary[mode]["confusion_matrix"]) for mode in MODES)
    labels = {"contact_short": "Short contact", "contact_dual": "Two contact scales",
              "contact_interval": "Contact + interval", "interval_only": "Interval masks only"}
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), dpi=180, sharex=True, sharey=True)
    for ax, mode in zip(axes.flat, MODES):
        metric = summary[mode]
        matrix = np.asarray(metric["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > vmax * .5 else "black")
        ax.set_xticks(range(3), CLASSES, rotation=25)
        ax.set_yticks(range(3), CLASSES)
        ax.set_title(f"{labels[mode]}\nmacro F1 {metric['macro_f1']:.3f}; CHID P/R "
                     f"{metric['chid_precision']:.3f}/{metric['chid_recall']:.3f}")
    for ax in axes[:, 0]:
        ax.set_ylabel("True class")
    for ax in axes[-1]:
        ax.set_xlabel("Predicted class")
    fig.suptitle("Five 40 kb spatial-group folds, 344 structures (exploratory)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def render_single_confusion(summary: dict, path: Path) -> None:
    metric = summary["contact_interval"]
    matrix = np.asarray(metric["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=180)
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label="Structures")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    fontsize=13, color="white" if matrix[i, j] > matrix.max() * .55 else "black")
    ax.set(xticks=range(3), xticklabels=CLASSES,
           yticks=range(3), yticklabels=CLASSES,
           xlabel="Predicted class", ylabel="True class",
           title=("Contact + annotated interval CNN\n"
                  f"40 kb spatial CV · macro F1 {metric['macro_f1']:.3f} · "
                  f"CHID precision {metric['chid_precision']:.3f}"))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def render_sensitivity(occlusion: dict, saliency: pd.DataFrame,
                       samples: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=180)
    rng = np.random.default_rng(20260928)
    sal = saliency[saliency.evaluable].groupby(["id", "true_type"], as_index=False).agg(
        attribution_enrichment=("attribution_enrichment", "mean"))
    for x, label, color in ((0, "CHID", "#bf4b3c"), (1, "near CHIN", "#326c9c")):
        kind = "CHID" if x == 0 else "CHIN"
        values = sal.loc[sal.true_type == kind, "attribution_enrichment"].to_numpy()
        axes[0].scatter(x + rng.uniform(-.17, .17, len(values)), values,
                        s=18, alpha=.7, color=color)
        axes[0].plot([x - .23, x + .23], [np.median(values)] * 2, color="black", lw=2)
    axes[0].axhline(1, color="grey", ls="--", lw=1)
    axes[0].set(xticks=[0, 1], xticklabels=["CHID", "paired CHIN"],
                ylabel="Matched-offset gradient enrichment",
                title="True-class contact saliency")

    per = samples.groupby(["id", "true_type"], as_index=False).agg(
        inside=("margin_drop_inside_long_contact", "mean"),
        outside=("margin_drop_matched_outside_long_contact", "mean"))
    per["difference"] = per.inside - per.outside
    groups = [per.loc[per.true_type == name, "difference"].to_numpy()
              for name in ("CHIN", "OPCID", "CHID")]
    axes[1].boxplot(groups, tick_labels=CLASSES, showfliers=False)
    for x, values in enumerate(groups, 1):
        axes[1].scatter(x + rng.uniform(-.15, .15, len(values)), values,
                        s=8, alpha=.3, color="#414e68")
    axes[1].axhline(0, color="grey", ls="--", lw=1)
    axes[1].set(ylabel="Inside − outside true-class margin drop",
                title="Equal-count, distance-matched contact occlusion")

    names = ("base", "no_short_contact", "no_long_contact", "no_contact",
             "no_interval", "inside_long_contact", "matched_outside_long_contact")
    label = ("Base", "No short", "No long", "No contact", "No interval", "Inside", "Outside")
    values = [occlusion["structure_scores"][name]["macro_f1"] for name in names]
    axes[2].bar(range(len(names)), values, color=["#2d7f70" if i == 0 else "#8ea1a7"
                                                    for i in range(len(names))])
    axes[2].set(xticks=range(len(names)), xticklabels=label, ylim=(0, 1),
                ylabel="Held-out macro F1", title="Channel and region perturbations")
    axes[2].tick_params(axis="x", labelrotation=55)
    fig.suptitle("Sensitivity only; zeroed inputs are out of distribution, not causal proof", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("outputs/task1_interval_conditioned_cnn"))
    args = ap.parse_args()
    summary = json.loads((args.dir / "summary.json").read_text(encoding="utf-8"))
    occlusion = json.loads((args.dir / "occlusion_summary.json").read_text(encoding="utf-8"))
    saliency = pd.read_csv(args.dir / "saliency_samples.csv")
    samples = pd.read_csv(args.dir / "occlusion_samples.csv")
    render_confusions(summary, args.dir / "confusion_ablation.png")
    render_single_confusion(summary, args.dir / "confusion_contact_interval.png")
    render_sensitivity(occlusion, saliency, samples, args.dir / "saliency_occlusion_diagnostics.png")
    print(args.dir / "confusion_ablation.png")
    print(args.dir / "confusion_contact_interval.png")
    print(args.dir / "saliency_occlusion_diagnostics.png")


if __name__ == "__main__":
    main()
