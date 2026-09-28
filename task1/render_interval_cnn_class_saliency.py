"""Three-class held-out saliency examples for the annotated-interval CNN.

One correctly recognized annotation per class is selected after OOF scoring
for illustration only. A fold-specific model is re-fitted for each example;
predictions are checked against the saved OOF table. Attribution is local
gradient sensitivity and cannot establish a causal biological structure.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from task1.evaluate_chid_overlap import spatial_groups
from task1.evaluate_interval_conditioned_cnn import make_modes
from task1.evaluate_pairwise import fit_fold
from task1.run_task1 import CLASSES
from task1.saliency_audit import distance_matched_enrichment


def select_examples(ann: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, int | None]:
    aligned = prediction.set_index("id").loc[ann.id]
    selected = {}
    for name in CLASSES:
        eligible = ann.index[(ann.type == name) &
                             (aligned.predicted_type.to_numpy() == name)].to_numpy()
        if len(eligible) == 0:
            selected[name] = None
            continue
        # Median annotation length among successes avoids choosing the most
        # visually dramatic example. Selection is still post-hoc, not random.
        lengths = ann.loc[eligible, "length_bp"].to_numpy()
        target = np.median(lengths)
        selected[name] = int(eligible[np.argmin(np.abs(lengths - target))])
    return selected


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", type=Path, default=Path("outputs/task1_multiscale/windows.npz"))
    ap.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    ap.add_argument("--dir", type=Path, default=Path("outputs/task1_interval_conditioned_cnn"))
    args = ap.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    ann = pd.read_csv(args.annotations).reset_index(drop=True)
    with np.load(args.windows) as bundle:
        contact = bundle["X"].astype(np.float32)
        ids = bundle["group"].astype(str)
    if not np.array_equal(ids, np.tile(ann.id.to_numpy(), 2)):
        raise ValueError("Cached replicate IDs do not match annotations")
    x = make_modes(contact, ann)["contact_interval"]
    y = ann.type.map({name: i for i, name in enumerate(CLASSES)}).to_numpy(dtype=int)
    groups = spatial_groups(ann, 40000)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260921)
                 .split(ann, y, groups))
    saved = pd.read_csv(args.dir / "crossfit_predictions.csv")
    saved = saved[saved["mode"] == "contact_interval"].set_index("id")
    choice = select_examples(ann, saved.reset_index())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    fig, axes = plt.subplots(3, 2, figsize=(9, 12), dpi=180)
    for row, name in enumerate(CLASSES):
        index = choice[name]
        if index is None:
            for ax in axes[row]:
                ax.text(.5, .5, f"No correctly classified {name} in held-out folds",
                        ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
            continue
        fold = next(k for k, (_, test) in enumerate(folds) if index in test)
        if fold not in models:
            models[fold] = fit_fold(x, y, folds[fold][0], np.empty((0, 2), dtype=int),
                                    12, 20260921 + fold, 0.0, device)
        model = models[fold]
        images, gradients, logits = [], [], []
        for rep in (0, 1):
            sample = x[index + rep * len(ann)]
            tensor = torch.from_numpy(sample[None]).to(device).requires_grad_(True)
            output = model(tensor)[0]
            gradient = torch.autograd.grad(output[y[index]], tensor)[0][0, 1].abs().detach().cpu().numpy()
            images.append(sample[1])
            gradients.append(gradient)
            logits.append(output.detach().cpu().numpy())
        logits = np.mean(logits, axis=0)
        identifier = ann.iloc[index].id
        if not np.allclose(logits, [saved.loc[identifier, f"logit_{label}"] for label in CLASSES], atol=1e-4):
            raise ValueError(f"Fold refit disagrees with saved prediction for {identifier}")
        image = np.mean(images, axis=0)
        gradient = np.mean(gradients, axis=0)
        interval = np.diagonal(x[index, 2]).astype(bool)
        left, right = np.flatnonzero(interval)[[0, -1]]
        enrichment = distance_matched_enrichment(gradient, image, interval)
        for col, ax in enumerate(axes[row]):
            ax.imshow(image, cmap="Greys_r", origin="lower", vmin=-2, vmax=3)
            if col == 1:
                high = max(float(np.quantile(gradient, .99)), 1e-8)
                strength = np.clip(gradient / high, 0, 1)
                ax.imshow(gradient, cmap="magma", origin="lower", vmin=0, vmax=high,
                          alpha=np.clip(strength * .8, 0, .8))
            ax.add_patch(Rectangle((left - .5, left - .5), right - left + 1,
                                   right - left + 1, fill=False, edgecolor="#17b783", lw=1.2))
            ax.set(xticks=[], yticks=[])
        axes[row, 0].set_ylabel(f"{name}: {identifier}\nOOF predicted {CLASSES[logits.argmax()]}")
        axes[row, 0].set_title("20.48 kb contact" if row == 0 else "")
        axes[row, 1].set_title("Contact + |true-class gradient| overlay" if row == 0 else "")
        axes[row, 1].text(.02, .98,
                          f"offset-matched enrichment {enrichment['attribution_enrichment']:.2f}",
                          va="top", transform=axes[row, 1].transAxes, fontsize=8,
                          color="white", bbox={"facecolor": "black", "alpha": .5, "pad": 2})
    fig.suptitle("Illustrative correct held-out examples, chosen after scoring; green = supplied interval\n"
                 "Input gradients indicate sensitivity, not causal or independent biological validation",
                 fontsize=10)
    fig.tight_layout()
    path = args.dir / "saliency_examples_by_class.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path}: " + ", ".join(f"{name}={ann.iloc[index].id if index is not None else 'none'}"
                                   for name, index in choice.items()))


if __name__ == "__main__":
    main()
