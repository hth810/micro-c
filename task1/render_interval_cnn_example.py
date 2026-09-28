"""Render one post-hoc illustrative CHID/nearby-CHIN saliency/occlusion pair.

The example is selected after seeing outer-fold predictions and must not be
treated as a randomly sampled case or a performance estimate. The same held-
out fold model is re-fitted and its logits are checked against saved OOF data.
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

from task1.audit_interval_cnn_occlusion import matched_long_contact_occlusions
from task1.evaluate_chid_overlap import spatial_groups
from task1.evaluate_interval_conditioned_cnn import make_modes
from task1.evaluate_pairwise import fit_fold, nearby_chid_chin_pairs
from task1.run_task1 import CLASSES


def choose_example(ann: pd.DataFrame, prediction: pd.DataFrame,
                   saliency: pd.DataFrame) -> tuple[int, int]:
    """Correct, <1 kb-overlapping CHID nearest median enrichment, then its CHIN."""
    pairs = nearby_chid_chin_pairs(ann, 1000)
    lookup = prediction.set_index("id")
    enrichment = saliency[saliency.evaluable].groupby("id").attribution_enrichment.mean()
    eligible = []
    for chid_index, chin_index in pairs:
        identifier = ann.iloc[chid_index].id
        if lookup.loc[identifier, "predicted_type"] == "CHID" and identifier in enrichment:
            eligible.append((int(chid_index), int(chin_index), float(enrichment[identifier])))
    if not eligible:
        raise ValueError("No correctly classified nearby CHID for illustrative figure")
    midpoint = float(np.median([row[2] for row in eligible]))
    selected = min(eligible, key=lambda row: (abs(row[2] - midpoint), row[0]))
    return selected[0], selected[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", type=Path, default=Path("outputs/task1_multiscale/windows.npz"))
    ap.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    ap.add_argument("--dir", type=Path, default=Path("outputs/task1_interval_conditioned_cnn"))
    args = ap.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    ann = pd.read_csv(args.annotations).reset_index(drop=True)
    with np.load(args.windows) as bundle:
        x = bundle["X"].astype(np.float32)
        ids = bundle["group"].astype(str)
    if not np.array_equal(ids, np.tile(ann.id.to_numpy(), 2)):
        raise ValueError("Cached replicate IDs do not match annotations")
    images = make_modes(x, ann)["contact_interval"]
    y = ann.type.map({name: i for i, name in enumerate(CLASSES)}).to_numpy(dtype=int)
    groups = spatial_groups(ann, 40000)
    saved = pd.read_csv(args.dir / "crossfit_predictions.csv")
    saved = saved[saved["mode"] == "contact_interval"]
    saliency = pd.read_csv(args.dir / "saliency_samples.csv")
    chid, chin = choose_example(ann, saved, saliency)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260921)
                 .split(ann, y, groups))
    matching = [number for number, (_, test) in enumerate(folds) if chid in test]
    if len(matching) != 1 or chin not in folds[matching[0]][1]:
        raise ValueError("CHID and nearby CHIN are not in the same held-out fold")
    fold = matching[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fit_fold(images, y, folds[fold][0], np.empty((0, 2), dtype=int),
                     12, 20260921 + fold, 0.0, device)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), dpi=180)
    chosen = (chid, chin)
    for row_index, index in enumerate(chosen):
        contacts, gradients, logits = [], [], []
        for rep in (0, 1):
            sample = images[index + rep * len(ann)]
            tensor = torch.from_numpy(sample[None]).to(device).requires_grad_(True)
            out = model(tensor)[0]
            gradient = torch.autograd.grad(out[y[index]], tensor)[0][0, 1].abs().detach().cpu().numpy()
            contacts.append(sample[1])
            gradients.append(gradient)
            logits.append(out.detach().cpu().numpy())
        observed = np.mean(logits, axis=0)
        reference = saved.set_index("id").loc[ann.iloc[index].id]
        if not np.allclose(observed, [reference[f"logit_{name}"] for name in CLASSES], atol=1e-4):
            raise ValueError("Refitted fold model disagrees with saved out-of-fold prediction")
        contact = np.mean(contacts, axis=0)
        gradient = np.mean(gradients, axis=0)
        sample = images[index].copy()
        inside, outside, count = matched_long_contact_occlusions(
            sample, np.random.default_rng(20260928 + 2 * index))
        selector = np.zeros_like(contact)
        selector[(sample[1] != 0) & (inside[1] == 0)] = 1
        selector[(sample[1] != 0) & (outside[1] == 0)] = -1
        mask = np.diagonal(sample[2]).astype(bool)
        left, right = np.flatnonzero(mask)[[0, -1]]
        panels = ((contact, "20.48 kb normalized contact", "RdBu_r", -2, 2),
                  (gradient, "|true-class gradient|", "magma", 0, None),
                  (selector, f"Matched occlusion ({count} pairs)", "bwr", -1, 1))
        for column, (data, title, color, low, high) in enumerate(panels):
            ax = axes[row_index, column]
            ax.imshow(data, cmap=color, origin="lower", vmin=low, vmax=high)
            ax.add_patch(Rectangle((left - .5, left - .5), right - left + 1,
                                   right - left + 1, fill=False, edgecolor="#35cf99", lw=1.2))
            ax.set(title=title if row_index == 0 else "", xticks=[], yticks=[])
        class_name = ann.iloc[index].type
        axes[row_index, 0].set_ylabel(
            f"{ann.iloc[index].id}\ntrue {class_name}; OOF {CLASSES[observed.argmax()]}")
    fig.suptitle("Illustrative held-out CHID and nearest CHIN (selected after scoring); "
                 "green = annotated interval\nOcclusion colors: red inside, blue matched outside; "
                 "gradients/perturbations are not causal proof", fontsize=10)
    fig.tight_layout()
    path = args.dir / "representative_saliency_occlusion.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path}: CHID {ann.iloc[chid].id}, CHIN {ann.iloc[chin].id}, held-out fold {fold}")


if __name__ == "__main__":
    main()
