"""Circular-bin normalization and evaluation for a ring chromosome.

Distances are in bin indices. A final partial bin is an approximation; exported
genomic coordinates must still clip its end to the actual chromosome length.
The legacy linear target is retained elsewhere for historical reproducibility.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def ring_offsets(n: int) -> np.ndarray:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 5:
        raise ValueError("A circular chromosome needs at least five bins")
    delta = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    return np.minimum(delta, n - delta)


def circular_pairs(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each unordered non-neighbor pair once, including across the origin."""
    offsets = ring_offsets(n)
    i, j = np.triu_indices(n, k=1)
    keep = offsets[i, j] >= 2
    return i[keep], j[keep], offsets[i[keep], j[keep]]


def cyclic_chain_variance(coordinates):
    """Penalize unequal neighbor lengths without privileging the genome origin.

    All n cyclic edges are included; the wrap edge is not forced to length zero.
    This term alone is not a physical restraint or a non-collapse guarantee.
    """
    import torch
    if coordinates.ndim != 2 or coordinates.shape[0] < 3:
        raise ValueError("Coordinates must contain at least three rows")
    lengths = torch.linalg.vector_norm(coordinates - torch.roll(coordinates, -1, 0), dim=1)
    return (lengths - lengths.mean()).square().mean()


def circular_contact_target(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    if (raw.ndim != 2 or raw.shape[0] != raw.shape[1] or
            not np.isfinite(raw).all() or (raw < 0).any() or
            not np.allclose(raw, raw.T, rtol=1e-8, atol=1e-8)):
        raise ValueError("Raw contact must be finite, nonnegative, symmetric and square")
    n = len(raw)
    offsets = ring_offsets(n)
    i, j = np.triu_indices(n)
    x = np.log1p(raw)
    expected = np.array([np.median(x[i, j][offsets[i, j] == offset])
                         for offset in range(n // 2 + 1)])
    residual = np.maximum(x - expected[offsets], 0)
    positive = residual[residual > 0]
    scale = float(np.quantile(positive, .95)) if len(positive) else 1.0
    return np.clip(residual / max(scale, 1e-8), 0, 1)


def circular_matched_auc(contact: np.ndarray, distances: dict[str, np.ndarray],
                         seed: int = 20261006) -> dict:
    """Balance positive/zero residual labels within every exact ring offset."""
    n = len(contact)
    if (contact.shape != (n, n) or not np.isfinite(contact).all() or
            any(value.shape != (n, n) or not np.isfinite(value).all()
                for value in distances.values())):
        raise ValueError("Contact and distance maps must be equally sized finite squares")
    i, j, offsets = circular_pairs(n)
    labels = contact[i, j] > 0
    rng = np.random.default_rng(seed)
    selections, groups = [], []
    for offset in np.unique(offsets):
        positive = np.flatnonzero((offsets == offset) & labels)
        zero = np.flatnonzero((offsets == offset) & ~labels)
        count = min(len(positive), len(zero))
        if count:
            selections.extend((rng.choice(positive, count, replace=False),
                               rng.choice(zero, count, replace=False)))
            groups.append({"circular_offset_bins": int(offset),
                           "positive": count, "zero": count})
    if not selections:
        return {"matched_pairs_per_class": 0, "offsets": [], "auc": {},
                "not_evaluable": "No ring offset contains both positive and zero residuals"}
    take = np.concatenate(selections)
    auc = {name: float(roc_auc_score(labels[take], -matrix[i[take], j[take]]))
           for name, matrix in distances.items()}
    # An exact offset score avoids floating-point differences in sin/cos chords.
    auc["exact_ring_offset_baseline"] = float(roc_auc_score(labels[take], -offsets[take]))
    return {"matched_pairs_per_class": int(labels[take].sum()),
            "selection_seed": seed, "offsets": groups, "auc": auc,
            "definition": "positive vs clipped-zero residual at the same circular bin offset"}
