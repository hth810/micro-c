"""Audit genomic separation of Task 1 train/validation/test annotations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def legacy_splits(frame: pd.DataFrame, seed: int) -> pd.Series:
    fit, test, labels_fit, _ = train_test_split(
        frame.id.values, frame.label.values, test_size=.2, random_state=seed,
        stratify=frame.label.values)
    train, validation = train_test_split(
        fit, test_size=.2, random_state=seed + 1, stratify=labels_fit)
    return pd.Series(np.select([frame.id.isin(train), frame.id.isin(validation)],
                               ["train", "validation"], default="test"), index=frame.index)


def audit(frame: pd.DataFrame, window_bp: int) -> dict:
    result = {"window_bp": window_bp, "structures": len(frame), "by_split": {}}
    for split in ("train", "validation", "test"):
        subset = frame[frame.split == split]
        other = frame[frame.split != split]
        nearest = []
        for _, row in subset.iterrows():
            neighbor = other[other.chrom == row.chrom]
            nearest.append(int((neighbor.center - row.center).abs().min()))
        nearest = np.asarray(nearest)
        result["by_split"][split] = {
            "structures": len(subset),
            "overlapping_other_split_windows": int((nearest < window_bp).sum()),
            "minimum_other_split_center_distance_bp": int(nearest.min()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--window-bp", type=int, default=5120)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--legacy", action="store_true", help="reconstruct original ID-only split")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest)
    if args.legacy:
        frame["split"] = legacy_splits(frame, args.seed)
    elif "split" not in frame:
        raise ValueError("Manifest needs a split column, or pass --legacy")
    result = audit(frame, args.window_bp)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
