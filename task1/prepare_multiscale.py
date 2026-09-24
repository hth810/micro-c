"""Pair the existing 5.12 kb patches with independently normalized 20.48 kb context."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from microc.data import open_cool
from task1.run_task1 import extract_window


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-windows", type=Path,
                        default=Path("outputs/task1_spatial/extracted_windows.npz"))
    parser.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    parser.add_argument("--rep1", type=Path, required=True)
    parser.add_argument("--rep2", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/task1_multiscale/windows.npz"))
    args = parser.parse_args()
    ann = pd.read_csv(args.annotations)
    with np.load(args.short_windows) as bundle:
        short = bundle["X"][:, 0].astype(np.float32)
        ids = bundle["group"].astype(str)
    if len(short) != 2 * len(ann) or not np.array_equal(ids[:len(ann)], ann.id.to_numpy()):
        raise ValueError("Short windows and annotation order do not match")
    large = []
    for path in (args.rep1, args.rep2):
        with open_cool(path) as cool:
            for _, row in ann.iterrows():
                large.append(extract_window(cool, int(row.center), 128, aggregate=16))
    channels = np.stack([short, np.stack(large)], axis=1).astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=channels, group=ids)
    print(f"Saved {len(channels)} two-scale windows to {args.out}")


if __name__ == "__main__":
    main()
