"""Record all generated 10-kb track images without committing 93 MB of PNGs."""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracks", type=Path, default=Path("outputs/task3_labeled"))
    ap.add_argument("--out", type=Path, default=Path("results/task3_track_manifest.csv"))
    ap.add_argument("--genome-bp", type=int, default=4_641_652)
    ap.add_argument("--interval-bp", type=int, default=10_000)
    args = ap.parse_args()
    rows = []
    for path in sorted(args.tracks.glob("*.png")):
        match = re.search(r"_(\d+)\.png$", path.name)
        if not match:
            raise ValueError(f"Unexpected track image name: {path.name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        start = int(match.group(1))
        rows.append({"start_bp": start, "end_bp": min(start + args.interval_bp, args.genome_bp),
                     "filename": path.name, "bytes": path.stat().st_size, "sha256": digest})
    frame = pd.DataFrame(rows)
    if frame.start_bp.tolist() != list(range(0, args.genome_bp, args.interval_bp)):
        raise ValueError("Tracks do not cover the genome exactly at the requested interval")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"Recorded {len(frame)} track images: {args.out}")


if __name__ == "__main__":
    main()
