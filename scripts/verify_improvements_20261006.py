"""Byte-level integrity receipt for the 2026-10-06 experimental improvements.

Writing a receipt is an explicit packaging operation, not an experiment or proof
of scientific validity. Verification never repairs or overwrites frozen records.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = Path("results/improvements_20261006_integrity.json")


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Receipt path escapes repository: {relative}")
    return path


def verify(root: Path, receipt: dict) -> dict:
    for relative, expected in receipt["sha256"].items():
        path = safe_path(root, relative)
        if not path.is_file() or digest(path) != expected:
            raise AssertionError(f"Missing or changed evidence: {relative}")
    return {"verified_files": len(receipt["sha256"]),
            "scope": "micro-c only; byte integrity, not independent scientific validation"}


def build(root: Path) -> dict:
    paths = {root / "data/structures.csv", Path(__file__)}
    for pattern in ("results/task1_improve_20261006_*", "results/task2_improve_20261006_*",
                    "results/task6_circular_20261006_*", "task1/*_20261006.py",
                    "scripts/*_20261006.py", "tests/*_20261006.py"):
        paths.update(path for path in root.glob(pattern) if path.is_file())
    paths.update(root / name for name in (
        "microc/data.py", "microc/deep.py", "microc/circular.py",
        "scripts/task6_circular_sensitivity.py", "scripts/check_deliverables.py",
        "tests/test_area_resampling.py", "tests/test_circular_contact.py",
        "tests/test_task2_residual_motif_screen.py", "tests/test_task2_resampling_comparison.py"))
    # Preserve actual helper-source byte hashes recorded at Task 1 run time,
    # including the historical Windows absolute path, but resolve portably.
    for protocol_path in root.glob("results/task1_improve_20261006_*protocol.json"):
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        sources = protocol.get("source_sha256", {})
        if not isinstance(sources, dict):
            continue
        for name, expected in sources.items():
            relative = name.replace("\\", "/").split("/micro-c/")[-1]
            path = safe_path(root, relative)
            if digest(path) != expected:
                raise AssertionError(f"Frozen Task 1 source changed: {relative}")
            paths.add(path)
    return {"packaged_utc": datetime.now(timezone.utc).isoformat(),
            "run_parent_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "scope": "Micro-C Tasks 1/2/6 improvements; historical negative controls retained",
            "caution": "This records published bytes, not new blind validation; raw COOL remains outside Git",
            "sha256": {path.relative_to(root).as_posix(): digest(path) for path in sorted(paths)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    receipt_path = safe_path(root, str(args.receipt))
    if args.write_manifest:
        if receipt_path.exists():
            raise ValueError("Receipt already exists; use a new path to preserve the prior packaging record")
        receipt = build(root)
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    print(json.dumps(verify(root, receipt), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
