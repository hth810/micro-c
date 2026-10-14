"""Post-run integrity and overwrite-guard checks for frozen Task5 artifacts.

This audit does not train, select, modify predictions, or replace any prior
receipt. Scientific numeric cross-checks are in the separate export receipt.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.task5_full_structure_verification_20261014 import ROOT, digest, utc, write_json
from scripts.task5_full_structure_train_20261014 import PREFIX


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/task5_full_structure_20261014/neural_adaptation")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    target = args.run / "complete_evidence/post_run_integrity_receipt.json"
    if target.exists() or (args.results_dir / (PREFIX + target.name)).exists():
        raise FileExistsError("Integrity audit refuses to overwrite an earlier audit")
    receipt = json.loads((args.run / "receipt.json").read_text(encoding="utf-8"))
    export = json.loads((args.run / "complete_evidence/export_receipt.json").read_text(encoding="utf-8"))
    freeze = json.loads((args.run / "selection_freeze.json").read_text(encoding="utf-8"))
    files = {args.run / name: value for name, value in receipt["output_sha256"].items()}
    files.update({args.run / "complete_evidence" / name: value for name, value in export["output_sha256"].items()})
    files.update({args.run / "receipt.json": digest(args.run / "receipt.json"),
                  args.run / "complete_evidence/export_receipt.json": digest(args.run / "complete_evidence/export_receipt.json")})
    for path, expected in files.items():
        if digest(path) != expected:
            raise ValueError(f"Frozen artifact changed: {path}")
    sources = {
        "script": ROOT / "scripts/task5_full_structure_train_20261014.py",
        "evaluator": ROOT / "scripts/task5_full_structure_verification_20261014.py",
        "scale_script": ROOT / "scripts/task5_full_structure_repair_20261014.py",
        "deep": ROOT / "microc/deep.py",
        "data": ROOT / "microc/data.py",
    }
    for name, path in sources.items():
        if digest(path) != freeze["source_and_input_sha256"][name]:
            raise ValueError(f"Frozen implementation changed: {name}")
    if digest(args.run / "selected_model.pt") != freeze["model_sha256"]:
        raise ValueError("Frozen final model changed")
    guards = []
    env = dict(os.environ, OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    commands = [
        ("duplicate_select", [sys.executable, "-m", "scripts.task5_full_structure_train_20261014", "--phase", "select", "--out", str(args.run), "--results-dir", str(args.results_dir)]),
        ("duplicate_test", [sys.executable, "-m", "scripts.task5_full_structure_train_20261014", "--phase", "test", "--out", str(args.run), "--results-dir", str(args.results_dir)]),
        ("duplicate_export", [sys.executable, "-m", "scripts.task5_full_structure_export_20261014", "--run", str(args.run), "--results-dir", str(args.results_dir)]),
    ]
    for name, command in commands:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=env, timeout=45)
        rejected = completed.returncode != 0 and "FileExistsError" in completed.stderr
        if not rejected:
            raise ValueError(f"Overwrite guard did not reject {name}: {completed.stderr}")
        guards.append({"name": name, "command": command, "returncode": completed.returncode,
                       "rejected_before_run": rejected, "error": completed.stderr.strip().splitlines()[-1]})
    for path, expected in files.items():
        if digest(path) != expected:
            raise ValueError(f"Overwrite-guard attempt changed frozen artifact: {path}")
    record = {"completed_at_utc": utc(), "audit_script_sha256": digest(Path(__file__)),
              "frozen_artifact_hash_checks_before_and_after": len(files),
              "source_hash_checks": {name: digest(path) for name, path in sources.items()},
              "overwrite_guards": guards, "all_frozen_artifacts_unchanged": True,
              "full_map_verification_receipt_sha256": digest(args.run / "complete_evidence/export_receipt.json"),
              "model_sha256": freeze["model_sha256"], "no_training_or_scientific_refitting": True}
    write_json(target, record)
    write_json(args.results_dir / (PREFIX + target.name), record)
    print(json.dumps({"checks": len(files), "overwrite_guards": len(guards), "unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
