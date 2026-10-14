"""Run the complete additive evidence audit and fixed stronger-SR comparisons."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import pandas as pd

from scripts.task5_full_structure_verification_20261014 import mean_interval

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def task2_check_path(root, check):
    # Summary-copy receipts store a published filename; other kinds already
    # store a repository-relative path. Preserve and interpret that schema.
    relative = Path("results") / check["path"] if check["kind"] == "summary_copy" else Path(check["path"])
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Task2 evidence path leaves its repository")
    return path


def strong_controls():
    records = []
    for replicate in ("rep1_test", "rep2_fixed_test"):
        path = ROOT / f"results/task5_full_structure_20261014_final_{replicate}.csv"
        frame = pd.read_csv(path)
        selected = frame[frame.method == "adapted_cnn_fraction_1"].sort_values("id")
        if len(selected) != 68 or selected.id.nunique() != 68:
            raise AssertionError("Missing or repeated full-window predictions")
        for control in ("bilinear", "bicubic", "train_calibrated_bilinear", "train_calibrated_bicubic"):
            other = frame[frame.method == control].sort_values("id")
            if selected.id.tolist() != other.id.tolist():
                raise AssertionError("Strong control prediction IDs differ")
            for metric in ("psnr", "ssim", "shape_r", "profile_r", "peak_error_bp", "top5_f1"):
                delta = selected[metric].to_numpy() - other[metric].to_numpy()
                if metric == "peak_error_bp":
                    delta = -delta
                low, high = mean_interval(delta, selected.overlap_component.to_numpy())
                records.append({"replicate": replicate, "control": control, "metric": metric,
                                "n_structures": len(selected), "n_overlap_components": selected.overlap_component.nunique(),
                                "cnn_mean": selected[metric].mean(), "control_mean": other[metric].mean(),
                                "gain_mean": delta.mean(), "gain_component_95ci_lower": low, "gain_component_95ci_upper": high,
                                "source_csv_sha256": digest(path)})
    return pd.DataFrame(records)


def main():
    output = ROOT / "results/final_verification_20261014_audit.json"
    control_path = ROOT / "results/task5_full_structure_20261014_final_strong_controls.csv"
    if output.exists() or control_path.exists():
        raise FileExistsError("Preserve the completed final audit")
    logs = []
    for args in (("-m", "pytest", "-q", "tests"),
                 ("-m", "scripts.verify_followups_20261006"),
                 ("-m", "scripts.audit_task6_final_20261014"),
                 ("-m", "scripts.check_deliverables")):
        run = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        logs.append({"argv": [sys.executable, *args], "exit_code": run.returncode,
                     "stdout": run.stdout, "stderr": run.stderr})
        if run.returncode:
            raise RuntimeError(f"Final audit failed: {args}: {run.stdout} {run.stderr}")
    match = re.search(r"(\d+) passed", logs[0]["stdout"])
    if not match:
        raise AssertionError("Cannot establish actual passing test count")
    task2 = json.loads((ROOT / "results/task2_course_verification_20261014_final_receipt.json").read_text(encoding="utf-8"))
    if task2["failures"] or task2["checks_passed"] != 93:
        raise AssertionError("Task2 final evidence audit incomplete")
    for check in task2["checks"]:
        path = task2_check_path(ROOT, check)
        if digest(path) != check["expected_sha256"]:
            raise AssertionError(f"Task2 evidence drift after receipt: {check['path']}")
    for relative, expected in task2["source_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise AssertionError(f"Task2 frozen source drift: {relative}")
    receipt_path = ROOT / "results/task5_full_structure_20261014_final_receipt.json"
    task5 = json.loads(receipt_path.read_text(encoding="utf-8"))
    folder = ROOT / "outputs/task5_full_structure_20261014/neural_adaptation"
    verified_sr = []
    for relative, expected in task5["output_sha256"].items():
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder.resolve()) or digest(path) != expected:
            raise AssertionError(f"Full-window SR evidence drift: {relative}")
        verified_sr.append(relative)
    comparisons = strong_controls()
    comparisons.to_csv(control_path, index=False)
    value = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "hth810/micro-c",
             "pytest_passed": int(match.group(1)), "checks": logs,
             "task2_frozen_receipt_sha256": digest(ROOT / "results/task2_course_verification_20261014_final_receipt.json"),
             "task5_verified_output_count": len(verified_sr), "task5_verified_output_files": verified_sr,
             "task5_receipt_sha256": digest(receipt_path), "strong_control_csv_sha256": digest(control_path),
             "interpretation": "Integrity and fixed comparisons; course and scientific claims are separately reviewed",
             "source_sha256": {"scripts/audit_final_experiments_20261014.py": digest(Path(__file__))}}
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pytest_passed": value["pytest_passed"], "task5_verified_outputs": len(verified_sr),
                      "strong_control_rows": len(comparisons)}, ensure_ascii=False))


if __name__ == "__main__": main()
