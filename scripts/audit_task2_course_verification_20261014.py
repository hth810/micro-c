"""Read-only evidence verification plus a new delivery receipt; no rerunning."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.task2_residual_motif_screen_20261006 import sha256, write_json


def main():
    groups = [
        (Path("outputs/task2_course_verification_20261014"), "task2_course_verification_20261014_"),
        (Path("outputs/task2_course_verification_20261014_qc_counterfactual_v2"), "task2_course_verification_20261014_qc_"),
        (Path("outputs/task2_course_verification_20261014_classifier_context"), "task2_course_verification_20261014_classifier_"),
    ]
    checks, inputs = [], {}
    for folder, prefix in groups:
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        protocol = json.loads((folder / "protocol_freeze.json").read_text(encoding="utf-8"))
        for relative, expected in summary["output_sha256"].items():
            path = folder / relative
            actual = sha256(path)
            checks.append({"kind": "output", "path": path.as_posix(), "expected_sha256": expected,
                           "actual_sha256": actual, "passed": expected == actual})
            published = Path("results") / (prefix + relative)
            if published.exists():
                actual_copy = sha256(published)
                checks.append({"kind": "published_copy", "path": published.as_posix(), "expected_sha256": expected,
                               "actual_sha256": actual_copy, "passed": expected == actual_copy})
        for name, expected in protocol["source_sha256"].items():
            source = Path(name)
            actual = sha256(source)
            checks.append({"kind": "frozen_source", "path": source.as_posix(), "expected_sha256": expected,
                           "actual_sha256": actual, "passed": actual == expected})
        for name, expected in protocol["input_sha256"].items():
            if name in inputs and inputs[name] != expected:
                raise ValueError("Conflicting frozen hashes for the same input")
            inputs[name] = expected
        # Summary itself is outside its own output hash list, but its published
        # copy must be exact and the resulting receipt hashes both versions.
        expected = sha256(folder / "summary.json")
        actual = sha256(Path("results") / (prefix + "summary.json"))
        checks.append({"kind": "summary_copy", "path": prefix + "summary.json",
                       "expected_sha256": expected, "actual_sha256": actual, "passed": expected == actual})
    for name, expected in inputs.items():
        actual = sha256(Path(name))
        checks.append({"kind": "frozen_input", "path": name, "expected_sha256": expected,
                       "actual_sha256": actual, "passed": actual == expected})
    failed = [check for check in checks if not check["passed"]]
    sources = [Path("scripts") / name for name in (
        "task2_course_verification_20261014.py", "task2_course_qc_counterfactual_20261014.py",
        "task2_course_classifier_context_20261014.py", "audit_task2_course_verification_20261014.py")]
    sources += [Path("tests") / name for name in (
        "test_task2_course_verification_20261014.py", "test_task2_course_verification_counterfactual_20261014.py",
        "test_task2_course_verification_classifier_20261014.py")]
    receipt = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Byte integrity and publication-copy receipt, not independent biological validation",
        "groups": [{"output": folder.as_posix(), "results_prefix": prefix} for folder, prefix in groups],
        "source_sha256": {source.as_posix(): sha256(source) for source in sources},
        "checks": checks, "checks_passed": len(checks) - len(failed), "failures": failed,
        "scientific_scope": {"fixed_candidates": ["A1", "A2", "A3", "A4"],
                             "historical_default_shape_outlier_candidates_replicated": ["A1", "A2", "A4"],
                             "confirmed_new_shape_types": 0, "p_or_fdr_claimed": False,
                             "provisional_roi_classifier_is_ood_test": False,
                             "artificial_coverage_masks_are_genomic_null": False},
        "interrupted_partial_run": "outputs/task2_course_verification_20261014_qc_counterfactual/interruption_receipt.json",
        "authoritative_common_depth_output": "outputs/task2_course_verification_20261014_qc_counterfactual_v2",
    }
    if failed:
        raise ValueError(f"Evidence hashes failed: {failed}")
    target = Path("results/task2_course_verification_20261014_final_receipt.json")
    if target.exists():
        raise ValueError("Do not overwrite an audit receipt")
    write_json(target, receipt)
    print(json.dumps({"checks_passed": receipt["checks_passed"], "failures": len(failed),
                      "receipt_sha256": sha256(target)}, indent=2))


if __name__ == "__main__":
    main()
