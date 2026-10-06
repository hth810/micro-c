"""Verify the additive Micro-C follow-up package without rewriting old evidence.

Only compact published results and sources are mandatory. Raw COOL/FASTA and
large local ranking-pair tables have recorded provenance, not a requirement
for inspecting a clean checkout. This is integrity checking, not validation.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from scripts.verify_improvements_20261006 import digest, safe_path, verify as verify_old


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = "results/followups_20261006_integrity.json"
PREFIXES = (
    "task1_followup_20261006_", "task2_followup_20261006_",
    "task2_reference_followup_20261006_", "task6_followup_20261006_",
    "task6_controlled_followup_20261006_",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_indexes(root: Path) -> int:
    count = 0
    for path in (root / "results").glob("*evidence_index.json"):
        if not path.name.startswith(PREFIXES):
            continue
        index = load(path)
        artifacts = index.get("files", index.get("artifacts_sha256", {}))
        if "artifacts" in index:
            records = index["artifacts"]
            artifacts = {Path(record["artifact"]).name: record["sha256"] for record in records}
            if len(artifacts) != len(records):
                raise AssertionError("Duplicate follow-up artifact in index")
        if not isinstance(artifacts, dict) or not artifacts:
            raise AssertionError(f"Unrecognized/empty artifact index: {path.name}")
        for relative, expected in artifacts.items():
            # Artifact keys are published basenames; a nested source is NOT an
            # artifact and must be listed in the source map instead.
            if Path(relative).name != relative:
                raise ValueError(f"Unsafe published artifact key: {relative}")
            actual = safe_path(root, "results/" + relative)
            if not actual.is_file() or digest(actual) != expected:
                raise AssertionError(f"Index checksum mismatch: {relative}")
        sources = index.get("source_sha256", {})
        if path.name == "task2_followup_20261006_evidence_index.json":
            sources = load(root / "results/task2_followup_20261006_diagnostic_plan_freeze.json")["source_sha256"]
        if "script_sha256" in index:
            sources = dict(sources)
            sources["scripts/task2_reference_uniqueness_followup.py"] = index["script_sha256"]
        for relative, expected in sources.items():
            relative = relative.replace("\\", "/").split("/micro-c/")[-1]
            actual = safe_path(root, relative)
            if not actual.is_file() or digest(actual) != expected:
                raise AssertionError(f"Frozen follow-up source changed: {relative}")
        count += 1
    return count


def reference_table_audit(root: Path) -> dict:
    path = root / "results/task2_reference_followup_20261006_reference_window_uniqueness.csv"
    if not path.exists():
        raise AssertionError("Missing reference uniqueness cohort table")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    # The original table records four candidates in addition to the original
    # 291 unknown representatives (which contain those candidate loci).
    if len(rows) != (4 + 291 + 344) * 3:
        raise AssertionError("Reference cohort rows do not cover every k/cohort")
    identities = {(row["cohort"], row["id"], int(row["k"])) for row in rows}
    if len(identities) != len(rows):
        raise AssertionError("Repeated cohort/id/k reference record")
    expected_cohorts = {"candidate": 4, "unannotated_reference": 291, "known_reference": 344}
    for cohort, count in expected_cohorts.items():
        for k in (31, 51, 101):
            if sum(row["cohort"] == cohort and int(row["k"]) == k for row in rows) != count:
                raise AssertionError(f"Reference cohort count differs: {cohort}/{k}")
    if any(not 0 <= int(row["start_bp"]) < int(row["end_bp"]) <= 4_641_652 for row in rows):
        raise AssertionError("Reference window outside genome")
    if any(int(row["contained_kmer_starts"]) != int(row["valid_acgt_starts"])
           for row in rows):
        raise AssertionError("Ambiguous/omitted reference starts need an explicit audit")
    return {"reference_rows": len(rows), "ambiguous_start_windows": 0, "edge_omitted_windows": 0,
            "expected_candidate_rows": 12,
            "note": "Reference uniqueness is not measured aligner mappability"}


def verify(root: Path, receipt: dict) -> dict:
    old = verify_old(root, load(root / "results/improvements_20261006_integrity.json"))
    for relative, expected in receipt["sha256"].items():
        actual = safe_path(root, relative)
        if not actual.is_file() or digest(actual) != expected:
            raise AssertionError(f"Missing or changed follow-up evidence: {relative}")
    indexes = verify_indexes(root)
    if indexes != receipt["evidence_indexes"]:
        raise AssertionError("Follow-up index count changed; keep a new receipt")
    return {"verified_followup_files": len(receipt["sha256"]),
            "verified_old_files": old["verified_files"], "evidence_indexes": indexes,
            **reference_table_audit(root),
            "scope": "Micro-C only; byte/coverage integrity, not independent scientific validation"}


def build(root: Path) -> dict:
    paths = {root / "scripts/verify_followups_20261006.py",
             root / "tests/test_followup_integrity.py",
             root / "results/improvements_20261006_integrity.json"}
    for directory in ("scripts", "task1", "tests"):
        paths.update((root / directory).glob("*followup.py"))
    paths.update(path for path in (root / "results").iterdir()
                 if path.is_file() and path.name.startswith(PREFIXES))
    if any(not path.is_file() for path in paths):
        raise AssertionError("A required follow-up source is missing")
    return {"packaged_utc": datetime.now(timezone.utc).isoformat(),
            "run_parent_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "scope": "Additive Micro-C Task 1/2/6 follow-up, preserving prior experiments",
            "caution": "Rep2 and labels historically inspected; no new blind validation. Local raw data/ranking pairs excluded.",
            "evidence_indexes": verify_indexes(root),
            "reference_cohort_audit": reference_table_audit(root),
            "sha256": {path.relative_to(root).as_posix(): digest(path) for path in sorted(paths)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--receipt", default=DEFAULT)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    destination = safe_path(root, args.receipt)
    if args.write_manifest:
        if destination.exists():
            raise FileExistsError("Keep the frozen receipt; use a new path")
        receipt = build(root)
        destination.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        receipt = load(destination)
    print(json.dumps(verify(root, receipt), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
