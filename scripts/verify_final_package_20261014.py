"""Integrity and schema checks for the additive Task 2/5/6 final evidence.

Scientific/course conclusions are explicit in the assessment; file hashes and
passing software tests are never treated as proof that every hypothesis passed.
This checker preserves both historical integrity manifests unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPT = "results/final_verification_20261014_integrity.json"
ASSESSMENT = "results/final_verification_20261014_assessment.json"
PREFIXES = ("task2_course_verification_20261014_", "task2_coverage_counterfactual_20261014_",
            "task5_full_structure_20261014_", "task5_full_adaptation_20261014_",
            "task6_full_geometry_20261014_", "task6_literature_regions_20261014_")


def digest(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or Path(relative).is_absolute():
        raise ValueError("Evidence paths must stay inside this repository")
    return path


def validate_assessment(value: dict) -> None:
    if value.get("scope") != "hth810/micro-c" or set(value.get("tasks", {})) != {"2", "5", "6"}:
        raise ValueError("Assessment must cover Micro-C Tasks 2/5/6 only")
    for task in value["tasks"].values():
        if not isinstance(task.get("artifact_complete"), bool) or not isinstance(task.get("course_success_supported"), bool):
            raise ValueError("Separate artifact completeness and course success")
        if not task.get("evidence_sources") or not task.get("conclusion"):
            raise ValueError("Every conclusion needs actual experiment sources")
    novelty = value["tasks"]["2"]
    if not isinstance(novelty.get("confirmed_new_types"), int) or novelty["confirmed_new_types"] < 0:
        raise ValueError("Missing actual confirmed-type count")
    if novelty.get("biological_confirmation") is not False:
        raise ValueError("This computational package does not provide a new wet-lab confirmation")


def build(root: Path) -> dict:
    assessment = json.loads(safe_path(root, ASSESSMENT).read_text(encoding="utf-8"))
    validate_assessment(assessment)
    paths = {safe_path(root, ASSESSMENT), root / "scripts/verify_final_package_20261014.py",
             root / "tests/test_verify_final_package_20261014.py"}
    for relative in (".gitattributes", "README.md", "REPORT.md", "METHODS.md", "AUDIT.md", "REPRODUCE.md", "ENVIRONMENT.md", "requirements.txt",
                     "scripts/build_submission_zip.py", "tests/test_submission_zip.py",
                     "scripts/render_final_report_20261014.ps1",
                     "results/final_verification_20261014_audit.json",
                     "results/final_report_20261014_layout_audit.json",
                     "output/report/MicroC_实验报告_最终版_20261014.docx",
                     "output/pdf/MicroC_实验报告_最终版_20261014.pdf"):
        paths.add(safe_path(root, relative))
    support = root / "output/report/support_final_verification_20261014"
    for name in ("report_reference.docx", "report_reference_content.json", "report_patch.json",
                 "report_language_edits.json", "final_source_index.json", "report_edit_receipt_v3.json"):
        paths.add(support / name)
    paths.update((support / "assets").glob("*.png"))
    llm_support = root / "output/report/support_20261014"
    paths.update(p for p in llm_support.iterdir() if p.is_file() and p.suffix in {".json", ".txt", ".py"})
    paths.update((root / "submission/archive/final_verification_20261014").glob("*.md"))
    for directory in ("scripts", "tests"):
        paths.update((root / directory).glob("*20261014*.py"))
    paths.update(p for p in (root / "results").iterdir() if p.is_file() and p.name.startswith(PREFIXES))
    for task in assessment["tasks"].values():
        for relative in task["evidence_sources"]:
            paths.add(safe_path(root, relative))
    for relative in ("results/improvements_20261006_integrity.json", "results/followups_20261006_integrity.json"):
        paths.add(safe_path(root, relative))
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError("An explicitly referenced final evidence file is missing")
    return {"packaged_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_parent_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "scope": "hth810/micro-c", "assessment": ASSESSMENT,
            "meaning": "Byte integrity and explicit conclusion completeness; not a scientific grading oracle",
            "sha256": {p.relative_to(root).as_posix(): digest(p) for p in sorted(paths)}}


def verify(root: Path, manifest: dict) -> dict:
    if manifest.get("scope") != "hth810/micro-c" or not manifest.get("sha256"):
        raise ValueError("Wrong repository or empty integrity package")
    for relative, expected in manifest["sha256"].items():
        path = safe_path(root, relative)
        if not path.is_file() or digest(path) != expected:
            raise AssertionError(f"Final evidence missing or changed: {relative}")
    assessment = json.loads(safe_path(root, manifest["assessment"]).read_text(encoding="utf-8"))
    validate_assessment(assessment)
    return {"verified_files": len(manifest["sha256"]), "scope": "hth810/micro-c",
            "course_success_supported": {key: task["course_success_supported"] for key, task in assessment["tasks"].items()},
            "confirmed_new_types": assessment["tasks"]["2"]["confirmed_new_types"],
            "note": "A supported negative result remains negative; passing integrity does not alter a scientific claim"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--receipt", default=DEFAULT_RECEIPT)
    p.add_argument("--write-manifest", action="store_true")
    args = p.parse_args()
    root = args.root.resolve()
    destination = safe_path(root, args.receipt)
    if args.write_manifest:
        if destination.exists():
            raise FileExistsError("Preserve existing integrity receipt")
        manifest = build(root)
        destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        manifest = json.loads(destination.read_text(encoding="utf-8"))
    print(json.dumps(verify(root, manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
