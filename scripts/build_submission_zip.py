"""Build and verify a compact, auditable course-code submission ZIP.

The archive intentionally omits raw contact matrices, complete generated
tracks, large intermediate arrays, local environments, and other files that
are unnecessary to inspect or rerun the source code. Raw GEO checksums remain
available in results/provenance.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = (
    "README.md",
    "REPORT.md",
    "METHODS.md",
    "AUDIT.md",
    "ENVIRONMENT.md",
    "REPRODUCE.md",
    "requirements.txt",
    "data/structures.csv",
    "submission/PRESENTATION_OUTLINE.md",
    "submission/GROUP_CONTRIBUTIONS_TEMPLATE.md",
)
SOURCE_DIRS = ("microc", "scripts", "task1", "tests")
SOURCE_SUFFIXES = {".py", ".md", ".txt"}
RESULT_SUFFIXES = {".csv", ".json"}
RESULT_PNGS = {
    "task1_spatial_confusion_matrix.png",
    "task1_spatial_saliency_CHID.png",
    "task1_spatial_saliency_CHIN.png",
    "task1_spatial_saliency_OPCID.png",
    "task2_novel_cluster_N1_3307520_rep1.png",
    "task2_novel_cluster_N1_3307520_rep2.png",
    "task2_novel_cluster_N1_4241920_rep1.png",
    "task2_novel_cluster_N1_4241920_rep2.png",
    "task3_example_005.png",
    "task3_example_250.png",
    "task4_CHIN_94_wt.png",
    "task4_CHIN_94_heat_44c.png",
    "task5_metric_comparison.png",
    "task5_known_CHID.png",
    "task5_ensemble_validation_tradeoff.png",
    "task6_all_positive_chromosome_3d.png",
    "task6_all_positive_contact_reconstruction.png",
    "task6_domain_overlay_chromosome_3d_macroregions.png",
}
RESULT_HTMLS = {"task2_annotation_review.html"}
RESULT_MODELS = {
    "task1_spatial_task1_cnn.pt",
    "task1_val_selected_cnn_selected.pt",
    "task1_groupnorm_final.pt",
    "task1_handcrafted_final.joblib",
    "task1_boundary_final.joblib",
    "task2_autoencoder.pt",
    "task5_super_resolution.pt",
    "task5_peak_weighted_super_resolution.pt",
    "task6_all_positive_genome_3d.pt",
    "task6_genome_3d.pt",
}
REQUIRED_RESULTS = {
    "README.md",
    "quality_audit.json",
    "provenance.json",
    "task1_spatial_metrics.json",
    "task1_spatial_saliency_audit_summary.json",
    "task1_val_selected_cnn_summary.json",
    "task2_novel_cluster_summary.json",
    "task2_novel_cluster_putative_novel_clusters.csv",
    "task2_all_pair_transfer_summary.json",
    "task3_track_manifest.csv",
    "task4_order_invariant_sensitivity.json",
    "task4_heat_stable_changes.csv",
    "task5_ensemble_summary.json",
    "task5_reproducibility_audit.json",
    "task5_spatial_blend_summary.json",
    "task6_all_positive_replicate_validation.json",
    "task6_domain_overlay_summary.json",
    "task6_macroregion_validation.json",
}
MANIFEST_NAME = "MANIFEST.json"
CHECKSUM_NAME = "SHA256SUMS.txt"
FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)


def _git_value(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=root, text=True, capture_output=True,
            check=True, timeout=10,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _inside_root(root: Path, path: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def select_files(root: Path) -> list[Path]:
    paths: set[Path] = set()
    for name in ROOT_FILES:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Required submission source missing: {path}")
        paths.add(path)

    for dirname in SOURCE_DIRS:
        directory = root / dirname
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing source directory: {directory}")
        for path in directory.iterdir():
            if path.is_file() and not path.is_symlink() and path.suffix in SOURCE_SUFFIXES:
                paths.add(path)

    results = root / "results"
    if not results.is_dir():
        raise FileNotFoundError(f"Missing result directory: {results}")
    for name in REQUIRED_RESULTS | RESULT_PNGS | RESULT_MODELS | RESULT_HTMLS:
        path = results / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Required result artifact missing: {path}")
    for path in results.iterdir():
        if (path.is_file() and not path.is_symlink() and
                (path.suffix in RESULT_SUFFIXES or path.name in RESULT_PNGS or
                 path.name in RESULT_HTMLS or
                 path.name in RESULT_MODELS or path.name == "README.md")):
            paths.add(path)

    for path in paths:
        if not _inside_root(root, path):
            raise ValueError(f"Submission path escapes repository: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Empty submission file: {path}")
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def _zipinfo(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def build(root: Path, output: Path, dry_run: bool = False) -> dict:
    paths = select_files(root)
    if _inside_root(root, output) and output in paths:
        raise ValueError("Output ZIP must not be one of its own payload files")
    payload = []
    entries = []
    for path in paths:
        name = path.relative_to(root).as_posix()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        payload.append({"path": name, "bytes": len(data), "sha256": digest})
        entries.append((name, data))
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch_at_build": _git_value(root, "branch", "--show-current"),
        "git_commit_at_build": _git_value(root, "rev-parse", "HEAD"),
        "scope": "Course code, compact review evidence, presentation outline, and unfilled group-contribution template; no raw matrices or generated track corpus",
        "raw_input_source": "NCBI GEO GSE272159; original-file hashes in results/provenance.json",
        "payload_file_count": len(payload),
        "payload_bytes": sum(record["bytes"] for record in payload),
        "files": payload,
    }
    if dry_run:
        return manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing ZIP: {output}")
    checksum_lines = [f'{record["sha256"]}  {record["path"]}' for record in payload]
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9, allowZip64=True) as archive:
        for name, data in entries:
            archive.writestr(_zipinfo(name), data, compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=9)
        archive.writestr(_zipinfo(MANIFEST_NAME),
                         (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        archive.writestr(_zipinfo(CHECKSUM_NAME),
                         ("\n".join(checksum_lines) + "\n").encode("utf-8"))
    return manifest


def verify(path: Path) -> dict:
    with zipfile.ZipFile(path, "r") as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC verification failed")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate path in ZIP")
        if MANIFEST_NAME not in names or CHECKSUM_NAME not in names:
            raise ValueError("ZIP lacks manifest or checksum list")
        manifest = json.loads(archive.read(MANIFEST_NAME))
        records = manifest["files"]
        expected = {record["path"] for record in records}
        if len(expected) != len(records) or set(names) != expected | {MANIFEST_NAME, CHECKSUM_NAME}:
            raise ValueError("Manifest and ZIP entries disagree")
        if manifest["payload_file_count"] != len(records):
            raise ValueError("Manifest count mismatch")
        if manifest["payload_bytes"] != sum(record["bytes"] for record in records):
            raise ValueError("Manifest byte total mismatch")
        checksum_lines = []
        for record in records:
            name = record["path"]
            if Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name:
                raise ValueError(f"Unsafe ZIP path: {name}")
            data = archive.read(name)
            if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError(f"Payload checksum mismatch: {name}")
            checksum_lines.append(f'{record["sha256"]}  {name}')
        if archive.read(CHECKSUM_NAME).decode("utf-8") != "\n".join(checksum_lines) + "\n":
            raise ValueError("SHA256SUMS does not match manifest")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path, help="Verify an existing ZIP instead of building")
    parser.add_argument("--dry-run", action="store_true", help="List payload size without writing")
    args = parser.parse_args()
    if args.verify is not None:
        manifest = verify(args.verify)
        print(f'Verified {len(manifest["files"])} payload files: {args.verify}')
        return
    root = args.root.resolve()
    output = (args.output or root / "submission" / "micro-c-code.zip").resolve()
    manifest = build(root, output, dry_run=args.dry_run)
    action = "Selected" if args.dry_run else "Built"
    print(f'{action} {manifest["payload_file_count"]} payload files, '
          f'{manifest["payload_bytes"]:,} uncompressed bytes: {output}')
    if not args.dry_run:
        verify(output)
        print(f'Verified SHA-256 manifest and ZIP CRC; archive bytes: {output.stat().st_size:,}')


if __name__ == "__main__":
    main()
