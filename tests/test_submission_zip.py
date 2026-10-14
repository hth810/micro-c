import zipfile
from pathlib import Path

from scripts.build_submission_zip import build, verify


def test_submission_archive_is_compact_and_verifiable(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "micro-c-code.zip"
    expected = build(root, output)
    actual = verify(output)
    assert actual["files"] == expected["files"]
    assert actual["payload_file_count"] > 100
    # The additive follow-up includes 75 more pinned files, including frozen
    # sensitivity models and both objective-control checkpoints. Keep these
    # auditable artifacts instead of discarding evidence to meet an old cap.
    # Final full-interval SR weights, morphology figures, coverage stress tests
    # and actual final-coordinate geometry are additional compact evidence.
    # Keep those outputs; the 777 MB native predictions remain excluded.
    assert output.stat().st_size < 32_000_000
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
    assert "MANIFEST.json" in names and "SHA256SUMS.txt" in names
    assert "data/structures.csv" in names
    assert "REPRODUCE.md" in names
    assert "submission/PRESENTATION_OUTLINE.md" in names
    assert "submission/GROUP_CONTRIBUTIONS_TEMPLATE.md" in names
    assert "scripts/build_submission_zip.py" in names
    assert "results/task2_annotation_review.html" in names
    assert "results/task5_reproducibility_audit.json" in names
    assert "results/task1_boundary_final.joblib" in names
    assert "results/task6_genome_3d.pt" in names
    assert not any(name.startswith(("data/raw/", "outputs/", ".git/")) for name in names)
    assert not any(name.endswith((".cool", ".cool.gz", ".pdf", ".zip")) for name in names)
    assert {name for name in names if name.endswith(".npz")} == {
        "results/task6_circular_20261006_coordinate_models.npz",
        "results/task2_followup_20261006_rep1_sensitivity_models.npz",
        "results/task6_followup_20261006_coordinate_models.npz",
        "results/task6_controlled_followup_20261006_coordinate_models.npz"}
    import json
    with zipfile.ZipFile(output) as archive:
        receipt = json.loads(archive.read("results/improvements_20261006_integrity.json"))
    assert set(receipt["sha256"]).issubset(names)
    with zipfile.ZipFile(output) as archive:
        followup = json.loads(archive.read("results/followups_20261006_integrity.json"))
    assert set(followup["sha256"]).issubset(names)
    assert "results/task2_reference_followup_20261006_reference_uniqueness.png" in names
