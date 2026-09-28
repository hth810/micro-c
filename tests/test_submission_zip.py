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
    assert output.stat().st_size < 15_000_000
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
    assert not any(name.endswith((".cool", ".cool.gz", ".npz", ".pdf", ".zip")) for name in names)
