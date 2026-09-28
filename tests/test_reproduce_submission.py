"""Fast, no-network contract tests for the six-task orchestration script."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts import reproduce_submission as repro


def test_requested_inputs_and_local_cool_preference(tmp_path: Path) -> None:
    assert repro.requested_inputs(5, 5) == ("wt1",)
    assert repro.requested_inputs(1, 6) == tuple(repro.GEO_FILES)
    gz = tmp_path / repro.GEO_FILES["wt1"]
    assert repro.input_path(tmp_path, "wt1") == gz
    cool = gz.with_suffix("")
    cool.write_bytes(b"nonempty")
    assert repro.input_path(tmp_path, "wt1") == cool


def test_all_core_steps_are_ordered_and_have_expected_outputs(tmp_path: Path) -> None:
    files = {name: tmp_path / filename for name, filename in repro.GEO_FILES.items()}
    steps = repro.build_steps(tmp_path, files, "python")
    assert len(steps) == 33
    assert [s.stage for s in steps] == sorted(s.stage for s in steps)
    assert len({s.name for s in steps}) == len(steps)
    assert all(s.command[0] == "python" and s.expected for s in steps)
    assert "--split-manifest" in steps[0].command
    assert str(tmp_path / "results" / "task1_spatial_annotation_manifest.csv") in steps[0].command
    names = [s.name for s in steps]
    assert names.index("task2-novel-cluster") < names.index("task2-transfer-hns")
    assert names.index("task2-hns-check") < names.index("task2-annotation-review")
    assert names.index("task4-rif-20260923") < names.index("task4-seed-sensitivity")
    assert names.index("task5-train-peak") < names.index("task5-blend-check")
    assert names.index("task6-all-positive") < names.index("task6-replicate-validation")


def test_dry_run_never_downloads_or_writes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert repro.main(["--dry-run", "--download", "--data-dir", str(tmp_path),
                       "--from-stage", "5", "--through-stage", "5"]) == 0
    output = capsys.readouterr().out
    assert "DOWNLOAD:" in output
    assert "task5-train-main" in output
    assert "MISSING INPUTS: wt1" in output
    assert list(tmp_path.iterdir()) == []


def test_missing_data_fails_before_any_training(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="2"):
        repro.main(["--data-dir", str(tmp_path), "--from-stage", "5", "--through-stage", "5"])
