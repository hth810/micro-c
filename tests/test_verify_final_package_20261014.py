import json

import pytest

from scripts.verify_final_package_20261014 import digest, safe_path, validate_assessment, verify


def assessment():
    tasks = {key: {"artifact_complete": True, "course_success_supported": True,
                   "evidence_sources": ["results/evidence.json"], "conclusion": "Measured evidence"}
             for key in ("2", "5", "6")}
    tasks["2"].update(confirmed_new_types=0, biological_confirmation=False)
    return {"scope": "hth810/micro-c", "tasks": tasks}


def test_negative_type_count_does_not_fail_integrity(tmp_path):
    folder = tmp_path / "results"
    folder.mkdir()
    path = folder / "assessment.json"
    path.write_text(json.dumps(assessment()), encoding="utf-8")
    manifest = {"scope": "hth810/micro-c", "assessment": "results/assessment.json",
                "sha256": {"results/assessment.json": digest(path)}}
    result = verify(tmp_path, manifest)
    assert result["confirmed_new_types"] == 0
    assert result["course_success_supported"]["2"] is True
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(AssertionError):
        verify(tmp_path, manifest)


def test_failures_remain_failures_and_provenance_required():
    value = assessment()
    value["tasks"]["5"]["course_success_supported"] = False
    validate_assessment(value)
    value["tasks"]["6"]["evidence_sources"] = []
    with pytest.raises(ValueError, match="sources"):
        validate_assessment(value)
    value = assessment()
    value["tasks"]["2"]["biological_confirmation"] = True
    with pytest.raises(ValueError, match="wet-lab"):
        validate_assessment(value)


def test_paths_and_repository_are_not_interchangeable(tmp_path):
    with pytest.raises(ValueError):
        safe_path(tmp_path, "../outside.json")
    value = assessment()
    value["scope"] = "GvpA/ESM-2"
    with pytest.raises(ValueError, match="Micro-C"):
        validate_assessment(value)
