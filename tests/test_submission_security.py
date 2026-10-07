import hashlib
import json
import zipfile

import pytest

from scripts.build_submission_zip import verify
from scripts.verify_followups_20261006 import verify_indexes


@pytest.mark.parametrize("name", ["../outside", "/absolute", "C:/absolute", "C:relative", "a\\b"])
def test_archive_rejects_cross_platform_unsafe_paths(tmp_path, name):
    payload = b"test"
    digest = hashlib.sha256(payload).hexdigest()
    manifest = {"payload_file_count": 1, "payload_bytes": 4,
                "files": [{"path": name, "bytes": 4, "sha256": digest}]}
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as out:
        out.writestr(name, payload)
        out.writestr("MANIFEST.json", json.dumps(manifest))
        out.writestr("SHA256SUMS.txt", f"{digest}  {name}\n")
    with pytest.raises(ValueError, match="Unsafe ZIP path"):
        verify(archive)


def test_raw_metadata_index_is_verified(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "task2_followup_20261006_raw_read_availability_audit_index.json").write_text(
        json.dumps({"files": {"metadata.json": "incorrect"}}), encoding="utf-8")
    (results / "metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(AssertionError, match="checksum mismatch"):
        verify_indexes(tmp_path)
