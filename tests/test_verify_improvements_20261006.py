import hashlib

import pytest

from scripts.verify_improvements_20261006 import safe_path, verify


def test_evidence_verification_detects_byte_change(tmp_path):
    path = tmp_path / "evidence.bin"
    path.write_bytes(b"real output")
    receipt = {"sha256": {"evidence.bin": hashlib.sha256(b"real output").hexdigest()}}
    assert verify(tmp_path, receipt)["verified_files"] == 1
    path.write_bytes(b"altered output")
    with pytest.raises(AssertionError, match="changed evidence"):
        verify(tmp_path, receipt)


def test_receipt_rejects_paths_outside_repository(tmp_path):
    with pytest.raises(ValueError, match="escapes repository"):
        safe_path(tmp_path, "../other.bin")
