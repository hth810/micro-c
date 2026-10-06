from pathlib import Path

import pytest

from scripts.verify_followups_20261006 import reference_table_audit, verify_indexes
from scripts.verify_improvements_20261006 import safe_path


ROOT = Path(__file__).resolve().parents[1]


def test_followup_indexes_preserve_published_and_source_bytes():
    assert verify_indexes(ROOT) >= 3


def test_reference_followup_covers_all_cohorts_and_lengths():
    audit = reference_table_audit(ROOT)
    assert audit["reference_rows"] == 1917
    assert audit["ambiguous_start_windows"] == 0


def test_receipt_paths_cannot_escape_checkout(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        safe_path(tmp_path, "../outside.txt")


def test_index_rejects_artifact_tampering(tmp_path):
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    (result_dir / "task1_followup_20261006_evidence_index.json").write_text(
        '{"files":{"payload.json":"wrong"}}', encoding="utf-8")
    (result_dir / "payload.json").write_text("{}", encoding="utf-8")
    with pytest.raises(AssertionError, match="checksum"):
        verify_indexes(tmp_path)


@pytest.mark.parametrize("seed", [20261006, 20261007, 20261008])
def test_zero_ranking_weight_exactly_reproduces_old_training(monkeypatch, seed):
    import numpy as np
    import torch
    from scripts import task6_zero_residual_controlled_followup as controlled
    from scripts.task6_circular_sensitivity import train as baseline_train
    from scripts.task6_zero_residual_followup import matched_ranking_pairs

    rng = np.random.default_rng(12)
    upper = np.triu(rng.uniform(.1, 1, (17, 17)) * (rng.random((17, 17)) > .5), 1)
    contact = (upper + upper.T).astype(np.float32)
    matches, _ = matched_ranking_pairs(contact)
    monkeypatch.setattr(controlled, "ALPHA", 0.)
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        old_model, old_xyz, _ = baseline_train(contact, seed, epochs=5, pair_count=37)
        new_model, new_xyz, _ = controlled.train(contact, matches, seed, epochs=5, pair_count=37)
        assert np.array_equal(old_xyz, new_xyz)
        for name, value in old_model.state_dict().items():
            assert torch.equal(value, new_model.state_dict()[name])
    finally:
        torch.set_num_threads(original_threads)
