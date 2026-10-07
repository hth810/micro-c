"""Delivery corrections must preserve the scientifically empty comparator."""
import numpy as np
import pandas as pd
import pytest

from scripts.task2_artifact_addendum_followup import (
    CONTROL_COLUMNS, factor_diagnostics, schema_control_manifest,
)


def test_empty_control_schema_is_readable_without_mutating_original(tmp_path):
    original = tmp_path / "original.csv"
    original.write_bytes(b"\r\n")
    before = original.read_bytes()
    schema = schema_control_manifest(original)
    assert list(schema.columns) == CONTROL_COLUMNS and schema.empty
    output = tmp_path / "schema.csv"
    schema.to_csv(output, index=False)
    assert pd.read_csv(output).empty
    assert original.read_bytes() == before


def test_nonempty_schema_rejects_unexpected_columns(tmp_path):
    p = tmp_path / "unexpected.csv"
    p.write_text("score\n3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected"):
        schema_control_manifest(p)


def test_posthoc_factor_counts_reproduce_joint_failure_without_selecting_controls():
    rows = []
    for i in range(30):
        rows.append({"center_bp": i * 20000, "role": "independent_unannotated",
                     "known_distance_bp": 5000, "native_symmetric_total": 100+i,
                     "native_upper_offdiag_total": 50+i,
                     "native_zero_offdiag_row_fraction": 0., "native_upper_nonzero_fraction": .3,
                     "native_top_1pct_mass_fraction": .1, "native_offdiag_row_cv": .2})
    rows.append({**rows[10], "center_bp": 1000000, "native_zero_offdiag_row_fraction": .4})
    qc = pd.DataFrame(rows)
    candidates = pd.DataFrame([{"candidate_id": "A2", "center": 1000000}])
    table = factor_diagnostics(qc, candidates)
    assert len(table) == 7
    assert (table.joint_original_caliper_controls == 0).all()
    zero = table[table.covariate == "native_zero_offdiag_row_fraction"].iloc[0]
    assert zero.within_2iqr_controls == 0
    assert zero.pool_q10 == zero.pool_q90 == 0
    assert zero.iqr_scale == 1e-8
