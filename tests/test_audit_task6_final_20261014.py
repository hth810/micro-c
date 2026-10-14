import numpy as np
import pandas as pd
import pytest

from scripts.audit_task6_final_20261014 import validate_geometry
from scripts.task6_full_geometry_verification_20261014 import MODEL_KEYS


def tables():
    keys = list(MODEL_KEYS) + ["exact_genome_circle"]
    macro = pd.DataFrame([{"scope": s, "trim_bp": t, "model": k, "same_domain_near_auc": .5,
                           "standardized_between_minus_within": 0.} for s in ("four_macrodomains", "all_six_regions")
                          for t in (0, 50000) for k in keys])
    domains = pd.DataFrame([{"trim_bp": t, "region": r, "model": k} for t in (0, 50000)
                            for r in ("Ori", "Ter", "Right", "Left") for k in keys])
    null = macro.loc[macro.index.repeat(928), ["scope", "trim_bp", "model"]].reset_index(drop=True)
    null["shift_bins"] = np.tile(np.arange(1, 929), 20)
    return macro, domains, null


def test_complete_schema_does_not_require_a_scientific_effect():
    # Integrity passes even though these synthetic effects are all exactly zero.
    assert validate_geometry(*tables())["no_significance_requirement_in_integrity_audit"] is True


def test_missing_cyclic_shift_rejected():
    a, b, c = tables()
    c.loc[0, "shift_bins"] = 2
    with pytest.raises(AssertionError, match="every nonzero"):
        validate_geometry(a, b, c)


def test_model_substitution_rejected():
    a, b, c = tables()
    a.loc[0, "model"] = "legacy_mlp"
    with pytest.raises(AssertionError, match="omitted"):
        validate_geometry(a, b, c)


def test_circle_false_discrimination_rejected():
    a, b, c = tables()
    a.loc[a.model == "exact_genome_circle", "same_domain_near_auc"] = .7
    with pytest.raises(AssertionError, match="pure-ring"):
        validate_geometry(a, b, c)
