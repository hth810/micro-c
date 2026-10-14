"""Deterministic checks of the saved LLM review against five frozen CSVs.

This script reads the review and experiment sources without changing them.
Its only generated file is the requested numeric-consistency receipt. It
checks reported facts, not biological novelty, expert approval, or free-form
interpretation. No scientific p value or FDR is computed.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


TOLERANCE = 1e-9
EXPECTED_SOURCES = {
    "candidate_validation": "results/task2_improve_20261006_individual_candidate_validation.csv",
    "native_qc": "results/task2_followup_20261006_candidate_native_qc.csv",
    "reference_uniqueness": "results/task2_reference_followup_20261006_candidate_reference_diagnostics.csv",
    "normalization_sensitivity": "results/task2_followup_20261006_candidate_normalization_sensitivity.csv",
    "matched_controls": "results/task2_followup_20261006_candidate_matched_descriptive_comparison.csv",
}
REGIMES = ("frozen_input_stress", "rep1_recalibrated_sensitivity")
METRICS = ("highpass", "rowcol")
EXPECTED_CATEGORIES = {
    "A1": "A1 截断敏感待查",
    "A2": "A2 高重复风险优先 QC",
    "A3": "A3 验证失败保留负对照",
    "A4": "A4 稳健异常待查",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_bool(value: str) -> bool:
    if value not in ("True", "False", "true", "false"):
        raise ValueError(f"Invalid source boolean {value!r}")
    return value.lower() == "true"


class Checks:
    def __init__(self) -> None:
        self.groups: Counter[str] = Counter()
        self.failures: list[dict] = []
        self.max_absolute_numeric_error = 0.0

    def equal(self, group: str, field: str, declared, observed) -> None:
        self.groups[group] += 1
        error = None
        try:
            if isinstance(declared, bool):
                actual = as_bool(observed) if isinstance(observed, str) else observed
                matches = isinstance(actual, bool) and declared is actual
            elif isinstance(declared, (int, float)):
                actual = float(observed)
                error = abs(float(declared) - actual)
                matches = math.isfinite(actual) and math.isfinite(float(declared)) and error <= TOLERANCE
                if math.isfinite(error):
                    self.max_absolute_numeric_error = max(self.max_absolute_numeric_error, error)
            else:
                matches = declared == observed
        except (ValueError, TypeError, OverflowError):
            matches = False
        if not matches:
            failure = {"group": group, "field": field, "declared": declared, "source_value": observed}
            if error is not None and math.isfinite(error):
                failure["absolute_error"] = error
            self.failures.append(failure)


def exactly_one(rows: list[dict], **criteria) -> dict:
    matches = [row for row in rows if all(row.get(key) == str(value) for key, value in criteria.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one source row for {criteria}; found {len(matches)}")
    return matches[0]


def selector_values(selector: str) -> dict[str, str]:
    return {key.strip(): value.strip() for key, value in re.findall(r"([^,=]+)=([^,]+)", selector)}


def sensitivity_fields(checks: Checks, tables: dict, candidate_id: str,
                       variant: str, regime: str, replicate: str,
                       fields: dict, metric: str | None = None) -> None:
    for declared_field, value in fields.items():
        field_metric, source_field = metric, declared_field
        match = re.fullmatch(r"(highpass|rowcol)_(.+)", declared_field)
        if match:
            field_metric, source_field = match.groups()
        if field_metric not in METRICS:
            raise ValueError(f"Missing sensitivity metric for {candidate_id}/{declared_field}")
        row = exactly_one(tables["normalization_sensitivity"], candidate_id=candidate_id,
                          variant=variant, model_regime=regime,
                          replicate=replicate, metric=field_metric)
        if source_field not in row:
            raise ValueError(f"Unknown sensitivity field {source_field}")
        context = f"{candidate_id}/{variant}/{regime}/{replicate}/{field_metric}/{source_field}"
        checks.equal("sensitivity", context, value, row[source_field])


def validate(review: dict, tables: dict, source_metadata: dict) -> Checks:
    checks = Checks()
    checks.equal("schema", "record_type", review.get("record_type"), "evidence_bound_llm_candidate_review")
    checks.equal("schema", "source_names", sorted(review["source_tables"]), sorted(EXPECTED_SOURCES))
    checks.equal("metadata", "human_expert_validation.completed", review["human_expert_validation"]["completed"], False)
    checks.equal("metadata", "model_provenance.exact_api_model_snapshot", review["model_provenance"]["exact_api_model_snapshot"], None)
    checks.equal("metadata", "model_provenance.snapshot_disclosed", review["model_provenance"]["snapshot_disclosed"], False)
    for name, path in EXPECTED_SOURCES.items():
        metadata = review["source_tables"][name]
        checks.equal("source_integrity", f"{name}.path", metadata["path"], path)
        checks.equal("source_integrity", f"{name}.rows", metadata["rows"], len(tables[name]))
        checks.equal("source_integrity", f"{name}.sha256", metadata["sha256"], source_metadata[name]["sha256"])
    records = review["records"]
    checks.equal("record_coverage", "record_count", len(records), 4)
    checks.equal("record_coverage", "candidate_ids", sorted(record["candidate_id"] for record in records), sorted(EXPECTED_CATEGORIES))
    status_mapping = {
        "default_rep2_dual_gate_pass": ("default_clip_5", "frozen_input_stress"),
        "unclipped_rep2_dual_gate_pass": ("unclipped", "frozen_input_stress"),
        "tight_clip_frozen_model_rep2_dual_gate_pass": ("explicit_clip_2_4", "frozen_input_stress"),
        "tight_clip_rep1_recalibrated_rep2_dual_gate_pass": ("explicit_clip_2_4", "rep1_recalibrated_sensitivity"),
    }
    for record in records:
        candidate_id = record["candidate_id"]
        candidate = exactly_one(tables["candidate_validation"], candidate_id=candidate_id)
        checks.equal("queue_labels", f"{candidate_id}.manual_followup_category", record["manual_followup_category"], EXPECTED_CATEGORIES[candidate_id])
        for declared_field, source_field in (("chrom", "chrom"), ("center_bp", "center"), ("start_bp", "start_bp"), ("end_bp", "end_bp")):
            checks.equal("coordinates", f"{candidate_id}.{declared_field}", record["coordinates"][declared_field], candidate[source_field])
        for status, (variant, regime) in status_mapping.items():
            dual_pass = all(as_bool(exactly_one(tables["normalization_sensitivity"], candidate_id=candidate_id,
                                              variant=variant, model_regime=regime,
                                              replicate="rep2", metric=metric)["passes_rep1_frozen_cutoff"])
                            for metric in METRICS)
            checks.equal("derived_gate_status", f"{candidate_id}.{status}", record["replication_status"][status], dual_pass)
        checks.equal("default_gate_status", f"{candidate_id}.default_source_gate", record["replication_status"]["default_rep2_dual_gate_pass"], candidate["rep2_passes_both_frozen_novelty_gates"])
        evidence_blocks = record["supporting_evidence"]
        checks.equal("evidence_coverage", f"{candidate_id}.source_names", sorted(block["source"] for block in evidence_blocks), sorted(EXPECTED_SOURCES))
        for block in evidence_blocks:
            source = block["source"]
            selection = selector_values(block["selector"])
            checks.equal("selector_identity", f"{candidate_id}/{source}.selector", selection.get("candidate_id"), candidate_id)
            if source == "candidate_validation":
                for field, value in block["fields"].items():
                    checks.equal("candidate_facts", f"{candidate_id}.{field}", value, candidate[field])
            elif source == "native_qc":
                for replicate, fields in block["fields_by_replicate"].items():
                    row = exactly_one(tables[source], candidate_id=candidate_id, replicate=replicate)
                    checks.equal("source_coordinate_alignment", f"{candidate_id}/{source}/{replicate}.center", record["coordinates"]["center_bp"], row["center_bp"])
                    for field, value in fields.items():
                        checks.equal("native_qc", f"{candidate_id}/{replicate}/{field}", value, row[field])
            elif source == "reference_uniqueness":
                for k, value in block["unique_fraction_by_k"].items():
                    row = exactly_one(tables[source], candidate_id=candidate_id, k=k)
                    checks.equal("source_coordinate_alignment", f"{candidate_id}/{source}/k{k}.center", record["coordinates"]["center_bp"], row["center_bp"])
                    checks.equal("reference_uniqueness", f"{candidate_id}/k{k}/unique_fraction", value, row["unique_fraction"])
                    checks.equal("reference_uniqueness", f"{candidate_id}/k{k}/background_median_unique_fraction", block["background_median_unique_fraction"], row["background_median_unique_fraction"])
            elif source == "matched_controls":
                rows = [row for row in tables[source] if row["candidate_id"] == candidate_id]
                checks.equal("control_coverage", f"{candidate_id}.matched_combinations", len(rows), 12)
                for row in rows:
                    context = "/".join(row[key] for key in ("variant", "model_regime", "metric"))
                    for field, value in block["fields"].items():
                        checks.equal("matched_controls", f"{candidate_id}/{context}/{field}", value, row[field])
            elif source == "normalization_sensitivity":
                if "fields_by_model_regime" in block:
                    for regime, fields in block["fields_by_model_regime"].items():
                        sensitivity_fields(checks, tables, candidate_id, selection["variant"], regime,
                                           selection["replicate"], fields, selection.get("metric"))
                elif "fields_by_model_regime_and_replicate" in block:
                    for context, fields in block["fields_by_model_regime_and_replicate"].items():
                        regime, replicate = context.rsplit("_", 1)
                        if regime not in REGIMES or replicate not in ("rep1", "rep2"):
                            raise ValueError(f"Unknown sensitivity context {context}")
                        sensitivity_fields(checks, tables, candidate_id, selection["variant"], regime,
                                           replicate, fields, selection.get("metric"))
                elif "fields_by_variant_and_model_regime" in block:
                    for context, fields in block["fields_by_variant_and_model_regime"].items():
                        regime = next((regime for regime in REGIMES if context.endswith("_" + regime)), None)
                        if regime is None:
                            raise ValueError(f"Unknown sensitivity context {context}")
                        variant = context[:-(len(regime) + 1)]
                        sensitivity_fields(checks, tables, candidate_id, variant, regime,
                                           selection["replicate"], fields, selection.get("metric"))
                else:
                    raise ValueError(f"Unknown sensitivity evidence schema for {candidate_id}")
            else:
                raise ValueError(f"Unknown source table {source}")
    return checks


def regression_checks(review: dict, tables: dict, metadata: dict) -> list[dict]:
    """In-memory negative cases; the saved response and CSVs are untouched."""
    cases = [
        ("coordinate_mismatch", ("coordinates", "center_bp"), 729601),
        ("default_gate_mismatch", ("replication_status", "default_rep2_dual_gate_pass"), False),
        ("qc_mismatch", ("supporting_evidence", 2, "fields_by_replicate", "rep1", "normalization_mad"), 0.001),
        ("uniqueness_mismatch", ("supporting_evidence", 3, "unique_fraction_by_k", "31"), 0.5),
        ("sensitivity_mismatch", ("supporting_evidence", 1, "fields_by_model_regime", "frozen_input_stress", "novelty_ratio"), 0.8),
        ("controls_mismatch", ("supporting_evidence", 4, "fields", "controls"), 1),
    ]
    output = []
    for name, path, value in cases:
        altered = copy.deepcopy(review)
        node = altered["records"][0]
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        rejected = bool(validate(altered, tables, metadata).failures)
        output.append({"case": name, "in_memory_only": True, "expected_rejected": True, "actual_rejected": rejected})
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    support = repo / "output/report/support_20261014"
    review_path = (args.review or support / "llm_review_response.json").resolve()
    receipt_path = (args.out or support / "llm_validation_receipt.json").resolve()
    source_paths = {name: (repo / path).resolve() for name, path in EXPECTED_SOURCES.items()}
    protected_paths = {review_path, Path(__file__).resolve(), *source_paths.values()}
    if receipt_path in protected_paths:
        parser.error("Receipt output must not overwrite review, validator, or frozen CSVs")
    before_hashes = {name: sha256(path) for name, path in source_paths.items()}
    review_hash = sha256(review_path)
    receipt = {
        "record_type": "llm_review_numeric_consistency_receipt",
        "validator_sha256": sha256(Path(__file__)),
        "review_sha256": review_hash,
        "absolute_numeric_tolerance": TOLERANCE,
        "deterministic": True,
        "scope": "CSV fact, identity, version, and gate checks only; not expert approval or biological validation",
        "scientific_p_values_generated": False,
        "source_tables": {},
    }
    try:
        review = json.loads(review_path.read_text(encoding="utf-8-sig"))
        tables = {}
        for name, path in source_paths.items():
            with path.open(encoding="utf-8-sig", newline="") as source:
                tables[name] = list(csv.DictReader(source))
            receipt["source_tables"][name] = {"path": EXPECTED_SOURCES[name], "rows": len(tables[name]), "sha256": before_hashes[name]}
        checks = validate(review, tables, receipt["source_tables"])
        self_tests = regression_checks(review, tables, receipt["source_tables"])
        receipt.update({"checks": sum(checks.groups.values()), "checks_by_group": dict(sorted(checks.groups.items())),
                        "max_absolute_numeric_error": checks.max_absolute_numeric_error,
                        "failures": checks.failures, "in_memory_negative_cases": self_tests})
    except (ValueError, KeyError, TypeError, OSError, IndexError) as exc:
        receipt.update({"checks": 0, "checks_by_group": {}, "failures": [{"execution_error": str(exc)}],
                        "in_memory_negative_cases": []})
    after_hashes = {name: sha256(path) for name, path in source_paths.items()}
    receipt["frozen_sources_unchanged"] = before_hashes == after_hashes
    receipt["saved_review_unchanged"] = review_hash == sha256(review_path)
    receipt["status"] = "passed" if (not receipt["failures"] and receipt["frozen_sources_unchanged"]
                                               and receipt["saved_review_unchanged"]
                                               and all(case["actual_rejected"] for case in receipt["in_memory_negative_cases"])) else "failed"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "checks": receipt["checks"], "failures": receipt["failures"],
                      "receipt": str(receipt_path), "receipt_sha256": sha256(receipt_path)}, ensure_ascii=False, indent=2))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
