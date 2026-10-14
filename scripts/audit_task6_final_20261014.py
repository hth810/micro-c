"""Audit byte provenance and complete Task6 geometry schemas, not significance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.task6_full_geometry_verification_20261014 import ROOT, MODEL_KEYS, dump, sha
from scripts.task6_literature_regions_20261014 import REGIONS, VERSION


def verify_receipt(root: Path, prefix: str, receipt_name: str, *, input_paths=None):
    receipt = json.loads((root / "results" / (prefix + receipt_name)).read_text(encoding="utf-8"))
    checked = []
    for relative, digest in receipt["source_sha256"].items():
        p = (root / relative).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p) != digest:
            raise AssertionError(f"Source missing or changed: {relative}")
        checked.append(p.relative_to(root).as_posix())
    for filename, digest in receipt["outputs_sha256"].items():
        p = root / "results" / (prefix + filename)
        if not p.is_file() or sha(p) != digest:
            raise AssertionError(f"Output missing or changed: {p.name}")
        checked.append(p.relative_to(root).as_posix())
    for key, filename in (input_paths or {}).items():
        p = Path(filename)
        if not p.is_absolute():
            p = root / p
        if sha(p) != receipt["input_sha256"][key]:
            raise AssertionError(f"Input changed: {key}")
        checked.append(str(p))
    return checked


def validate_geometry(macro, domains, null):
    keys = set(MODEL_KEYS) | {"exact_genome_circle"}
    if len(macro) != 20 or len(domains) != 40 or len(null) != 20 * 928:
        raise AssertionError("Incomplete declared scope/trim/model/control outputs")
    for scope in ("four_macrodomains", "all_six_regions"):
        for trim in (0, 50000):
            subset = macro[(macro.scope == scope) & (macro.trim_bp == trim)]
            if set(subset.model) != keys or subset.model.duplicated().any():
                raise AssertionError("A model was omitted or selected after evaluation")
            for model in keys:
                shifts = null[(null.scope == scope) & (null.trim_bp == trim) & (null.model == model)].shift_bins
                if sorted(shifts.tolist()) != list(range(1, 929)):
                    raise AssertionError("Cyclic control must include every nonzero shift exactly once")
    for trim in (0, 50000):
        for region in ("Ori", "Ter", "Right", "Left"):
            subset = domains[(domains.trim_bp == trim) & (domains.region == region)]
            if set(subset.model) != keys or subset.model.duplicated().any():
                raise AssertionError("An individual domain or seed was omitted")
    circle = macro[macro.model == "exact_genome_circle"]
    if not np.allclose(circle.same_domain_near_auc, .5, atol=1e-12) or not np.allclose(circle.standardized_between_minus_within, 0, atol=1e-12):
        raise AssertionError("Exact-offset matching did not remove the pure-ring baseline")
    return {"macro_rows": len(macro), "individual_domain_rows": len(domains), "cyclic_control_rows": len(null),
            "all_seed_and_baseline_coverage": True, "no_significance_requirement_in_integrity_audit": True}


def run(root: Path):
    checked = []
    geometry = {}
    for version, prefix in (("prior_approximate", "task6_full_geometry_20261014_"),
                            (VERSION, "task6_literature_regions_20261014_")):
        protocol = json.loads((root / "results" / (prefix + "protocol.json")).read_text(encoding="utf-8"))
        checked.extend(verify_receipt(root, prefix, "receipt.json", input_paths=protocol["input_paths"]))
        tables = [pd.read_csv(root / "results" / (prefix + name)) for name in
                  ("macrogeometry.csv", "individual_domains.csv", "cyclic_shift_effects.csv")]
        geometry[version] = validate_geometry(*tables)
    prefix = "task6_literature_regions_20261014_"
    picture = json.loads((root / "results" / (prefix + "primary_views_receipt.json")).read_text(encoding="utf-8"))
    checked.extend(verify_receipt(root, prefix, "primary_views_receipt.json", input_paths={
        "bundle": "results/task6_controlled_followup_20261006_coordinate_models.npz",
        "coordinates": "results/task6_controlled_followup_20261006_coordinates.csv",
        "geometry_protocol": "results/task6_literature_regions_20261014_protocol.json",
        "gff": "data/raw/GCF_000005845.2_ASM584v2_genomic.gff.gz"}))
    if picture["model"] != MODEL_KEYS[0] or not picture["equal_axis_and_view_scales"]:
        raise AssertionError("Main figure must retain the first declared seed and equal scales")
    boundaries = pd.read_csv(root / "results" / (prefix + "literature_boundaries.csv"))
    if tuple(boundaries[["region", "start_bp", "end_exclusive_bp"]].itertuples(index=False, name=None)) != REGIONS:
        raise AssertionError("The whole source version changed")
    labels = pd.read_csv(root / "results" / (prefix + "coordinate_labels.csv"))
    if len(labels) != 929 or labels.start.iloc[0] != 0 or labels.end.iloc[-1] != 4641652 or labels.literature_region.iloc[1] != "Ori":
        raise AssertionError("Whole chromosome or circular Ori mapping is inconsistent")
    replication = pd.read_csv(root / "results/task6_full_geometry_20261014_rep2_reverification.csv")
    if set(replication.model) != set(MODEL_KEYS) or replication.max_abs_difference_from_frozen.max() > 1e-10:
        raise AssertionError("Fresh rep2 validation disagrees with original final-model evidence")
    landmarks = pd.read_csv(root / "results/task6_full_geometry_20261014_landmarks.csv")
    order = landmarks[landmarks.check == "Ori4_NSR5_closer_than_Right2"]
    if set(order.model) != set(MODEL_KEYS) or not order.raw_published_order_reproduced.all():
        raise AssertionError("Three-locus statements disagree with measured distances")
    return {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "hth810/micro-c Task6 additive final verification",
            "checked_paths": sorted(set(checked)), "verified_path_count": len(set(checked)), "geometry_versions": geometry,
            "rep2_frozen_max_abs_difference": float(replication.max_abs_difference_from_frozen.max()),
            "full_chromosome_bins": 929, "genome_bp": 4641652, "last_bin_bp": 1652,
            "figure_seed": MODEL_KEYS[0], "literature_region_version": VERSION,
            "old_scientific_freezes_unchanged": True,
            "scientific_interpretation": "Three-seed contact results and positive four-macrodomain trend support course-level global organization. Individual domains remain mixed and adjusted cyclic-shift q is not significant. Not full physical reconstruction.",
            "meaning": "Byte provenance and complete stated comparisons; not a hypothesis significance or grading oracle",
            "audit_source_sha256": sha(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    value = run(args.root.resolve())
    if args.out:
        if args.out.exists():
            raise FileExistsError("Keep the prior audit receipt")
        dump(args.out, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
