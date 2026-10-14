"""Run additive final verification in isolated output/result directories.

Requires the WT COOLs and the previous area-discovery and trained-model inputs
documented in REPRODUCE.md. This is the final Task 2/5/6 evidence extension, not
a replacement of the six-task core data preparation pipeline. Negative checks
are valid outputs and are never silently relabelled as positive conclusions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def steps(root: Path, output: Path, tasks: set[str], threads: int = 2):
    published = output / "published_results"
    values = []
    def add(name, module, args):
        values.append({"name": name, "argv": [sys.executable, "-m", module, *map(str, args)]})
    if "2" in tasks:
        base, qc = output / "task2", output / "task2_qc"
        add("task2-score-free-background-and-sensitivity", "scripts.task2_course_verification_20261014",
            ["--annotations", root / "data/structures.csv", "--out", base, "--results", published])
        add("task2-coverage-counterfactual-and-common-depth", "scripts.task2_course_qc_counterfactual_20261014",
            ["--base", base, "--out", qc, "--results", published])
        add("task2-fixed-classifier-context", "scripts.task2_course_classifier_context_20261014",
            ["--out", output / "task2_classifier", "--results", published])
    if "5" in tasks:
        scale, neural = output / "task5_scale", output / "task5_neural"
        add("task5-train-only-full-window-scale-fitting", "scripts.task5_full_structure_repair_20261014",
            ["--phase", "select", "--out", scale, "--threads", threads])
        for phase in ("select", "test"):
            add("task5-neural-" + phase, "scripts.task5_full_structure_train_20261014",
                ["--phase", phase, "--calibration", scale / "trained_scale_parameters.json",
                 "--out", neural, "--results-dir", published, "--threads", threads])
        add("task5-full-case-export-and-recalculation", "scripts.task5_full_structure_export_20261014",
            ["--run", neural, "--calibration", scale / "trained_scale_parameters.json",
             "--results-dir", published])
    if "6" in tasks:
        add("task6-final-coordinate-geometry-and-rep2", "scripts.task6_full_geometry_verification_20261014",
            ["--out", output / "task6"])
        add("task6-sourced-circular-region-definition", "scripts.task6_literature_regions_20261014",
            ["--prior-macro", output / "task6/macrogeometry.csv",
             "--prior-domains", output / "task6/individual_domains.csv",
             "--out", output / "task6_literature"])
        add("task6-first-declared-seed-two-views", "scripts.task6_primary_views_20261014",
            ["--geometry-protocol", output / "task6_literature/protocol.json",
             "--out", output / "task6_views"])
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=("all", "2", "5", "6"), default="all")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    output = args.out.resolve()
    if output.exists() or not output.is_relative_to(ROOT):
        raise ValueError("Choose a new output directory inside the Micro-C repository")
    if not 1 <= args.threads <= 8:
        raise ValueError("Use a bounded CPU thread budget")
    plan = steps(ROOT, output, {"2", "5", "6"} if args.task == "all" else {args.task}, args.threads)
    print(json.dumps({"scope": "hth810/micro-c", "dry_run": args.dry_run,
                      "out": str(output), "steps": plan}, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    output.mkdir(parents=True)
    for step in plan:
        subprocess.run(step["argv"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
