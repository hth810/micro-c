"""Rebuild the six core Micro-C tasks from source data, in dependency order.

The script itself uses only the Python standard library. Install
``requirements.txt`` before starting a real run. Outputs are written under
``outputs/`` (plus the Task 3 manifest in ``results/``); existing files are
not deleted. The default full run is computationally expensive.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GEO_FILES = {
    "wt1": "GSE272159_37C_rep1.mapq_30.10.cool.gz",
    "wt2": "GSE272159_37C_rep2.mapq_30.10.cool.gz",
    "heat": "GSE272159_44C_rep1.mapq_30.10.cool.gz",
    "heat2": "GSE272159_44C_rep2.mapq_30.10.cool.gz",
    "rif": "GSE272159_rif25_rep1.MG1655.mapq_30.10.cool.gz",
    "rif2": "GSE272159_rif25_rep2.MG1655.mapq_30.10.cool.gz",
    "hns1": "GSM9678478_HNS_mut_rep1.MG1655.mapq_30.10.cool.gz",
    "hns2": "GSM9678479_HNS_mut_rep2.MG1655.mapq_30.10.cool.gz",
    "gff": "GCF_000005845.2_ASM584v2_genomic.gff.gz",
}
STAGE_INPUTS = {
    1: ("wt1", "wt2"),
    2: ("wt1", "wt2", "heat", "heat2", "rif", "rif2", "hns1", "hns2"),
    3: ("wt1", "heat", "rif", "gff"),
    4: ("wt1", "wt2", "heat", "heat2", "rif", "rif2"),
    5: ("wt1",),
    6: ("wt1", "wt2"),
}


@dataclass(frozen=True)
class Step:
    stage: int
    name: str
    command: tuple[str, ...]
    expected: tuple[Path, ...]


def requested_inputs(first: int, last: int) -> tuple[str, ...]:
    wanted = {name for stage in range(first, last + 1) for name in STAGE_INPUTS[stage]}
    return tuple(name for name in GEO_FILES if name in wanted)


def input_path(data_dir: Path, name: str) -> Path:
    """Prefer a locally decompressed file, otherwise the official .gz name."""
    compressed = data_dir / GEO_FILES[name]
    if name != "gff":
        plain = compressed.with_suffix("")
        if plain.is_file() and plain.stat().st_size > 0:
            return plain
    elif compressed.with_suffix("").is_file():
        plain = compressed.with_suffix("")
        if plain.stat().st_size > 0:
            return plain
    return compressed


def build_steps(root: Path, files: dict[str, Path], python: str) -> list[Step]:
    out = root / "outputs"
    results = root / "results"
    ann = root / "data" / "structures.csv"
    steps: list[Step] = []

    def module(stage: int, name: str, mod: str, args: list[object], *expected: Path) -> None:
        steps.append(Step(stage, name,
                          (python, "-m", mod, *(str(a) for a in args)), expected))

    def script(stage: int, name: str, source: str, args: list[object], *expected: Path) -> None:
        steps.append(Step(stage, name,
                          (python, str(root / source), *(str(a) for a in args)), expected))

    wt1, wt2 = files["wt1"], files["wt2"]
    heat1, heat2 = files["heat"], files["heat2"]
    rif1, rif2 = files["rif"], files["rif2"]
    hns1, hns2 = files["hns1"], files["hns2"]
    task1 = out / "task1_spatial"
    script(1, "task1-cnn", "task1/run_task1.py",
           ["--rep1", wt1, "--rep2", wt2, "--annotations", ann,
            "--split-manifest", results / "task1_spatial_annotation_manifest.csv",
            "--out-dir", task1], task1 / "metrics.json", task1 / "saliency_audit_summary.json")

    strict = out / "task2_strict_current"
    revised = out / "task2_strict_revised"
    candidate_csv = out / "task2_strict_individual.csv"
    loci_csv = out / "task2_strict_individual_loci.csv"
    candidate_common = ["--rep1", wt1, "--rep2", wt2, "--annotations", ann,
                        "--top-fraction", 0.2, "--min-replicate-r", 0.3]
    module(2, "task2-strict-cluster-scan", "microc.analysis",
           ["discover", *candidate_common, "--out", strict],
           strict / "all_candidates.csv", strict / "summary.json")
    module(2, "task2-strict-individual-scan", "microc.analysis",
           ["discover", *candidate_common, "--candidate-rule", "individual", "--out", revised],
           revised / "new_candidates.csv", revised / "all_windows.csv")
    module(2, "task2-merge-candidates", "scripts.prepare_individual_candidates",
           ["--discovery", strict, "--out", candidate_csv], candidate_csv, loci_csv)
    module(2, "task2-wt-validation", "scripts.validate_discovery",
           ["--rep1", wt1, "--rep2", wt2, "--annotations", ann,
            "--discovery", strict, "--candidate-file", candidate_csv,
            "--out", out / "task2_strict_individual_validation"],
           out / "task2_strict_individual_validation" / "summary.json")
    conditions = out / "task2_strict_individual_conditions.csv"
    module(2, "task2-condition-validation", "scripts.validate_candidates_conditions",
           ["--candidates", candidate_csv, "--heat1", heat1, "--heat2", heat2,
            "--rif1", rif1, "--rif2", rif2, "--out", conditions], conditions)
    evidence = out / "task2_strict_locus_evidence.csv"
    module(2, "task2-locus-evidence", "scripts.summarize_strict_candidates",
           ["--loci", loci_csv, "--windows", candidate_csv, "--conditions", conditions,
            "--wt-null", out / "task2_strict_individual_validation" / "candidate_shift_null.csv",
            "--out", evidence], evidence)
    module(2, "task2-candidate-images", "scripts.render_individual_candidates",
           ["--loci", loci_csv, "--rep1", wt1, "--rep2", wt2,
            "--out", out / "task2_strict_individual_heatmaps"],
           out / "task2_strict_individual_heatmaps")
    novel = out / "task2_novel_cluster"
    module(2, "task2-novel-cluster", "scripts.validate_novel_cluster",
           ["--rep1", wt1, "--rep2", wt2, "--loci", loci_csv,
            "--candidates", revised / "new_candidates.csv", "--scan", revised / "all_windows.csv",
            "--annotations", ann, "--out", novel], novel / "summary.json")
    for name, first, second in (("44c", heat1, heat2), ("rif25", rif1, rif2),
                                ("hns", hns1, hns2)):
        target = novel / f"transfer_{name}.json"
        module(2, f"task2-transfer-{name}", "scripts.validate_novel_condition_transfer",
               ["--rep1", first, "--rep2", second, "--name", name,
                "--members", novel / "putative_novel_clusters.csv",
                "--scan", revised / "all_windows.csv", "--loci", loci_csv,
                "--out", target], target)
    module(2, "task2-hns-check", "scripts.validate_hns_candidates",
           ["--wt1", wt1, "--wt2", wt2, "--hns1", hns1, "--hns2", hns2,
            "--loci", loci_csv, "--scan", revised / "all_windows.csv",
            "--annotations", ann, "--out", out / "task2_hns_validation"],
           out / "task2_hns_validation" / "summary.json")
    module(2, "task2-annotation-review", "scripts.build_annotation_review",
           ["--evidence", evidence,
            "--images", out / "task2_strict_individual_heatmaps",
            "--checkpoint", task1 / "task1_cnn.pt",
            "--metrics", task1 / "metrics.json",
            "--rep1", wt1, "--rep2", wt2,
            "--suggestions", out / "task2_annotation_review_model_suggestions.csv",
            "--out", out / "task2_annotation_review.html"],
           out / "task2_annotation_review_model_suggestions.csv",
           out / "task2_annotation_review.html")

    tracks = out / "task3_labeled"
    module(3, "task3-genome-tracks", "microc.analysis",
           ["tracks", "--cool", wt1, heat1, rif1, "--labels", "WT_37C", "Heat_44C", "Rif_25",
            "--annotations", ann, "--gff", files["gff"], "--out", tracks],
           tracks / "NC_000913.3_0000000.png")
    module(3, "task3-track-manifest", "scripts.manifest_tracks",
           ["--tracks", tracks, "--out", results / "task3_track_manifest.csv"],
           results / "task3_track_manifest.csv")

    for name, first, second, label in (("heat", heat1, heat2, "heat_44c"),
                                       ("rif", rif1, rif2, "rifampicin_25")):
        for seed in (20260921, 20260922, 20260923):
            suffix = "" if seed == 20260921 else f"_seed{seed}"
            dest = out / f"task4_{name}_order_invariant{suffix}"
            module(4, f"task4-{name}-{seed}", "microc.analysis",
                   ["differential", "--wt", wt1, "--wt2", wt2,
                    "--condition", first, "--condition2", second,
                    "--name", label, "--annotations", ann,
                    "--seed", seed, "--out", dest], dest / "reproducible_changes.csv")
    module(4, "task4-wt-control", "microc.analysis",
           ["differential", "--wt", wt1, "--condition", wt2,
            "--name", "wt_rep2_control", "--annotations", ann,
            "--out", out / "task4_repeat_depth_final"],
           out / "task4_repeat_depth_final" / "structure_changes.csv")
    module(4, "task4-seed-sensitivity", "scripts.summarize_task4_thinning",
           ["--root", root, "--out", out / "task4_order_invariant_sensitivity.json"],
           out / "task4_order_invariant_sensitivity.json",
           out / "task4_heat_stable_changes.csv", out / "task4_rif_stable_changes.csv")

    sr = out / "task5_corrected"
    peak = out / "task5_peak_weighted"
    for name, dest, extra in (("main", sr, []), ("peak", peak, ["--peak-weight", 0.5])):
        module(5, f"task5-train-{name}", "microc.deep",
               ["superresolve", "--cool", wt1, "--annotations", ann,
                "--factor", 4, "--epochs", 12, *extra, "--out", dest],
               dest / "super_resolution.pt", dest / "known_structure_recovery.csv")
        module(5, f"task5-known-{name}", "scripts.validate_sr_known_shape",
               ["--cool", wt1, "--checkpoint", dest / "super_resolution.pt",
                "--known-file", dest / "known_structure_recovery.csv",
                "--out", out / ("task5_known_shape" if name == "main" else "task5_peak_known_shape")],
               out / ("task5_known_shape" if name == "main" else "task5_peak_known_shape") / "summary.json")
    module(5, "task5-blend-check", "scripts.evaluate_sr_ensemble",
           ["--cool", wt1, "--annotations", ann,
            "--main-checkpoint", sr / "super_resolution.pt",
            "--peak-checkpoint", peak / "super_resolution.pt",
            "--main-patches", sr / "reconstructed_test_patches.npz",
            "--peak-patches", peak / "reconstructed_test_patches.npz",
            "--out", out / "task5_ensemble"], out / "task5_ensemble" / "summary.json")

    independent = out / "task6_independent"
    positive = out / "task6_all_positive"
    for name, dest, extra in (("top20", independent, []),
                              ("all-positive", positive, ["--contact-quantile", 0])):
        module(6, f"task6-{name}", "microc.deep",
               ["reconstruct3d", "--cool", wt1, "--annotations", ann,
                "--pretrain-epochs", 0, "--anchor-weight", 0,
                *extra, "--out", dest],
               dest / "coordinates.csv", dest / "contact_and_baseline.npz")
    module(6, "task6-replicate-validation", "scripts.validate_3d_replicate",
           ["--rep2", wt2, "--original", independent / "contact_and_baseline.npz",
            "--all-positive", positive / "contact_and_baseline.npz",
            "--out", positive / "replicate_validation.json"],
           positive / "replicate_validation.json")
    module(6, "task6-domain-overlay", "scripts.annotate_3d_domains",
           ["--coordinates", positive / "coordinates.csv", "--annotations", ann,
            "--out", out / "task6_domain_overlay"],
           out / "task6_domain_overlay" / "summary.json",
           out / "task6_domain_overlay" / "chromosome_3d_macroregions.png")
    return steps


def format_command(command: tuple[str, ...]) -> str:
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def main(argv: list[str] | None = None) -> int:
    if os.name == "nt" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw",
                    help="directory containing GEO .cool.gz/.cool files and genome GFF")
    ap.add_argument("--download", action="store_true",
                    help="verify/download needed official GEO data before running")
    ap.add_argument("--workers", type=int, default=1,
                    help="range-download workers per GEO file when --download is used")
    ap.add_argument("--dry-run", action="store_true", help="print every command; do not write or train")
    ap.add_argument("--from-stage", type=int, choices=range(1, 7), default=1)
    ap.add_argument("--through-stage", type=int, choices=range(1, 7), default=6)
    ap.add_argument("--start-at", help="resume at a named step shown by --dry-run; prior outputs must exist")
    args = ap.parse_args(argv)
    if args.from_stage > args.through_stage:
        ap.error("--from-stage must not exceed --through-stage")
    if args.workers < 1:
        ap.error("--workers must be positive")
    data_dir = args.data_dir.resolve()
    names = requested_inputs(args.from_stage, args.through_stage)
    paths = {name: input_path(data_dir, name) for name in GEO_FILES}
    annotation_file = ROOT / "data" / "structures.csv"
    missing = [name for name in names if not paths[name].is_file() or paths[name].stat().st_size == 0]
    if not annotation_file.is_file() or annotation_file.stat().st_size == 0:
        ap.error(f"missing annotated structure CSV: {annotation_file}")

    if args.download:
        # The downloader checks remote sizes and SHA256 even for existing .gz files.
        download_names = [name for name in names if paths[name].suffix == ".gz"]
        if download_names:
            download = (sys.executable, str(ROOT / "scripts" / "download_geo.py"),
                        *download_names, "--out", str(data_dir), "--workers", str(args.workers))
            if args.dry_run:
                print("DOWNLOAD:", format_command(download))
            else:
                subprocess.run(download, cwd=ROOT, check=True)
                paths = {name: input_path(data_dir, name) for name in GEO_FILES}
                missing = [name for name in names if not paths[name].is_file() or paths[name].stat().st_size == 0]
    if missing and not args.dry_run:
        details = ", ".join(f"{name} ({GEO_FILES[name]})" for name in missing)
        ap.error(f"missing GEO inputs: {details}. Use --download or place files in {data_dir}")
    if args.dry_run and missing:
        print("MISSING INPUTS:", ", ".join(missing))

    steps = [step for step in build_steps(ROOT, paths, sys.executable)
             if args.from_stage <= step.stage <= args.through_stage]
    if args.start_at:
        matches = [index for index, step in enumerate(steps) if step.name == args.start_at]
        if not matches:
            ap.error(f"unknown --start-at step: {args.start_at}")
        steps = steps[matches[0]:]
    print(f"Repository: {ROOT}\nData: {data_dir}\nSteps: {len(steps)}", flush=True)
    for index, step in enumerate(steps, 1):
        print(f"[{index}/{len(steps)}] {step.name} (Task {step.stage})", flush=True)
        print("  " + format_command(step.command), flush=True)
        if args.dry_run:
            continue
        subprocess.run(step.command, cwd=ROOT, check=True)
        absent = [path for path in step.expected if not path.exists() or
                  (path.is_file() and path.stat().st_size == 0) or
                  (path.is_dir() and not any(path.iterdir()))]
        if absent:
            raise RuntimeError(f"{step.name} completed but expected outputs are missing/empty: {absent}")
    print("Dry run complete." if args.dry_run else "Core six-task run complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
