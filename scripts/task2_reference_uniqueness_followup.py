"""Exact circular reference k-mer uniqueness for the four frozen Task 2 loci.

This is a reference-sequence diagnostic, NOT aligner mappability or fragment
coverage. Read lengths, errors, paired-end constraints and experimental strain
variants are not available in processed COOL files. No candidates are selected.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
BASE = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/"
FILENAME = "GCF_000005845.2_ASM584v2_genomic.fna.gz"
KS = (31, 51, 101)


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def download(path: Path) -> dict:
    if path.name != FILENAME:
        raise ValueError("Official download must retain its assembly filename")
    # Retrieve the checksum from the same official assembly; never trust only a
    # filename or silently reuse an incomplete download.
    checksum_response = requests.get(BASE + "md5checksums.txt", timeout=(20, 60))
    checksum_response.raise_for_status()
    matches = [line.split()[0] for line in checksum_response.text.splitlines()
               if len(line.split()) == 2 and line.split()[1].lstrip("./") == FILENAME]
    if len(matches) != 1:
        raise ValueError("Official MD5 manifest does not uniquely identify the FASTA")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(BASE + FILENAME, stream=True, timeout=(20, 60))
        response.raise_for_status()
        with path.open("xb") as handle:
            for block in response.iter_content(1024 * 1024):
                handle.write(block)
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "md5").hexdigest()
    if actual != matches[0]:
        raise ValueError("Reference checksum mismatch; existing file retained for inspection")
    return {"url": BASE + FILENAME, "official_md5_url": BASE + "md5checksums.txt",
            "official_md5": matches[0], "sha256": sha(path)}


def read_reference(path: Path) -> tuple[str, bytes]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="ascii") as handle:
        records, accession, parts = [], None, []
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if accession is not None:
                    records.append((accession, "".join(parts).upper().encode("ascii")))
                accession, parts = line[1:].split()[0], []
            elif line:
                if accession is None:
                    raise ValueError("Sequence before FASTA header")
                parts.append(line)
        if accession is not None:
            records.append((accession, "".join(parts).upper().encode("ascii")))
    if len(records) != 1 or records[0][0] != "NC_000913.3" or len(records[0][1]) != 4_641_652:
        raise ValueError("Reference accession/length differs from processed MG1655 contacts")
    return records[0]


def circular_unique_starts(sequence: bytes, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact packed canonical k-mers; no probabilistic hash collisions.

    Return uniqueness and valid-ACGT flags for every possible circular start.
    Reverse complements share a canonical code, and each genomic start counts
    once. Ambiguous windows never count as unique.
    """
    n = len(sequence)
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= n:
        raise ValueError("Require integer 1 <= k <= sequence length")
    table = bytearray([4] * 256)
    for base, code in zip(b"ACGT", range(4)):
        table[base] = code
        table[base + 32] = code
    codes = (sequence + sequence[:k - 1]).translate(bytes(table))
    width = (2 * k + 7) // 8
    encoded = np.zeros(n, dtype=f"V{width}")
    valid = np.zeros(n, dtype=bool)
    mask, shift = (1 << (2 * k)) - 1, 2 * (k - 1)
    forward, reverse, last_invalid = 0, 0, -1
    for end, code in enumerate(codes):
        if code > 3:
            last_invalid = end
            code = 0
        forward = ((forward << 2) | code) & mask
        reverse = (reverse >> 2) | ((3 - code) << shift)
        start = end - k + 1
        if start >= 0 and last_invalid < start:
            valid[start] = True
            encoded[start] = min(forward, reverse).to_bytes(width, "big")
    unique = np.zeros(n, dtype=bool)
    if valid.any():
        _, inverse, counts = np.unique(encoded[valid], return_inverse=True, return_counts=True)
        unique[valid] = counts[inverse] == 1
    return unique, valid


def window_statistics(unique: np.ndarray, valid: np.ndarray, center: int, k: int) -> dict:
    # Match the actual 10-bp-bin alignment used by microc.data.window().
    start = center // 10 * 10 - 2560
    end = start + 5120
    if start < 0 or end > len(unique):
        raise ValueError("Diagnostic window is padded/outside the reference; do not invent sequence")
    stops = slice(start, end - k + 1)
    valid_n = int(valid[stops].sum())
    total = end - start - k + 1
    return {"start_bp": start, "end_bp": end, "k": k, "contained_kmer_starts": total,
            "valid_acgt_starts": valid_n, "unique_starts": int(unique[stops].sum()),
            "unique_fraction_all_starts": float(unique[stops].mean()),
            "unique_fraction_valid_starts": float(unique[stops].sum() / valid_n) if valid_n else None}


def run(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError("Use a fresh diagnostic output directory")
    protocol = {"experiment": "task2-reference-uniqueness-followup", "frozen_utc": datetime.now(timezone.utc).isoformat(),
                "run_parent_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "source_sha256": sha(Path(__file__)), "k": list(KS), "topology": "circular",
                "canonical_both_strands": True, "sequence_encoding": "exact packed two-bit code, no hashed collisions",
                "window": "same 5120bp native 10bp-aligned window as contact extraction; fully contained read starts",
                "candidate_manifest_sha256": sha(args.candidates), "background_manifest_sha256": sha(args.background),
                "annotations_sha256": sha(ROOT / "data/structures.csv"),
                "selection": "All four previously frozen candidates; no replacement or ranking rule",
                "caution": "Exact reference uniqueness is not aligner/paired-read mappability, experimental fragment coverage, or a significance test"}
    save_json(args.out / "protocol.json", protocol)
    source = download(args.reference) if args.download else {"url": BASE + FILENAME, "sha256": sha(args.reference)}
    accession, sequence = read_reference(args.reference)
    save_json(args.out / "reference_receipt.json", {**source, "accession": accession, "bases": len(sequence),
        "sequence_sha256": hashlib.sha256(sequence).hexdigest(), "protocol_sha256": sha(args.out / "protocol.json")})
    candidates = pd.read_csv(args.candidates)
    if len(candidates) != 4 or candidates.center.nunique() != 4:
        raise ValueError("The existing freeze must contain all four candidates")
    unknown = pd.read_csv(args.background)
    known = pd.read_csv(ROOT / "data/structures.csv")
    if len(unknown) != 291 or len(known) != 344:
        raise ValueError("Original reference cohorts changed")
    rows, genome = [], []
    for k in KS:
        unique, valid = circular_unique_starts(sequence, k)
        genome.append({"k": k, "genomic_starts": len(sequence), "valid_starts": int(valid.sum()),
                       "unique_starts": int(unique.sum()), "unique_fraction": float(unique.mean())})
        for cohort, frame, id_column in (("candidate", candidates, "candidate_id"),
                                        ("unannotated_reference", unknown, None), ("known_reference", known, "id")):
            for index, row in frame.iterrows():
                center = int(row.center)
                # Unknown representatives are defined by actual contact windows;
                # a partial edge window cannot be called a sequence-valid window.
                if center // 10 * 10 - 2560 < 0 or center // 10 * 10 + 2560 > len(sequence):
                    continue
                identifier = str(row[id_column]) if id_column else str(center)
                rows.append({"cohort": cohort, "id": identifier, "center_bp": center,
                             **window_statistics(unique, valid, center, k)})
        print(f"finished exact reference uniqueness k={k}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "reference_window_uniqueness.csv", index=False)
    summaries = []
    for row in table[table.cohort.eq("candidate")].itertuples():
        pool = table[(table.cohort.eq("unannotated_reference")) & table.k.eq(row.k)]
        permitted = np.all(np.abs(pool.center_bp.to_numpy()[:, None] - candidates.center.to_numpy()[None]) >= 10240, axis=1)
        values = pool.loc[permitted, "unique_fraction_all_starts"].to_numpy()
        summaries.append({"candidate_id": row.id, "center_bp": row.center_bp, "k": row.k,
                          "unique_fraction": row.unique_fraction_all_starts,
                          "independent_background_loci": len(values),
                          "background_median_unique_fraction": float(np.median(values)),
                          "descriptive_lower_tail_fraction_with_ties": float(np.mean(values <= row.unique_fraction_all_starts))})
    pd.DataFrame(summaries).to_csv(args.out / "candidate_reference_diagnostics.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=True)
    for ax, k in zip(axes, KS):
        frame = table[table.cohort.eq("candidate") & table.k.eq(k)]
        ax.bar(frame.id, frame.unique_fraction_all_starts, color="#24769c")
        ax.set(ylim=(0, 1.02), title=f"Exact canonical {k}-mer uniqueness", ylabel="Unique / all contained starts")
    fig.suptitle("Frozen windows: reference-sequence diagnostic, not aligner mappability")
    fig.savefig(args.out / "reference_uniqueness.png", dpi=160)
    plt.close(fig)
    summary = {"protocol_sha256": sha(args.out / "protocol.json"), "reference": source,
               "genome": genome, "candidates": summaries,
               "caution": protocol["caution"], "outcomes_do_not_reselect_candidates": True}
    save_json(args.out / "summary.json", summary)
    prefix = "task2_reference_followup_20261006_"
    args.results.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for path in args.out.iterdir():
        if path.is_file():
            target = args.results / (prefix + path.name)
            if target.exists():
                raise ValueError(f"Published artifact already exists: {target}")
            shutil.copy2(path, target)
            artifacts[target.name] = sha(target)
    save_json(args.results / (prefix + "evidence_index.json"), {"artifacts_sha256": artifacts,
              "script_sha256": sha(Path(__file__)), "reference_sha256": sha(args.reference)})
    print(json.dumps(summaries, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("data/raw") / FILENAME)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--candidates", type=Path, default=Path("results/task2_improve_20261006_individual_candidate_validation.csv"))
    parser.add_argument("--background", type=Path, default=Path("results/task2_improve_20261006_area_rep1_independent_unannotated.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/task2_reference_followup_20261006"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
