"""Small, public metadata audit of the four WT 37 C GSE272159 SRA runs.

Default fetches metadata only. Optional FASTQ sampling reads at most 64 KiB
per paired file into memory, never full FASTQs. This is a metadata query
receipt, NOT an experiment or candidate freeze. Processed COOL-to-run mapping
is not inferred from matching biological-replicate labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
import zlib

import requests

ROOT = Path(__file__).resolve().parents[1]
RUNS = (("GSM8394770", "SRX25308045", "SRR29808782", "t1", 1),
        ("GSM8394772", "SRX25308047", "SRR29808780", "t1", 2),
        ("GSM8394774", "SRX25308049", "SRR29808778", "t2", 1),
        ("GSM8394776", "SRX25308051", "SRR29808776", "t2", 2))
PREFIX = "task2_followup_20261006_raw_read_availability_"
FIELDS = "run_accession,experiment_accession,study_accession,instrument_model,library_layout,read_count,base_count,fastq_ftp,fastq_bytes,fastq_md5"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fetch_metadata(url: str, params: dict | None = None) -> dict:
    with requests.get(url, params=params, timeout=30, stream=True) as response:
        response.raise_for_status()
        content = bytearray()
        for block in response.iter_content(65536):
            content.extend(block)
            if len(content) > 2 * 1024 * 1024:
                raise ValueError("Metadata response exceeded the fixed 2 MiB safety bound")
        return {"url": response.url, "http_status": response.status_code,
                "retrieved_at_utc": now(), "response_bytes": len(content),
                "response_sha256": hashlib.sha256(content).hexdigest(),
                "body_utf8": content.decode("utf-8")}


def parse_geo(text: str) -> dict:
    values: dict[str, list[str]] = {}
    for line in text.splitlines():
        if line.startswith("!") and " = " in line:
            key, value = line[1:].split(" = ", 1)
            values.setdefault(key, []).append(value)
    return values


def parse_ena(text: str, expected_run: str) -> dict:
    rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))
    if len(rows) != 1 or rows[0]["run_accession"] != expected_run:
        raise ValueError("ENA run identity or number of records differs")
    row = rows[0]
    paths, sizes, md5 = (row[key].split(";") for key in ("fastq_ftp", "fastq_bytes", "fastq_md5"))
    if not paths or len(paths) != len(sizes) or len(paths) != len(md5):
        raise ValueError("ENA file vectors differ")
    files = []
    for path, size, checksum in zip(paths, sizes, md5):
        name = path.rsplit("/", 1)[-1]
        if not path.startswith("ftp.sra.ebi.ac.uk/") or len(checksum) != 32:
            raise ValueError("Unexpected public ENA host or MD5")
        role = "R1" if name.endswith("_1.fastq.gz") else "R2" if name.endswith("_2.fastq.gz") else "unpaired"
        files.append({"filename": name, "role": role, "https_url": "https://" + path,
                      "ftp_url": "ftp://" + path, "bytes": int(size), "md5": checksum})
    if sorted(f["role"] for f in files if f["role"] != "unpaired") != ["R1", "R2"]:
        raise ValueError("Missing or duplicate paired FASTQ file")
    return {"run_accession": row["run_accession"], "experiment_accession": row["experiment_accession"],
            "study_accession": row["study_accession"], "platform": row["instrument_model"],
            "library_layout": row["library_layout"], "ena_read_count_field": int(row["read_count"]),
            "base_count": int(row["base_count"]), "fastq_files": files,
            "paired_fastq_gzip_bytes": sum(f["bytes"] for f in files if f["role"] != "unpaired")}


def sample_prefix(file: dict) -> dict:
    """HEAD plus exactly one bounded, anonymous range; no file is saved."""
    with requests.head(file["https_url"], timeout=30, allow_redirects=True) as head:
        head.raise_for_status()
        length = int(head.headers.get("Content-Length", -1))
        if length != file["bytes"]:
            raise ValueError("HEAD length differs from ENA metadata")
    with requests.get(file["https_url"], headers={"Range": "bytes=0-65535"},
                      timeout=30, stream=True) as response:
        if response.status_code != 206:
            raise ValueError("Server did not honor bounded byte range; refusing full download")
        content_range = response.headers.get("Content-Range", "")
        if content_range != f"bytes 0-65535/{file['bytes']}":
            raise ValueError("Unexpected range response")
        content = response.raw.read(65536)
    if len(content) != 65536:
        raise ValueError("Short compressed prefix")
    lines = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(content).decode("ascii").splitlines()
    lengths = []
    for offset in range(0, len(lines) - 3, 4):
        if not lines[offset].startswith("@") or not lines[offset + 2].startswith("+"):
            raise ValueError("Malformed FASTQ prefix")
        if len(lines[offset + 1]) != len(lines[offset + 3]):
            continue  # The final compressed prefix may end inside a quality line.
        lengths.append(len(lines[offset + 1]))
        if len(lengths) == 100:
            break
    if len(lengths) != 100:
        raise ValueError("Bounded prefix did not contain 100 complete records")
    return {"https_url": file["https_url"], "head_http_status": 200,
            "head_bytes": length, "range_http_status": 206, "content_range": content_range,
            "compressed_bytes_read_into_memory": len(content),
            "prefix_sha256": hashlib.sha256(content).hexdigest(), "records_checked": len(lengths),
            "observed_prefix_read_lengths_bp": sorted(set(lengths)), "checked_at_utc": now(),
            "limit": "Only first 100 complete records; not proof of full-run read-length distribution or full-file MD5 verification"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/task2_followup_20261006_raw_read_availability"))
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--sample-fastq-prefix", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Use a new output directory; never overwrite a metadata receipt")
    sources = [Path(__file__).resolve(), ROOT / "tests/test_raw_read_availability_followup.py"]
    args.out.mkdir(parents=True)
    plan = {"query_plan_recorded_at_utc": now(), "record_type": "metadata query plan, NOT experiment freeze",
            "code_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in sources},
            "dependency_versions": {"requests": importlib.metadata.version("requests")},
            "runs": [list(row) for row in RUNS], "metadata_only_default": True,
            "prefix_sampling_requested": args.sample_fastq_prefix,
            "bounds": "Metadata <=2 MiB per request; optional 64 KiB per R1/R2, first100 records in memory; no large FASTQ/SRA downloads or installs"}
    write_json(args.out / "query_plan.json", plan)
    metadata, rows = {}, []
    for sample, experiment, run, label, rep in RUNS:
        geo = fetch_metadata("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi",
                             {"acc": sample, "targ": "self", "form": "text", "view": "full"})
        ena = fetch_metadata("https://www.ebi.ac.uk/ena/portal/api/filereport",
                             {"accession": run, "result": "read_run", "fields": FIELDS, "format": "tsv"})
        sra = fetch_metadata("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                             {"db": "sra", "id": run, "rettype": "full", "retmode": "xml"})
        values = parse_geo(geo["body_utf8"])
        row = parse_ena(ena["body_utf8"], run)
        if values["Sample_geo_accession"] != [sample] or row["experiment_accession"] != experiment:
            raise ValueError("Cross-source sample or experiment identity mismatch")
        if not any(experiment in value for value in values["Sample_relation"]):
            raise ValueError("GEO does not link the expected SRA experiment")
        tree = ET.fromstring(sra["body_utf8"])
        record = tree.find(f".//RUN[@accession='{run}']")
        if record is None or record.attrib.get("is_public") != "true":
            raise ValueError("Expected public SRA run")
        row.update({"geo_sample": sample, "geo_title": values["Sample_title"][0],
                    "geo_characteristics": values.get("Sample_characteristics_ch1", []),
                    "geo_growth_protocol": values.get("Sample_growth_protocol_ch1", []),
                    "t_label_from_original_title": label, "biological_replicate_label": rep,
                    "sra_spots": int(record.attrib["total_spots"]),
                    "sra_archive_bytes": int(record.attrib["size"]),
                    "sra_public": True, "geo_url": geo["url"], "ena_metadata_url": ena["url"],
                    "sra_metadata_url": sra["url"],
                    "sra_original_fastq_files": [{"filename": f.attrib["filename"],
                                                  "bytes": int(f.attrib["size"]), "md5": f.attrib["md5"]}
                                                 for f in record.findall("./SRAFiles/SRAFile")
                                                 if f.attrib.get("supertype") == "Original"]})
        if row["sra_spots"] != row["ena_read_count_field"]:
            raise ValueError("ENA count field differs from SRA spots")
        row["prefix_checks"] = [sample_prefix(f) for f in row["fastq_files"]
                                if args.sample_fastq_prefix and f["role"] != "unpaired"]
        metadata[run] = {"geo": geo, "ena": ena, "sra": sra}
        rows.append(row)
        print(f"Verified metadata {sample} {run} {row['platform']} {label}; paired gzip {row['paired_fastq_gzip_bytes']} bytes", flush=True)
    processed = []
    import cooler
    for rep in (1, 2):
        path = ROOT / "data/raw" / f"GSE272159_37C_rep{rep}.mapq_30.10.cool"
        if path.exists():
            info = cooler.Cooler(str(path)).info
            processed.append({"path": path.relative_to(ROOT).as_posix(),
                              "header_metadata": info.get("metadata", {}),
                              "upper_triangle_count_sum": int(info["sum"]),
                              "binsize_bp": int(info["bin-size"]),
                              "check": "Header only; no run mapping manifest present and no processed-file hash generated by this metadata audit"})
    receipt = {"record_type": "raw-read availability metadata receipt, NOT experiment/candidate freeze",
               "completed_at_utc": now(), "query_plan_sha256": digest(args.out / "query_plan.json"),
               "runs": rows, "local_processed_cool_headers": processed,
               "t1_paired_gzip_total_bytes": sum(row["paired_fastq_gzip_bytes"] for row in rows if row["t_label_from_original_title"] == "t1"),
               "t2_paired_gzip_total_bytes": sum(row["paired_fastq_gzip_bytes"] for row in rows if row["t_label_from_original_title"] == "t2"),
               "source_boundary": "Exact WT37 C t1/t2 run/sample links are verified. Series-level processed COOL metadata are empty and count sums exceed t1-only spots; cannot prove these t1 runs are sole raw inputs or reconstruct t1/t2 merging from filenames. Obtain original processing manifest or rebuild before claiming raw-to-current-COOL identity. Do not treat t2 as an additional independent blind biological replicate.",
               "next_step": "Public reads can support remapping/unique-mapping/MAPQ and coverage-gap QC after handling paired and unpaired reads and obtaining source manifests; no installation or large download performed here.",
               "scientific_limit": "Read-level artifact QC is feasible but does not independently establish a new biological structural type, remove historical label reuse, or replace orthogonal biological validation."}
    write_json(args.out / "metadata_responses.json", metadata)
    write_json(args.out / "receipt.json", receipt)
    if any(digest(ROOT / name) != value for name, value in plan["source_sha256"].items()):
        raise ValueError("Auditor source changed during metadata collection")
    files = ("query_plan.json", "metadata_responses.json", "receipt.json")
    if args.publish:
        targets = [ROOT / "results" / (PREFIX + name) for name in (*files, "audit_index.json")]
        if any(p.exists() for p in targets):
            raise ValueError("Refusing to overwrite published raw-read metadata")
        for name in files:
            shutil.copyfile(args.out / name, ROOT / "results" / (PREFIX + name))
        index = {"schema": "small_metadata_audit_index_v1", "created_at_utc": now(),
                 "files": {PREFIX + name: digest(ROOT / "results" / (PREFIX + name)) for name in files},
                 "source_sha256": plan["source_sha256"],
                 "scope": "Separate metadata package; does not alter existing experiment integrity receipts",
                 "reproduce": "python -m scripts.audit_raw_read_availability_followup --out outputs/task2_followup_20261006_raw_read_availability --sample-fastq-prefix --publish"}
        write_json(ROOT / "results" / (PREFIX + "audit_index.json"), index)
    print(json.dumps({"t1_paired_gzip_total_bytes": receipt["t1_paired_gzip_total_bytes"],
                      "t2_paired_gzip_total_bytes": receipt["t2_paired_gzip_total_bytes"],
                      "source_boundary": receipt["source_boundary"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
