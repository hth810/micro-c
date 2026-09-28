"""Download selected GSE272159 Micro-C files from the official GEO repository."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE272nnn/GSE272159/suppl/"
FILES = {
    "wt1": "GSE272159_37C_rep1.mapq_30.10.cool.gz",
    "wt2": "GSE272159_37C_rep2.mapq_30.10.cool.gz",
    "heat": "GSE272159_44C_rep1.mapq_30.10.cool.gz",
    "heat2": "GSE272159_44C_rep2.mapq_30.10.cool.gz",
    "rif": "GSE272159_rif25_rep1.MG1655.mapq_30.10.cool.gz",
    "rif2": "GSE272159_rif25_rep2.MG1655.mapq_30.10.cool.gz",
    "hns1": "GSM9678478_HNS_mut_rep1.MG1655.mapq_30.10.cool.gz",
    "hns2": "GSM9678479_HNS_mut_rep2.MG1655.mapq_30.10.cool.gz",
    "fix37": "GSM8950767_37C_37C_fix_rep1.MG1655.mapq_30.10.cool.gz",
    "gff": "GCF_000005845.2_ASM584v2_genomic.gff.gz",
}
GFF_URL = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/"


def download_parallel(url: str, target: Path, existing: int, size: int, workers: int) -> None:
    """Resume in ordered range batches so an interrupted file stays contiguous."""
    chunk_size = 8 * 1024 * 1024
    starts = list(range(existing, size, chunk_size))

    def fetch(start: int) -> bytes:
        end = min(start + chunk_size, size) - 1
        for attempt in range(5):
            try:
                response = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                    raise ValueError(f"Server returned an unexpected range for {start}-{end}")
                if len(response.content) != end - start + 1:
                    raise ValueError(f"Incomplete range for {start}-{end}")
                return response.content
            except (requests.RequestException, ValueError):
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        raise AssertionError("Unreachable")

    with target.open("ab") as stream, ThreadPoolExecutor(max_workers=workers) as executor:
        for offset in range(0, len(starts), workers):
            for chunk in executor.map(fetch, starts[offset:offset + workers]):
                stream.write(chunk)
            stream.flush()
            print(f"Downloaded {target.name}: {stream.tell():,}/{size:,} bytes", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("names", nargs="*", choices=FILES, default=list(FILES))
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--workers", type=int, default=1, help="parallel HTTP range requests per file")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "geo_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    with requests.Session() as session:
        for name in args.names:
            filename = FILES[name]
            target = args.out / filename
            if name == "fix37":
                url = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8950nnn/GSM8950767/suppl/{filename}"
            elif name.startswith("hns"):
                accession = filename.split("_", 1)[0]
                url = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9678nnn/{accession}/suppl/{filename}"
            else:
                url = (GFF_URL if name == "gff" else BASE) + filename
            remote = session.head(url, timeout=30)
            remote.raise_for_status()
            size = int(remote.headers["Content-Length"])
            existing = target.stat().st_size if target.exists() else 0
            if existing > size:
                raise ValueError(f"Local file exceeds remote size: {target}")
            if existing < size:
                if args.workers > 1:
                    download_parallel(url, target, existing, size, args.workers)
                else:
                    headers = {"Range": f"bytes={existing}-"} if existing else {}
                    with session.get(url, headers=headers, stream=True, timeout=120) as response:
                        response.raise_for_status()
                        if existing and response.status_code != 206:
                            raise ValueError("Server did not honor resume range; refusing to corrupt local file")
                        with target.open("ab" if existing else "wb") as stream:
                            for chunk in response.iter_content(1024 * 1024):
                                if chunk: stream.write(chunk)
            if target.stat().st_size != size:
                raise ValueError(f"Incomplete download: {target}")
            digest = hashlib.sha256()
            with target.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            manifest[name] = {"file": filename, "source": url, "bytes": size, "sha256": digest.hexdigest()}
            print(f"Downloaded {filename}: {size:,} bytes")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
