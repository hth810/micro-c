"""Download selected GSE272159 Micro-C files from the official GEO repository."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import requests


BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE272nnn/GSE272159/suppl/"
FILES = {
    "wt1": "GSE272159_37C_rep1.mapq_30.10.cool.gz",
    "wt2": "GSE272159_37C_rep2.mapq_30.10.cool.gz",
    "heat": "GSE272159_44C_rep1.mapq_30.10.cool.gz",
    "rif": "GSE272159_rif25_rep1.MG1655.mapq_30.10.cool.gz",
    "rif2": "GSE272159_rif25_rep2.MG1655.mapq_30.10.cool.gz",
    "gff": "GCF_000005845.2_ASM584v2_genomic.gff.gz",
}
GFF_URL = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("names", nargs="*", choices=FILES, default=list(FILES))
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with requests.Session() as session:
        for name in args.names:
            filename = FILES[name]
            target = args.out / filename
            url = (GFF_URL if name == "gff" else BASE) + filename
            remote = session.head(url, timeout=30)
            remote.raise_for_status()
            size = int(remote.headers["Content-Length"])
            existing = target.stat().st_size if target.exists() else 0
            if existing > size:
                raise ValueError(f"Local file exceeds remote size: {target}")
            if existing < size:
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
    (args.out / "geo_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
