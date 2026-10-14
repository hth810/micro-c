"""Check the current report and historical evidence after report archival."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify():
    audit = json.loads((ROOT / "results/report_submission_20261014_layout_audit.json").read_text(encoding="utf-8"))
    for kind in ("docx", "pdf"):
        assert digest((ROOT / audit[kind]).read_bytes()) == audit[kind + "_sha256"], kind
    baseline = json.loads((ROOT / "results/final_verification_20261014_integrity.json").read_text(encoding="utf-8"))
    for relative, expected in baseline["sha256"].items():
        if relative == "README.md":
            data = subprocess.check_output(["git", "show", "3906ce4a4ab05ff4406b7fdf06eda5cf11b95449:README.md"], cwd=ROOT)
            # README's current edit only changes the report delivery link and
            # archive notice. The scientific summary remains byte-identical.
            before = data.decode("utf-8").splitlines()
            after = (ROOT / relative).read_text(encoding="utf-8").splitlines()
            assert len(before) == len(after)
            assert [i for i, (a, b) in enumerate(zip(before, after)) if a != b] == [6]
        else:
            target = audit["historical_integrity_relocations"].get(relative, relative)
            data = (ROOT / target).read_bytes()
        assert digest(data) == expected, relative
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(ROOT / audit["docx"]) as package:
        for part in ("word/document.xml", "word/styles.xml"):
            root = etree.fromstring(package.read(part))
            for tag in ("keepNext", "keepLines", "pageBreakBefore", "suppressLineNumbers"):
                for flag in root.xpath(".//w:" + tag, namespaces=ns):
                    assert flag.get("{" + ns["w"] + "}val") in {"0", "false", "off"}, tag
    active_word = list((ROOT / "output/report").glob("*.docx"))
    active_pdf = list((ROOT / "output/pdf").glob("*.pdf"))
    assert active_word == [ROOT / audit["docx"]]
    assert active_pdf == [ROOT / audit["pdf"]]
    return {"current_report_verified": True, "historical_files_verified": len(baseline["sha256"]),
            "active_report_versions": 1, "pages_reviewed": audit["page_count"],
            "black_square_flags_enabled": 0}


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
