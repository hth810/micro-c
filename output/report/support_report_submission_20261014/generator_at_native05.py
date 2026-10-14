"""Revise report prose in place while preserving every non-body DOCX part.

The frozen experiment package and the earlier report remain byte-exact.
This is an editorial revision, with no change to experimental measurements.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/report/MicroC_实验报告_最终版_20261014.docx"
SOURCE_SHA = "9ead07ead1801bdfd5e3b01af7063bf278506035e80482435a3990a296ef7a38"
SUPPORT = ROOT / "output/report/support_report_submission_20261014"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def set_text(paragraph, value):
    if not paragraph.runs:
        paragraph.add_run(value)
    else:
        paragraph.runs[0].text = value
        for run in paragraph.runs[1:]:
            run.text = ""


def build(destination, receipt_path):
    destination, receipt_path = destination.resolve(), receipt_path.resolve()
    if not destination.is_relative_to(ROOT) or not receipt_path.is_relative_to(ROOT):
        raise ValueError("Report revision stays inside this repository")
    if destination.exists() or receipt_path.exists():
        raise FileExistsError("Preserve source and prior editorial revisions")
    source = SOURCE.read_bytes()
    if sha(source) != SOURCE_SHA:
        raise ValueError("Source changed; check user edits before revision")
    recipe_path = SUPPORT / "editorial_recipe.json"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    document = Document(io.BytesIO(source))
    paragraphs = list(document.paragraphs)
    assert len(paragraphs) == 213
    changes = []
    for index, replacement in recipe["paragraphs"].items():
        paragraph = paragraphs[int(index)]
        before = paragraph.text
        if replacement is None:
            paragraph._p.getparent().remove(paragraph._p)
        else:
            set_text(paragraph, replacement)
        changes.append({"paragraph_index": int(index), "before": before, "after": replacement})
    for entry in recipe.get("cells", []):
        cell = document.tables[entry["table"]].cell(entry["row"], entry["column"])
        before = cell.text
        assert before == entry["expected"], "Table source changed"
        set_text(cell.paragraphs[0], entry["text"])
        for paragraph in cell.paragraphs[1:]:
            paragraph._p.getparent().remove(paragraph._p)
        changes.append({"table": entry["table"], "row": entry["row"], "column": entry["column"],
                        "before": before, "after": entry["text"]})
    for index, entry in recipe.get("headings", {}).items():
        paragraph = paragraphs[int(index)]
        paragraph.style = entry["style"]
        set_text(paragraph, entry["text"])
        paragraph.paragraph_format.left_indent = Pt(0)
        paragraph.paragraph_format.first_line_indent = Pt(0)
        for run in paragraph.runs:
            run.font.color.rgb = RGBColor(0, 0, 0)
            run.bold = True
            if entry["style"] == "Heading 3":
                run.font.size = Pt(12)
                run.font.name = "Microsoft YaHei"
                run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "Microsoft YaHei")
    # Objective laboratory-report voice, including table text and figure alt.
    for element in document._element.xpath(".//w:t"):
        for before, after in recipe.get("global_text_replacements", {}).items():
            element.text = (element.text or "").replace(before, after)
    content_indent = Pt(recipe["layout"]["content_indent_pt"])
    section = document.sections[0]
    body_width = section.page_width - section.left_margin - section.right_margin
    content_width = body_width - content_indent
    under_subheading = False
    for paragraph in document.paragraphs:
        if paragraph.style.name == "Heading 1":
            under_subheading = False
        elif paragraph.style.name in ("Heading 2", "Heading 3"):
            under_subheading = True
        elif under_subheading:
            paragraph.paragraph_format.left_indent = content_indent
            paragraph.paragraph_format.right_indent = Pt(0)
            paragraph.paragraph_format.first_line_indent = Pt(0)
        if paragraph.style.name == "Caption" and paragraph.text.startswith("图 "):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    # Keep the entire content column within the right margin after indentation.
    scale = content_width / body_width
    for table in document.tables:
        table.alignment = WD_TABLE_ALIGNMENT.LEFT
        properties = table._tbl.tblPr
        indent = properties.find(qn("w:tblInd"))
        if indent is None:
            indent = OxmlElement("w:tblInd")
            properties.append(indent)
        indent.set(qn("w:type"), "dxa")
        indent.set(qn("w:w"), str(round(content_indent / 635)))
        width = properties.find(qn("w:tblW"))
        if width is not None:
            width.set(qn("w:type"), "dxa")
            width.set(qn("w:w"), str(round(content_width / 635)))
        for grid_column in table._tbl.tblGrid:
            grid_column.set(qn("w:w"), str(round(int(grid_column.get(qn("w:w"))) * scale)))
        for cell_width in table._tbl.xpath(".//w:tcW"):
            if cell_width.get(qn("w:type")) == "dxa":
                cell_width.set(qn("w:w"), str(round(int(cell_width.get(qn("w:w"))) * scale)))
    for shape in document.inline_shapes:
        if shape.width > content_width:
            old_width, old_height = shape.width, shape.height
            shape.width = int(content_width)
            shape.height = round(old_height * content_width / old_width)
    # Remove the paragraph-pagination switches that Word marks with a black
    # square in its editing margin. Preserve page positions with ordinary
    # page-break runs, not paragraph PageBreakBefore properties.
    pagination_tags = ("keepNext", "keepLines", "pageBreakBefore", "suppressLineNumbers")
    converted_breaks = 0
    for paragraph in list(document.paragraphs):
        if paragraph.paragraph_format.page_break_before:
            break_paragraph = OxmlElement("w:p")
            properties = OxmlElement("w:pPr")
            spacing = OxmlElement("w:spacing")
            for name, value in (("before", "0"), ("after", "0"), ("line", "20"), ("lineRule", "exact")):
                spacing.set(qn("w:" + name), value)
            properties.append(spacing)
            break_paragraph.append(properties)
            run = OxmlElement("w:r")
            run_properties = OxmlElement("w:rPr")
            size = OxmlElement("w:sz")
            size.set(qn("w:val"), "2")
            run_properties.append(size)
            run.append(run_properties)
            page_break = OxmlElement("w:br")
            page_break.set(qn("w:type"), "page")
            run.append(page_break)
            break_paragraph.append(run)
            paragraph._p.addprevious(break_paragraph)
            converted_breaks += 1
    for root in (document._element, document.styles.element):
        for properties in root.xpath(".//w:pPr"):
            for tag in pagination_tags:
                for flag in list(properties.findall(qn("w:" + tag))):
                    properties.remove(flag)
                flag = OxmlElement("w:" + tag)
                flag.set(qn("w:val"), "0")
                properties.append(flag)
    # Explicit zero overrides also cover paragraphs that inherited controls.
    for paragraph in document.paragraphs:
        properties = paragraph._p.get_or_add_pPr()
        for tag in pagination_tags:
            if properties.find(qn("w:" + tag)) is None:
                flag = OxmlElement("w:" + tag)
                flag.set(qn("w:val"), "0")
                properties.append(flag)
    for element in document._element.xpath(".//wp:docPr"):
        old = element.get("descr", "")
        for before, after in recipe.get("alt_replacements", {}).items():
            old = old.replace(before, after)
        element.set("descr", old)
    text = "\n".join(p.text for p in document.paragraphs)
    assert not any(s in text for s in ("不再把", "并不等于", "不等于", "不把", "不声称", "未把", "不自动解释", "不能互相替代", "不互相替代"))
    assert text.count("共同新结构类型尚未建立") == 1
    assert "342/344" in text
    assert "我们" not in "".join(document._element.itertext())
    assert len(document.tables) == 30 and len(document.inline_shapes) == 15
    for root in (document._element, document.styles.element):
        for tag in pagination_tags:
            assert all(x.get(qn("w:val")) == "0" for x in root.xpath(".//w:" + tag))
    stream = io.BytesIO()
    document.save(stream)
    destination.parent.mkdir(parents=True, exist_ok=True)
    preserved = {}
    with zipfile.ZipFile(io.BytesIO(source)) as original, zipfile.ZipFile(stream) as updated, zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as final:
        for info in original.infolist():
            data = updated.read(info.filename) if info.filename in {"word/document.xml", "word/styles.xml"} else original.read(info.filename)
            final.writestr(info, data)
            if info.filename not in {"word/document.xml", "word/styles.xml"}:
                preserved[info.filename] = sha(data)
    with zipfile.ZipFile(destination) as final:
        assert all(sha(final.read(name)) == expected for name, expected in preserved.items())
    assert sha(SOURCE.read_bytes()) == SOURCE_SHA
    receipt = {"revised_at_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "Micro-C report editorial revision; experiment data and models unchanged",
               "source_sha256": SOURCE_SHA, "source_path": SOURCE.relative_to(ROOT).as_posix(),
               "output_path": destination.relative_to(ROOT).as_posix(), "output_sha256": sha(destination.read_bytes()),
               "recipe_sha256": sha(recipe_path.read_bytes()),
               "source_script_sha256": sha(Path(__file__).read_bytes()),
               "changes": changes, "heading_revision_count": len(recipe.get("headings", {})),
               "content_indent_pt": recipe["layout"]["content_indent_pt"], "figure_captions_centered": True,
               "black_square_pagination_flags_enabled": 0, "ordinary_page_breaks": converted_breaks,
               "preserved_package_parts": preserved,
               "visual_status": "pending complete page review"}
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"changed_blocks": len(changes), "output_sha256": receipt["output_sha256"], "preserved_parts": len(preserved)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    build(args.out, args.receipt)
