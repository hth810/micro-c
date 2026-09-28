"""Build the Chinese course report from the audited REPORT.md.

Usage:
    python -m pip install "reportlab>=4,<6"
    python scripts/build_report_pdf.py

The detailed report body is sourced directly from REPORT.md, while the cover
summarizes the audited results. Representative figures are included from
results/; all linked result files remain in the repository.
"""

from __future__ import annotations

import argparse
import html
import re
from datetime import date
from pathlib import Path
from urllib.parse import quote

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "REPORT.md"
DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "microc_experiment_report.pdf"
FONT_FILE = Path(r"C:\Windows\Fonts\simhei.ttf")
REPO_BASE = "https://github.com/hth810/micro-c/blob/feature/complete-microc-tasks/"
INK = colors.HexColor("#182638")
BLUE = colors.HexColor("#165A88")
PALE = colors.HexColor("#EAF3F7")
GRAY = colors.HexColor("#526173")
RULE = colors.HexColor("#C8D4DC")

FIGURES = {
    "任务一": (
        ("task1_spatial_confusion_matrix.png",),
        "图 1. 严格空间隔离 CNN 混淆矩阵；CHID 测试样本未被正确识别。",
    ),
    "任务二": (
        ("task2_novel_cluster_N1_3307520_rep1.png", "task2_novel_cluster_N1_3307520_rep2.png"),
        "图 2. 待验证 N1 位点之一在两个 WT 生物学重复中的局部接触图；不是新结构的独立验证。",
    ),
    "任务三": (
        ("task3_example_005.png",),
        "图 3. WT、44℃、利福平及基因/结构标注的代表性 10 kb 多轨道图。",
    ),
    "任务四": (
        ("task4_CHIN_94_wt.png", "task4_CHIN_94_heat_44c.png"),
        "图 4. 主种子筛出的 CHIN_94 位点 WT 与 44℃ 热图；跨种子稳定性须以交集表审阅。",
    ),
    "任务五": (
        ("task5_metric_comparison.png",),
        "图 5. 4 倍降分辨率任务的重建指标；像素质量改善不代表热点定位改善。",
    ),
    "任务六": (
        ("task6_domain_overlay_chromosome_3d_macroregions.png",),
        "图 6. 学得的三维坐标与外部基因组宏区标签；颜色不证明真实空间分区。",
    ),
}


def register_font() -> str:
    if not FONT_FILE.is_file():
        raise FileNotFoundError(f"Chinese font not found: {FONT_FILE}")
    pdfmetrics.registerFont(TTFont("MicroCChinese", str(FONT_FILE)))
    pdfmetrics.registerFontFamily(
        "MicroCChinese", normal="MicroCChinese", bold="MicroCChinese"
    )
    return "MicroCChinese"


def styles(font: str) -> dict[str, ParagraphStyle]:
    return {
        "title": ParagraphStyle(
            "Title", fontName=font, fontSize=24, leading=34, textColor=INK,
            spaceAfter=14 * mm, alignment=TA_LEFT, wordWrap="CJK",
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", fontName=font, fontSize=12.5, leading=21,
            textColor=BLUE, spaceAfter=5 * mm, wordWrap="CJK",
        ),
        "cover_body": ParagraphStyle(
            "CoverBody", fontName=font, fontSize=10, leading=18,
            textColor=INK, spaceAfter=3 * mm, wordWrap="CJK",
        ),
        "cover_table": ParagraphStyle(
            "CoverTable", fontName=font, fontSize=8.8, leading=14.2,
            textColor=INK, wordWrap="CJK",
        ),
        "cover_label": ParagraphStyle(
            "CoverLabel", fontName=font, fontSize=9, leading=14.2,
            textColor=BLUE, wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "H1", fontName=font, fontSize=15, leading=23, textColor=BLUE,
            spaceBefore=8 * mm, spaceAfter=3 * mm, keepWithNext=True,
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "H2", fontName=font, fontSize=11.5, leading=19, textColor=BLUE,
            spaceBefore=5 * mm, spaceAfter=2 * mm, keepWithNext=True,
            wordWrap="CJK",
        ),
        "body": ParagraphStyle(
            "Body", fontName=font, fontSize=9.3, leading=16.2,
            textColor=INK, spaceAfter=2.7 * mm, wordWrap="CJK",
            allowWidows=0, allowOrphans=0,
        ),
        "table": ParagraphStyle(
            "Table", fontName=font, fontSize=8.1, leading=13.1,
            textColor=INK, wordWrap="CJK",
        ),
        "table_header": ParagraphStyle(
            "TableHeader", fontName=font, fontSize=8.4, leading=13.1,
            textColor=colors.white, wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "Caption", fontName=font, fontSize=8.4, leading=14,
            textColor=GRAY, alignment=TA_CENTER, spaceBefore=1.5 * mm,
            spaceAfter=5 * mm, wordWrap="CJK",
        ),
    }


TOKEN = re.compile(r"(\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\)|`[^`]+`)")


def markup(source: str) -> str:
    """Convert the small Markdown subset used by REPORT.md to Paragraph XML."""
    source = source.replace("‑", "-").replace("–", "-").replace("—", "-")
    chunks: list[str] = []
    for part in TOKEN.split(source):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            chunks.append(f"<b>{html.escape(part[2:-2])}</b>")
        elif part.startswith("`") and part.endswith("`"):
            chunks.append(f'<font color="#165A88">{html.escape(part[1:-1])}</font>')
        elif part.startswith("[") and "](" in part and part.endswith(")"):
            label, destination = part[1:-1].split("](", 1)
            if not destination.startswith(("https://", "http://")):
                destination = REPO_BASE + quote(destination, safe="/-_.")
            chunks.append(
                f'<link href="{html.escape(destination, quote=True)}" color="#165A88">'
                f"{html.escape(label)}</link>"
            )
        else:
            chunks.append(html.escape(part))
    return "".join(chunks)


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(markup(text), style)


def figure_image(path: Path, max_w: float, max_h: float) -> Image:
    with PILImage.open(path) as source:
        width, height = source.size
    scale = min(max_w / width, max_h / height)
    picture = Image(str(path), width=width * scale, height=height * scale)
    picture.hAlign = "CENTER"
    return picture


def figure(section: str, text_styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    filenames, caption = FIGURES[section]
    paths = [ROOT / "results" / name for name in filenames]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Representative figure(s) missing: {missing}")
    if len(paths) == 1:
        visual: Flowable = figure_image(paths[0], 168 * mm, 77 * mm)
    else:
        visuals = [figure_image(path, 76 * mm, 68 * mm) for path in paths]
        visual = Table([[visuals[0], visuals[1]]], colWidths=[83 * mm, 83 * mm])
        visual.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
    return [KeepTogether([visual, para(caption, text_styles["caption"])])]


def result_table(rows: list[list[str]], text_styles: dict[str, ParagraphStyle]) -> Table:
    parsed = [
        [para(cell, text_styles["table_header" if row_index == 0 else "table"])
         for cell in row]
        for row_index, row in enumerate(rows)
    ]
    item = Table(parsed, colWidths=[28 * mm, 67 * mm, 73 * mm], repeatRows=1, hAlign="LEFT")
    item.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, BLUE),
        ("LINEBELOW", (0, 1), (-1, -1), 0.2, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return item


def parse_report(text: str, text_styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    story: list[Flowable] = []
    lines = text.splitlines()
    previous_task: str | None = None
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("# "):
            # The cover already carries the report title.
            index += 1
            continue
        if line.startswith("### "):
            if previous_task is not None:
                story.extend(figure(previous_task, text_styles))
            title = line[4:]
            match = next((key for key in FIGURES if title.startswith(key)), None)
            previous_task = match
            story.append(para(title, text_styles["h2"]))
            index += 1
            continue
        if line.startswith("## "):
            if previous_task is not None:
                story.extend(figure(previous_task, text_styles))
                previous_task = None
            story.append(para(line[3:], text_styles["h1"]))
            index += 1
            continue
        if line.startswith("|"):
            table_lines: list[str] = []
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            rows = [[cell.strip() for cell in item.strip().strip("|").split("|")]
                    for item in table_lines]
            rows = [row for row in rows if not all(re.fullmatch(r"-+", c) for c in row)]
            story.append(result_table(rows, text_styles))
            story.append(Spacer(1, 2 * mm))
            continue
        paragraph_lines = [line]
        index += 1
        while index < len(lines) and lines[index].strip() and not lines[index].startswith(("#", "|")):
            paragraph_lines.append(lines[index].strip())
            index += 1
        story.append(para(" ".join(paragraph_lines), text_styles["body"]))
    if previous_task is not None:
        story.extend(figure(previous_task, text_styles))
    return story


class ReportDoc(BaseDocTemplate):
    def __init__(self, filename: str, font: str):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
            topMargin=20 * mm,
            bottomMargin=19 * mm,
            title="基于深度学习的 Micro-C 接触矩阵分析：实验报告",
            author="Micro-C 项目组",
            subject="六项课程任务与审计结果",
        )
        self.font = font
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        )
        self.addPageTemplates(PageTemplate(id="report", frames=frame, onPage=self.decorate))

    def decorate(self, canvas, doc):
        canvas.saveState()
        width, height = A4
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.45)
        canvas.line(20 * mm, height - 13.5 * mm, width - 20 * mm, height - 13.5 * mm)
        canvas.line(20 * mm, 13.8 * mm, width - 20 * mm, 13.8 * mm)
        canvas.setFont(self.font, 8)
        canvas.setFillColor(GRAY)
        canvas.drawString(20 * mm, height - 11 * mm, "Micro-C 接触矩阵分析 | 综合实验报告")
        canvas.drawString(20 * mm, 9.5 * mm, "六项任务 · 可复核结果")
        canvas.drawRightString(width - 20 * mm, 9.5 * mm, f"第 {doc.page} 页")
        canvas.restoreState()


def cover(text_styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    story: list[Flowable] = [Spacer(1, 20 * mm)]
    story.append(para("基于深度学习的 Micro-C 接触矩阵分析", text_styles["title"]))
    story.append(para("三项必做 + 三项选做 · 完整实验报告", text_styles["subtitle"]))
    story.append(para(f"数据：大肠杆菌 MG1655 · GEO GSE272159　|　版本：{date.today().isoformat()}", text_styles["cover_body"]))
    story.append(Spacer(1, 8 * mm))
    story.append(para("结果总览", text_styles["h1"]))
    overview = [
        ("任务一", "空间隔离 CNN 准确率 0.870，但 CHID F1=0；边界感知混合模型五折 CHID 精确率/召回率 0.600/0.577。"),
        ("任务二", "7 个未注释位点，组成 1 个待验证形态簇 N1；严格旧簇规则为 0，跨处理条件不稳定。"),
        ("任务三", "覆盖全基因组的 465 张 10 kb 多轨道图，附原始图像清单与哈希。"),
        ("任务四", "44℃ 主种子筛出 21 条注释，三种子交集 17 条；属于探索性差异候选。"),
        ("任务五", "4 倍模拟超分辨率 PSNR 较双线性插值提升 1.942 dB，热点定位仍未达标。"),
        ("任务六", "WT rep2 接触-距离 Spearman 0.712；精确距离匹配 AUC 0.748，未超过 MDS 0.753。"),
    ]
    table = Table(
        [[para(a, text_styles["cover_label"]), para(b, text_styles["cover_table"])] for a, b in overview],
        colWidths=[25 * mm, 143 * mm],
    )
    table.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [PALE, colors.white]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(table)
    story.append(Spacer(1, 8 * mm))
    story.append(para(
        "证据边界：本报告如实保留未通过的阈值、少数类识别、跨条件迁移、热点定位和三维基线对照；计算候选不等同于已证实的新生物学结构。完整数据来源、代码、逐样本指标及权重可由仓库链接复核。",
        text_styles["cover_body"],
    ))
    story.append(PageBreak())
    return story


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    font = register_font()
    text_styles = styles(font)
    story = cover(text_styles)
    story.extend(parse_report(args.source.read_text(encoding="utf-8"), text_styles))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ReportDoc(str(args.output), font).build(story)
    print(args.output)


if __name__ == "__main__":
    main()
