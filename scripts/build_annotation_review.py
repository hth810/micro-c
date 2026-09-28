"""Build an offline, human-in-the-loop review board for Task 2 candidates.

The known-class CNN is shown only as a visual resemblance suggestion. It was
trained on annotated structures and is not a calibrated novelty detector.
No review label is inferred from model output; the reviewer exports it.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


CLASSES = ("CHIN", "OPCID", "CHID")


def predict_known_resemblance(
    centers: list[int], rep1: Path, rep2: Path, checkpoint: Path, metrics: Path
) -> pd.DataFrame:
    from microc.data import open_cool
    from task1.run_task1 import StructureCNN, extract_window

    params = json.loads(metrics.read_text(encoding="utf-8"))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = StructureCNN(len(CLASSES), int(saved.get("spatial_grid", 1)))
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    window_bins = int(params["window_bins"])
    aggregate = int(params["aggregate"])
    rows = []
    torch.set_num_threads(min(4, torch.get_num_threads()))
    with open_cool(rep1) as first, open_cool(rep2) as second:
        for center in centers:
            matrices = [extract_window(cool, center, window_bins, aggregate)
                        for cool in (first, second)]
            inputs = torch.from_numpy(np.stack(matrices)[:, None].astype(np.float32))
            with torch.no_grad():
                # Do not add any post-hoc class bias to an OOD suggestion.
                probabilities = model(inputs).softmax(dim=1).numpy().mean(axis=0)
            rows.append({"center_bp": center, "suggested_known_shape": CLASSES[int(probabilities.argmax())],
                         **{f"shape_score_{name}": float(probabilities[i])
                            for i, name in enumerate(CLASSES)}})
    return pd.DataFrame(rows)


def image_data(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def review_image(images: Path, center: int, replicate: str) -> Path:
    packaged = images / f"task2_strict_individual_{center}_{replicate}.png"
    generated = images / f"{center}_{replicate}.png"
    if packaged.is_file():
        return packaged
    return generated


def prepare_cards(evidence: pd.DataFrame, images: Path, suggestions: pd.DataFrame | None) -> list[dict]:
    required = {"chrom", "start", "end", "representative_center_bp", "wt_replicate_r",
                "known_distance_bp", "max_candidate_score"}
    if not required.issubset(evidence.columns):
        raise ValueError(f"Missing candidate evidence columns: {sorted(required - set(evidence.columns))}")
    if evidence.representative_center_bp.duplicated().any():
        raise ValueError("Review candidates must be independent representative loci")
    lookup = suggestions.set_index("center_bp") if suggestions is not None else None
    cards = []
    for row in evidence.itertuples():
        center = int(row.representative_center_bp)
        suggestion = None
        if lookup is not None:
            if center not in lookup.index:
                raise ValueError(f"Missing CNN suggestion for {center}")
            item = lookup.loc[center]
            suggestion = {"class": str(item.suggested_known_shape),
                          "scores": {name: float(item[f"shape_score_{name}"]) for name in CLASSES}}
        cards.append({"center": center, "chrom": str(row.chrom), "start": int(row.start),
                      "end": int(row.end), "wt_r": float(row.wt_replicate_r),
                      "known_distance_bp": int(row.known_distance_bp),
                      "candidate_score": float(row.max_candidate_score),
                      "putative_n1": center in (3_307_520, 4_241_920),
                      "images": {rep: image_data(review_image(images, center, rep))
                                 for rep in ("rep1", "rep2")},
                      "suggestion": suggestion})
    return cards


def render_html(cards: list[dict]) -> str:
    if not cards:
        raise ValueError("No candidates to review")
    data = json.dumps(cards, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Micro-C 候选结构人工复核</title>
<style>
body{{font:15px/1.55 system-ui,sans-serif;margin:0;background:#f5f7fa;color:#1c2833}}
header{{background:#12344b;color:white;padding:1.2rem max(1rem,calc((100vw - 1200px)/2))}}
main{{max-width:1200px;margin:auto;padding:1rem}}.warning{{background:#fff3d3;border-left:4px solid #dd9a1b;padding:.7rem 1rem}}
.toolbar{{display:flex;gap:1rem;align-items:center;flex-wrap:wrap;margin:1rem 0}}button{{background:#126a70;color:white;border:0;padding:.6rem .9rem;border-radius:5px;cursor:pointer}}
.card{{background:white;border:1px solid #d6dfe7;border-radius:8px;padding:1rem;margin:1rem 0;box-shadow:0 2px 8px #0001}}
.images{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.8rem}}img{{width:100%;max-height:490px;object-fit:contain;background:#f7f7f7}}
.meta{{display:flex;gap:.5rem 1.3rem;flex-wrap:wrap}}label{{display:block;margin:.6rem 0}}select,textarea{{font:inherit;padding:.35rem;max-width:100%}}textarea{{width:100%;min-height:3.2rem;box-sizing:border-box}}
small{{color:#465a68}}.badge{{background:#dbeef1;border-radius:4px;padding:.15rem .4rem}}
@media(max-width:700px){{.images{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>Micro-C 疑似新结构人工复核</h1><div>7 个独立位点 · 双生物学重复 · 离线运行</div></header>
<main><p class="warning">这些都是计算候选，不是已证实的新生物学结构。已知类 CNN 在 CHID 上表现较差，且没有接受“新类型”训练；其输出只是形态相似性提示，不能自动决定标签。N1 两成员在跨条件检验中未保持同类配对形态。</p>
<div class="toolbar"><button id="export">导出人工复核 JSON</button><button id="clear">清空本机暂存</button><small id="status"></small></div>
<div id="cards"></div><p><small>本页不上传数据。选择和备注临时保存在当前浏览器的 localStorage；请导出 JSON 作为人工审阅记录。候选坐标和统计量来自 Task 2 结果。</small></p></main>
<script>
const candidates={data};
const storageKey='microc-review-v1';
let saved={{}};try{{saved=JSON.parse(localStorage.getItem(storageKey)||'{{}}')}}catch(e){{saved={{}}}}
const escapeHtml=s=>String(s).replace(/[&<>\"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}}[c]));
const root=document.getElementById('cards');
root.innerHTML=candidates.map(c=>{{
 const p=c.suggestion;
 const model=p?`已知类 CNN 形态提示：${{escapeHtml(p.class)}}（CHIN ${{p.scores.CHIN.toFixed(2)}} / OPCID ${{p.scores.OPCID.toFixed(2)}} / CHID ${{p.scores.CHID.toFixed(2)}}；非新颖性概率）`:'未计算 CNN 形态提示';
 return `<section class="card" data-center="${{c.center}}"><h2>${{escapeHtml(c.chrom)}}:${{c.start.toLocaleString()}}–${{c.end.toLocaleString()}} ${{c.putative_n1?'<span class="badge">WT 疑似 N1 成员</span>':''}}</h2>
 <div class="meta"><span>中心 ${{c.center.toLocaleString()}} bp</span><span>候选分数 ${{c.candidate_score.toFixed(3)}}</span><span>WT 重复相关 ${{c.wt_r.toFixed(3)}}</span><span>距已知注释 ${{c.known_distance_bp.toLocaleString()}} bp</span></div>
 <p><small>${{model}}</small></p><div class="images"><figure><img src="${{c.images.rep1}}" alt="rep1 接触矩阵"><figcaption>WT rep1</figcaption></figure><figure><img src="${{c.images.rep2}}" alt="rep2 接触矩阵"><figcaption>WT rep2</figcaption></figure></div>
 <label>人工判定 <select class="decision"><option value="unreviewed">未审阅</option><option value="retain_candidate">保留为待验证候选</option><option value="known_like">更像已知类</option><option value="artifact">疑似噪声/伪影</option></select></label>
 <label>证据与疑虑 <textarea class="note" placeholder="例如：两重复形态、背景密度、邻近已知结构、需补实验"></textarea></label></section>`
}}).join('');
function persist(){{const out={{}};document.querySelectorAll('.card').forEach(el=>{{const center=el.dataset.center;out[center]={{decision:el.querySelector('.decision').value,note:el.querySelector('.note').value}}}});saved=out;try{{localStorage.setItem(storageKey,JSON.stringify(out));document.getElementById('status').textContent='已在本机暂存'}}catch(e){{document.getElementById('status').textContent='浏览器未允许本机暂存，请及时导出 JSON'}}}}
document.querySelectorAll('.card').forEach(el=>{{const v=saved[el.dataset.center]||{{}};el.querySelector('.decision').value=v.decision||'unreviewed';el.querySelector('.note').value=v.note||'';el.addEventListener('change',persist);el.addEventListener('input',persist)}});
document.getElementById('export').onclick=()=>{{persist();const records=candidates.map(c=>({{chrom:c.chrom,start:c.start,end:c.end,center_bp:c.center,review: saved[c.center]}}));const blob=new Blob([JSON.stringify({{schema:'microc-candidate-human-review-v1',caution:'Human judgments, not validated biological discoveries',records}},null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='microc_candidate_human_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}};
document.getElementById('clear').onclick=()=>{{if(confirm('清空本机的人工判定与备注？')){{try{{localStorage.removeItem(storageKey)}}catch(e){{}};document.querySelectorAll('.card').forEach(el=>{{el.querySelector('.decision').value='unreviewed';el.querySelector('.note').value=''}});saved={{}};document.getElementById('status').textContent='已清空当前页面记录'}}}};
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", type=Path, default=Path("results/task2_strict_locus_evidence.csv"))
    ap.add_argument("--images", type=Path, default=Path("results"))
    ap.add_argument("--checkpoint", type=Path, default=Path("results/task1_spatial_task1_cnn.pt"))
    ap.add_argument("--metrics", type=Path, default=Path("results/task1_spatial_metrics.json"))
    ap.add_argument("--rep1", type=Path)
    ap.add_argument("--rep2", type=Path)
    ap.add_argument("--suggestions", type=Path, default=Path("results/task2_annotation_review_model_suggestions.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/task2_annotation_review.html"))
    args = ap.parse_args()
    evidence = pd.read_csv(args.evidence)
    if bool(args.rep1) != bool(args.rep2):
        raise ValueError("Supply both WT replicates or neither")
    if args.rep1:
        suggestions = predict_known_resemblance(
            evidence.representative_center_bp.astype(int).tolist(),
            args.rep1, args.rep2, args.checkpoint, args.metrics)
        args.suggestions.parent.mkdir(parents=True, exist_ok=True)
        suggestions.to_csv(args.suggestions, index=False)
    else:
        suggestions = pd.read_csv(args.suggestions) if args.suggestions.is_file() else None
    cards = prepare_cards(evidence, args.images, suggestions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_html(cards), encoding="utf-8")
    print(f"Wrote {len(cards)} candidate review cards to {args.out}")


if __name__ == "__main__":
    main()
