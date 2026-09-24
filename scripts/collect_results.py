"""Collect small, reviewable real-data results for the Git branch."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs"
DEST = ROOT / "results"


def copy(relative: str, output_name: str | None = None) -> None:
    source = SOURCE / relative
    if not source.exists():
        raise FileNotFoundError(source)
    target = DEST / (output_name or relative.replace("/", "_"))
    shutil.copy2(source, target)


def hash_file(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main() -> None:
    DEST.mkdir(exist_ok=True)
    required = [
        "task1/metrics.json", "task1/confusion_matrix.csv", "task1/confusion_matrix.png",
        "task1/task1_cnn.pt", "task1/test_predictions.csv", "task1/annotation_manifest.csv",
        "task2/summary.json", "task2/new_candidates.csv", "task2/all_candidates.csv",
        "task2/cluster_centers.csv", "task2/embedding.png", "task2/autoencoder.pt",
        "task2_strict/summary.json",
        "task4/class_summary.csv", "task4/structure_changes.csv",
        "task4_repeat/class_summary.csv",
        "task5/summary.json", "task5/comparison.csv", "task5/metric_comparison.png",
        "task5/known_structure_recovery.csv", "task5/reconstructed_test_patches.npz",
        "task5/super_resolution.pt", "task6/metrics.json", "task6/coordinates.csv",
        "task6/chromosome_3d.png", "task6/contact_reconstruction.png", "task6/genome_3d.pt",
    ]
    for entry in required:
        copy(entry)
    for kind in ("CHIN", "OPCID", "CHID"):
        copy(f"task1/saliency_{kind}.png")
        copy(f"task5/known_{kind}.png")
    for index, image in enumerate(sorted((SOURCE / "task3").glob("*.png"))):
        if index in (0, 5, 250, 450):
            shutil.copy2(image, DEST / f"task3_example_{index:03d}.png")
    for image in sorted((SOURCE / "task2" / "heatmaps").glob("*.png"))[:4]:
        shutil.copy2(image, DEST / f"task2_{image.name}")
    for image in sorted((SOURCE / "task4" / "heatmaps").glob("*.png"))[:4]:
        shutil.copy2(image, DEST / f"task4_{image.name}")
    provenance = [hash_file(path) for pattern in ("*.cool.gz", "*.gff.gz", "*.xlsx")
                  for path in sorted((ROOT / "data/raw").glob(pattern))]
    provenance += [hash_file(ROOT / "data/structures.csv")]
    (DEST / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    t1 = json.loads((SOURCE / "task1/metrics.json").read_text(encoding="utf-8"))
    t2 = json.loads((SOURCE / "task2/summary.json").read_text(encoding="utf-8"))
    strict = json.loads((SOURCE / "task2_strict/summary.json").read_text(encoding="utf-8"))
    t5 = json.loads((SOURCE / "task5/summary.json").read_text(encoding="utf-8"))
    t6 = json.loads((SOURCE / "task6/metrics.json").read_text(encoding="utf-8"))
    changes = pd.read_csv(SOURCE / "task4/class_summary.csv")
    count_tracks = len(list((SOURCE / "task3").glob("*.png")))
    lines = ["# 真实数据运行结果", "",
        "数据：NCBI GEO GSE272159；结构注释：课程工作簿 Supplementary Tables 4/5/6。文件哈希见 `provenance.json`。",
        "所有原始矩阵和完整逐窗口图仍在本地 `data/raw/` 与 `outputs/`，本目录保留便于审阅的小型结果。", "",
        "## 任务一：已知结构识别", "",
        f"留出集准确率 {t1['accuracy']:.3f}，宏平均 F1 {t1['macro_f1']:.3f}；结构数训练/验证/测试为 "
        f"{t1['train_structures']}/{t1['validation_structures']}/{t1['test_structures']}。",
        "CHID 样本少，逐类召回请结合混淆矩阵阅读。", "",
        "## 任务二：新结构候选", "",
        f"扫描 {t2['windows']} 个窗口，入选 {t2['selected']} 个；按预设规则得到 "
        f"{t2['new_candidates']} 个未注释候选。已知结构跨重复筛选后召回 "
        f"{t2.get('known_structure_recall_after_reproducibility', float('nan')):.3f}。",
        f"较严格的预设阈值（前 20%、重复相关至少 0.3）产生 {strict['new_candidates']} 个新候选；"
        "对该参数选择应做进一步敏感性验证。",
        f"形态特征区分已知注释与背景的空间留出 AUC 为 "
        f"{t2.get('shape_representation_known_vs_background_auc', float('nan')):.3f}，证据较弱。",
        "这些是计算候选，不是经实验确认的新结构。", "",
        "## 任务三：全基因组轨道", "",
        f"生成 {count_tracks} 张 10 kb 区间图，覆盖 WT 37℃、44℃ 和 rifampicin 25。", "",
        "## 任务四：实验条件变化", "",
        "rifampicin 25 相对于 WT 的归一化结构强度平均变化：", "" ]
    for _, row in changes.iterrows():
        lines.append(f"- {row['type']}（{int(row['count'])} 个）：{row['mean']:.3f}")
    lines += ["两条件各使用两个生物学重复；WT 重复对照汇总见 `task4_repeat_class_summary.csv`。", "",
        "## 任务五：超分辨", "",
        f"测试窗口 {t5['test_patches']} 个；PSNR 双线性/CNN 为 "
        f"{t5['comparison']['psnr']['bilinear']:.3f}/{t5['comparison']['psnr']['cnn']:.3f}，"
        f"SSIM 为 {t5['comparison']['ssim']['bilinear']:.3f}/{t5['comparison']['ssim']['cnn']:.3f}。",
        "低分辨率输入由 Micro-C 合并 bin 模拟，不能外推到独立 Hi-C。", "",
        "## 任务六：三维重建", "",
        f"{t6['bins']} 个 {t6['bin_bp']} bp bin；接触与反距离 Spearman 相关为 "
        f"MLP {t6['mlp_contact_distance_spearman']:.3f}、MDS {t6['mds_contact_distance_spearman']:.3f}。",
        "MLP 使用 MDS 初始化，因此两者不是独立建模路线；坐标并非唯一的真实物理构象。", ""]
    (DEST / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Collected {len(list(DEST.iterdir()))} review files in {DEST}")


if __name__ == "__main__": main()
