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
        "task1_calibrated/metrics.json",
        "task1_scale8/metrics.json",
        "task2/summary.json", "task2/new_candidates.csv", "task2/all_candidates.csv",
        "task2/all_windows.csv",
        "task2/cluster_centers.csv", "task2/embedding.png", "task2/autoencoder.pt",
        "task2_strict/summary.json", "task2_validation/summary.json",
        "task2_validation/heldout_predictions.csv", "task2_validation/candidate_shift_null.csv",
        "task6_independent/metrics.json", "task6_independent/coordinates.csv",
        "task6_independent/contact_and_baseline.npz", "task6_independent/chromosome_3d.png",
        "task6_independent/contact_reconstruction.png", "task6_independent/genome_3d.pt",
    ]
    for entry in required:
        if entry == "task1_calibrated/metrics.json":
            copy(entry, "task1_calibration_trial_metrics.json")
        elif entry == "task1_scale8/metrics.json":
            copy(entry, "task1_scale8_trial_metrics.json")
        elif entry.startswith("task6_independent/"):
            copy(entry, entry.replace("task6_independent/", "task6_"))
        else:
            copy(entry)
    for source, output in (
        ("task4_heat_final/class_summary.csv", "task4_class_summary.csv"),
        ("task4_heat_final/structure_changes.csv", "task4_structure_changes.csv"),
        ("task4_heat_final/reproducible_changes.csv", "task4_reproducible_changes.csv"),
        ("task4_rif_final/class_summary.csv", "task4_rif_class_summary.csv"),
        ("task4_rif_final/structure_changes.csv", "task4_rif_structure_changes.csv"),
        ("task4_rif_final/reproducible_changes.csv", "task4_rif_reproducible_changes.csv"),
        ("task4_repeat_depth_final/class_summary.csv", "task4_repeat_class_summary.csv"),
    ):
        copy(source, output)
    sr_output = SOURCE / "task5_corrected"
    if sr_output.exists():
        for name in ("summary.json", "comparison.csv", "metric_comparison.png",
                     "known_structure_recovery.csv", "reconstructed_test_patches.npz",
                     "super_resolution.pt"):
            copy(f"task5_corrected/{name}", f"task5_{name}")
        for kind in ("CHIN", "OPCID", "CHID"):
            copy(f"task5_corrected/known_{kind}.png", f"task5_known_{kind}.png")
    for kind in ("CHIN", "OPCID", "CHID"):
        copy(f"task1/saliency_{kind}.png")
        copy(f"task1/example_{kind}.png")
    for index, image in enumerate(sorted((SOURCE / "task3_labeled").glob("*.png"))):
        if index in (0, 5, 250, 450):
            shutil.copy2(image, DEST / f"task3_example_{index:03d}.png")
    for image in sorted((SOURCE / "task2" / "heatmaps").glob("*.png")):
        shutil.copy2(image, DEST / f"task2_{image.name}")
    for image in sorted((SOURCE / "task4_heat_final" / "heatmaps").glob("*.png")):
        shutil.copy2(image, DEST / f"task4_{image.name}")
    provenance = [hash_file(path) for pattern in ("*.cool.gz", "*.gff.gz", "*.xlsx")
                  for path in sorted((ROOT / "data/raw").glob(pattern))]
    provenance += [hash_file(ROOT / "data/structures.csv")]
    (DEST / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    t1 = json.loads((SOURCE / "task1/metrics.json").read_text(encoding="utf-8"))
    t1_trial = json.loads((SOURCE / "task1_calibrated/metrics.json").read_text(encoding="utf-8"))
    t1_scale8 = json.loads((SOURCE / "task1_scale8/metrics.json").read_text(encoding="utf-8"))
    t2 = json.loads((SOURCE / "task2/summary.json").read_text(encoding="utf-8"))
    t2_validation = json.loads((SOURCE / "task2_validation/summary.json").read_text(encoding="utf-8"))
    strict = json.loads((SOURCE / "task2_strict/summary.json").read_text(encoding="utf-8"))
    t5 = json.loads((DEST / "task5_summary.json").read_text(encoding="utf-8"))
    t5_validation = json.loads((DEST / "task5_validation.json").read_text(encoding="utf-8"))
    t6 = json.loads((SOURCE / "task6_independent/metrics.json").read_text(encoding="utf-8"))
    t6_regularized = json.loads((DEST / "task6_regularized_metrics.json").read_text(encoding="utf-8"))
    changes = pd.read_csv(SOURCE / "task4_heat_final/class_summary.csv")
    reproducible = pd.read_csv(SOURCE / "task4_heat_final/reproducible_changes.csv")
    merged_loci, right_edge = 0, -1
    for _, row in reproducible.sort_values("start").iterrows():
        if int(row.start) >= right_edge:
            merged_loci += 1
        right_edge = max(right_edge, int(row.end))
    rif_changes = pd.read_csv(SOURCE / "task4_rif_final/class_summary.csv")
    count_tracks = len(list((SOURCE / "task3_labeled").glob("*.png")))
    if count_tracks != 465:
        raise ValueError(f"Expected 465 whole-genome track images, found {count_tracks}")
    lines = ["# 真实数据运行结果", "",
        "数据：NCBI GEO GSE272159；结构注释：课程工作簿 Supplementary Tables 4/5/6。文件哈希见 `provenance.json`。",
        "所有原始矩阵和完整逐窗口图仍在本地 `data/raw/` 与 `outputs/`，本目录保留便于审阅的小型结果。", "",
        "## 任务一：已知结构识别", "",
        f"留出集准确率 {t1['accuracy']:.3f}，宏平均 F1 {t1['macro_f1']:.3f}；结构数训练/验证/测试为 "
        f"{t1['train_structures']}/{t1['validation_structures']}/{t1['test_structures']}。",
        f"CHID 精确率/召回率为 {t1['classification_report']['CHID']['precision']:.3f}/"
        f"{t1['classification_report']['CHID']['recall']:.3f}；样本少，需结合混淆矩阵阅读。", "",
        f"验证集类别阈值校准的额外试验使留出集准确率/宏 F1 降至 "
        f"{t1_trial['accuracy']:.3f}/{t1_trial['macro_f1']:.3f}。"
        f"10.24 kb 宽窗口实验验证集宏 F1 从 0.589 降至 "
        f"{max(row['validation_macro_f1'] for row in t1_scale8['history']):.3f}；"
        "两种变体均不采用。", "",
        "## 任务二：新结构候选", "",
        f"扫描 {t2['windows']} 个窗口，入选 {t2['selected']} 个；按当前操作性规则得到 "
        f"{t2['new_candidates']} 个未注释候选。已知结构跨重复筛选后召回 "
        f"{t2.get('known_structure_recall_after_reproducibility', float('nan')):.3f}。",
        f"较严格的敏感性对照阈值（前 20%、重复相关至少 0.3）产生 {strict['new_candidates']} 个新候选；"
        "对该参数选择应做进一步敏感性验证。",
        f"独立的 344 个已知中心与 344 个非重叠背景对照：按 100 kb 空间区段留出，rep1 训练、rep2 测试；"
        f"形态线性/随机森林 AUC 为 {t2_validation['auc']['shape_logistic']:.3f}/"
        f"{t2_validation['auc']['shape_forest']:.3f}，密度单特征为 "
        f"{t2_validation['auc']['density_logistic']:.3f}。",
        f"随机森林按空间组重采样的 95% 区间为 "
        f"{t2_validation['auc_spatial_group_bootstrap_95ci']['shape_forest'][0]:.3f}–"
        f"{t2_validation['auc_spatial_group_bootstrap_95ci']['shape_forest'][1]:.3f}。",
        f"5 个候选中，远距离错位和邻近错位零分布检验分别有 "
        f"{t2_validation['candidates_shift_null_q_below_005']} 和 "
        f"{t2_validation['candidates_local_shift_q_below_005']} 个通过 BH q<0.05。",
        "原全窗口重叠标签诊断 AUC 仅约 0.598；独立中心对照回答的是较窄、较明确的问题，不能抹去该局限。",
        "这些是计算候选，不是经实验确认的新结构。", "",
        "## 任务三：全基因组轨道", "",
        f"生成 {count_tracks} 张 10 kb 区间图，覆盖 WT 37℃、44℃ 和 rifampicin 25。", "",
        "## 任务四：实验条件变化", "",
        "原始计数在比较前进行二项抽稀，匹配两条件测序深度。44℃ 相对于 37℃ WT 的局部归一化强度差值：", "" ]
    for _, row in changes.iterrows():
        lines.append(f"- {row['type']}（{int(row['count'])} 个）：平均 {row['mean']:+.3f}；"
                     f"重复方向一致率 {row['repeat_direction_agreement_rate']:.1%}")
    lines += ["", f"按重复差异第 95 百分位、两重复方向一致和效应大于本位点重复波动的探索性规则，"
              f"有 {len(reproducible)} 条注释入选，合并重叠后为 {merged_loci} 个位点；"
              "清单见 `task4_reproducible_changes.csv`。"
              "该筛选不是正式的多重检验显著性结论。",
              "rifampicin 25 相对于 WT 的结果见 `task4_rif_class_summary.csv`；"
              "该条件原始深度仅约 WT 的 18%，必须匹配深度。"
              "未匹配深度的旧分析曾得到相反方向，故不作为结论。",
        "两条件各有两个生物学重复；WT 重复对照汇总见 `task4_repeat_class_summary.csv`。"
        "方向仅属于此局部强度指标，不等于整个结构形成或消失。", "",
        "## 任务五：超分辨", "",
        f"4 倍降分辨率（40 bp→10 bp）测试窗口 {t5['test_patches']} 个；PSNR 双线性/CNN 为 "
        f"{t5['comparison']['psnr']['bilinear']:.3f}/{t5['comparison']['psnr']['cnn']:.3f}，"
        f"SSIM 为 {t5['comparison']['ssim']['bilinear']:.3f}/{t5['comparison']['ssim']['cnn']:.3f}。",
        f"成对 PSNR/SSIM 提升 {t5_validation['test']['psnr']['mean_gain']:.3f}/"
        f"{t5_validation['test']['ssim']['mean_gain']:.3f}；已知结构中心 68 个窗口的 "
        f"SSIM 胜率 {t5_validation['known_structures']['ssim']['win_fraction']:.1%}。",
        "输入先在原始计数上合并 bin，再单独归一化；不使用目标分辨率的背景统计量。",
        "低分辨率输入由 Micro-C 合并 bin 模拟，不能外推到独立 Hi-C。", "",
        "## 任务六：三维重建", "",
        f"{t6['bins']} 个 {t6['bin_bp']} bp bin；接触与反距离 Spearman 相关为 "
        f"MLP {t6['mlp_contact_distance_spearman']:.3f}、MDS {t6['mds_contact_distance_spearman']:.3f}。",
        f"独立模型不使用 MDS 预训练或正则。另保留弱 MDS 正则的消融对照 "
        f"{t6_regularized['mlp_contact_distance_spearman']:.3f}，该对照不应称为独立基线。"
        "坐标并非唯一的真实物理构象。", ""]
    (DEST / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Collected {len(list(DEST.iterdir()))} review files in {DEST}")


if __name__ == "__main__": main()
