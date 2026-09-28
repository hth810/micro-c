# Micro-C 染色质结构分析（六项任务）

本仓库实现课程方案中的任务一至六。任务一使用有监督 CNN 和显著性图；任务二进行全基因组候选发现、形态表征、聚类和跨重复核验；任务三绘制全基因组多条件轨道；任务四定位条件变化；任务五进行接触矩阵超分辨重建；任务六输出三维坐标并与 MDS 比较。方法和验收边界见 [方法说明](METHODS.md)。

逐项交付核查、失败对照及课程正式提交尚缺的材料见 [六任务核查表](AUDIT.md)。从原始数据按依赖顺序运行的命令与断点续跑说明见 [一键复现指南](REPRODUCE.md)。当前最强的已知区间分类器是手工特征加梯度提升，不能称为纯深度学习；疑似新结构 N1 尚未获独立生物学验证；超分辨虽提升图像指标，结构热点定位仍不优于插值。

课程加分项的[离线辅助标注页](results/task2_annotation_review.html)展示 7 个候选位点的双重复热图、已知类 CNN 形态提示和人工复核导出。它不会自动把候选判定为新结构；模型对未注释区域属于分布外应用。报告 PDF 可用 `python scripts/build_report_pdf.py` 从 [Markdown 报告](REPORT.md)重建（另需 `reportlab>=4,<6` 及本机中文字体）；[已生成 PDF](output/pdf/microc_experiment_report.pdf)随分支保存。讲解视频的[分镜口播稿](submission/PRESENTATION_OUTLINE.md)供组员制作，不能代替录制完成的视频。

按课程要求，代码 ZIP、报告 PDF、PPT 讲解视频要分别提交；运行 `python scripts/build_submission_zip.py` 生成本地 `submission/micro-c-code.zip`，用 `python scripts/build_submission_zip.py --verify submission/micro-c-code.zip` 校验。ZIP 故意不上传 Git，避免与分支源码重复；提交时须按课程的“组号-姓名-学号”命名要求重命名三个文件。[组员贡献模板](submission/GROUP_CONTRIBUTIONS_TEMPLATE.md)尚待填写真实身份与百分比，录屏视频也须由小组完成，不能把模板或口播稿当作正式完成的提交物。

## 数据

原始矩阵来自 [NCBI GEO GSE272159](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE272159)。`data/structures.csv` 是从课程提供的 `标注数据.xlsx` 的 Supplementary Table 4/5/6 导出的 344 条标注，来源工作簿不进入 Git。下载脚本获取六个主分析矩阵：两个 37℃ 重复、两个 44℃ 重复、两个 rifampicin 25 重复；H-NS 独立条件检查另取两个缺失突变体重复。另获取一个基因注释 GFF 文件。

```bash
python -m pip install -r requirements.txt
python scripts/download_geo.py wt1 wt2 heat heat2 rif rif2 gff
python scripts/download_geo.py hns1 hns2
python scripts/prepare_annotations.py data/raw/标注数据.xlsx
```

工作簿在课程 PDF 给出的夸克分享中。`data/raw/` 为大文件目录，已在 `.gitignore` 中排除。`gff` 对应 NCBI 的 `GCF_000005845.2_ASM584v2_genomic.gff.gz`，供基因轨道使用。标注 CSV 已随仓库提供，因此复跑主要分析无需原工作簿。

## 运行

以下命令在仓库根目录执行。所有生成结果写入被 Git 忽略的 `outputs/`。

```bash
python task1/run_task1.py --data-dir data/raw --annotations data/structures.csv --out-dir outputs/task1_spatial
python -m task1.saliency_audit --run-dir outputs/task1_spatial

python scripts/audit_task1_split.py outputs/task1_spatial/annotation_manifest.csv --out outputs/task1_spatial/split_audit.json

python task1/run_task1.py --annotations data/structures.csv --aggregate 4 --split-gap-bp 10240 --out-dir outputs/task1_spatial_gap10k_scale4

python task1/run_task1.py --annotations data/structures.csv --aggregate 8 --out-dir outputs/task1_spatial_scale8

python task1/run_task1.py --annotations data/structures.csv --spatial-grid 4 --out-dir outputs/task1_spatial_grid4

python -m task1.evaluate_chid_overlap --windows outputs/task1_spatial/extracted_windows.npz --annotations data/structures.csv --out outputs/task1_chid_overlap

python -m scripts.summarize_chid_overlap --predictions outputs/task1_chid_overlap/crossfit_predictions.csv --out outputs/task1_chid_overlap/bootstrap.json

python -m task1.evaluate_pairwise --windows outputs/task1_spatial/extracted_windows.npz --annotations data/structures.csv --out outputs/task1_pairwise

python -m task1.prepare_multiscale --short-windows outputs/task1_spatial/extracted_windows.npz --rep1 data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --rep2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool.gz --out outputs/task1_multiscale/windows.npz

python -m task1.evaluate_multiscale --windows outputs/task1_multiscale/windows.npz --out outputs/task1_multiscale_eval

python -m scripts.summarize_task1_cv --predictions outputs/task1_multiscale_eval/crossfit_predictions.csv --out outputs/task1_multiscale_eval/uncertainty.json

python -m task1.train_final_groupnorm --windows outputs/task1_spatial/extracted_windows.npz --out outputs/task1_groupnorm_final.pt

python -m task1.evaluate_length_context --out outputs/task1_length_context

python -m task1.evaluate_handcrafted --out outputs/task1_handcrafted

python -m task1.evaluate_nested_threshold --out outputs/task1_nested_threshold

python -m scripts.summarize_task1_cv --predictions outputs/task1_handcrafted/crossfit_predictions.csv --mode-column model --resampling-unit "non-overlapping 40 kb genomic component" --out outputs/task1_handcrafted/uncertainty.json

python -m task1.train_final_handcrafted --out outputs/task1_handcrafted_final.joblib

python -m task1.predict_handcrafted --rep1 data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --rep2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool.gz --annotations data/structures.csv --model outputs/task1_handcrafted_final.joblib --out outputs/task1_handcrafted_in_sample_smoke.csv

python -m task1.evaluate_context_features --out outputs/task1_context_features
python -m task1.evaluate_scale_ensemble --out outputs/task1_scale_ensemble
python -m task1.evaluate_interval_boundaries --out outputs/task1_interval_boundaries
python -m scripts.summarize_task1_cv --predictions outputs/task1_interval_boundaries/crossfit_predictions.csv --resampling-unit "non-overlapping 40 kb genomic component" --out outputs/task1_interval_boundaries/uncertainty.json
python -m scripts.compare_task1_boundary --out outputs/task1_interval_boundaries/paired_uncertainty.json
python -m task1.train_final_boundaries --out outputs/task1_boundary_final.joblib
python -m task1.predict_handcrafted --rep1 data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --rep2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool.gz --annotations data/structures.csv --model outputs/task1_boundary_final.joblib --out outputs/task1_boundary_in_sample_smoke.csv
python -m task1.evaluate_replicate_transfer --out outputs/task1_replicate_transfer
python -m task1.evaluate_balanced_cnn --out outputs/task1_balanced_cnn

python task1/run_task1.py --data-dir data/raw --annotations data/structures.csv --split-mode structure --out-dir outputs/task1

python scripts/audit_task1_split.py outputs/task1/annotation_manifest.csv --legacy --out outputs/task1_legacy_split_audit.json

python task1/run_task1.py --data-dir data/raw --annotations data/structures.csv --split-mode structure --calibrate --out-dir outputs/task1_calibrated

python task1/run_task1.py --data-dir data/raw --annotations data/structures.csv --split-mode structure --aggregate 8 --out-dir outputs/task1_scale8

python -m microc.analysis discover --rep1 data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --rep2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool.gz --annotations data/structures.csv --out outputs/task2

python -m microc.analysis discover --rep1 data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --rep2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool.gz --annotations data/structures.csv --top-fraction 0.2 --min-replicate-r 0.3 --out outputs/task2_strict_current

python -m microc.analysis discover --rep1 data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --rep2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool.gz --annotations data/structures.csv --top-fraction 0.2 --min-replicate-r 0.3 --candidate-rule individual --out outputs/task2_strict_revised

python -m scripts.audit_candidate_sensitivity --scan outputs/task2/all_windows.csv --candidates outputs/task2/new_candidates.csv --out outputs/task2_sensitivity

python -m scripts.prepare_individual_candidates --discovery outputs/task2_strict_current --out outputs/task2_strict_individual.csv

python -m microc.analysis tracks --cool data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz data/raw/GSE272159_44C_rep1.mapq_30.10.cool.gz data/raw/GSE272159_rif25_rep1.MG1655.mapq_30.10.cool.gz --labels WT_37C Heat_44C Rif_25 --annotations data/structures.csv --gff data/raw/GCF_000005845.2_ASM584v2_genomic.gff.gz --out outputs/task3_labeled
python -m scripts.manifest_tracks --tracks outputs/task3_labeled --out results/task3_track_manifest.csv

python -m microc.analysis differential --wt data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --wt2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool.gz --condition data/raw/GSE272159_44C_rep1.mapq_30.10.cool.gz --condition2 data/raw/GSE272159_44C_rep2.mapq_30.10.cool.gz --name heat_44c --annotations data/structures.csv --seed 20260921 --heatmaps 50 --out outputs/task4_heat_order_invariant

python -m microc.analysis differential --wt data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --wt2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool.gz --condition data/raw/GSE272159_rif25_rep1.MG1655.mapq_30.10.cool.gz --condition2 data/raw/GSE272159_rif25_rep2.MG1655.mapq_30.10.cool.gz --name rifampicin_25 --annotations data/structures.csv --seed 20260921 --out outputs/task4_rif_order_invariant

以上两个条件须用 `--seed 20260922` 和 `20260923` 各再运行一次，输出目录依次加 `_seed20260922`、`_seed20260923`；随后运行 `python -m scripts.summarize_task4_thinning`，主运行和三种子交集才构成任务四的当前结果。旧 `task4_heat_final` / `task4_rif_final` 只作历史对照。

python -m microc.analysis differential --wt data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --condition data/raw/GSE272159_37C_rep2.mapq_30.10.cool.gz --name wt_rep2_control --annotations data/structures.csv --out outputs/task4_repeat_depth_final

python -m microc.deep superresolve --cool data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --annotations data/structures.csv --factor 4 --epochs 12 --out outputs/task5_corrected
python -m scripts.validate_sr_known_shape --cool data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --checkpoint outputs/task5_corrected/super_resolution.pt --known-file outputs/task5_corrected/known_structure_recovery.csv --out outputs/task5_known_shape
python -m microc.deep superresolve --cool data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --annotations data/structures.csv --factor 4 --epochs 12 --peak-weight 0.5 --out outputs/task5_peak_weighted
python -m scripts.validate_sr_known_shape --cool data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --checkpoint outputs/task5_peak_weighted/super_resolution.pt --known-file outputs/task5_peak_weighted/known_structure_recovery.csv --out outputs/task5_peak_known_shape
python -m scripts.evaluate_sr_ensemble --out outputs/task5_ensemble

python -m microc.deep reconstruct3d --cool data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --annotations data/structures.csv --pretrain-epochs 0 --anchor-weight 0 --out outputs/task6_independent
python -m microc.deep reconstruct3d --cool data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --annotations data/structures.csv --pretrain-epochs 0 --anchor-weight 0 --contact-quantile 0 --out outputs/task6_all_positive
python -m scripts.validate_3d_replicate --rep2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool.gz
python -m scripts.annotate_3d_domains --coordinates outputs/task6_all_positive/coordinates.csv --out outputs/task6_domain_overlay
```

`.cool.gz` 会临时解压后读取。重复运行时可自行保留同名 `.cool` 并将参数指向解压后的文件，避免反复解压。

任务二与任务五的独立验证：

```bash
python -m scripts.validate_discovery --rep1 data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --rep2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool.gz --annotations data/structures.csv --discovery outputs/task2 --out outputs/task2_validation
python -m scripts.validate_discovery --rep1 data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --rep2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool.gz --annotations data/structures.csv --discovery outputs/task2_strict_current --candidate-file outputs/task2_strict_individual.csv --out outputs/task2_strict_individual_validation
python -m scripts.validate_candidates_conditions --candidates outputs/task2_strict_individual.csv --heat1 data/raw/GSE272159_44C_rep1.mapq_30.10.cool.gz --heat2 data/raw/GSE272159_44C_rep2.mapq_30.10.cool.gz --rif1 data/raw/GSE272159_rif25_rep1.MG1655.mapq_30.10.cool.gz --rif2 data/raw/GSE272159_rif25_rep2.MG1655.mapq_30.10.cool.gz --out outputs/task2_strict_individual_conditions.csv
python -m scripts.summarize_strict_candidates --loci outputs/task2_strict_individual_loci.csv --windows outputs/task2_strict_individual.csv --conditions outputs/task2_strict_individual_conditions.csv --wt-null outputs/task2_strict_individual_validation/candidate_shift_null.csv --out outputs/task2_strict_locus_evidence.csv
python -m scripts.render_individual_candidates --loci outputs/task2_strict_individual_loci.csv --rep1 data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --rep2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool.gz --out outputs/task2_strict_individual_heatmaps
python -m scripts.build_annotation_review --rep1 data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --rep2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool.gz
python -m scripts.validate_novel_cluster --rep1 data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --rep2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool.gz --out outputs/task2_novel_cluster
python -m scripts.validate_novel_condition_transfer --rep1 data/raw/GSE272159_44C_rep1.mapq_30.10.cool.gz --rep2 data/raw/GSE272159_44C_rep2.mapq_30.10.cool.gz --name 44C --out outputs/task2_novel_cluster/transfer_44c.json
python -m scripts.validate_novel_condition_transfer --rep1 data/raw/GSE272159_rif25_rep1.MG1655.mapq_30.10.cool.gz --rep2 data/raw/GSE272159_rif25_rep2.MG1655.mapq_30.10.cool.gz --name rif25 --out outputs/task2_novel_cluster/transfer_rif25.json
python -m scripts.validate_novel_condition_transfer --rep1 data/raw/GSM9678478_HNS_mut_rep1.MG1655.mapq_30.10.cool.gz --rep2 data/raw/GSM9678479_HNS_mut_rep2.MG1655.mapq_30.10.cool.gz --name HNS_mut --out outputs/task2_novel_cluster/transfer_hns.json
python -m scripts.validate_hns_candidates --wt1 data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --wt2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool.gz --hns1 data/raw/GSM9678478_HNS_mut_rep1.MG1655.mapq_30.10.cool.gz --hns2 data/raw/GSM9678479_HNS_mut_rep2.MG1655.mapq_30.10.cool.gz --out outputs/task2_hns_validation
python scripts/collect_results.py
python -m scripts.evaluate_saved_sr --bundle results
python scripts/collect_results.py
python -m scripts.check_deliverables --out results/quality_audit.json
```

## 验证

```bash
python -m pytest -q tests
```

测试使用合成 `.cool` 矩阵覆盖读取、归一化、聚类、轨道、条件变化、超分辨和三维重建，不代表真实数据性能。真实结果必须以 `results/` 中实际运行的指标、图表与数据哈希为准。任务四在原始计数上匹配测序深度；任务五以合并 bin 构造模拟低分辨率输入，不等同于独立 Hi-C 实验；任务六的三维坐标不能单凭接触一致性视作真实物理形态。
