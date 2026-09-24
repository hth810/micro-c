# Micro-C 染色质结构分析（六项任务）

本仓库实现课程方案中的任务一至六。任务一使用有监督 CNN 和显著性图；任务二进行全基因组候选发现、形态表征、聚类和跨重复核验；任务三绘制全基因组多条件轨道；任务四定位条件变化；任务五进行接触矩阵超分辨重建；任务六输出三维坐标并与 MDS 比较。方法和验收边界见 [方法说明](METHODS.md)。

## 数据

原始矩阵来自 [NCBI GEO GSE272159](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE272159)。`data/structures.csv` 是从课程提供的 `标注数据.xlsx` 的 Supplementary Table 4/5/6 导出的 344 条标注，来源工作簿不进入 Git。下载脚本获取六个主分析矩阵：两个 37℃ 重复、两个 44℃ 重复、两个 rifampicin 25 重复；H-NS 独立条件检查另取两个缺失突变体重复。另获取一个基因注释 GFF 文件。

```bash
python -m pip install -r requirements.txt
python scripts/download_geo.py wt1 wt2 heat heat2 rif rif2 gff
python scripts/prepare_annotations.py data/raw/标注数据.xlsx
```

工作簿在课程 PDF 给出的夸克分享中。`data/raw/` 为大文件目录，已在 `.gitignore` 中排除。`gff` 对应 NCBI 的 `GCF_000005845.2_ASM584v2_genomic.gff.gz`，供基因轨道使用。标注 CSV 已随仓库提供，因此复跑主要分析无需原工作簿。

## 运行

以下命令在仓库根目录执行。所有生成结果写入被 Git 忽略的 `outputs/`。

```bash
python task1/run_task1.py --data-dir data/raw --annotations data/structures.csv --out-dir outputs/task1_spatial

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

python -m microc.analysis differential --wt data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --wt2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool.gz --condition data/raw/GSE272159_44C_rep1.mapq_30.10.cool.gz --condition2 data/raw/GSE272159_44C_rep2.mapq_30.10.cool.gz --name heat_44c --annotations data/structures.csv --heatmaps 60 --out outputs/task4_heat_final

python -m microc.analysis differential --wt data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --wt2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool.gz --condition data/raw/GSE272159_rif25_rep1.MG1655.mapq_30.10.cool.gz --condition2 data/raw/GSE272159_rif25_rep2.MG1655.mapq_30.10.cool.gz --name rifampicin_25 --annotations data/structures.csv --out outputs/task4_rif_final

python -m microc.analysis differential --wt data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --condition data/raw/GSE272159_37C_rep2.mapq_30.10.cool.gz --name wt_rep2_control --annotations data/structures.csv --out outputs/task4_repeat_depth_final

python -m microc.deep superresolve --cool data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --annotations data/structures.csv --factor 4 --epochs 12 --out outputs/task5_corrected

python -m microc.deep reconstruct3d --cool data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --annotations data/structures.csv --pretrain-epochs 0 --anchor-weight 0 --out outputs/task6_independent
```

`.cool.gz` 会临时解压后读取。重复运行时可自行保留同名 `.cool` 并将参数指向解压后的文件，避免反复解压。

任务二与任务五的独立验证：

```bash
python -m scripts.validate_discovery --rep1 data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --rep2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool.gz --annotations data/structures.csv --discovery outputs/task2 --out outputs/task2_validation
python -m scripts.validate_discovery --rep1 data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --rep2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool.gz --annotations data/structures.csv --discovery outputs/task2_strict_current --candidate-file outputs/task2_strict_individual.csv --out outputs/task2_strict_individual_validation
python -m scripts.validate_candidates_conditions --candidates outputs/task2_strict_individual.csv --heat1 data/raw/GSE272159_44C_rep1.mapq_30.10.cool.gz --heat2 data/raw/GSE272159_44C_rep2.mapq_30.10.cool.gz --rif1 data/raw/GSE272159_rif25_rep1.MG1655.mapq_30.10.cool.gz --rif2 data/raw/GSE272159_rif25_rep2.MG1655.mapq_30.10.cool.gz --out outputs/task2_strict_individual_conditions.csv
python -m scripts.summarize_strict_candidates --loci outputs/task2_strict_individual_loci.csv --windows outputs/task2_strict_individual.csv --conditions outputs/task2_strict_individual_conditions.csv --wt-null outputs/task2_strict_individual_validation/candidate_shift_null.csv --out outputs/task2_strict_locus_evidence.csv
python -m scripts.render_individual_candidates --loci outputs/task2_strict_individual_loci.csv --rep1 data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --rep2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool.gz --out outputs/task2_strict_individual_heatmaps
python -m scripts.download_geo hns1 hns2
python -m scripts.validate_hns_candidates --wt1 data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --wt2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool.gz --hns1 data/raw/GSM9678478_HNS_mut_rep1.MG1655.mapq_30.10.cool.gz --hns2 data/raw/GSM9678479_HNS_mut_rep2.MG1655.mapq_30.10.cool.gz --out outputs/task2_hns_validation
python scripts/collect_results.py
python -m scripts.evaluate_saved_sr --bundle results
python scripts/collect_results.py
```

## 验证

```bash
python -m pytest -q tests
```

测试使用合成 `.cool` 矩阵覆盖读取、归一化、聚类、轨道、条件变化、超分辨和三维重建，不代表真实数据性能。真实结果必须以 `results/` 中实际运行的指标、图表与数据哈希为准。任务四在原始计数上匹配测序深度；任务五以合并 bin 构造模拟低分辨率输入，不等同于独立 Hi-C 实验；任务六的三维坐标不能单凭接触一致性视作真实物理形态。
