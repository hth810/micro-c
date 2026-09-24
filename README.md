# Micro-C 染色质结构分析（六项任务）

本仓库实现课程方案中的任务一至六。任务一使用有监督 CNN 和显著性图；任务二进行全基因组候选发现、形态表征、聚类和跨重复核验；任务三绘制全基因组多条件轨道；任务四定位条件变化；任务五进行接触矩阵超分辨重建；任务六输出三维坐标并与 MDS 比较。方法和验收边界见 [方法说明](METHODS.md)。

## 数据

原始矩阵来自 [NCBI GEO GSE272159](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE272159)。`data/structures.csv` 是从课程提供的 `标注数据.xlsx` 的 Supplementary Table 4/5/6 导出的 344 条标注，来源工作簿不进入 Git。下载脚本获取六个矩阵：两个 37℃ 重复、两个 44℃ 重复、两个 rifampicin 25 重复；另获取一个基因注释 GFF 文件。

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

python task1/run_task1.py --data-dir data/raw --annotations data/structures.csv --split-mode structure --out-dir outputs/task1

python scripts/audit_task1_split.py outputs/task1/annotation_manifest.csv --legacy --out outputs/task1_legacy_split_audit.json

python task1/run_task1.py --data-dir data/raw --annotations data/structures.csv --split-mode structure --calibrate --out-dir outputs/task1_calibrated

python task1/run_task1.py --data-dir data/raw --annotations data/structures.csv --split-mode structure --aggregate 8 --out-dir outputs/task1_scale8

python -m microc.analysis discover --rep1 data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --rep2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool.gz --annotations data/structures.csv --out outputs/task2

python -m microc.analysis discover --rep1 data/raw/GSE272159_37C_rep1.mapq_30.10.cool.gz --rep2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool.gz --annotations data/structures.csv --top-fraction 0.2 --min-replicate-r 0.3 --out outputs/task2_strict

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
python scripts/collect_results.py
python -m scripts.evaluate_saved_sr --bundle results
python scripts/collect_results.py
```

## 验证

```bash
python -m pytest -q tests
```

测试使用合成 `.cool` 矩阵覆盖读取、归一化、聚类、轨道、条件变化、超分辨和三维重建，不代表真实数据性能。真实结果必须以 `results/` 中实际运行的指标、图表与数据哈希为准。任务四在原始计数上匹配测序深度；任务五以合并 bin 构造模拟低分辨率输入，不等同于独立 Hi-C 实验；任务六的三维坐标不能单凭接触一致性视作真实物理形态。
