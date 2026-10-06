# 六项任务一键复现

在仓库根目录执行（Windows PowerShell、Linux/macOS 均可）：

```powershell
python -m pip install -r requirements.txt
python scripts/reproduce_submission.py --dry-run
python scripts/reproduce_submission.py --download --workers 4
```

首行仅供全新环境安装依赖；当前环境若已装好，无须重复安装。复现脚本自身不会安装软件。

## 2026-10-06 改良实验补充

上面的37步是历史核心流水线，不自动运行本节新实验。新实验全部只属于 `hth810/micro-c`，不使用GvpA/ESM-2数据。冻结协议拒绝覆盖，以下输出目录在本机已存在；从零复现应在新检出仓库运行，或为每个命令显式指定新的 `--out` 并同步后续输入路径，**不要删除或覆盖现有冻结文件**。

任务一前提是 `outputs/task1_multiscale/windows.npz`，可由既有 `prepare_multiscale` 命令生成；两标准WT重复、344个已知区间，短/长窗口原始计数聚合到128×128，并各自归一化。它不是任务二的旧插值测量。改良命令：

```powershell
python -m task1.evaluate_nested_neural_20261006 --seed 20260921 --out outputs/task1_improve_20261006/nested_neural_seed21
python -m task1.evaluate_roi_cnn_20261006 --seed 20260921 --out outputs/task1_improve_20261006/roi_cnn_seed21
python -m task1.evaluate_roi_cnn_20261006 --seed 20260922 --out outputs/task1_improve_20261006/roi_cnn_seed22
python -m task1.evaluate_hybrid_20261006 --phase freeze --seed 20260921 --cnn outputs/task1_improve_20261006/roi_cnn_seed21 --out outputs/task1_improve_20261006/hybrid_seed21
python -m task1.evaluate_hybrid_20261006 --phase score --seed 20260921 --cnn outputs/task1_improve_20261006/roi_cnn_seed21 --out outputs/task1_improve_20261006/hybrid_seed21
python -m task1.evaluate_hybrid_20261006 --phase freeze --seed 20260922 --cnn outputs/task1_improve_20261006/roi_cnn_seed22 --out outputs/task1_improve_20261006/hybrid_seed22
python -m task1.evaluate_hybrid_20261006 --phase score --seed 20260922 --cnn outputs/task1_improve_20261006/roi_cnn_seed22 --out outputs/task1_improve_20261006/hybrid_seed22
python -m task1.evaluate_roi_ablation_20261006
python -m task1.audit_roi_cnn_20261006
python -m task1.publish_improvement_20261006
python -m task1.predict_roi_cnn_20261006
```

最后一步只核对全标注推理完整性，不作为泛化评分。混合对照冻结在CNN结果后，属于回顾性辅助比较；显著图/遮挡来自外折权重，最终部署权重不能用于留出归因。

任务二沿用仓库 `results/task2_all_windows.csv` 中仅rep1派生的代表点优先级，重新读取全部原生窗口。先完成两个发现臂并冻结，再运行验证；历史rep2已经查看，不称前瞻盲测：

```powershell
python -m scripts.task2_residual_motif_screen_20261006 discover --rep1 data/raw/GSE272159_37C_rep1.mapq_30.10.cool --resampling legacy --out outputs/task2_improve_20261006/legacy
python -m scripts.task2_residual_motif_screen_20261006 discover --rep1 data/raw/GSE272159_37C_rep1.mapq_30.10.cool --resampling area --out outputs/task2_improve_20261006/area
python -m scripts.task2_residual_motif_screen_20261006 validate --rep2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool --discovery outputs/task2_improve_20261006/legacy --out outputs/task2_improve_20261006/legacy/validation
python -m scripts.task2_residual_motif_screen_20261006 validate --rep2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool --discovery outputs/task2_improve_20261006/area --out outputs/task2_improve_20261006/area/validation
python -m scripts.task2_resampling_comparison_20261006 --rep2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool
python -m scripts.task2_publish_improvement_20261006 --rep2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool
```

`.cool.gz` 也可传给读取接口。任务六最终发表的是完整循环链的第三次运行；前两次为保留在本地的修正前诊断，不混入结果包：

```powershell
python -m scripts.task6_circular_sensitivity --rep1 data/raw/GSE272159_37C_rep1.mapq_30.10.cool --rep2 data/raw/GSE272159_37C_rep2.mapq_30.10.cool --out outputs/task6_circular_20261006_003
python -m scripts.verify_improvements_20261006
python -m scripts.check_deliverables
python -m pytest -q tests
```

`results/improvements_20261006_integrity.json` 用于验证发表的小结果和实际源代码字节。全新运行的时间、权重序列化和硬件可能产生不同哈希，不能拿旧receipt声称新运行逐字节一致；应保留新receipt与旧版本并列。`.gitattributes` 为冻结源码和结果关闭换行转换，少量历史helper的纯换行差异是为了保存运行时哈希，不代表新增算法修改。原始大型COOL不上传Git。

前面的 `python scripts/reproduce_submission.py --download --workers 4` 会检查或从官方 GEO 下载本仓库需要的 8 个 Micro-C 矩阵及 1 个基因 GFF，然后按依赖顺序执行 37 步，覆盖任务一至六，包括纯 CNN 验证选模失败对照、任务二全候选配对跨条件审计、任务四三个测序深度抽样种子、任务五的双模型与局部融合复核、任务六的重复样本和宏区域标签平移检验。原始数据默认放在 `data/raw/`，注释使用已随仓库提供的 `data/structures.csv`。如已将 `.cool.gz` 解压为同名 `.cool`，脚本优先使用 `.cool`，免去反复解压。可用 `--data-dir D:\path\to\raw` 指定其他数据目录。

`--dry-run` 只列出命令、缺失输入，不下载、不训练、不改文件；正式运行很耗时，尤其是全基因组轨道图、候选验证和神经网络训练。脚本逐步运行，失败即停，并核查每步关键输出。结果主要写入被 Git 忽略的 `outputs/`，任务三的 465 张轨道图清单写入 `results/task3_track_manifest.csv`。脚本不会清理既有输出，因此重跑前应自行保留需要比较的版本。程序和模型有固定默认随机种子，但不同硬件或依赖版本的神经网络数值可能有小幅差异。

中断后可以从指定任务或步骤继续，前序产物须已存在：

```powershell
python scripts/reproduce_submission.py --from-stage 4 --through-stage 6
python scripts/reproduce_submission.py --from-stage 5 --through-stage 5 --start-at task5-train-peak
python scripts/reproduce_submission.py --dry-run --from-stage 2 --through-stage 2
```

步骤名由 `--dry-run` 列出。只跑某个后段任务时，可通过 `--from-stage N --through-stage N` 缩小范围。`--download` 会对使用的压缩 GEO 文件核对远端大小并记录 SHA-256；直接提供本地 `.cool` 时视其为用户自行验证的数据。

这是一条从原始矩阵重建六项**核心分析**和新审计的流水线，不是“结果均已得到独立实验确认”的保证。任务二的 N1 仅为待验证形态候选对，旧 0.0002 经验 p 因对照不匹配不可作显著性；任务四是敏感于深度抽样的启发式结果，不是正式 FDR；任务五使用模拟低分辨率输入且热点增益区间跨 0；任务六的三维形态不可仅靠接触图或宏区域标签关联认定为真实物理构象。具体结果、失败对照与边界见 `REPORT.md`、`AUDIT.md` 和 `results/`。`scripts/collect_results.py` 用于维护现有的完整展示包，还引用额外消融实验，不属于此处从零开始的核心流水线。
