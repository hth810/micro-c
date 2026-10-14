# 六项任务一键复现

## 最终完整区间与几何验证

最终补强只作用于本仓库 Micro-C，不使用 GvpA/ESM-2 数据。保留原六任务核心流水线，再执行以下独立扩展；当前已安装依赖，不需重复安装。

```powershell
python -m scripts.reproduce_final_verification_20261014 --out outputs/replay_final_20261014 --dry-run
python -m scripts.reproduce_final_verification_20261014 --out outputs/replay_final_20261014
python -m scripts.verify_final_package_20261014
```

`--out` 必须尚不存在。新输入后的结果发布到该目录内部的 `published_results`，不会覆盖仓库当前冻结结果。复跑依赖两个 WT COOL、已有面积发现包 `outputs/task2_improve_20261006/area`、任务一多尺度窗口及已发布的最终识别器/局部 SR 权重和三维坐标包；对应前置构建命令见下文。Task2 显式使用随仓库提供的 `data/structures.csv`，与原工作簿科学内容相同但文件哈希不同，记录实际输入版本。

扩展依次执行完整无偏背景和尺度验证、覆盖缺口反事实/三个共同深度、固定识别器辅助推理、训练区全窗尺度拟合、完整区间 CNN 的验证选择及冻结测试、全部 136 预测重算和图库、最新三维坐标几何、整套文献区域版本及第一声明种子双视角。Task5 参数范围为 2/4/6 轮 × 四种融合比例，只用 validation 选择；Task6 不用宏域标签训练或挑选种子。历史 test/rep2 已被查看，不称新前瞻盲测。

最终主结果分别位于 `task2_course_verification_20261014_*`、`task5_full_structure_20261014_final_*`、`task6_full_geometry_20261014_*` 和 `task6_literature_regions_20261014_*`。Task5 全部 136 个 10 bp 预测约 777 MB，在本地 `outputs/task5_full_structure_20261014/neural_adaptation/`，逐文件哈希见 `final_all_case_manifest.json`，小包不携带这些可重新生成的大数组。旧 640 bp 局部指标不用于评价新全窗模型。Task2 共同深度主证据使用 QC 的 10000/20000/30000 档；原首轮 34548 档对 A2 rep2 缺共同支持，不作为等深度主证据。

上述检查分别记录产物完整性、课程目标和新类型证据。最终结果为三个跨重复形态异常候选、完整区间主要形态与轮廓的稳定重建，以及宏域相关的总体距离组织；共同新类型、精细像素恢复和各分域表现各有独立评价结果。

## 最终报告复建

报告源模板、最终内容配方、语言修订和加分项 LLM 证据均随代码小包提供。现有环境已具备依赖，复建使用新的输出文件，不覆盖已验收的 Word：

```powershell
python -m scripts.update_final_report_20261014 --source output/report/support_final_verification_20261014/report_reference.docx --patch output/report/support_final_verification_20261014/report_patch.json --out output/report/MicroC_report_rebuilt.docx --receipt output/report/rebuilt_receipt.json
```

Windows 上使用 `scripts/render_final_report_20261014.ps1` 导出独立副本并生成逐页 PNG；参数指定 `-Source`、新的 `-OutputDirectory` 和已有 Poppler 的 `-PopplerDirectory`。脚本使用独立最小化 Word 实例，仅关闭自身临时副本，保留用户其他文档。最终验收记录见 `results/final_report_20261014_layout_audit.json`，已交付 Word/PDF 与代码 ZIP 分开提交。

在仓库根目录执行（Windows PowerShell、Linux/macOS 均可）：

```powershell
python -m pip install -r requirements.txt
python scripts/reproduce_submission.py --dry-run
python scripts/reproduce_submission.py --download --workers 4
```

首行仅供全新环境安装依赖；当前环境若已装好，无须重复安装。复现脚本自身不会安装软件。

## 本轮追加核查（不覆盖上轮冻结结果）

以下命令使用独立新输出路径。需要上轮任务一窗口包及任务二面积发现包、两个原始WT矩阵；参考FASTA不存在时，唯一性脚本从官方NCBI下载并验证MD5。当前环境无需重新安装依赖。新检出中的小结果可直接核查，不必下载原始COOL：

```powershell
python -m scripts.verify_improvements_20261006
python -m scripts.verify_followups_20261006
python -m pytest -q tests
```

从数据重跑（应使用尚不存在的输出目录，不覆盖本机冻结文件）：

```powershell
python -m task1.evaluate_roi_transfer_followup --out outputs/replay_task1_transfer
python -m scripts.task2_reference_uniqueness_followup --download --out outputs/replay_reference --results outputs/replay_reference_published
python -m scripts.task2_candidate_artifact_followup --out outputs/replay_task2_artifact --results outputs/replay_task2_published
python -m scripts.task2_artifact_addendum_followup --results outputs/replay_task2_published
python -m scripts.task6_zero_residual_followup --out outputs/replay_task6_ranking
python -m scripts.task6_zero_residual_controlled_followup --out outputs/replay_task6_controlled
python -m scripts.audit_raw_read_availability_followup --out outputs/replay_raw_availability
```

上面重跑命令不向已带冻结结果的 `results/` 覆盖发布：任务一/六不传 `--publish`，任务二用独立 `--results`，补充图再读取该新目录。`replay_*` 目录也须尚不存在；再次重跑应换新名称。任务六最后一项隔离排序项随机流，使正接触minibatch与旧训练严格一致；参数不因第一次追加实验的rep2结果改变。两个追加实验都保留，历史rep2已被查看，不冒称新盲测。大型排序配对表只在本地outputs保存并记录哈希；可由固定rep1和种子重新生成。任务二默认[-5,5]与额外截断敏感性分开，参考唯一性和原生接触覆盖仅为风险诊断，不能补足原始reads或新实验。

最后的raw-availability命令只查询四个WT37℃ t1/t2 run的GEO/ENA/SRA metadata；加 `--sample-fastq-prefix` 时，每个配对文件最多取64KiB压缩前缀，只检验前100完整记录，不核验全FASTQ的MD5，也不证明全run读长分布。发布的[小包审计](results/task2_followup_20261006_raw_read_availability_audit_index.json)另列源码/响应/输出哈希，不修改六份实验索引或75项完整性记录。原始reads可获取，但raw-to-current-COOL处理manifest仍未明确，未开始约198GB WT压缩reads的全量重比对。

## 上轮改良实验

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
