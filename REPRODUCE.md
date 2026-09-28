# 六项任务一键复现

在仓库根目录执行（Windows PowerShell、Linux/macOS 均可）：

```powershell
python -m pip install -r requirements.txt
python scripts/reproduce_submission.py --dry-run
python scripts/reproduce_submission.py --download --workers 4
```

首行仅供全新环境安装依赖；当前环境若已装好，无须重复安装。复现脚本自身不会安装软件。

最后一条命令会检查或从官方 GEO 下载本仓库需要的 8 个 Micro-C 矩阵及 1 个基因 GFF，然后按依赖顺序执行 33 步，覆盖任务一至六，包括任务二的新候选严格筛选、跨条件核查与离线审阅页面、任务四的三个测序深度抽样种子、任务五的双模型对照和任务六的重复样本验证。原始数据默认放在 `data/raw/`，注释使用已随仓库提供的 `data/structures.csv`。如已将 `.cool.gz` 解压为同名 `.cool`，脚本优先使用 `.cool`，免去反复解压。可用 `--data-dir D:\path\to\raw` 指定其他数据目录。

`--dry-run` 只列出命令、缺失输入，不下载、不训练、不改文件；正式运行很耗时，尤其是全基因组轨道图、候选验证和神经网络训练。脚本逐步运行，失败即停，并核查每步关键输出。结果主要写入被 Git 忽略的 `outputs/`，任务三的 465 张轨道图清单写入 `results/task3_track_manifest.csv`。脚本不会清理既有输出，因此重跑前应自行保留需要比较的版本。程序和模型有固定默认随机种子，但不同硬件或依赖版本的神经网络数值可能有小幅差异。

中断后可以从指定任务或步骤继续，前序产物须已存在：

```powershell
python scripts/reproduce_submission.py --from-stage 4 --through-stage 6
python scripts/reproduce_submission.py --from-stage 5 --through-stage 5 --start-at task5-train-peak
python scripts/reproduce_submission.py --dry-run --from-stage 2 --through-stage 2
```

步骤名由 `--dry-run` 列出。只跑某个后段任务时，可通过 `--from-stage N --through-stage N` 缩小范围。`--download` 会对使用的压缩 GEO 文件核对远端大小并记录 SHA-256；直接提供本地 `.cool` 时视其为用户自行验证的数据。

这是一条从原始矩阵重建六项**核心分析**的流水线，不是“结果均已得到独立实验确认”的保证。任务二的新结构仍为计算候选；任务四是敏感于深度抽样的启发式结果，不是正式 FDR；任务五使用模拟低分辨率输入；任务六的三维形态不可仅靠接触图一致性认定为真实物理构象。具体结果、失败对照与边界见 `REPORT.md`、`AUDIT.md` 和 `results/`。`scripts/collect_results.py` 用于维护现有的完整展示包，还引用额外消融实验，不属于此处从零开始的核心流水线。
