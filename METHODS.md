# Micro-C 最终实验方法与评价规则

本文件描述最终交付方法，更新于 2026 年 10 月 14 日。结果见 [REPORT.md](REPORT.md)，逐项核查见 [AUDIT.md](AUDIT.md)，运行入口见 [REPRODUCE.md](REPRODUCE.md)。历史试验方法原文保存在[方法归档](submission/archive/final_verification_20261014/METHODS_before_final_verification.md)，不与当前方法混用。

## 1 共用输入、坐标与证据记录

使用 NC_000913.3（4,641,652 bp）的 10 bp Micro-C 原始计数矩阵和 344 条结构注释：CHIN 250、OPCID 68、CHID 26。主分析由两个 WT 37℃ 重复构成，条件变化另有 44℃、利福平和 H-NS 缺失双重复。来源和 SHA-256 见 [provenance.json](results/provenance.json)。

所有降分辨率步骤基于原始计数聚合；接触图按对角距离背景进行归一化。输入统计量与高分辨率目标统计量分别计算，低分辨率输入使用自身统计量。脚本、输入、运行基础代码 SHA、配置、冻结与输出哈希记录于各实验协议/回执。

既有标签、候选及测试区在本项目历史上曾被查看，所以当前结果标为回顾性空间留出与跨重复复核。此次选择文件在读取对应测试结果前冻结，仍不据此抹去历史查看情况。

## 2 任务一：已知区间分类

### 2.1 模型与训练

最终网络使用 64×64 的短/长尺度接触图、两张已知区间掩膜和 ROI 接触支路；卷积通道为 8/16/32，采用 GroupNorm。15 维边界/跨度描述符作为侧支路，只用训练折拟合标准化。类别权重为训练折平方根逆频率，固定 24 轮、两个初始化（偏移 0/31）。反转增强同步转换接触图与边界特征。

按照 40 kb 基因组连通组进行空间五折，同一注释的两个重复始终在同一折。全 344 条注释的最终权重只用于推理，泛化指标取外折交叉预测。ROI 和跨度依赖已知起止点，模型用于给定区间的分类。[冻结协议](results/task1_improve_20261006_roi_cnn_seed21_protocol.json)、[部署模型](results/task1_improve_20261006_final_interval_roi_cnn.json)

### 2.2 评价与归因

报告结构级准确率、宏 F1、逐类精确率/召回率及混淆矩阵。在相同外折上比较接触图＋区间基线、最终模型和去描述符侧支路消融；成对区间按空间组重采样。另分别只用 rep1/rep2 训练，测试另一重复的留出区段，保留同源重复测试作为对照。[分类评价与消融](results/task1_improve_20261006_evidence_index.json)、[单源双向迁移](results/task1_followup_20261006_transfer_protocol.json)

显著性图来自正确分类的外折模型。遮挡比较非近对角、相同原始对角距离、等数量的区间内外像素，保持区间/跨度通道并同步重算接触描述符；无可匹配背景的记录单列。梯度与遮挡用于描述模型对输入接触信息的依赖。[OOF 归因协议](results/task1_improve_20261006_audit_protocol.json)

## 3 任务二：候选发现与固定验证

### 3.1 发现流程

按 5,120 bp 窗口、2,560 bp 步长扫描全基因组，共 1,812 窗。对原始计数进行计数守恒面积聚合与距离背景归一化，自编码器、形态及密度分数组成发现表示；形态残差包括去 σ=4 平滑背景和去行列均值的两个视图。以 rep1 的 344 个已知参考拟合标准化/PCA，并由最近独立参考距离的第 90 百分位校准候选新颖性。

单窗口异常与跨位点共同 motif 分别定义：前者要求固定形态异常门槛的重复延续；后者还要求不同位点间的双视图相似度、背景比较及优于各自已知最佳匹配。候选冻结后不依据此次 rep2、尺度或质量对照重选位点/阈值。[发现摘要与冻结证据](results/task2_improve_20261006_summary.json)

### 3.2 背景与重复敏感性

验证使用四个固定候选 A1–A4；完整背景为 1,103 个未注释窗口，不按照分数优先抽取。按染色体坐标顺序、至少 10,240 bp 间隔去重，得到 333 个独立代表。原生接触 QC 包括唯一上三角计数、零行比例、非零像素比例、top1% 质量和行轮廓波动等；匹配仅用 rep1 的协变量。共同支持不足时标记为匹配不可评价，保留相应支持缺口。

尺度设定为 3,840/5,120/7,680 bp，中心偏移为 −320/−160/0/160/320 bp。参考序列唯一性与接触行稀疏是质量代理，不替代实际双端比对质量或片段级覆盖。[最终固定验证协议](results/task2_course_verification_20261014_protocol_freeze.json)、[完整背景摘要](results/task2_course_verification_20261014_summary.json)

### 3.3 共同深度、覆盖反事实与分类器参照

最终共同深度使用 10,000/20,000/30,000 个预期唯一上三角非对角计数，每档五个固定种子；344 个已知模板与四候选、两个重复全部支持。二项抽稀匹配期望而非恰好实现的计数，种子最小－最大范围不标为置信区间。

覆盖反事实先冻结 rep1 候选的零行模式，再原样施加到所有已知模板的两重复，仍使用原新颖性模型和阈值。该人工干预用于检验缺失模式诱导评分的机制；基因组假阳性率需要另外的真实背景数据。[质量协议](results/task2_course_verification_20261014_qc_protocol_freeze.json)、[掩膜冻结](results/task2_course_verification_20261014_qc_mask_freeze.json)、[共同深度及反事实摘要](results/task2_course_verification_20261014_qc_summary.json)

任务一固定分类器另用相同 5.12 kb 临时 ROI 对候选提供闭集样式参照，并对已知样本施加同宽 ROI 以检查输入分布变化。softmax 提供已知样式参照，新颖性阈值保持冻结。[分类器参照协议](results/task2_course_verification_20261014_classifier_protocol_freeze.json)

## 4 任务三：连续轨道图

每 10 kb 一个图像区间，四面板依次展示三条件深度校正强度曲线、条件均值、基因和 CHIN/OPCID 注释。全长 465 图共用基因组坐标，末区间裁剪到参考长度。图像清单记录坐标、文件大小和 SHA-256；仓库提供代表图，本地保留全图。[图像清单](results/task3_track_manifest.csv)

## 5 任务四：条件变化

原始整数计数聚合到 64×64 后，依据全局最小测序深度独立二项抽稀；每个样本、位点与全局种子生成独立固定随机流，文件枚举顺序不影响结果。距离归一化后比较非近对角正信号。

变化规则要求全部四个处理－WT 重复比较同方向、平均效应超过本位点两组内重复差异及全体内部重复差异的第 95 百分位。种子固定为 20260921/22/23，三者交集为稳定候选，再合并重叠注释为位点。该规则用于探索性强度变化筛选。H-NS 与 WT 的平台/批次与条件混杂，抽稀只控制测序深度。[44℃/利福平敏感性](results/task4_order_invariant_sensitivity.json)、[H-NS 敏感性](results/task4_hns_sensitivity.json)

## 6 任务五：完整区间超分辨

### 6.1 输入、队列与神经适配

10 bp 原始计数每 4×4 个像素聚合，以 40 bp LR 计数均值形成输入，并恢复到原尺寸；LR 和 HR 独立归一化。完整已知结构两侧各保留 160 bp，上下文边界对齐 40 bp。保留 248 个训练、28 个验证和 68 个测试完整区间，训练/验证/测试窗口没有跨集合重叠。测试类别为 CHIN 52、OPCID 10、CHID 6。

从既有残差超分辨权重初始化 48 通道、四残差块 CNN，附可训练正仿射输出尺度。只在训练完整区间抽取 96 像素上下文、64 像素核心，每轮 800 个 patch，batch 8，Adam 学习率 1e−4；损失为 MSE＋0.15 梯度损失＋0.5 局部 SSIM，梯度裁剪 5，使用转置/基因组反转增强。非对角 patch 不强行局部对称，含 halo 的完整推理拼接后全局对称化。[完整训练协议](results/task5_full_structure_20261014_final_protocol.json)、[队列清单](results/task5_full_structure_20261014_final_cohort_manifest.csv)

### 6.2 验证选择与对照

固定候选网格为 2/4/6 轮和 CNN 比例 0.25/0.5/0.75/1.0，共 12 个候选。只由验证完整区间选择：图像指标正增益并保护 CHIN/OPCID 形态/轮廓，优先比较 SSIM 再 PSNR。最终选定第 6 轮、比例 1.0 的纯 CNN，模型、队列与选择记录冻结后才评估本次测试及 rep2。

对照包括原始双线性/双三次、相同训练集合拟合两个仿射参数的校准双线性/双三次，以及原冻结 CNN。全部候选和对照保留，不只展示最终模型。[选择冻结](results/task5_full_structure_20261014_final_selection_freeze.json)、[各方法分类型汇总](results/task5_full_structure_20261014_final_method_type_summary.csv)

### 6.3 指标、图像与复核

PSNR/SSIM 在完整原生 10 bp 输出上计算。非近对角形态先以 σ=4 原生像素（40 bp）高斯平滑，面积聚合到 64×64 后取上三角 k=2 的形态向量并计算相关；正信号行轮廓在原生平滑图排除偏移小于 8 像素（80 bp）区域后求行和，再进行相同 σ=4 的一维平滑。所有图像采用相同显示流程和颜色范围，轮廓不逐图重缩放。原图的平滑峰位置误差和 top5 F1 单列，评价直接基于重建图像。

两个重复各 68 个结构完整保留；重叠输入区间形成 41 个连通组，成对自助重采样 2,000 次计算指标增益 95% 区间。独立导出从完整预测重算六个指标并检查结果/图像哈希。主形态与细微峰分别解释，当前验证范围为模拟降分辨率 Micro-C，真实 Hi-C 跨技术泛化需要独立数据检验。[最终指标](results/task5_full_structure_20261014_final_summary.json)、[完整预测清单](results/task5_full_structure_20261014_final_all_case_manifest.json)、[独立重算回执](results/task5_full_structure_20261014_final_export_receipt.json)

## 7 任务六：环形全基因组构象

### 7.1 固定模型

5 kb 全基因组计数得到 929 个 bin，末 bin 导出到 4,641,652 bp。按精确环形 bin 间距去背景、正截断并缩放，全部非邻接正残差对用于距离训练，完整循环链相邻距离方差作为连续性约束。额外排序项在同一环形间距内平衡正残差与截断为零的残差：每轮随机 20,000 对、softplus margin 0.2、权重 0.1。额外抽样使用 seed＋7919 独立 CPU 随机流，不改变网络初始化及原正边抽样。

固定 20261006/07/08 三种子全部报告，首种子用于部署与主图；不使用 MDS 预训练或锚定，MDS 仅作同目标基线。坐标冻结后进行本轮 rep2 评价，截断为零的残差不解释为物理不接触。[模型协议](results/task6_controlled_followup_20261006_protocol.json)、[坐标冻结](results/task6_controlled_followup_20261006_coordinate_freeze.json)

### 7.2 接触与文献几何评价

rep2 评价在精确相同环形间距内匹配正/零残差，以预测距离 AUC 和正接触/反距离 Spearman 分别衡量近邻排序与强度一致性；保留强、弱及跨原点层全部结果，不只报道最佳层。[独立重算](results/task6_full_geometry_20261014_rep2_reverification.csv)

文献几何采用 Zarei2013 补充表 1 全六区边界，Ori 3.76–0.05 Mb 跨零点；全分析采用同一完整边界版本。主评价为四个结构化宏域，排除边界 ±50 kb，同/异域 bin 对按精确环形基因组间距配平。六区和无裁剪结果作为补充，原近似版本保留。每项距离效应与全部 928 个非零循环标签位移比较，完整报告 p 和 16 项检验的 BH q，统计解释以多重校正结果为准。[文献协议](results/task6_literature_regions_20261014_protocol.json)、[边界转录](results/task6_literature_regions_20261014_literature_boundaries.csv)、[整体源表](results/task6_literature_regions_20261014_macrogeometry.csv)、[单域源表](results/task6_literature_regions_20261014_individual_domains.csv)

文献 Ori4（9.9 kb）、NSR5（515 kb）、Right2（738 kb）相对次序单列；oriC 按官方 GFF，dif 约为 1.604 Mb。首种子两视角使用等比例坐标轴、全部 bin 和统一图例。宏域参考是独立文献坐标而非本样品单细胞真值，边界转录精度约 10 kb。解的旋转、镜像和绝对尺度不唯一。[位点定义](results/task6_literature_regions_20261014_coordinate_labels.csv)、[主图回执](results/task6_literature_regions_20261014_primary_views_receipt.json)

## 8 课程结论、科学结论与大语言模型辅助

六项任务分别通过实际模型、源表、基线和图像评价已知识别、候选发现与验证、多轨道与条件变化、主要完整形态恢复及整体构象组织。候选共同类型、细微峰位和单域效果按对应专项检验结果报告，生物学功能与绝对物理坐标留待独立实验检验。[最终判定](results/final_verification_20261014_assessment.json)

大语言模型用于组织冻结候选证据、设计覆盖缺口反事实、实现可复跑检查和辅助解释；每项数值回到真实源表核对。人工备注与导出辅助标注记录解释过程，候选判断依据实际形态和质量对照结果。[候选实际运行回执](results/task2_course_verification_20261014_final_receipt.json)
