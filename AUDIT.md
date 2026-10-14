# 六项任务最终核查（2026-10-14）

本页对照课程任务和实际实验，分别核查交付完整性、方法/基线证据及科学结论范围。课程的定性要求并未规定“所有 CHID 全找回”“每个微观峰零误差”或“全部宏域严格显著”等硬门槛；项目中的操作性阈值用于内部复核，课程评价仍按教师要求进行。最终报告见 [REPORT.md](REPORT.md)，最终方法见 [METHODS.md](METHODS.md)，结果判定见[实验判定清单](results/final_verification_20261014_assessment.json)。

## 1 当前核查结果

| 任务 | 已完成的实验与交付 | 最终结果及可支持的结论 |
| --- | --- | --- |
| 1 已知结构识别 | 多尺度 CNN＋ROI＋边界描述符；空间五折、两初始化、逐类指标、消融、OOF 显著性/遮挡、双向跨重复测试和部署权重 | 准确率 0.924、宏 F1 0.839；CHID 精确率/召回率 0.696/0.615。两方向跨重复分别为 0.9157/0.9070 准确率，均识别 17/26 个 CHID。支持已知区间三分类与重复稳健性，模型适用范围为给定起止点的区间。[分类证据](results/task1_improve_20261006_evidence_index.json)、[跨重复证据](results/task1_followup_20261006_transfer_summary.json) |
| 2 候选发现 | 已知结构再发现、全 1,812 窗扫描、冻结候选和第二重复；完整 1,103 个未注释背景、333 个按坐标独立代表；尺度/偏移、三档共同深度、覆盖缺口反事实和固定分类器核查 | 得到 A1/A2/A4 三个默认规则下双重复延续的未注释形态异常候选，完成候选验证流程。A2 有突出的覆盖缺口混杂；三个候选已完成逐位点验证，共同新类型尚未建立。[验证摘要](results/task2_course_verification_20261014_summary.json)、[质量对照](results/task2_course_verification_20261014_qc_summary.json)、[最终回执](results/task2_course_verification_20261014_final_receipt.json) |
| 3 多轨道图 | 465 张连续 10 kb 四面板图，覆盖 4,641,652 bp；三条件、基因和结构轨道；逐图坐标/哈希及代表图 | 完成长染色体的连续多轨道可视化。Git 保存代表图与全图清单，本地保留全部 PNG。[全图清单](results/task3_track_manifest.csv) |
| 4 条件变化 | WT/44℃/利福平/H-NS 双重复、深度匹配、全部四组跨重复比较、固定每样本/位点随机流、三抽稀种子、稳定交集与位点合并 | 44℃/利福平/H-NS 的交集为 17/98/82 条注释，合并 12/86/64 个位点。完成处理相关变化的探索性定位及稳定性检验；H-NS 与平台/批次存在混杂，因果机制仍待独立检验。[44℃/利福平](results/task4_order_invariant_sensitivity.json)、[H-NS](results/task4_hns_sensitivity.json) |
| 5 超分辨 | 248/28/68 个完整结构训练/验证/测试；训练区适配；验证冻结 6 轮纯 CNN；两重复完整 136 个预测；原始及训练校准插值、图像/形态/轮廓/原图峰指标、连通组配对区间、全案例图集和独立重算 | 每个重复 68/68 个结构的 PSNR、SSIM、非近对角形态相关优于原始双线性，主形态和行轮廓恢复得到支持；CNN 也优于等训练校准插值。OPCID 细近对角带的恢复程度有限，峰定位增益区间跨零。满足课程图像改善及已知主形态恢复的评价目标。[协议](results/task5_full_structure_20261014_final_protocol.json)、[选择冻结](results/task5_full_structure_20261014_final_selection_freeze.json)、[完整结果](results/task5_full_structure_20261014_final_summary.json) |
| 6 三维重建 | 929 bin 环形全基因组坐标；三固定种子、MDS、第二重复接触复算；完整发表边界版本、四/六区和边界敏感性、928 个非零循环位移、各单域结果、文献位点次序及双视角图 | 匹配接触 AUC 0.7995–0.8048 > MDS 0.7813；完整文献边界四宏域 ±50 kb 排除下 AUC 0.6722/0.7171/0.7550 > MDS 0.6390。支持宏域相关整体距离组织，文献三位点次序全部复现；调整后循环位移 q 均 >0.05，单域效果有所差异，结果表示接触约束下的相对几何组织。[接触复算](results/task6_full_geometry_20261014_rep2_reverification.csv)、[文献版本结果](results/task6_literature_regions_20261014_summary.json)、[最终核查](results/task6_literature_regions_20261014_final_audit.json) |

## 2 关键验收边界

### 2.1 候选发现与共同类型检验

A1/A2/A4 延续默认异常判定，A3 的新颖性未在第二重复中延续。全部候选已完成尺度/偏移和质量对照。全量背景仍缺可比质量支持，人工掩膜能够诱导相似评分，因此现有结果为“可复现候选＋质量解释”，共同新类型确证数为 0。[逐候选验证](results/task2_course_verification_20261014_summary.json)、[覆盖反事实](results/task2_course_verification_20261014_qc_counterfactual_summary.csv)

共同深度最终采用 10,000/20,000/30,000 的预期上三角计数，全部已知模板和候选的两重复均支持；旧首轮 34,548 深度诊断中 A2 rep2 支持不足，最终共同深度结果以三档全量支持的方案为准。[最终共同深度摘要](results/task2_course_verification_20261014_qc_summary.json)

### 2.2 完整形态恢复与微观峰定位分别评价

任务五最终模型使用完整区间上下文进行神经适配，选择冻结先于此次测试读取；两重复各 68 个结构全部保留，完整形态、行轮廓和原图峰指标均可追溯。主要非近对角形态与精细热点分别评价，当前报告统一采用新完整区间的 PSNR。[136 案例清单](results/task5_full_structure_20261014_final_all_case_manifest.json)、[独立指标重算](results/task5_full_structure_20261014_final_export_receipt.json)、[运行后完整性](results/task5_full_structure_20261014_final_post_run_integrity_receipt.json)

### 2.3 三维整体趋势与严格统计特异性分别评价

分析采用文献完整边界版本，Ori 明确跨越坐标零点。最终三种子和 MDS 全部呈现，原近似边界版本及各单域结果保留。整体 AUC 的基线优势、三位点关系与循环位移校正分别报告，同时呈现整体与单域效果的差异。[完整转录边界](results/task6_literature_regions_20261014_literature_boundaries.csv)、[版本比较](results/task6_literature_regions_20261014_coordinate_version_comparison.png)、[单域结果](results/task6_literature_regions_20261014_individual_domains.csv)

## 3 复现与正式提交

结果包校验与重新计算的命令见 [REPRODUCE.md](REPRODUCE.md)。已冻结的输入、脚本和结果不因更新报告而改写；原运行基础提交与新增脚本的实际字节哈希一并保留在协议和回执。检查工具核验文件、数量、坐标、哈希与数值一致性，科学结论由相应实验和对照支持。

完整原始矩阵、465 张轨道图和约 777 MB 的完整结构预测在本地，不放入紧凑代码 ZIP；公开结果保存源表、最终模型、代表图和完整输出清单。正式身份和真实分工/贡献比例仍由组员填写，讲解视频录制及课程平台提交由小组完成。上述事项属于实验结果之外的正式提交手续。

本轮未执行约 198 GB 原始 reads 的全量重比对；该步骤可补实际 MAPQ/片段覆盖，属于额外的数据质量核查；现有 COOL 和参考 k-mer 提供接触覆盖与序列唯一性代理。[原数据来源](results/provenance.json)、[候选验证限制](results/task2_course_verification_20261014_qc_summary.json)

原审核记录未删除，见[更新前审核原件](submission/archive/final_verification_20261014/AUDIT_before_final_verification.md)；任务五/六的最终状态以本页及最新实验源表为准。
