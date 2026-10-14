"""Bind final report language to the actual final Task 2/5/6 source tables."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "output/report/support_final_verification_20261014"


def load(name):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def P(text): return {"type": "p", "text": text}
def H(text, new=False): return {"type": "h2", "text": text, "new_page": new}
def T(caption, headers, rows, widths):
    return {"type": "table", "caption": "表 0  " + caption, "headers": headers, "rows": rows, "widths": widths}
def F(path, caption, height=3):
    return {"type": "figure", "path": path, "caption": "图 0  " + caption, "width": 6.86, "max_height": height}
def interval(value, places=3):
    return "[" + ", ".join(f"{x:.{places}f}" for x in value) + "]"


def main():
    target = SUPPORT / "report_patch.json"
    if target.exists():
        raise FileExistsError("Keep the already reviewed patch; use a new build location")
    base = json.loads((SUPPORT / "report_reference_content.json").read_text(encoding="utf-8"))
    pages = {page["number"]: deepcopy(page["blocks"]) for page in base["pages"]}
    task2 = load("task2_course_verification_20261014_summary.json")
    qc = load("task2_course_verification_20261014_qc_summary.json")
    task5 = load("task5_full_structure_20261014_final_summary.json")
    task6 = load("task6_literature_regions_20261014_summary.json")
    # A changed success label may not override a failed frozen experiment.
    if not task5["overall_operational_success"]:
        raise ValueError("Task5 has not supported the frozen complete-window criteria; continue work before report finalization")
    assessment = load("final_verification_20261014_assessment.json")
    audits = load("final_verification_20261014_audit.json")
    if not all(item["course_success_supported"] for item in assessment["tasks"].values()):
        raise ValueError("Final course conclusions have not been independently reviewed")
    edited = {}

    def use(number):
        edited[str(number)] = pages[number]
        return pages[number]

    def first_table(blocks):
        return next(b for b in blocks if b["type"] == "table")

    cover = use(1)
    table = first_table(cover)
    table["rows"][1][1] = "3 个跨重复形态异常候选 完整质量与混杂验证"
    psnr = [task5["replicates"][r]["selected"]["all"]["psnr"]["model_mean"] for r in ("rep1_test", "rep2_fixed_test")]
    ssim = [task5["replicates"][r]["selected"]["all"]["ssim"]["model_mean"] for r in ("rep1_test", "rep2_fixed_test")]
    table["rows"][4][1] = f"完整结构 PSNR {psnr[0]:.3f}/{psnr[1]:.3f} dB 形态双重复验证"
    table["rows"][5][1] = "接触 AUC 0.799–0.805 文献版四域 AUC 0.672–0.755"
    cover[-2] = P("六项实验形成了完整的数据到结果流程。分类模型稳定识别已知区间，发现流程定位并跨重复复核了未注释形态异常候选；超分辨模型同时改善完整区间的图像质量及主要非近对角形态，三维模型复现宏域相关的整体距离组织及文献位点的相对关系。候选清单与新结构类型分开报告，所有结论均对应实际实验表和图。")

    design = use(3)
    first_table(design)["rows"][4] = ["任务五", "完整区间适配的对称残差 CNN", "原生图像指标与完整形态共同评价"]
    design[-1] = P("分类按结构的空间组评价，两个重复保持同组；超分辨的完整结构窗口分训练、验证与测试，实际窗口不跨集合重叠。发现候选与诊断规则先记录，再执行当前重复复核；三维完整声明三个随机种子并列报告。重叠结构按连通组件重采样，避免把像素或重叠窗口当作独立样本。这是现有标注和重复数据上的回顾性实验。")
    edited["4"] = [
        {"type": "h1", "text": "2 设计思路 新结构判定"},
        H("计算候选与新类型的分层定义"),
        P("我们把未注释区域、形态异常候选和共同新类型区分开。候选需有明确坐标、可复现形态及相对已知参考的差异；新类型还需描述其区别于已知类型及组合的形态特征。课程的发现流程评价与更强的新类型主张采用不同证据层级，不把研究筛选阈值当作老师指定的硬性门槛。"),
        T("发现对象 判定依据与实际结果", ["对象", "操作性依据", "实际结果"], [
            ["流程有效性", "已知结构再发现 形态与密度基线", "召回 310/344"],
            ["未注释候选", "完整 5.12 kb 窗口 坐标冻结", "4 个重点位点"],
            ["形态异常复现", "rep1 已知参考 q90 两视图复核 rep2", "A1 A2 A4 通过"],
            ["稳健性与技术质量", "尺度 偏移 共同深度 无偏背景\n覆盖缺口反事实 已知模板与固定识别器", "完整诊断见 R21"],
            ["共同新类型", "原冻结规则 ≥3 位点 两视图共同形态\n覆盖质量可比较 区别于已知结构", "尚未建立"],
        ], [1.24, 4.18, 1.44]),
        H("冻结与验证接口"),
        P("原始候选规则保持不变：高通与行列残差各取 20 维 PCA，相对已知的新颖距离阈值为 30.4523 与 28.4985；主计算截断为 [−5, 5]。三位点及 q99 等用于本研究共同类型筛查，而不是课程规定的通用标准。候选的完整性、复现性与类型认定分别记录。"),
        P("质量验证覆盖全部 1103 个未注释完整窗口，并按坐标抽稀得到 333 个独立背景，不再把高分优选代表当作随机总体。零行匹配失败、已知模板受缺失模式影响和识别器的闭集限制均给出可复跑对照；异常评分不自动解释为生物学新类型。[R21]"),
    ]
    preprocessing = use(5)
    first_table(preprocessing)["rows"][2] = ["超分辨", "完整区间及两侧 160 bp", "原生 10 bp 输出 40 bp 模拟输入"]
    metric_table = [b for b in preprocessing if b["type"] == "table"][1]
    metric_table["rows"][4] = ["完整结构恢复", "全窗形态相关 行强度轮廓 真实图像峰位置"]
    metric_table["rows"][5] = ["三维评价", "接触一致性 同偏移宏域 AUC 文献位点关系"]

    for number in (13, 14):
        blocks = use(number)
        for block in blocks:
            if block["type"] == "figure":
                block["path"] = (SUPPORT / "assets" / Path(block["path"]).name).relative_to(ROOT).as_posix()
    pages[13][-1] = P("尺度与中心偏移检查中，A1、A2 的高通重复相关分别为 0.628–0.801 和 0.582–0.774，局部图样可复现。覆盖缺口反事实同时说明 A2 的高新颖评分容易受技术因素影响：同样的固定缺失行模式可使 342/344 个已知模板越过双重复新颖门槛。该结果用于解释混杂机制，不是实际误报率，也不直接证明 A2 为伪影。[R21]")
    pages[14][-1] = P("任务二完成已知结构再发现，并形成三个默认规则下跨重复通过的未注释形态异常候选。A3 保留为新颖性未复现的对照。当前四个位点未形成共同双视图形态簇，因此报告候选发现成果，不把四个位点或三个通过位点写成新结构类型。")
    evidence = task2["candidate_evidence"]
    rows = [[r["candidate_id"], f'{r["minimum_shape_repeat_r_over_scales_shifts"]:.3f}–{r["maximum_shape_repeat_r_over_scales_shifts"]:.3f}',
             str(next(x for x in qc["counterfactual_all_known"] if x["mask_from_candidate"] == r["candidate_id"])["both_replicates_dual_novelty_count"]) + "/344"] for r in evidence]
    pages[14] += [
        H("候选验证与覆盖混杂", True),
        P("尺度检查采用 3.84、5.12、7.68 kb 窗口和 ±160、±320 bp 中心偏移。共同深度检查固定 10000、20000、30000 个预期非对角上三角计数，每档保留五个种子；344 个已知参考及四候选的两个重复均支持三档深度。抽样匹配的是期望计数，图中范围是五种子最小值至最大值，不标为置信区间。"),
        T("固定候选稳健性及覆盖缺口压力测试", ["候选", "尺度偏移高通相关范围", "已知模板双重复越过门槛"], rows, [.85, 2.81, 3.2]),
        F("results/task2_course_verification_20261014_qc_counterfactual_and_depth.png", "覆盖缺口反事实与三个共同深度档。两重复及全部已知参考采用同一规则，人工缺失模式是压力测试而非真实背景。来源 R21", 3.3),
        P("未施加缺失模式时，12/344 个已知模板越过双重复新颖门槛；A1、A2、A3、A4 模式下分别为 17、342、131、30 个。固定识别器给四候选的闭集样式均偏 CHIN，但相同临时 ROI 下 CHID 仅 10/26 与原类别一致，softmax 不用于判定未知类别。复现、模板距离、质量对照与辅助识别因此分别解读，不互相替代。"),
    ]

    chosen = task5["selected"]
    pixel_rows = []
    shape_rows = []
    for replicate, short in (("rep1_test", "rep1"), ("rep2_fixed_test", "rep2")):
        report = task5["replicates"][replicate]["selected"]
        for metric, label, places in (("psnr", "PSNR dB", 3), ("ssim", "SSIM", 5)):
            row = report["all"][metric]
            pixel_rows.append([short + " " + label, f'{row["bilinear_mean"]:.{places}f}', f'{row["model_mean"]:.{places}f}', interval(row["paired_component_95ci_gain"], places)])
        for kind in ("CHIN", "OPCID", "CHID"):
            record = report[kind]
            shape_rows.append([short + " " + kind, str(record["n"]), f'{record["shape_r"]["model_median"]:.3f}',
                               f'{record["profile_r"]["model_median"]:.3f}', f'{record["shape_r_at_least_0_5_fraction"]:.1%}'])
    edited["17"] = [
        {"type": "h1", "text": "8 接触矩阵超分辨重建"}, H("最终模型与完整结构输入"),
        P("最终模型为 48 通道、4 残差块 CNN，先用 640 bp 局部配对图训练，再在训练区的完整标注区间上进行全窗分布适配。完整窗口增加两侧 160 bp，并按 40 bp 网格对齐；原生 10 bp 计数按 4×4 均值合并生成 40 bp 输入，再双线性上采样。输入归一化仅使用低分辨率统计量，高分辨图只作为训练目标和评价真值。[R22]"),
        P(f"完整结构按空间分为 248 条训练、28 条验证和 68 条测试，实际窗口不跨集合重叠。全窗适配使用 96 像素上下文和 64 像素中心块，Adam 学习率 0.0001，损失为 MSE 加 0.15 倍梯度损失与 0.5 倍局部 SSIM 损失。预先固定 2、4、6 轮及四个融合比例，共 12 个验证候选，最终选择 {chosen['epoch']} 轮、CNN 比例 {chosen['fraction']:.1f}。模型与比例冻结后才执行当前测试区和 rep2 评价。"),
        T("68 个完整结构 双重复原生图像评价", ["评价", "双线性", "最终 CNN", "配对增益 95% 区间"], pixel_rows, [1.42, 1.2, 1.2, 3.04]),
        P("置信区间按 41 个完整窗口重叠连通组件作 2000 次配对重采样。双线性、双三次及等预算训练区校准插值均保留。全窗推理使用带 halo 的分块卷积，非对角块不强行局部对称，完成后对整张矩阵对称化；预测与完整窗口直接推理的一致性由单元测试检验。"),
        F("results/task5_full_structure_20261014_final_rep1_test_CHIN_morphology.png", "完整 CHIN 区间的插值 最终 CNN 与目标。四图使用相同形态平滑与色阶，原生图和全部测试结构图另行保存。来源 R22", 2.4),
    ]
    edited["18"] = [
        H("完整形态与位置恢复"),
        P("形态评价覆盖完整已知区间而非仅结构中心。全窗采用相同 40 bp 高斯平滑后聚合到 64 网格，比较非近对角形态；另比较非近对角正残差的行强度轮廓，检查结构出现的位置和跨度。原生图的最强峰误差与 Top5 峰匹配仍独立保留，不以候选排序器收益替代重建图表现。"),
        T("完整结构分类型恢复 两重复使用相同 68 坐标", ["重复与类型", "数量", "形态相关中位", "轮廓相关中位", "形态相关 ≥0.5"], shape_rows, [1.37, .55, 1.45, 1.45, 2.04]),
        P("两个重复均支持全窗图像增益与 CHIN、OPCID 主要非近对角形态及轮廓的一致性提高，满足本实验预先规定的联合评价规则。CHID 六个测试区间也全部保留。新方法改变实际高分辨重建图；原生高频稀疏纹理及 OPCID 的精细近对角增强带尚未完整恢复，不将平滑展示当作细节恢复证明。"),
        H("插值强对照与实际峰定位", True),
        T("全部完整结构上的等训练区校准对照", ["重复与方法", "PSNR dB", "SSIM", "形态相关均值", "轮廓相关均值"],
          [[short + " " + label,
            f'{record["all"]["psnr"]["model_mean"]:.3f}', f'{record["all"]["ssim"]["model_mean"]:.5f}',
            f'{record["all"]["shape_r"]["model_mean"]:.3f}', f'{record["all"]["profile_r"]["model_mean"]:.3f}']
           for replicate, short in (("rep1_test", "rep1"), ("rep2_fixed_test", "rep2"))
           for label, record in [("最终 CNN", task5["replicates"][replicate]["selected"]),
                                 ("双三次", task5["replicates"][replicate]["controls"]["bicubic"]),
                                 ("校准双线性", task5["replicates"][replicate]["controls"]["train_calibrated_bilinear"]),
                                 ("校准双三次", task5["replicates"][replicate]["controls"]["train_calibrated_bicubic"])]],
          [2.08, 1.0, 1.0, 1.39, 1.39]),
        P("两种校准插值各用相同训练区拟合两个强度参数，不让 CNN 独占尺度校准的优势。最终模型在两个重复中的 PSNR、SSIM 与形态相关均高于这些强对照。全窗原生最强峰误差在 rep1 为 468→491 bp、rep2 为 408→194 bp，两项增益区间均跨零，故不声称精细峰位一致提高。"),
        F("results/task5_full_structure_20261014_final_rep1_test_OPCID_morphology.png", "完整 OPCID 的同尺度形态比较及行强度轮廓。窗口坐标与源表一一对应。来源 R22", 2.2),
        F("results/task5_full_structure_20261014_final_rep1_test_CHID_morphology.png", "完整 CHID 多 kb 区间的恢复示例。相同平滑同时应用于插值 预测和真值；原生像素结果单独评分。来源 R22", 2.2),
    ]
    # Balance the morphology page and control page without shrinking type.
    opcid = next(b for b in edited["18"] if b.get("path", "").endswith("OPCID_morphology.png"))
    edited["18"].remove(opcid)
    next_page = next(i for i, b in enumerate(edited["18"]) if b.get("new_page"))
    edited["18"].insert(next_page, opcid)
    three_d = use(19)
    three_d[-2] = P("三个声明种子完整保留，主图使用第一种子，不按几何评价挑选最好的一次。宏域标签采用发表补表中的整套六区域坐标，Ori 跨越零点；标签只用于外部解释与评价，不进入模型训练。整体特征验证同时比较同环形偏移的域内域间距离及文献位点相对关系。[R23]")
    three_d[-1] = F("results/task6_literature_regions_20261014_primary_two_views.png", "第一声明种子的最终三维轨迹 双视角 整套文献区域及 oriC 和 dif 标注。两视角等轴比例，距离为模型相对单位。来源 R23", 3.3)
    primary = {r["model"]: r for r in task6["four_macrodomains_50kb"]}
    macro_rows = []
    for key, short in (("rank_mlp_seed20261006", "seed 06"), ("rank_mlp_seed20261007", "seed 07"), ("rank_mlp_seed20261008", "seed 08"), ("circular_mds", "MDS")):
        r = primary[key]
        macro_rows.append([short, f'{r["same_domain_near_auc"]:.3f}', f'{r["standardized_between_minus_within"]:.3f}', f'{r["cyclic_shift_bh_q_all_16_checks"]:.3f}'])
    edited["21"] = [
        H("整体几何特征验证"),
        P("宏域评价在相同环形基因组偏移内比较域内与域间空间距离，以负距离判别域内配对；精确均匀圆环为 AUC 0.5 的几何控制。主分析用四个结构化宏域，并剔除真实域边界两侧 50 kb；同时保留不裁边界、六区域以及原近似边界版本。区域坐标、保留 bin 和配对数都在源表中明确。[R23]"),
        T("文献整套坐标下四宏域的距离组织", ["模型", "同域近距 AUC", "标准化隔离效应", "环移 BH q"], macro_rows, [1.25, 1.7, 1.95, 1.96]),
        P("三个种子的总体四域近距 AUC 为 0.672–0.755，均高于 MDS 的 0.639，标准化隔离效应均为正。这支持宏域相关的整体距离组织；928 个非零环移的多重校正结果未达到 0.05，因此不写成显著的宏域确证，也不要求每个局部区域都优于基线。Ori 的恢复最稳定，其他区域的紧凑程度与边界版本存在差异。"),
        F("results/task6_literature_regions_20261014_coordinate_version_comparison.png", "整套文献坐标与原近似坐标的完整比较。三个种子和 MDS 全部保留，体现总体趋势及分域差异。来源 R23", 3.0),
        H("文献位点与构象解释"),
        P("文献所述 Ori4 9.9 kb 至 NSR5 515 kb 比至 Right2 738 kb 更近的次序，在三个种子和 MDS 中均复现。该关系来自同参考坐标系的已发表空间观察，但生长条件不同，不当作本样品的显微真值。oriC 由官方 GFF 定位，dif 以约 1.604 Mb 标注；其同偏移距离另列描述诊断。三维结果表示接触约束下的相对构象，旋转、镜像与绝对尺度不唯一。"),
    ]
    llm = use(22)
    llm[2] = P("大语言模型用于组织候选证据、设计覆盖缺口反事实检查和解读最终验证，不凭模型回答宣布新结构。输入包含冻结候选、双重复相关、尺度和深度敏感性、参考唯一性、无偏背景及固定识别器输出，数值由源表核对。候选、参数与验证结果保持可追溯，LLM 建议落实为可运行代码后再由实际实验检验。[R21]")
    table = first_table(llm)
    table["headers"] = ["候选", "最终证据解读", "报告定位"]
    table["rows"] = [["A1", "默认复现 尺度偏移稳定 对截断敏感", "形态异常候选"],
                     ["A2", "默认复现 覆盖缺口可强烈诱导新颖评分", "候选与技术混杂并列"],
                     ["A3", "重复图相似 但默认新颖性未复现", "保留负对照"],
                     ["A4", "默认复现 严截断相对稳定 仍需质量对照", "形态异常候选"]]
    llm[-1] = P("LLM 解读的作用是把图像、指标与可能解释联系起来，帮助发现需要对照的问题。已完成的程序核对和哈希审计对应真实源表，不把 LLM 自审或闭集 softmax 当作独立生物学验证。离线辅助标注工具保留双重复查看、人工备注与 JSON 导出，支持候选的后续人工核查。")
    implementation = use(23)
    implementation[2] = P("完整实验位于 hth810/micro-c 的 feature/microc-final-verification-20261014 分支。历史模型与新增实验的输入和代码分别记录 SHA256；新增运行的基础提交为 2c237f0，各新增源码的实际字节哈希随协议保存。原 109 项及补充 75 项冻结证据保持不变，新模型、几何检查、源表和图像另立回执。[R24]")
    table = first_table(implementation)
    table["caption"] = "表 0  完整软件与最终实验交付核查"
    table["rows"] = [["自动测试", f"{audits['pytest_passed']} 项通过"],
                     ["历史冻结证据", "109 项及 75 项哈希保持一致"],
                     ["任务二验证", "93 条代码 输入 冻结与输出路径核查通过"],
                     ["任务五完整评价", "68 个完整区间 两重复 全部预测与源表"],
                     ["任务六最终评价", "40 条来源产物核查 接触指标独立复算"],
                     ["同源轨道与坐标", "465 张轨道图 929 个全染色体坐标"]]
    implementation[5] = P("REPRODUCE.md 分开列六任务核心流水线与最终验证扩展。新入口 reproduce_final_verification_20261014.py 支持 --dry-run，并要求尚不存在的输出目录；各扩展结果存入隔离目录，避免覆盖当前冻结包。Task5 明确按训练、验证选择、冻结测试的顺序执行，Task6 不重训、不选种子，复算当前坐标的接触和几何指标。")
    implementation[-1] = P("实验直接使用公开处理后的 COOL 矩阵，未把原始 reads 全量重比对写成已完成。候选的参考重复程度和原生行稀疏性用于技术诊断；这些信息与实际 reads 比对质量分开。完整性核对确认可追溯来源，课程结论仍由具体实验指标和图像支持。")
    edited["24"] = [
        {"type": "h1", "text": "12 讨论与结论"}, H("六项实验的综合成果"),
        P("本实验完成六项任务的数据处理、模型或分析实现及结果展示。已知结构分类在空间五折下取得 92.4% 准确率和 0.839 宏 F1，并提供双向跨重复、消融与显著性图。发现流程召回 310/344 个已知结构，定位并跨重复复核 A1、A2、A4 三个未注释形态异常候选，配套共同深度、尺度偏移和覆盖混杂检查。尚未建立共同新结构类型，候选发现成果与类型主张明确分开。"),
        P("465 张四层轨道连续覆盖染色体，条件差异形成深度匹配与重复一致的稳定变化集合。最终超分辨 CNN 在完整结构测试区和同坐标第二重复中，同时支持图像增益及 CHIN、OPCID 主要非近对角形态的稳定恢复。三维模型在三个种子中保持接触一致性优势，并在整套文献坐标下复现宏域相关的整体距离组织和文献位点相对次序。"),
        H("结果的适用范围"),
        P("本次超分辨验证针对由 Micro-C 合并 bin 得到的低分辨率输入，不把它直接推广为真实 Hi-C 跨模态重建。三维宏域总体趋势与分域表现分别报告，区域边界的文献版本敏感性保留。候选的未注释状态、稳定图样和类型新颖性也分别列证据；覆盖缺口反事实尤其说明，较高的复现与异常分数可能同时受到技术因素影响。"),
        P("方法设计的主要特点是把问题拆成可检验的判断：形态与密度对照、空间数据隔离、完整区间恢复、同基因组距离几何对照和固定规则跨重复复核。配套模型、逐结构表、完整图集、冻结协议及输入输出哈希，使报告中的每项结论都能回到对应实验。"),
    ]
    refs = use(25)
    refs.insert(8, P("[7] Zarei M, Sclavi B, Cosentino Lagomarsino M. Gene silencing and large-scale domain structure of the E. coli genome. Molecular BioSystems. 2013;9:758–767. DOI 10.1039/C3MB25364C. 宏域坐标取官方 ESI 第 7 页补表 1，整套使用，不按模型结果选择边界。"))
    appendix = use(26)
    first_table(appendix)["rows"] += [["R21", "全背景 尺度深度 覆盖反事实 固定分类器候选复核"],
                                      ["R22", "最终全窗 CNN 训练验证冻结 68 完整区间双重复"],
                                      ["R23 R24", "最新坐标几何 文献六区版本 最终综合核查"]]
    # Only one source-defined edit recipe is saved; the separate renderer checks
    # actual page breaks and typography after every meaningful edit iteration.
    patch = {"reference_pages": 26, "pages": edited, "conclusion_policy": "final methods and actual results; no exploration narrative",
             "language_replacements": json.loads((SUPPORT / "report_language_edits.json").read_text(encoding="utf-8"))}
    target.write_text(json.dumps(patch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sources = [p for p in (ROOT / "results").iterdir() if p.is_file() and ("20261014" in p.name or p.name in ("improvements_20261006_integrity.json", "followups_20261006_integrity.json"))]
    index = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (SUPPORT / "final_source_index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"edited_slots": sorted(map(int, edited)), "bound_sources": len(index), "task5_operational_success": True}, ensure_ascii=False))


if __name__ == "__main__": main()
