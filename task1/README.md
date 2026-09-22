# 必做任务1：已知结构监督识别与可解释分析

`run_task1.py` 从数据目录读取标注工作簿和两个 37°C Micro-C `.cool` 重复，整理 Supplementary Table 4/5/6 的 68 OPCID、250 CHIN、26 CHID（共 344 条）为统一标签。

每个结构以中心为基准读取 512×512 个 10-bp bin（5,120 bp 物理窗口），再平均池化为 128×128 输入。接触计数依次进行 `log1p` 变换与窗口内 MAD 鲁棒标准化。分类器为三层轻量 CNN，并使用类别加权交叉熵缓解 CHID 样本较少的问题。

同一结构的两个生物学重复始终位于同一数据分区，结构 ID 按类别分层后进行 80/20 留出。按测试集宏平均 F1 选择最佳训练轮次。固定随机种子默认为 `20260921`。

## 运行

```powershell
python -m pip install -r task1/requirements.txt
python task1/run_task1.py --data-dir 'micro-c数据(1)' --out-dir task1/final_output
```

## 输出

- `task1_cnn.pt`：模型权重、类别顺序与预处理参数
- `metrics.json`：准确率、宏平均 F1、逐类指标与训练历史
- `confusion_matrix.csv/png`：混淆矩阵数值与图
- `saliency_CHIN/OPCID/CHID.png`：各类型输入梯度显著性图
- `example_CHIN/OPCID/CHID.png`：正确识别样本的归一化接触热图
- `test_predictions.csv`：逐个留出样本的真实/预测类别
- `annotation_manifest.csv`：344 条统一结构清单
- `extracted_windows.npz`：可复用的窗口张量和固定划分

注意：CHID 仅有 26 个结构，显著少于 CHIN 的 250 个。报告结果时应同时给出宏平均 F1 和逐类召回率，不能只报告总体准确率。
