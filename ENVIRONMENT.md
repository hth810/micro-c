# 复核环境

本次复核使用 D 盘 Anaconda 的独立复制环境：

```text
D:\anaconda3\envs\microc-review
```

该环境从 `microc-task1` 克隆，并通过清华 PyPI 镜像补充了 `scikit-image` 和 `pytest`。关键版本如下：

```text
Python 3.11
PyTorch 2.6.0+cu124
CUDA: NVIDIA GeForce RTX 4060 Laptop GPU
cooler 0.10.4
NumPy 2.3.3
SciPy 1.16.1
pandas 2.3.1
scikit-learn 1.7.2
scikit-image 0.26.0
```

复核命令：

```powershell
D:\anaconda3\envs\microc-review\python.exe -m pytest -q tests
```

测试结果：`13 passed`（另有 1 个 scikit-learn 关于小样本分层的 warning）。任务1 CUDA 训练需要脚本自动设置 `CUBLAS_WORKSPACE_CONFIG`，否则 PyTorch 确定性反向传播会失败。

## 后续整合复核（E 盘工作区）

本分支在另一套运行环境继续整合，前述 D 盘路径属于历史复核。最终环境为 Python 3.14.3、PyTorch 2.14.0+cpu、cooler 0.10.4、NumPy 2.4.4、pandas 3.0.3、scikit-learn 1.8.0、scikit-image 0.26.0。完整测试结果为 `228 passed`，记录于 `results/final_verification_20261014_audit.json`。最终完整区间模型实际在 CPU 上训练和评价，模型 revision、参数与输入输出哈希另见对应协议。原始数据的来源和哈希记录在 `results/provenance.json`。

报告复建额外使用 python-docx、Pillow、pypdf；最终 Word 在 Microsoft Word 中导出并逐页检查。新增依赖已列入 `requirements.txt`，当前环境无需重复安装。
