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

本分支在另一台/另一套运行环境继续整合，不要求前述 D 盘路径在所有机器存在。后续复核环境为 Python 3.14.3、PyTorch 2.14.0+cpu、cooler 0.10.4、NumPy 2.4.4、pandas 3.0.3、scikit-learn 1.8.0、scikit-image 0.26.0。原始数据的来源和哈希记录在 `results/provenance.json`。
