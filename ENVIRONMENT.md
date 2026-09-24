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

测试结果：`3 passed`。
