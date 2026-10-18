# V2.3 GPU 独立训练线：Kaggle T4 XGBoost

- 比赛：playground-series-s6e10
- 状态：kernel_queued
- Notebook：https://www.kaggle.com/code/mjw3177109/kas-v23-gpu-playground-series-s6e10
- GPU 实际配置：未核验
- 旧冠军 V2.1 3-Fold OOF：0.9582786744608702
- 旧冠军 Public LB：0.95774
- 本轮 GPU 3-Fold OOF：暂无
- CV 与旧冠军差值：暂无
- CV 门槛通过：False
- Kaggle 已提交：False
- 新 Public LB：未提交或未评分
- 线上冠军：V2.1 ensemble

## GPU 训练模型

原有 CPU CatBoost/LightGBM/XGBoost Ensemble 保留不变；GPU 训练仅是新的候选模型。
使用 NVIDIA T4 ×2 实例，但当前单进程 XGBoost 仅使用一个 CUDA 设备，
不能声称同时使用了两张 GPU。


## 排队说明

Kaggle 报告 QUEUED，仅表示已排队，不能视为 GPU 正在训练。

## 诊断

mjw3177109/kas-v23-gpu-playground-series-s6e10 has status "KernelWorkerStatus.QUEUED"
