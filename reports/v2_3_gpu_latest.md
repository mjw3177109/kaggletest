# V2.3 GPU 独立训练线：Kaggle T4 XGBoost

- 比赛：playground-series-s6e10
- 状态：complete
- Notebook：https://www.kaggle.com/code/mjw3177109/kas-v23-gpu-playground-series-s6e10
- GPU 实际配置：['Tesla T4, 15360 MiB', 'Tesla T4, 15360 MiB']
- 旧冠军 V2.1 3-Fold OOF：0.9582786744608702
- 旧冠军 Public LB：0.95774
- 本轮 GPU 3-Fold OOF：0.9585317063733847
- CV 与旧冠军差值：0.00025303191251446755
- CV 门槛通过：True
- Kaggle 已提交：True
- 新 Public LB：0.95819
- 线上冠军：V2.3 GPU gpu_xgb_depth7

## GPU 训练模型

原有 CPU CatBoost/LightGBM/XGBoost Ensemble 保留不变；GPU 训练仅是新的候选模型。
使用 NVIDIA T4 ×2 实例，但当前单进程 XGBoost 仅使用一个 CUDA 设备，
不能声称同时使用了两张 GPU。

| 模型 | 3-Fold OOF ROC-AUC |
|---|---:|
| gpu_xgb_depth5 | 0.95816175 |
| gpu_xgb_depth7 | 0.95853171 |

## 诊断

mjw3177109/kas-v23-gpu-playground-series-s6e10 has status "KernelWorkerStatus.QUEUED"
