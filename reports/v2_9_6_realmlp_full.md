# V2.9.6 RealMLP 完整三折 GPU 实验

- 核验 UTC：2026-10-19T13:55:39.390863+00:00
- Kaggle kernel：https://www.kaggle.com/code/mjw3177109/kas-v296-realmlp-full-s6e10
- 当前阶段：kernel_queued
- 验证协议：StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)
- 方法：固定 4 子模型、3 epochs 的 RealMLP-TD；训练集内 5fold 交叉目标编码。
- 新预测不修改既有 V2.5、V2.8、V2.9.3 的实验输出。
- RealMLP 独立三折 OOF：尚未训练
- 历史最佳三折 OOF：0.9607535043719558
- 新固定 80/20 融合 OOF：尚未完成
- 融合相对提升：尚未完成
- 单折改变量：尚未完成
- 真正分配 Tesla T4：False
- 是否已经 Kaggle 提交：False
- 正式 Public LB：尚无本版本新成绩；不能将 CV 当成 Public LB。
- 本版本只负责训练与 CV gate；提交由独立预留后的一次性控制器执行。

