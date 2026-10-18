# Kaggle Auto Scientist V2.3 - 独立 TPU 神经网络实验

## 训练线状态

- 树模型 V2.1 Ensemble 保留不变。
- 独立的 TPU 线用 JAX MLP；统一 3 折验证、再判断是否提交。
- 比赛：playground-series-s6e10
- 当前阶段：kernel_queued
- Kaggle Notebook：https://www.kaggle.com/code/mjw3177109/kas-v23-tpu-playground-series-s6e10

## 硬件核验

- 请求硬件：TpuV5E8
- 真实 JAX backend：未知
- 真正使用 TPU：False
- JAX device count：未知

## 3-Fold OOF 对比

- V2.1 树模型 CV：0.9606599604603964
- V2.1 Public LB：0.96035
- V2.3 JAX TPU CV：尚未完成
- CV 提升差值：未知
- 提交门槛：未知
- 达到提交条件：False

## 结果与保底

- 已提交：False
- 新 Public LB：未提交或未评分
- 当前线上冠军：V2.5 blend_rank

V2.1 未存储逐行 OOF 预测，因此不能声称已比较了 TPU/树模型的联合融合分数。
TPU 线单独验证，不覆盖 V2.1 树模型。

## Kaggle 当前排名（和 Public LB 一起报告）

- 此名次关联的 Public LB：0.96035
- 当前 Public 排名：**第 958 名 / 2224 支队伍**
- 排名核验 UTC：2026-10-18T08:46:00.889497+00:00
- 排名证据：kaggle_authenticated_userRank
- 官方排行榜：https://www.kaggle.com/competitions/playground-series-s6e10/leaderboard

## Kaggle 仍在排队

KernelWorkerStatus.QUEUED 表示 Kaggle 尚未安排执行，
因此现在没有 TPU JAX 检测或 CV 结果；绝不能将它写成训练中或训练失败。
可在 Kaggle Notebook 页面查看排队状态，若排队持续很久再检查 TPU 额度。

## 诊断信息

mjw3177109/kas-v23-tpu-playground-series-s6e10 has status "KernelWorkerStatus.QUEUED"
