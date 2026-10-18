# V2.5｜研究高分选手后的 GPU 特征工程实验

- 生成时间 UTC：2026-10-18T08:17:45.594189+00:00
- 比赛：playground-series-s6e10
- Kaggle Notebook：https://www.kaggle.com/code/mjw3177109/kas-v25-gpu-te-playground-series-s6e10
- 运行阶段：kernel_queued
- 真实 T4：False；尚未验证

## 为什么改进？

V2.3 GPU XGBoost 只针对原始特征和少量统计做调参，Public LB 0.95819。
公开高分实验提示，精确数值交叉拟合 Target Encoding、航线距离画像和旅客细分群特征更有价值。
V2.5 新增 5 折内部交叉拟合 TE；每个外层验证折只使用对应训练折生成的特征，禁止目标泄漏。

## GPU/CPU 两模型候选与 OOF

- 原冠军：V2.3 GPU gpu_xgb_depth7
- 原冠军 CV：0.9585317063733847
- 原冠军 Public LB：0.95819
- LightGBM + XGBoost GPU 最佳模型：尚未完成
- 新 3-Fold OOF：尚未完成
- CV 超越差值：暂无
- 门槛：0.0001
- 达标：False
- 是否已提交：False
- 新 Public LB：尚未提交/未评分
- 线上冠军：V2.3 GPU gpu_xgb_depth7

## 本次已提交成绩对应的官方排名

- 当前 Public 排名：**未核实（需按最新成绩重新查询）**
- 官方排名查询：reports/score_rank_latest.md

## 同行参考（思路参考，不照搬外部预测文件）

- GitHub: https://github.com/alvis1111/s6e10-airline-solution
- Kaggle ROC-AUC 评测：https://www.kaggle.com/competitions/playground-series-s6e10/overview
- 官方排行榜：https://www.kaggle.com/competitions/playground-series-s6e10/leaderboard

任何外部公开方法或成绩都不是本站实测。上线是否有效，只以本次真实 OOF 和官方 Public LB 判断。

说明：Notebook 还在等待 Kaggle 分配 GPU；并非训练中。

## 错误/状态诊断
mjw3177109/kas-v25-gpu-te-playground-series-s6e10 has status "KernelWorkerStatus.QUEUED"
