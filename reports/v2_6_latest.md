# V2.6｜客群条件频率交叉拟合 GPU 实验

- 生成时间 UTC：2026-10-18T08:45:48.265885+00:00
- 比赛：playground-series-s6e10
- Kaggle Notebook：https://www.kaggle.com/code/mjw3177109/kas-v26-gpu-te-playground-series-s6e10
- 运行阶段：kernel_queued
- 真实 T4：尚未确认（Kernel 尚未执行）；尚未验证

## 为什么改进？

V2.5 使用无泄漏 Target Encoding + 航线画像获得 Public LB 0.96035，已成为当前正式冠军。
公开高分实验提示，精确数值交叉拟合 Target Encoding、航线距离画像和旅客细分群特征更有价值。
V2.6 在原有内部 5 折无泄漏 TE 基础上，增加 12 个按航程、年龄和客群评分组合的内部 5 折频率编码。

## GPU/CPU 两模型候选与 OOF

- 原冠军：V2.5 blend_rank
- 原冠军 CV：0.9606599604603964
- 原冠军 Public LB：0.96035
- LightGBM + XGBoost GPU 最佳模型：尚未完成
- 新 3-Fold OOF：尚未完成
- CV 超越差值：暂无
- 门槛：0.0001
- 达标：False
- 是否已提交：False
- 新 Public LB：尚未提交/未评分
- 线上冠军：V2.5 blend_rank

## 当前已评分冠军对应的官方排名

- 此名次关联的 Public LB：0.96035
- 当前 Public 排名：**第 958 名 / 2225 支队伍**
- 排名核验 UTC：2026-10-18T09:08:43.718208+00:00
- 排名证据：kaggle_authenticated_userRank
- 官方排行榜：https://www.kaggle.com/competitions/playground-series-s6e10/leaderboard

## 同行参考（思路参考，不照搬外部预测文件）

- GitHub: https://github.com/alvis1111/s6e10-airline-solution
- Kaggle ROC-AUC 评测：https://www.kaggle.com/competitions/playground-series-s6e10/overview
- 官方排行榜：https://www.kaggle.com/competitions/playground-series-s6e10/leaderboard

任何外部公开方法或成绩都不是本站实测。上线是否有效，只以本次真实 OOF 和官方 Public LB 判断。

说明：Notebook 还在等待 Kaggle 分配 GPU；并非训练中。

## 错误/状态诊断
mjw3177109/kas-v26-gpu-te-playground-series-s6e10 has status "KernelWorkerStatus.QUEUED"
