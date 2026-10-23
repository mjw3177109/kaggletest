# V2.6｜客群条件频率交叉拟合 GPU 实验

- 生成时间 UTC：2026-10-18T09:27:24.597140+00:00
- 比赛：playground-series-s6e10
- Kaggle Notebook：https://www.kaggle.com/code/mjw3177109/kas-v26-gpu-te-playground-series-s6e10
- 运行阶段：skipped_no_cv_gain
- 真实 T4：True；['Tesla T4, 15360 MiB', 'Tesla T4, 15360 MiB']

## 为什么改进？

V2.5 使用无泄漏 Target Encoding + 航线画像获得 Public LB 0.96035，已成为当前正式冠军。
公开高分实验提示，精确数值交叉拟合 Target Encoding、航线距离画像和旅客细分群特征更有价值。
V2.6 在原有内部 5 折无泄漏 TE 基础上，增加 12 个按航程、年龄和客群评分组合的内部 5 折频率编码。

## GPU/CPU 两模型候选与 OOF

- 原冠军：V2.5 blend_rank
- 原冠军 CV：0.9606599604603964
- 原冠军 Public LB：0.96035
- LightGBM + XGBoost GPU 最佳模型：blend_rank
- 新 3-Fold OOF：0.960614633696302
- CV 超越差值：-4.5326764094411054e-05
- 门槛：0.0001
- 达标：False
- 是否已提交：False
- 新 Public LB：尚未提交/未评分
- 线上冠军：V2.5 blend_rank

## 当前已评分冠军对应的官方排名

- 此名次关联的 Public LB：0.96035
- 当前 Public 排名：**第 931 名 / 2809 支队伍**
- 排名核验 UTC：2026-10-23T15:13:59.072679+00:00
- 排名证据：kaggle_authenticated_userRank
- 官方排行榜：https://www.kaggle.com/competitions/playground-series-s6e10/leaderboard

## 同行参考（思路参考，不照搬外部预测文件）

- GitHub: https://github.com/alvis1111/s6e10-airline-solution
- Kaggle ROC-AUC 评测：https://www.kaggle.com/competitions/playground-series-s6e10/overview
- 官方排行榜：https://www.kaggle.com/competitions/playground-series-s6e10/leaderboard

任何外部公开方法或成绩都不是本站实测。上线是否有效，只以本次真实 OOF 和官方 Public LB 判断。

| 候选 | 同3折 OOF AUC |
|---|---:|
| lightgbm | 0.96046789 |
| xgboost | 0.96049326 |
| blend_mean | 0.96060921 |
| blend_rank | 0.96061463 |
