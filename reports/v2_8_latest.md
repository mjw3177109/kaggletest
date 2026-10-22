# V2.8｜CatBoost GPU 多样化建模实验

- 生成时间 UTC：2026-10-18T09:39:53.668160+00:00
- 比赛：playground-series-s6e10
- Kaggle Notebook：https://www.kaggle.com/code/mjw3177109/kas-v28-catboost-playground-series-s6e10
- 运行阶段：skipped_no_cv_gain
- 真实 T4：True；['Tesla T4, 15360 MiB', 'Tesla T4, 15360 MiB']

## 为什么改进？

V2.5 使用无泄漏 Target Encoding + 航线画像获得 Public LB 0.96035，已成为当前正式冠军。
公开高分实验提示，精确数值交叉拟合 Target Encoding、航线距离画像和旅客细分群特征更有价值。
V2.8 保留 V2.5 已验证的内层 5 折无泄漏目标编码，新增真正运行在 NVIDIA T4 上的 CatBoost 算法分支，寻找不同树模型的互补性。

## GPU/CPU 两模型候选与 OOF

- 原冠军：V2.5 blend_rank
- 原冠军 CV：0.9606599604603964
- 原冠军 Public LB：0.96035
- LightGBM CPU + CatBoost GPU 最佳模型：blend_rank
- 新 3-Fold OOF：0.9607259172925492
- CV 超越差值：6.595683215282477e-05
- 门槛：0.0001
- 达标：False
- 是否已提交：False
- 新 Public LB：尚未提交/未评分
- 线上冠军：V2.5 blend_rank

## 当前已评分冠军对应的官方排名

- 此名次关联的 Public LB：0.96035
- 当前 Public 排名：**第 872 名 / 2647 支队伍**
- 排名核验 UTC：2026-10-22T05:04:38.446251+00:00
- 排名证据：kaggle_authenticated_userRank
- 官方排行榜：https://www.kaggle.com/competitions/playground-series-s6e10/leaderboard

## 同行参考（思路参考，不照搬外部预测文件）

- GitHub: https://github.com/alvis1111/s6e10-airline-solution
- Kaggle ROC-AUC 评测：https://www.kaggle.com/competitions/playground-series-s6e10/overview
- 官方排行榜：https://www.kaggle.com/competitions/playground-series-s6e10/leaderboard

任何外部公开方法或成绩都不是本站实测。上线是否有效，只以本次真实 OOF 和官方 Public LB 判断。

| 候选 | 同3折 OOF AUC |
|---|---:|
| lightgbm | 0.96053020 |
| catboost | 0.96050902 |
| blend_mean | 0.96070751 |
| blend_rank | 0.96072592 |
