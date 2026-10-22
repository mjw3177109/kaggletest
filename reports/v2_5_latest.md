# V2.5｜研究高分选手后的 GPU 特征工程实验

- 生成时间 UTC：2026-10-18T08:34:55.555020+00:00
- 比赛：playground-series-s6e10
- Kaggle Notebook：https://www.kaggle.com/code/mjw3177109/kas-v25-gpu-te-playground-series-s6e10
- 运行阶段：complete
- 真实 T4：True；['Tesla T4, 15360 MiB', 'Tesla T4, 15360 MiB']

## 为什么改进？

V2.3 GPU XGBoost 只针对原始特征和少量统计做调参，Public LB 0.95819。
公开高分实验提示，精确数值交叉拟合 Target Encoding、航线距离画像和旅客细分群特征更有价值。
V2.5 新增 5 折内部交叉拟合 TE；每个外层验证折只使用对应训练折生成的特征，禁止目标泄漏。

## GPU/CPU 两模型候选与 OOF

- 原冠军：V2.3 GPU gpu_xgb_depth7
- 原冠军 CV：0.9585317063733847
- 原冠军 Public LB：0.95819
- LightGBM + XGBoost GPU 最佳模型：blend_rank
- 新 3-Fold OOF：0.9606599604603964
- CV 超越差值：0.0021282540870116717
- 门槛：0.0001
- 达标：True
- 是否已提交：True
- 新 Public LB：0.96035
- 线上冠军：V2.5 blend_rank

## 当前已评分冠军对应的官方排名

- 此名次关联的 Public LB：0.96035
- 当前 Public 排名：**第 873 名 / 2648 支队伍**
- 排名核验 UTC：2026-10-22T05:22:57.919312+00:00
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
| xgboost | 0.96050532 |
| blend_mean | 0.96065257 |
| blend_rank | 0.96065996 |
