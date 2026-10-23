# V2.7｜零新增 GPU 算力的双模型 OOF 融合

- 评估时间 UTC：2026-10-18T09:28:12.219519+00:00
- 比赛：playground-series-s6e10
- 状态：skipped_no_cv_gain
- 本轮不创建新的 Kaggle GPU / TPU Notebook，只复用自己已训练模型的 OOF。
- 训练行数：699635
- 两个模型同一个 StratifiedKFold(3, seed=20261018)，且标签/样本顺序需逐行完全一致。
- 融合比例：预设 50% V2.5 + 50% V2.6，先排名归一化，不用排行榜反复搜权重。

## 对比与提交安全门槛

- V2.5 3折 OOF：0.9606599601500012
- V2.6 3折 OOF：0.9606146334272929
- V2.7 融合 OOF：0.9606946829000839
- 当前最强独立 CV 基准：0.9606599604603964
- 需要超越的阈值：0.9607599605
- CV 门槛是否通过：False
- 是否提交 Kaggle：False
- 此次正式 Public LB：尚未提交或评分
- 旧冠军 Public LB：0.96035

## 当前冠军分数对应的官方名次

- 此名次关联的 Public LB：0.96035
- 当前 Public 排名：**第 939 名 / 2830 支队伍**
- 排名核验 UTC：2026-10-23T18:30:10.376032+00:00
- 排名证据：kaggle_authenticated_userRank
- 官方排行榜：https://www.kaggle.com/competitions/playground-series-s6e10/leaderboard

