# V2.9.3｜Kaggle 真实 OOF 模型融合实验

- 检查 UTC：2026-10-19T07:47:38.981646+00:00
- 阶段：submission_reserved
- 竞赛：playground-series-s6e10
- 训练来源：本人 V2.5 和 V2.8 Kaggle Notebook 的既有 GPU OOF；本轮不重复 GPU 训练。
- CV 最强历史基准：0.9607259172925492
- 本轮最佳 CV：0.9607535043719558
- CV 提升：2.758707940664351e-05
- 最低提升门槛：2e-05
- 候选：v25_35_v28_65
- 是否已发送新的 Kaggle 提交：False
- 新 Public LB：未评分/未提交
- 旧 Public LB：0.96035
- 官方名次需独立通过认证排行追踪器核验，本报告不按分数推测。
- Public 榜前三不等于最终 Private 榜前三或获奖。

## 预注册候选 CV（相同三折、相同训练行顺序）

| 方法 | 3-fold OOF AUC | vs 最强 CV | 状态 |
|---|---:|---:|---|
| v25_60_v28_40 | 0.9607400931315118 | 1.4175838962571241e-05 | evaluated |
| v25_35_v28_65 | 0.9607535043719558 | 2.758707940664351e-05 | evaluated |
| xgb_50_cat_50 | — | — | unavailable_test_predictions |
| v25_25_v28_50_cat_25 | — | — | unavailable_test_predictions |

- 唯一提交标识：KAS V293 BLEND 033383e58777
- Kaggle notebook V2.5：https://www.kaggle.com/code/mjw3177109/kas-v25-gpu-te-playground-series-s6e10
- Kaggle notebook V2.8：https://www.kaggle.com/code/mjw3177109/kas-v28-catboost-playground-series-s6e10
