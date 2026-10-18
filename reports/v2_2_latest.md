# Kaggle Auto Scientist V2.2 — 自动参数搜索与实验复盘

- 比赛：playground-series-s6e10
- 当前阶段：skipped_no_cv_gain
- Kaggle Notebook：https://www.kaggle.com/code/mjw3177109/kas-v22-playground-series-s6e10
- 上一轮 V2.1 Public LB：0.95774
- 上一轮 V2.1 3-Fold OOF：0.9582786744608702

## 本轮做了什么

Optuna TPE 搜索 LightGBM 超参数；然后用与 V2.1 相同的 3-Fold
StratifiedKFold 和随机种子，分别验证旧特征方案与扩展特征方案。
两套方案都完成后才决定是否提交。重要性使用 LightGBM Gain。

- Optuna trials：8 / 8
- 本轮最好的候选：tuned_lightgbm

| 方案 | Full 3-Fold OOF ROC-AUC |
|---|---:|
| tuned_lightgbm | 0.95794477 |
| tuned_extended | 0.95788612 |

- 与 V2.1 3-Fold 的差值：-0.0003339029020230644
- 提交阈值：必须提高超过 8e-05
- 是否通过：否（保留原冠军）

## Top 15 特征重要性（Gain）

| 特征 | Gain |
|---|---:|
| Online boarding | 3285135.870921 |
| Inflight wifi service | 1232790.514248 |
| Class_Business | 877965.215496 |
| Type of Travel_Business travel | 634553.986636 |
| Customer Type_Loyal Customer | 292692.026415 |
| __auto_rating_max | 288917.150908 |
| __auto_rating_mean | 241595.428526 |
| Type of Travel_Personal Travel | 236085.939411 |
| Inflight entertainment | 235834.027811 |
| Class_Eco | 165725.384358 |
| Baggage handling | 153858.352398 |
| Seat comfort | 120425.402543 |
| Ease of Online booking | 117872.448999 |
| Checkin service | 114822.555796 |
| On-board service | 73974.226514 |

## Kaggle 结果

- V2.2 提交：False
- V2.2 Public LB：未提交
- 新旧线上分数差：None
- 最终冠军：V2.1 ensemble

## 下一轮建议

若本轮未通过 3-Fold 门槛，可重点分析服务评分、出行类型、延误时长相关交互特征；
避免仅根据一次 Public LB 调参，以免过拟合公共榜。
