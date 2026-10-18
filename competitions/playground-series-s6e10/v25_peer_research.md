# Kaggle S6E10｜V2.5 公开高分方案调查与自有改进实验

> 2026-10-18 调研。以下外部作者成绩为其自行公开说明，并非我们账户的成绩或官方核验名次。

## 为什么 0.95819 仍排名 1597/2221？

- 我们已经正式提交 0.95819 ROC-AUC（V2.3 CUDA XGBoost depth7，提交编号 57345271）。
- Kaggle 认证账户的最新公开榜排名曾为 1597 / 2221 队，核验时间 2026-10-18 08:08 UTC。
- 模型已使用真实 T4 GPU CUDA，说明瓶颈不在是否使用 GPU，而更可能在特征工程、数据模式发现及融合多样性。
- ROC-AUC 衡量概率排序，不是分类准确率。高密度排行榜中微小 AUC 差异也可能对应许多名次；最终以 Kaggle 官方名次为准。

## 参考公开高分实验

| 方案 | 作者公开报告 | 关键思路 |
|---|---|---|
| [alvis1111 S6E10 Solution](https://github.com/alvis1111/s6e10-airline-solution) | 0.96100 三路融合、后续自有改进 0.96126 Public | 全 21 字段 target encoding、精确航线距离画像、客群条件特征、神经网络融合 |
| [mrjohnsonsea baseline](https://github.com/mrjohnsonsea/playground-series-s6e10) | 5-fold XGB 0.9585、Optuna CV 0.9588 | 大量 XGBoost 超参数搜索仅获小幅提升 |
| [Kaggle 标签噪声探索](https://www.kaggle.com/code/dariushafshar/s6e10-the-gap-is-3-8-label-noise-not-features) | 标签噪声分析，不是我们成绩 | 接近天花板时小幅改进需要严格配对验证 |

以上外部成绩来自作者自述，未经我方复测，不可宣传成我们已达到的成绩。

## V2.5 已加入的可重复实验

1. 对原 21 字段的精确类别/数值字符串使用 sklearn TargetEncoder，严格采用外层 3 折和每个训练折内部 5 折 fit_transform 交叉拟合；验证和测试只调用 transform。
2. 按精确 Flight Distance 分组，计算仅由训练折得到的服务评分均值、标准差、残差、出现次数。
3. 条件客群特征：服务评分结合 Travel Type、Customer Type、Class；年龄/航程与客群的组合。
4. 两个独立模型：CPU LightGBM 与 T4 CUDA XGBoost，测试各自分数以及预先固定的 50/50 原始概率/排名融合（不对 OOF 大量搜融合权重）。
5. 不使用外部原始标签、他人预测文件、私榜信息，也不在验证折计算带标签统计。
6. 相同 StratifiedKFold (3 folds, seed=20261018) 与原 GPU 冠军 0.958531706 CV 比较：只有提升超过 0.00010 才提交 Kaggle。
7. 每次成绩报告附官方账号公开榜名次、总队伍数和核验时间，查不到名次就注明未核实。

## 为什么需要严格防止标签泄漏？

如果用 fit + transform 直接在训练数据上计算每行自己的目标均值，会让模型学习到包含真实标签的信息，导致训练分数失真；sklearn TargetEncoder 的 fit_transform 包含内部交叉拟合。参考 [Kaggle 的抗泄漏说明](https://www.kaggle.com/competitions/playground-series-s5e2/discussion/561801)。

## 后续实验顺序

第一阶段优先证实全字段 TE + 航线画像本身能否超过 V2.3；若确实提升，再评估 Kaggle 允许的数据源与使用条款下的原始数据特征、更多折数和可验证的 OOF 融合。绝不保证进入前三；先追求稳定的真实榜分提升。

[比赛官方评测与截止](https://www.kaggle.com/competitions/playground-series-s6e10/overview) · [官方排行榜](https://www.kaggle.com/competitions/playground-series-s6e10/leaderboard)
