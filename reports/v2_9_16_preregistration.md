# V2.9.16 — 训练前冻结实验设计（2026-10-21）

## 目标
基于已完成、官方 Public LB = 0.96099（submission 57446286）的 V2.9.15，训练 **一个独立的** 3-fold T4 RealMLP 特征交互模型，检验其能否为原有最优融合带来真实 OOF 增益。本文件与训练代码、工作流同一初始提交冻结；不会使用测试标签或官方 Public LB 寻找融合权重。

## 固定配置
- 竞赛：`playground-series-s6e10`
- 新 Notebook 唯一 ID：`mjw3177109/kas-v2916-interaction-realmlp-ens8-t4-s6e10`
- 模型：RealMLP-TD，`n_ens=8`、5 epochs、`lr=0.038`、`batch_size=256`
- 外层：`StratifiedKFold(n_splits=3, shuffle=True, random_state=20261018)`
- 模型种子：`20261024`；每折使用 `MODEL_SEED+fold`
- 既有交互字段、5-fold 内层 TargetEncoder 继续沿用 V2.9.15；新加 9 个输入派生字段：5 项机上服务的均值、标准差、最小值、低评分计数、WiFi-登机评分差、延误合计对数、延误差、服务均值/延误对数，以及飞行距离对数。
- 外层验证标签不参与 encoder fitting / model.fit；不利用测试集统计量拟合特征。
- 固定且只检验一个候选融合：`0.85 * rank(V2915 OOF) + 0.15 * rank(V2916 OOF)`
- 旧完整 3fold CV 精确基线：`0.9612850399332336`
- 晋级门槛：`candidate_auc >= 0.9613350399332336`（gain >= 0.00005），至少 2 折正增益，最差折增益严格大于 `-0.00005`。
- 复核 V2913/V2914/V2915 及更早总计 8 个历史来源的 SHA、OOF、test ID、预测哈希。
- 上一次提交记录必须为 `57446286 / 0.96099`；若不一致则 fail closed。

## 权限与停止条件
只允许真实 GPU 训练一次（通过 Kaggle owner 校验防重）。Notebook 以新的 slug 存在后，后续只能 `--recover-only`；任何状态都**不允许**本训练控制器直接调用 Kaggle competition submit。若未达标，仅保存真实结果并保持 V2915 官方冠军，不创建提交预约、不占用提交额度。GPU 不可用、三折不完整、校验失败、比赛截止或主人身份存疑时 fail closed。历史 Public LB 不用于推算 V2916 的成绩。
