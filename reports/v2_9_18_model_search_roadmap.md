# Kaggle S6E10 — V2.9.18+ 多架构模型搜索计划

研究日：2026-10-21。分支：feature/v29-autonomous-evolution。
本文件是实验计划，不是新榜单分数；未训练与通过 CV 但未提交的模型不可宣称为 Kaggle 官方成绩。

## 当前证据与模型边际收益
- 真实冠军 V2.9.15：3fold CV = 0.9612850399332336；官方 Public LB = 0.96099；submission ID = 57446286。
- V2.9.16：交互 RealMLP，OOF = 0.9605256773；85/15 融合 = 0.9613243650（+0.00003933），未过预设门槛。
- V2.9.17：7-epoch RealMLP，OOF = 0.9594506240；85/15 融合 = 0.9613002570（+0.00001522），未过门槛。
- 策略调整：优先研究不同模型和信息来源；不重复无限扫描 RealMLP 轮数与 Kaggle Public 权重。

## 多模型路线与当前阶段

| 版本 | 新模型 / 方法 | 预算 | 当前状态 |
| --- | --- | --- | --- |
| V2.9.18 | LightGBM + Exact Flight Distance route-profile 12 无标签特征 + 所有 21 原始字段及 2 类别组合的折内交叉 TargetEncoder（TE23） | Kaggle CPU，约 900 树 / 3 折 | 已编写 Kaggle 脚本 / GitHub Actions，等待实际 OOF |
| V2.9.19 | TabM 参数高效模型集成（区别于 RealMLP），固定航程画像 | T4 需 GPU 预算预检 | 研究，尚未训练 |
| V2.9.20 | XGBoost hist + 路线画像 + 频率编码，用不同树学习归纳偏差 | 优先 CPU | 研究，尚未训练 |
| V2.9.21 | 子群稳定性分析（Business、Eco、Delay、Distance）与有限的外部 CV 复验，筛掉不稳定提升 | CPU | 研究，尚未执行 |

## 外部证据，只用于选方向
- Kaggle 官方赛题：满意度二分类 ROC-AUC，截止 2026-10-31 23:59 UTC。https://www.kaggle.com/competitions/playground-series-s6e10/overview/about-the-tabular-playground-series
- 公开研究 alvis1111/s6e10-airline-solution：Exact Flight Distance、折内 TE、route profile、LightGBM、XGBoost、TabM 为独立思路。其自述分数基于不同训练数据/拆分，不能直接与本项目 CV 比较，也不等于我们的官方成绩。https://github.com/alvis1111/s6e10-airline-solution
- 公开 S6E10 多树算法基线：仅更换 CatBoost/LightGBM/XGBoost 的基线差异较小，仍需看特征与融合互补性。https://github.com/mrjohnsonsea/playground-series-s6e10
- pytabkit 提供 TabM、RealMLP 与 GBDT 统一接口，但高成本 TabM 需先做小规模 pilot。https://github.com/dholzmueller/pytabkit

## V2.9.18 实验预注册（禁止结果后改动）
1. 仅官方比赛 train/test/sample_submission：699635 训练行、299844 测试行；无外部标签。
2. 固定 StratifiedKFold 3 folds，shuffle=True，seed 20261018；每条训练 OOF 必须在其外层验证折以外完成训练。
3. 23 列 TargetEncoder(smooth=auto, cv=5)，fit_transform 仅对当前外层训练折做内层 5 折 cross-fit；outer-validation/test 只 transform，不碰标签；target 与 id 严禁入模。
4. Exact Flight Distance 的 12 个无标签 route-profile 统计仅在训练折计算；陌生航程默认训练折全局均值/route_count=0。
5. LightGBM：n_estimators=900，learning_rate=.035，num_leaves=47，min_child_samples=80，seed=20261026；不使用 outer validation 做 early stopping。
6. 新模型输出完整 3fold OOF、test vector，复核原 8 个 Notebook SHA、旧 V2913/V2915 官方候选预测 SHA、ID、标签、各折 OOF 一致性。
7. 只评估固定 rank 85% V2915 +15% V2918；基线 CV 0.9612850399332336。只有 OOF 增益 >= 0.00005，至少 2 折提高、最差折 > -0.00005，才晋级未提交候选。
8. 唯一 Kaggle CPU Notebook：mjw3177109/kas-v2918-routeprofile-lightgbm-cpu-s6e10。可安全只读恢复，不得重复创建。训练和三折验证管线绝不正式提交比赛。
9. 提交另需具备 SHA、额度、持久化一次性预约与去重的专门受控工作流；未授权不得直接提交。

## 交付文件
- kaggle_agent/v2918_route_profile_lgbm_train.py
- kaggle_agent/v2918_pipeline.py
- tests/test_v2918_route_profile.py
- .github/workflows/kaggle_v2918_route_profile_cpu.yml
- state/v2_9_18_status.json（实际运行时创建）
- reports/v2_9_18_route_profile_lgbm.md（实际运行时创建）

V2.9.19 TabM 应先做 GPU 显存、时间、pytabkit 版本与 CV pilot 验证；不直接尝试高成本 10 折/128 Epoch，避免免费 T4 作业被抢占或超时。

后续定期报告严格区分：研究、Kaggle 排队、Kaggle 训练、完整 OOF、晋级未提交、官方提交、Public LB 与认证排名。

## 新增真实实验记录（2026-10-21 UTC）
- V2.9.18 **完整三折已完成，不晋级**：独立 LightGBM OOF=0.9602322592841711；固定 85/15 融合 CV=0.9612515540913926；相对官方冠军 OOF=-0.00003348584184093，三折均负，skipped_no_cv_gain，无 Kaggle 提交。
- V2.9.19 **首轮 Kaggle ERROR，非模型分数**：原始 Notebook mjw3177109/kas-v2919-tabm-route-t4-s6e10 的日志显示 /kaggle/src/script.py 导入 feature_engineering 模块失败、其次导入 v2918_route_profile_lgbm_train 失败；故障唯一原因为 Kaggle 单脚本入口未发现随附辅助文件，尚无 TabM OOF。详情 reports/v2_9_19_failure_diagnostic.md。
- V2.9.19R **一次性兼容修复**：新的唯一 Notebook slug mjw3177109/kas-v2919r-tabm-selfcontained-t4-s6e10；把 V2918 自研航程画像/TE23 特征代码完整内嵌 main.py，保留 TabM-D k8 8epochs、seed 20261027、三折和固定85/15融合。必须首先验证前版本 kernel_failed 再开始，绝不覆盖旧 Kernel、绝不提交比赛。实际 CV 待 Kaggle 完成后才有结论。
- V2.9.20 仍作为第三模型架构 XGBoost hist CPU 的候选；不因同族 RealMLP 多次失败就宣称它会提高 AUC。

再次确认：CI SUCCESS 表示脚本任务无断言失败，不代表 Kaggle Notebook 训练完成；模型指标只以经 OOF 哈希及同折验证的状态文件为准。

## 多架构实测更新（2026-10-21 UTC）

### V2.9.18 LightGBM — 完整三折，不晋级
- Standalone OOF ROC-AUC: 0.9602322592841711.
- Frozen 85/15 rank blend: 0.9612515540913926; delta vs V2915: -0.00003348584184093.
- All folds were negative. No leaderboard submission.

### V2.9.19R2 TabM-D — 完整 T4 三折，不晋级
- Independent OOF ROC-AUC: 0.9592979143322833.
- Frozen 85/15 rank blend: 0.9612207697305537; delta vs V2915: -0.00006427020267985029.
- Fold deltas: -0.00008382486404445721, -0.00003698494477444303, -0.0000616598879653596; 0/3 folds improved.
- Original V2919 import error and V2919R missing-constant error remain archived; V2919R2 used a single-file script and completed correctly.

### V2.9.20 XGBoost Hist — 完整 CPU 三折，不晋级
- Independent OOF ROC-AUC: 0.9600177680486823.
- Frozen 85/15 rank blend: 0.9612378193529525; delta vs V2915: -0.00004722058028105991.
- Fold deltas: -0.00006602942093902886, -0.00004187696493418436, -0.00003495102258621596; 0/3 folds improved.

### V2.9.21 CatBoost Native Categorical — 正在执行
- New information path: 21 original predictors + 12 fold-fitted unlabelled exact-distance profile features = 33 total features.
- Four original categorical columns ('Gender', 'Customer Type', 'Type of Travel', 'Class') remain native strings for CatBoost's categorical statistics; unlike V2918/V2920 this model does not consume the custom TE23 matrix.
- Frozen parameters: 1,000 iterations, depth 7, learning rate 0.04, CPU only, seed 20261029, 3 outer folds.
- Current stage is recorded in state/v2_9_21_status.json. No result should be claimed until full OOF, ID/label alignment and historical SHA replay complete.

### V2.9.22 Subgroup Stability — Analysis-only
- kaggle_agent/v2922_subgroup_stability.py and tests/test_v2922_subgroup_stability.py are added.
- After V2921 reaches a terminal complete-OOF state, the diagnostic retrieves the exact OOF artifacts for LightGBM, TabM, XGBoost and CatBoost, aligns each by official id and target, reports subgroup AUCs (class, travel type, customer type, gender, distance quintile, delay bands and age bands) and pairwise Spearman rank correlations.
- This is descriptive analysis only: it does not train, submit, or search fusion weights. It will inform whether a future subgroup-specialist or residual-correction model is worth testing.

### Strategy change
Three new model families tested so far (LightGBM, TabM and XGBoost) all failed the pre-registered blend gate. The next useful step is native categorical learning (CatBoost) and subgroup/OOF diversity analysis, rather than trying more epochs or hand-tuning blend weights against the same OOF.
