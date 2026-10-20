# V2.9.7｜S6E10 提分路线：条件目标编码 × 上下文频率

## 当前基准（2026-10-19 UTC）

- 官方比赛：<https://www.kaggle.com/competitions/playground-series-s6e10>
- 最佳已验证可比完整三折 OOF：**0.9607535043719558**（V2.9.3 CPU rank blend）
- 正式已评分 Public LB：**0.96035**（V2.5 和 V2.9.3 均持平）
- V2.9.4 Original Teacher + Route T4：**0.9607525307822926**，已完整训练，未超过旧 CV；外部 Teacher 特征重要性较高但独立收益未体现。
- V2.9.5 RealMLP 首折：旧模型 0.96033171 → 固定 80/20 融合 0.96052992，单折改善 0.00019821；**不可冒充完整 CV**。
- V2.9.6 RealMLP 全三折：Kaggle GPU Notebook 已创建，最近状态仍 QUEUED，正式 CV 未核实。

## 从公开方案提炼的新证据

[公开复盘](https://github.com/alvis1111/s6e10-airline-solution) 记录：
1. **正确 cross-fit TargetEncoder**，平滑参数及精确数值编码的重要性；
2. **条件 Target Encoding**，尤其 rating × traveler_segment 和服务评分对 × 旅客子群体；
3. **5 折内层 frequency OOF**，避免同一个训练样本参与自己的频率表；
4. 另一条条件 TE + 频率树模型路线自述 OOF 约 **0.96114（与本仓库 3-fold 不可直接比较）**；
5. 外部 Teacher、航线统计、TabM、RealMLP 可以构成不同互补路线，但混合多个已高度类似的 GBDT 并不保证有效。

对比现有 [V2.9.4](../kaggle_agent/v294_original_teacher_train.py)：
- 已有全部原列的精确 TE、13 个服务评分 × 3 因子 segment、单一 pair TE，以及少量指定 key 的频率统计；
- **未完整包含**公开路线里的 5 组 *pair service ratings × (Travel|Customer)* 条件 TE；
- 目前频率特征以外层训练子集全量计数映射训练行，未对这些新条件组合执行内层训练行 OOF cross-fit；
- V2.9.4 还使用外部数据 Teacher，无法判断去掉其强特征后一个独立条件模型的互补性。

## 为什么不直接使用其他人的 Public LB 或模型预测

- 不同外层 folds、样本顺序、外部数据信号不能当作本站可比 CV；
- 不下载第三方预训练 OOF/test 后盲目加权，也不混合来源不清楚的 test 概率；
- 需每折记录 AUC 和原历史同折对照，Public LB 只有 Kaggle 提交后官方认证才能确认。
- Kaggle 该赛事官方的最终截止时间为 2026-10-31 23:59 UTC，Public 前三不代表 Private 前三或获奖。

## V2.9.7 受限、独立、可复现 CPU 实验

1. **模型**：单个 CPU LightGBM，预注册固定 1,100 trees、47 leaves，避免多轮 hyperparameter sweep。
2. **输入**：13 组服务分 × Travel/Customer/Class，5 组双服务分 × Travel/Customer，6 组 Age/Flight Distance × 类别组合；共 24 个高阶条件 keys。
3. **训练折编码**：外层固定 3 折（与 V2.5 / V2.8 同一 seed 20261018），训练行 TargetEncoder 5 折 fit_transform，验证/测试只 transform；另选 6 组条件键做 5 折 OOF frequency。
4. **高优先级互补信号**：模型自身完整三折 AUC + 旧基线 85% / 新条件模型 15% 的**唯一预注册融合**，不对验证集穷举权重。
5. **门槛**：融合 full CV 必须相对原最佳至少 +0.00006，至少两折改善，最差折不能低于 -0.00005。未达到则 `skipped_no_cv_gain`。
6. **资源**：**CPU-only Kaggle Notebook**，不占 V2.9.6 正排队的 Tesla T4；控制器强制比赛有效与唯一 kernel；可 recover-only，禁止自动提交。
7. **进化记录**：三折 CV、独立模型 AUC、各折配对增量、运行时间和失败都应持久化进入 `state/v2_9_experiments.json`。

## 本轮实现位置

- [训练端](../kaggle_agent/v297_conditional_cpu_train.py)
- [真实 CV 核验与模型选择](../kaggle_agent/v297_pipeline.py)
- [防泄漏 / 无提交 / 资源测试](../tests/test_v297_conditional.py)
- [CPU GitHub Actions](../.github/workflows/kaggle_v297_conditional_cpu.yml)
- 运行结果：`state/v2_9_7_status.json` 与 `reports/v2_9_7_conditional_cpu.md`（尚未完成时可能不存在）

## 成功与失败判定

**GitHub Actions 成功只是调度与控制器执行成功，不代表 Kaggle 完成训练。**
只有 Kaggle 显式 kernel_complete 且验证 3 折 CV、所有行、目标顺序、官方旧 OOF 对齐后，才能宣告新模型成果。若训练期间排队或失败，保留真实状态并在后续检查中只恢复已有 kernel；绝不创建相同 Notebook 的重复版本。
