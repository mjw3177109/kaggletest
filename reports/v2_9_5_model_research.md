# V2.9.5 模型优选研究（S6E10）

研究日期：2026-10-19。原则：不能从更大/更贵/更新的模型推断更高 Kaggle ROC-AUC；必须对齐同一 OOF 划分，分别验证独立模型表现与互补价值。

## 有来源的模型证据

| 候选 | 公开可核对的结果 | 适合本站的定位 |
|---|---|---|
| RealMLP-TD | alvis1111 独立复现称 10 折单模型 OOF 0.961309，将 RealMLP 加进自身融合使 Public LB 从 0.96100 到 0.96121 | **P0**：不同于 LightGBM/XGBoost/CatBoost 的神经网络模型 |
| TabM | 同一公开仓库的三路融合中引入 TabM 作为结构互补，单模型表现不一定优于 GBDT | P1：RealMLP 提升稳定后考虑更高计算成本 |
| 深层 XGBoost | 独立复现报告在其融合中仅 +0.00001 Public LB | P2：V2.9.4 已使用独立深树试验，暂缓重复 |
| TabPFN 大规模 | 公开复现显示 32k 小样本版本不理想，全数据高上下文资源需求高 | 预算限制，不优先 |
| 继续 CatBoost 深度调整 | 本项目 V2.9.3 Depth-8 CV 0.9607375，弱于我们的 0.9607535 | 暂停 |

来源：
- https://github.com/alvis1111/s6e10-airline-solution/blob/main/independent_repro/S6E10%E7%8B%AC%E7%AB%8B%E5%A4%8D%E7%8E%B0%E6%80%BB%E7%BB%93.md
- https://github.com/alvis1111/s6e10-airline-solution/blob/main/independent_repro/realmlp_repro_lean.py
- https://github.com/dholzmueller/pytabkit
- https://pytabkit.readthedocs.io/en/latest/models/01_sklearn_interfaces.html

**以上分数来自别人的 10 折和模型组合，不可与我们的 3 折 CV 直接作优劣结论；公开作者不是本届榜单前三本人。**

## V2.9.5 一阶段：资源受限的 RealMLP 首折 pilot

使用与历史 V2.5 / V2.8 一致的 StratifiedKFold(3, shuffle=True, random_state=20261018)，仅测试其中第 0 折：

1. Kaggle T4 实际 GPU 证明，RealMLP_TD_Classifier GPU 模式，4 个子模型、3 个固定 epoch，不用外层验证标签早停。
2. 21 原始字段、数值列 dual categorical 表示、少数固定组合、训练折-only count、内层 5-fold cross-fit TargetEncoder。
3. 输出唯一折 val indices、val target、val 概率和硬件时间，不输出 Kaggle submission。
4. 使用此前已归档的 V2.5 / V2.8 官方 Kaggle Notebook OOF，重建本站 V2.9.3 35/65 基线（整体 OOF 应再现 0.96075350437）。
5. **只在相同第 0 折**比较 (a) RealMLP AUC、(b) 预注册 80/20 rank 融合 AUC、(c) 相同基线 AUC 和配对增量。
6. 若单折融合 +0.00005，标记「值得扩到剩余两折」；**绝不**据单折实验晋升冠军、修改完整 CV 或提交 Kaggle。
7. 排队规则：在 `state/v2_9_4_status.json` 仍 queued/running 时严格阻断启动 RealMLP，防止抢占免费 GPU。

## 下一阶段（仅一阶段通过后）

完成相同 3-fold 的完整 RealMLP OOF，锁定两组预设融合策略，验证每折一致性与 Bootstrap / DeLong 置信范围。在全部证据通过、训练日志与候选 test 概率齐备之后，才评估是否有足够证据做一次 Kaggle 正式提交。

现有正式 Public LB 0.96035；当前最佳已测可比 3fold OOF 0.96075350437。严禁将别人 Public LB、我们的单折 AUC 或 GPU Github Actions success 误称为本站新成绩。

### 代码入口

- `kaggle_agent/v295_realmlp_pilot_train.py`：独立 Kaggle T4 首折训练。
- `kaggle_agent/v295_realmlp_pilot.py`：安全推送与恢复、同步 Kaggle OOF 和标准比较。
- `tests/test_v295_realmlp.py`：折内映射、资源锁、无自动提交测试。
- `.github/workflows/kaggle_v295_realmlp_pilot.yml`：开发分支专项测试及资源门控。
