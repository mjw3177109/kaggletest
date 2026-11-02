# Kaggle S6E11｜只读立项与参赛门禁

- 记录时间：2026-11-02 03:04 UTC（北京时间 11:04）
- 官方比赛：https://www.kaggle.com/competitions/playground-series-s6e11
- Kaggle 官方 CLI 认证发现：`playground-series-s6e11`，`Playground`，`Predicting Credit Card Fraud`
- Kaggle CLI 显示最终截止：**2026-11-30 23:59:00 UTC**（北京时间 2026-12-01 07:59）
- Kaggle CLI 显示奖品：`3 Swag`；**不是现金奖金，也不等于已经获奖**
- 2026-11-02 03:04 UTC 认证账户查询：`userHasEntered=false`；当时未加入比赛，无法合法下载受限数据或提交
- 来源：`state/competitions.json`（由 GitHub Actions Kaggle Scout #41380988948 通过配置的 KAGGLE_API_TOKEN 只读取得）
- 竞争优先级：优先于宝可梦对战智能体赛（同为 Playground，但表格欺诈预测更适合既有免费算力和可复用无泄漏 CV 工程）

## 当前状态

**blocked_awaiting_user_rule_acceptance**。用户已授权自动研究、训练、合规提交，但不授权代替其接受新比赛法律条款。需用户本人在官方 Kaggle 比赛页面阅读并接受规则。下一次自动巡检必须再次读取 `--group entered`，仅 `userHasEntered=true` 才解除这一门禁；不能凭此文档代替认证。

## 只读预研（非已验证比赛指标）

1. 进入比赛后读取官方 Overview、Rules、Evaluation、Data 页面，确认实际评分指标、目标列、样本 ID、训练测试行数、提交格式、允许的外部数据/预训练模型/生成式 AI 使用限制、提交次数及奖品资格。
2. 建立独立 `competitions/playground-series-s6e11/` 实验目录，禁止复用 S6E10 的提交文件、目标列、模型训练结果或门禁预约。
3. 先做数据审计与防泄漏基线：分层 CV（是否适用需看目标分布）、LightGBM/CatBoost/Logistic 等候选；所有预处理、编码、特征选择折内拟合；独立 OOF 及每折指标；数据和候选 SHA 留档。
4. 严格零额外付费：只用 Kaggle 免费配额或已授权的免费计算资源；未知配额或比赛规则时不启动训练。
5. 首次正式提交前重新认证：比赛已加入、规则允许、截止未过、额度足够、文件 schema 和 SHA 验收通过、官方提交历史无重复、GitHub Actions 全仓库五态空闲。提交后记录官方 submission ID、状态和 Public LB，赛后再核对最终 Private 榜和官方奖品通知。

## 与 S6E10 的隔离

S6E10 赛后报告见 `reports/2026-11-02_S6E10_赛后归纳.md`。V2941 永久禁止重试；V2942 已有唯一 run 41289631378，不再触发训练或逾期提交。S6E10 最终 Private 排名及奖品资格仍未认证，不得臆测。

> 此文件仅为只读立项，未接受规则、未下载数据、未训练、未提交、未产生费用。
