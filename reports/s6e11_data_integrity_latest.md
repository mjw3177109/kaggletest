# S6E11 真实数据完整性只读验收

- 竞赛：playground-series-s6e11
- 状态：data_integrity_verified
- 当前验收阶段：finished
- Kaggle 规则哈希：b09ffa98473a049145fb76ab9246d182169010662c684114b36b3bbce2132b7e
- 额外付费：禁止（GitHub 公共仓库已有 Actions 环境，未使用云端付费算力）
- 本流程不训练、不下载外部数据、不提交、不写入训练代码/样本/预测到公共仓库

## train.csv
- SHA256：87eaa10b8d17a7e4f657b8da1c66242431338bdd8b6ad45861521ac0c0e9fbd5
- 行数：699960
- 列：["id", "amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km", "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count", "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes", "merchant_category", "card_type", "auth_method", "channel", "device_type", "is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt", "is_fraud"]
- 缺失统计：{}

## test.csv
- SHA256：10f516959020357b565abaaa4fd25baae97fc2f536d0805695bb7cc15c1660f0
- 行数：299984
- 列：["id", "amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km", "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count", "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes", "merchant_category", "card_type", "auth_method", "channel", "device_type", "is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
- 缺失统计：{}

## sample_submission.csv
- SHA256：13fc6ed8b73de5329884ec490f2a0b750cf29a7cda772b78cfa0cf486f6f35fa
- 行数：299984
- 列：["id", "is_fraud"]
- 缺失统计：{}

- 欺诈样本占比：0.016699525687182123
- train/test ID 无交叉，sample ID 集合与 test 精确相等：是

此结果仅证明数据结构/文件完整性，不能证明无目标泄漏、分布无偏或模型可提交。
后续训练/提交须独立的 CV/OOF 和来源 SHA 门禁。
