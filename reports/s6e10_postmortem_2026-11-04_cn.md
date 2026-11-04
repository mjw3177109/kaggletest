# Kaggle S6E10 赛后归纳（核验中）

更新：2026-11-04（UTC+8）。比赛：playground-series-s6e10，Predicting Airline Satisfaction；截止：2026-10-31 23:59 UTC。禁止逾期提交。

## 已核验的 GitHub 证据
- V2942 GitHub Actions run 41289631378：job 124797974966 已 completed/failure；独立冠军重建及完整 OOF 验证（step 12）失败；step 13 终止失败预约，step 14 持久化结果，step 15 上传证据成功。证据 artifact 13077536838。不能把这次训练算作通过验证或已提交。
- 持久化 state/v2_9_42_status.json：stage=validation_mismatch，submitted=false，submission_attempted=false，run_reservation_id=41289631378，error=Competition deadline expired。状态中仍有过时字段 training_started=false / execution=not_implemented，和实际已执行训练的 job steps 冲突，需以 run 日志为执行证据，不可依赖单一状态字段。
- V2941 retry_prohibited=true，永久不得重试。
- 历史 V2915：CV 0.9612850399332336、Public LB 0.96099、submission ID 57446286；均不等于最终成绩。历史 Public rank 984/2931（2026-10-25），不代表最终名次。

## 未核实，严禁推断
- 未取得 Kaggle 账户认证最终 Public/Private 排名、最终提交列表、奖品资格或领奖通知；均记为 **未知**。不能宣布获奖。
- 尚未完成全仓库五态 Actions 与所有分支 HEAD 的独立全量分页核验；本报告不是全面无冲突证明。

## 技术复盘
- 计划：37 基础特征，三折 OOF，数值 RFF128 + LogisticRegression，严格折内拟合与目标编码。
- V2942 安全测试通过，但独立 OOF 验证失败；未形成可提升的正式候选。优先从失败日志定位独立重建差异，不重复训练，不重新提交。
- 经验：持久化状态字段必须与实际 job/step 对账；失败不能被安全测试成功掩盖；所有实验必须有 SHA、OOF 和终态证据。

## 下一场门禁
官方竞赛目录已检索到其他公开竞赛，但开放资格、具体规则、账户接受状态、数据规模、模型与外部数据政策、截止时间均需逐项核实。未完成前不接受条款、不训练、不提交。

来源：https://github.com/mjw3177109/kaggletest/actions/runs/41289631378 ; https://www.kaggle.com/competitions/playground-series-s6e10 ; https://www.kaggle.com/competitions
