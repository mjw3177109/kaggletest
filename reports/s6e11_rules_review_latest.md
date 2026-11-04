# Kaggle S6E11 官方规则只读复核

- Kaggle 参赛认证：True，来自已有认证门禁记录（此项未重新接受规则）
- 复核状态：**rules_text_fetched_manual_clause_audit_required**
- 原则：没有官方 S6E11 专属规则正文的完整证据，不自动批准训练、数据下载、外部模型或提交
- 禁止动作：自动接受规则、训练、下载受限数据、正式提交、使用付费资源

## 官网已公开确认
- 比赛：Predicting Credit Card Fraud（Playground Series S6E11）
- 指标：ROC AUC；正式提交列：\`id,is_fraud\`，预测概率在 [0,1]
- 最终期限：2026-11-30 23:59 UTC（以最新官方赛程为准）
- 奖励：前三名周边；不授 Kaggle Points / Medals；2026 同系列每人仅有一次周边奖励
- 公开资料：https://www.kaggle.com/competitions/playground-series-s6e11/overview

## Kaggle 认证元数据查询
- 官方 Page API：success，页面名：['About the Tabular Playground Series', 'Evaluation', 'Prizes', 'Timeline', 'abstract', 'data-description', 'rules']
- 是否取得完整专属 rules 内容：True
- 比赛规则 SHA256：b09ffa98473a049145fb76ab9246d182169010662c684114b36b3bbce2132b7e
- 专属队伍上限表达：three (3)
- 专属每日提交限制表达：待核实
- 专属最终计分选中数量表达：two (2)
- 数据清单查询：success；仅文件元信息，没有下载
- 文件元信息：[{'name': 'sample_submission.csv', 'total_bytes': 8399564}, {'name': 'test.csv', 'total_bytes': 41068069}, {'name': 'train.csv', 'total_bytes': 98533176}]
- 文件列表是否可能有下一页：False
- 官方账户当下提交额度接口：{'status': 'success', 'num_today': 0, 'num_total': 0, 'num_allowed_now': 10, 'limited_by_total': False}
- 已有正式提交历史接口：{'status': 'success', 'visible_count_first_page': 0, 'has_any': False}

## 专属规则可核查片段（非完整法律文本）
### external_data

EXTERNAL DATA AND TOOLS a. You may use data other than the Competition Data (“External Data”) to develop and test your Submissions. However, you will ensure the External Data is either publicly available and equally accessible to use by all Participants of the Competition for purposes of the competition at no cost to the other Participants, or satisfies the Reasonableness criteria as outlined in Section 2.6.b below. The ability to use External Data under this Section does not limit your other obligations under these Competition Rules, including but not limited to Section 2.8 (Winners Obligations). b. The use of external data and models is acceptable unless specifically prohibited by the Host. Because of the potential costs or restrictions (e.g., “geo restrictions”) associated with obtaining rights to use external data or certain software and associated tools, their use must be “reasonably accessible to all” and of “minimal cost”. Also, regardless of the cost challenges as they might affect all Participants during the course of the competition, the costs of potentially procuring a license for software used to generate a Submission, m

### team_limits

TEAM LIMITS a. The maximum Team size is three (3). b. Team mergers are allowed and can be performed by the Team leader. In order to merge, the combined Team must have a total Submission count less than or equal to the maximum allowed as of the Team Merger Deadline. The maximum allowed is the number of Submissions per day multiplied by the number of days the competition has been running. ####2. SUBMISSION LIMITS a. You may submit a maximum of ten (10) Submissions per day. b. You may select up to two (2) Final Submissions for judging. ####3. COMPETITION TIMELINE a. Competition Timeline dates (including Entry Deadline, Final Submission Deadline, Start Date, and Team Merger Deadline, as applicable) are reflected on the competition’s Overview > Timeline page. ####4. COMPETITION DATA a. Data Access and Use. *Competition Use and Commercial are checked*: You may access and use the Competition Data for any purpose, whether commercial or non-commercial, including for participating in the Competition and on Kaggle.com forums, and for academic research and education. The Competition Sponsor reserves the right to disqualify any Participant who u

### submission_limits

SUBMISSION LIMITS a. You may submit a maximum of ten (10) Submissions per day. b. You may select up to two (2) Final Submissions for judging. ####3. COMPETITION TIMELINE a. Competition Timeline dates (including Entry Deadline, Final Submission Deadline, Start Date, and Team Merger Deadline, as applicable) are reflected on the competition’s Overview > Timeline page. ####4. COMPETITION DATA a. Data Access and Use. *Competition Use and Commercial are checked*: You may access and use the Competition Data for any purpose, whether commercial or non-commercial, including for participating in the Competition and on Kaggle.com forums, and for academic research and education. The Competition Sponsor reserves the right to disqualify any Participant who uses the Competition Data other than as permitted by the Competition Website and these Rules. The Competition Data is also subject to the following terms and conditions: [Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). b. Data Security. 1. You agree to use reasonable and suitable measures to prevent persons who have not formally agreed to these Rules fro

### data_license

DATA ACCESS AND USE Attribution 4.0 International (CC BY 4.0) ###2. COMPETITION-SPECIFIC RULES In addition to the provisions of the General Competition Rules below, you understand and agree to these Competition-Specific Rules required by the Competition Sponsor: ####1. TEAM LIMITS a. The maximum Team size is three (3). b. Team mergers are allowed and can be performed by the Team leader. In order to merge, the combined Team must have a total Submission count less than or equal to the maximum allowed as of the Team Merger Deadline. The maximum allowed is the number of Submissions per day multiplied by the number of days the competition has been running. ####2. SUBMISSION LIMITS a. You may submit a maximum of ten (10) Submissions per day. b. You may select up to two (2) Final Submissions for judging. ####3. COMPETITION TIMELINE a. Competition Timeline dates (including Entry Deadline, Final Submission Deadline, Start Date, and Team Merger Deadline, as applicable) are reflected on the competition’s Overview > Timeline page. ####4. COMPETITION DATA a. Data Access and Use. *Competition Use and Commercial are checked*: You may access and us

### eligibility

ELIGIBILITY a. Unless otherwise stated in the Competition-Specific Rules above or prohibited by internal policies of the Competition Entities, employees, interns, contractors, officers and directors of Competition Entities may enter and participate in the Competition, but are not eligible to win any Prizes. "Competition Entities" means the Competition Sponsor, Kaggle Inc., and their respective parent companies, subsidiaries and affiliates. If you are such a Participant from a Competition Entity, you are subject to all applicable internal policies of your employer with respect to your participation. ####8. WINNER’S OBLIGATIONS a. As a condition to being awarded a Prize, a Prize winner must fulfill the following obligations: 1. Deliver to the Competition Sponsor the final model's software code as used to generate the winning Submission and associated documentation. The delivered software code should follow [these documentation guidelines][5], must be capable of generating the winning Submission, and contain a description of resources required to build and/or run the executable code successfully. For avoidance of doubt, delivered softw

## 当前门禁与待补证
- 账户加入不代表已经逐项审计所有比赛特定条款；本报告不替用户接受协议。
- 专属规则全文、资格及地域限制、外部数据、预训练模型、生成式 AI 使用许可：未明确验证前不擅自允许。
- 数据集许可证、实际文件格式/哈希、行数及目标分布：需要先通过获准的官方只读元数据/许可核验；目前不可假定。
- 训练只使用比赛允许的官方数据和零增费资源；任何不明确环节都 fail-closed。
- 后续：验证完成后另行办理独立数据/OOF/提交门禁；本次不启动训练，也不修改原报名门禁。

- 查询阶段：待核实
- 异常类（不包含凭据）：待核实
