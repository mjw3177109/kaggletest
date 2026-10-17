# Kaggle Auto Scientist V1

这是一个把 GitHub 与 Kaggle 串起来的自动参赛实验框架。

当前 V1 已完成第一条最小闭环：

1. GitHub Actions 读取仓库 Secret `KAGGLE_API_TOKEN`
2. 调用 Kaggle CLI 验证认证
3. 拉取当前比赛列表
4. 自动排序并筛选候选比赛
5. 生成 `state/competitions.json`
6. 生成 `reports/scout_latest.md`
7. 自动 commit + push 回当前仓库

> V1 暂时不会自动接受比赛规则，也不会自动提交比赛结果。

## 第一次运行

进入 GitHub 仓库：

`Actions -> Kaggle Scout -> Run workflow`

等待工作流执行结束。

成功后，仓库会自动新增/刷新：

- `state/competitions.json`
- `reports/scout_latest.md`

## Secret

仓库必须存在：

`Settings -> Secrets and variables -> Actions -> KAGGLE_API_TOKEN`

不要把真实 token 写入代码、README、issue 或 commit。

## 当前结构

```text
.github/workflows/kaggle_scout.yml
config/agent.yaml
kaggle_agent/scout.py
reports/
state/
requirements.txt
```

## V1 后续阶段

认证与 Scout 验证通过后继续加入：

- Competition Analyzer：读取题目、评价指标、数据说明
- Data Profiler：自动 EDA
- Solution Planner：自动生成 baseline / CV / 特征工程方案
- Training Generator：生成 Kaggle Notebook / Python 训练代码
- Kaggle Runner：推送 Notebook 并等待训练结束
- Submission Manager：生成并提交 submission.csv
- Result Reviewer：读取 leaderboard score
- Experiment Loop：根据 CV + LB 自动提出下一轮实验
- GitHub Writer：把每轮代码、报告、分数写回仓库

## 安全原则

- 不自动接受比赛 Rules
- 未验证前不自动提交
- Token 只从 GitHub Secret 注入
- `.gitignore` 已忽略 `kaggle.json`、`access_token`、`.kaggle/` 和 `.env*`
