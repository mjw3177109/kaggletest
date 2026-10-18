# Kaggle Auto Scientist

This repository is an automated Kaggle competition lab.

## 新手先看这里

每个比赛项目都会自动生成：

`competitions/<比赛名>/小白说明书.md`

如果你完全不懂 Kaggle、机器学习、CV、Public LB、submission，**先看这份文件，不要先看代码**。它会用白话说明：这题在做什么、数据是什么、模型怎么训练、分数怎么看、每轮实验改了什么、怎么自己重跑、常见报错怎么办。

统一说明书规范在：`docs/BEGINNER_GUIDE_STANDARD.md`。

## Current version: V2 + V2.1

V2 can authenticate with Kaggle, find currently-open competitions, automatically prefer Playground / Getting Started tasks, inspect competition pages, download data after you have accepted the competition rules, profile the dataset, generate a baseline, run it as a Kaggle Kernel, submit `submission.csv`, collect the latest score, and write the run artifacts back to this GitHub repository.

V2.1 continues from a completed V2 project and performs multi-model leaderboard optimization. It compares boosted-tree candidates with cross-validation, selects the best local candidate, submits it, compares the new Public LB against the previous online baseline, records every experiment, and refreshes the beginner guide.

### Required GitHub Secret

`KAGGLE_API_TOKEN`

Optional: `KAGGLE_USERNAME`. If omitted, the kernel owner falls back to the GitHub repository owner. Add `KAGGLE_USERNAME` only if your Kaggle username is different from your GitHub username.

## 已完成的实战成绩

- 比赛：[`playground-series-s6e10`](https://www.kaggle.com/competitions/playground-series-s6e10)
- Run 001（V2）Public LB：**0.95738**
- Run 005（V2.1 Ensemble）Public LB：**0.95774**（提升 **+0.00036**）
- 最佳 V2.1 3-Fold OOF ROC-AUC：**0.95827867**
- 自动生成的[小白说明书](competitions/playground-series-s6e10/小白说明书.md)
- [成绩和模型比较报告](reports/v2_1_latest.md)

V2.1 已完成并归档。再次运行相同配置时默认避免重复训练和提交；需要新的实验应修改实验设计并明确启用重新训练，而不是盲目重复提交。

## Run V2

Open **Actions → Kaggle Auto Scientist V2 → Run workflow**.

V2 and V2.1 currently start only when you explicitly choose **Run workflow**, to avoid duplicate long training runs and Kaggle submissions after a code or documentation edit. Daily scouting and beginner-guide updates remain automated.

### If the run stops at `rules_required`

This is expected for a competition you have not joined yet. Open `reports/v2_latest.md`, follow the selected Kaggle competition link, click **Join Competition**, accept the competition rules yourself, then rerun the workflow.

After rules are accepted, V2 continues automatically:

```text
select competition
-> download data
-> analyze data
-> generate baseline
-> Kaggle Kernel training
-> produce submission.csv
-> submit
-> collect score
-> generate/refresh 小白说明书.md
-> write results back to GitHub
```

## Run V2.1

After V2 has completed, open **Actions → Kaggle Auto Scientist V2.1 → Run workflow**.

V2.1 performs:

```text
load V2 champion
-> run multiple candidate models
-> cross-validation / OOF comparison
-> build ensemble candidate
-> choose best CV candidate
-> submit one candidate
-> compare Public LB with V2
-> record experiments
-> refresh 小白说明书.md
```

## Output

- `state/v2_candidates.json` — ranked open competitions
- `state/v2_selection.json` — selected competition
- `state/v2_status.json` — current V2 stage/status
- `state/v2_1_status.json` — current V2.1 optimizer status
- `reports/v2_latest.md` — latest V2 report
- `reports/v2_1_latest.md` — latest V2.1 leaderboard optimization report
- `competitions/<slug>/小白说明书.md` — beginner-first explanation for this project
- `competitions/<slug>/problem_analysis.md`
- `competitions/<slug>/data_analysis.md`
- `competitions/<slug>/solution_plan.md`
- `competitions/<slug>/experiments/` — per-run experiment records
- `competitions/<slug>/kaggle/` — V2 Kaggle training package
- `competitions/<slug>/kaggle_v21/` — V2.1 optimizer package
- `competitions/<slug>/results/` — V2 results
- `competitions/<slug>/results_v21/` — V2.1 results

Raw competition data, credentials, models, and large `submission.csv` files are intentionally excluded from Git.

Recovery workflow can re-read an already pushed Kaggle kernel without launching new training; it is available under **Actions → Kaggle Auto Scientist V2.1 Recovery**.
