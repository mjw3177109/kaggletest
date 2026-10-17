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

## Run V2

Open **Actions → Kaggle Auto Scientist V2 → Run workflow**.

The workflow is also triggered automatically when V2 code/config changes.

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
