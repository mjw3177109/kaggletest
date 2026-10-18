# Kaggle Auto Scientist

This repository is an automated Kaggle competition lab.

## 新手先看这里

每个比赛项目都会自动生成：

`competitions/<比赛名>/小白说明书.md`

如果你完全不懂 Kaggle、机器学习、CV、Public LB、submission，**先看这份文件，不要先看代码**。它会用白话说明：这题在做什么、数据是什么、模型怎么训练、分数怎么看、每轮实验改了什么、怎么自己重跑、常见报错怎么办。

统一说明书规范在：`docs/BEGINNER_GUIDE_STANDARD.md`。

## Current version: V2 + V2.1 + V2.2 + V2.3 TPU

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

## V2.2 — Optuna 自动调参、特征分析、保底提交

V2.2 仅在已完成的 V2.1 比赛上运行；沿用相同的 3 折划分（SEED=20261018），所以 CV 对比有意义。

1. Kaggle 云端 Optuna TPE 在 12 万行分层样本上搜索 8 组 LightGBM 参数（2 折）。
2. 在全部 699,635 行数据上验证 tuned_lightgbm 与 tuned_extended 特征方案（3 折）。
3. 输出 Optuna 搜索历史、3-Fold OOF、LightGBM Gain 特征重要性和实验复盘。
4. 若全量 CV 没有比 V2.1 最佳 OOF 提高超过 0.00008，自动跳过提交。
5. 若通过门槛，最多自动提交一次，核对 Public LB 后保留真正的线上冠军。
6. 将每个比赛的《小白说明书.md》一起更新。

**如何开始：** Actions → Kaggle Auto Scientist V2.2 → Run workflow。首次升级完成后，会保留其运行状态；重复点击同一配置不会重复训练或乱提交。

**主要文件：** kaggle_agent/v22_train.py（云端训练脚本）、kaggle_agent/v22_pipeline.py（控制与成绩回写）、config/agent.yaml（参数）、reports/v2_2_latest.md、state/v2_2_status.json、competitions/<slug>/results_v22/（指标/试验/特征重要性）。

本项目当前只对二分类表格赛启用 V2.2，非二分类数据需要独立适配；Kaggle 环境与计算额度也可能影响是否训练成功。

## V2.3：独立 TPU 神经网络训练线

原有的 V2.1 Ensemble（CV 0.95827867，Public LB 0.95774）**不作修改**。新增一条 TPU 独立实验线：

- Kaggle CPU/GPU：继续保留 CatBoost、LightGBM、XGBoost；不会强行将树模型放进 TPU。
- Kaggle TPU v5e-8：运行 JAX MLP 神经网络，沿用同一随机种子和 3-Fold StratifiedKFold。
- 上传 TPU kernel 时使用 Kaggle CLI --accelerator TpuV5E8 与 machine_shape: TpuV5E8。
- 训练开始前，脚本必须检查 jax.default_backend() 确实为 tpu；若实际是 CPU/GPU 则立即停止，保留原有线上冠军。
- TPU 神经网络 OOF 分数只有超过 V2.1 0.00010 才能提交；真正的线上成绩只有更高才升级冠军。
- 不会虚构 TPU+树模型融合成绩：当前未保存两者可配对的逐行 OOF，融合需要以后新增这部分输出再测试。

**使用：** GitHub → Actions → Kaggle Auto Scientist V2.3 TPU → Run workflow → mode=run。

**如果 Kaggle API 没真正分配 TPU：** 打开报告中的 notebook 链接；在 Kaggle 编辑器选 Settings → Accelerator → TPU v5e-8 → Save & Run All；确认 Notebook 运行完后，回 GitHub Actions 选择 mode=recover_existing。这只读取已经运行好的 Notebook，不重复消耗一次训练。Kaggle API 的 TPU 请求成功不等于硬件实际分配成功，必须以 JAX 运行时检测为准。

**V2.3 输出：**
- state/v2_3_status.json（真实阶段、硬件证据及提交决策）
- reports/v2_3_latest.md（TPU 与树模型对比和错误解释）
- competitions/<slug>/kaggle_tpu_v23/main.py（云端训练）
- competitions/<slug>/results_v23/metrics.json 和 hardware.json（若运行成功）
- competitions/<slug>/小白说明书.md（新增零基础 TPU 教学章节）

V2.3 是针对当前二分类表格比赛的首次实验版；TPU 对小模型不一定比 CPU 快，是否有益由实际 OOF 和运行时间决定。

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
