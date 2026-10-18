## V2.4｜GPU + TPU 双线冲榜与自动换赛

自动执行的两个旧赛 Notebook 保持独立：
- GPU XGBoost：Kaggle NVIDIA T4，按它自己的 3-Fold 计算并保护原冠军。
- TPU JAX MLP：只在 JAX 检测到真实 TPU 时计算，否则安全停止。

新增长期有效的 GitHub Actions：Kaggle Dual GPU-TPU Top3 or Deadline Ladder。每 2 小时读取 Kaggle 实际公开榜名次和最终提交截止时间；以下**任一**条件触发下一场候选：

1. 已认证的 Kaggle Public 榜名次为第 1、2 或 3 名。
2. 距离官方最终提交截止 <=48 小时。
3. 官方最终提交截止已过（比赛结束）。

**重要边界：** Public 榜前三不是最终获奖；需要等最终 Private 榜和官方通知独立核对。只有 Kaggle 已确定的官方截止日期才能触发时间规则；不因未取到 userRank 就忽略截止时间。

**换赛机制：**
- 优先在 Playground / Getting Started 中寻找截止仍有 >=72 小时的比赛，防止刚切换又到最后提交日。
- 下一场已报名：自动把选赛状态迁移到新赛，但保留 GPU/TPU 原 Notebook 和原分数；必须通过新数据类型、官方评分指标和提交格式预检后才能运行下一场的训练器。
- 下一场未报名：选出候选并记录等待用户加入/接受规则，**不自动替用户接受 Kaggle 条款**。
- 没有合格候选：继续记录待换赛状态，不会凭空发明比赛。
- 当前 GPU/TPU 训练器最初面向 S6E10 二分类 ROC-AUC；V2.4 当前已经实现自动选赛及时间/排名触发，但**尚未实现可适配任意比赛的通用 GPU+TPU 新赛训练器**，不能把换赛选中等同于已开始新的 GPU 或 TPU 训练。

查看自动决策：reports/dual_ladder_latest.md 和 state/dual_ladder.json。
每个比赛的《小白说明书.md》会自动增加 V2.4 新手章节。
领奖单独跟踪，不会因换赛遗忘旧比赛。


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
