from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def fmt(value: Any, digits: int = 6) -> str:
    if value in (None, ""):
        return "暂无"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def render_guide(slug: str) -> str:
    project = ROOT / "competitions" / slug
    v2 = load_json(ROOT / "state" / "v2_status.json", {}) or {}
    v21 = load_json(ROOT / "state" / "v2_1_status.json", {}) or {}
    profile = load_json(project / "data_profile.json", {}) or {}
    v2_metrics = load_json(project / "results" / "metrics.json", {}) or {}
    v21_metrics = load_json(project / "results_v21" / "metrics.json", {}) or {}

    train_info = profile.get("train") or {}
    test_info = profile.get("test") or {}
    target = profile.get("target") or v2_metrics.get("target") or "尚未识别"
    id_candidates = profile.get("id_candidates") or []
    baseline_lb = (v2.get("latest_submission") or {}).get("publicScore")
    baseline_cv = v2_metrics.get("validation_score") or (v2.get("metrics") or {}).get("validation_score")

    v21_latest = v21.get("latest_submission") or {}
    v21_lb = v21_latest.get("publicScore")
    v21_best = v21_metrics.get("best_model")
    v21_best_cv = v21_metrics.get("best_cv_auc")
    v21_delta = v21.get("public_score_delta")

    current_lb = v21_lb if v21.get("stage") == "complete" and v21_lb not in (None, "") else baseline_lb
    current_cv = v21_best_cv if v21.get("stage") == "complete" and v21_best_cv is not None else baseline_cv
    current_model = v21_best if v21.get("stage") == "complete" and v21_best else "HistGradientBoosting baseline"

    rows = []
    if baseline_cv is not None or baseline_lb not in (None, ""):
        rows.append(("Run 001", "HistGradientBoosting baseline", fmt(baseline_cv), str(baseline_lb or "暂无"), "V2 第一条可提交基线"))
    ranking = v21_metrics.get("ranking") or []
    run_map = {"catboost": "Run 002", "lightgbm": "Run 003", "xgboost": "Run 004", "ensemble": "Run 005"}
    for row in ranking:
        name = row.get("model")
        cv = row.get("cv_auc")
        lb = v21_lb if name == v21_best and v21.get("stage") == "complete" else "未单独提交"
        conclusion = "V2.1 CV 最优并已提交" if name == v21_best and v21.get("stage") == "complete" else "作为候选模型保留"
        rows.append((run_map.get(str(name), "Run ?"), str(name), fmt(cv), str(lb), conclusion))

    exp_table = ["| Run | 模型 | 本地 CV | Kaggle Public LB | 结论 |", "|---|---|---:|---:|---|"]
    for r in rows:
        exp_table.append("| " + " | ".join(r) + " |")
    if not rows:
        exp_table.append("| - | 尚未完成训练 | - | - | 先等待自动流程完成 |")

    stage = v21.get("stage") if v21 else v2.get("stage")
    stage_cn = {
        "rules_required": "等待你本人接受 Kaggle 比赛规则",
        "download_failed": "数据下载阶段出现问题",
        "kernel_failed": "Kaggle 云端训练失败，系统会根据日志继续修复",
        "kernel_timeout": "Kaggle 云端训练超过当前等待时限",
        "complete": "已经完成训练、提交和结果回写",
        "submitting_best_cv": "正在提交 V2.1 中 CV 最好的模型",
        "kernel_push": "正在把训练脚本推到 Kaggle",
    }.get(stage, str(stage or "尚未开始"))

    delta_text = "暂无 V2.1 线上对比"
    if isinstance(v21_delta, (int, float)):
        delta_text = f"{v21_delta:+.6f}"

    return f'''# 小白说明书｜{slug}

> 这份文档专门给第一次接触 Kaggle、机器学习、排行榜的人看。你不需要先懂算法，照着这里看就能知道这个项目正在做什么。

## 01｜这场比赛是干什么的

你可以把这场比赛理解成：**给 AI 一大批“已经有答案的历史样本”，让它学会规律，再去猜“还没有答案的新样本”。**

这个项目当前自动识别出的预测目标是：`{target}`。

也就是说，AI 最后真正要做的事情，就是根据其它字段，预测 `{target}`。

## 02｜你现在做到哪一步了

当前阶段：**{stage_cn}**。

目前系统记录到的最好结果：

- 当前模型：`{current_model}`
- 本地验证分数（CV）：`{fmt(current_cv)}`
- Kaggle Public LB：`{current_lb or '暂无'}`
- V2.1 相比 V2 的线上变化：`{delta_text}`

只要状态已经是 `complete`，就表示“分析题目 → 训练 → 生成 submission → 提交 Kaggle → 取回分数 → 写回 GitHub”这一整条链已经跑完。

## 03｜5分钟看懂 Kaggle

### train.csv 是什么？

`train.csv` 就像一套**带答案的练习题**。AI 可以同时看到题目和正确答案，用它来学习。

### test.csv 是什么？

`test.csv` 就像一套**考试题**。只有题目，没有答案。AI 训练完以后要给这些数据预测答案。

### target 是什么？

`target` 就是**我们最终要猜的答案列**。这个项目里当前识别的是 `{target}`。

### feature 是什么？

`feature` 就是**帮助 AI 做判断的线索**，比如年龄、距离、服务评分、类别等。

### model 是什么？

`model` 就是 AI 用来“从历史样本里总结规律”的算法。不同模型像不同学生，有的人擅长记规律，有的人更擅长处理复杂组合。

### CV 是什么？

CV（Cross Validation）可以理解成**我们自己先组织的模拟考试**。不直接碰 Kaggle 的隐藏答案，而是把已有训练数据分成几份，轮流拿一部分当模拟考试。

### Public LB 是什么？

Public LB（Public Leaderboard）就是**Kaggle 公布给你的线上排行榜分数**。它使用 Kaggle 掌握的一部分隐藏答案来评分。

### submission.csv 是什么？

`submission.csv` 就是**答题卡**。AI 对 test.csv 的预测会被写进这个文件，然后提交给 Kaggle。

## 04｜这份数据有哪些东西

当前数据概况：

- train 抽样分析行数：`{train_info.get('sampled_rows', '暂无')}`
- train 列数：`{train_info.get('columns', '暂无')}`
- test 抽样分析行数：`{test_info.get('sampled_rows', '暂无')}`
- test 列数：`{test_info.get('columns', '暂无')}`
- 目标字段：`{target}`
- ID 字段候选：`{id_candidates}`

系统还会把更详细的字段类型、缺失值情况写进：

`data_analysis.md`

你暂时不需要看懂全部字段，先记住：**train 比 test 多出来的那一个关键答案列，通常就是 target。**

## 05｜AI 是怎么训练的

当前自动流程大致是：

```text
读取比赛
→ 下载数据
→ 找 target
→ 处理空值和类别字段
→ 建模型
→ 做本地 CV
→ 用全部训练数据重新训练
→ 预测 test
→ 生成 submission.csv
→ 提交 Kaggle
→ 读取 Public LB
```

V2 先负责建立一个可靠 baseline；V2.1 再比较 CatBoost、LightGBM、XGBoost 和 Ensemble，并尽量只把本地 CV 最好的方案提交出去，减少乱交答案。

## 06｜怎么看 CV 和排行榜分数

这场比赛当前使用 ROC-AUC 一类“越大越好”的指标时，可以粗暴理解为：

```text
0.50 左右  ≈ 接近乱猜
0.70+      ≈ 开始学到规律
0.90+      ≈ 已经比较强
越接近 1   ≈ 区分能力越强
```

真正看模型是否靠谱，不要只看一个数字，而要同时看：

```text
本地 CV 是否提高
        ↓
Kaggle Public LB 是否也提高
        ↓
两边趋势是否基本一致
```

如果 CV 很高、LB 很低，往往说明本地验证方式有问题、过拟合，或者不小心学到了线上不存在的规律。

## 07｜每一轮实验做了什么

{chr(10).join(exp_table)}

以后新 Run 也会继续追加到 `experiments/`，不会把老结果覆盖掉。

## 08｜项目文件夹怎么看

你最常看的文件只需要这几个：

```text
competitions/{slug}/
├─ 小白说明书.md            ← 先看这个
├─ problem_analysis.md       ← 题目正式分析
├─ data_analysis.md          ← 数据正式分析
├─ solution_plan.md          ← 模型方案
├─ data_profile.json         ← 程序读的数据统计
├─ experiments/              ← 每一轮实验记录
├─ kaggle/                   ← V2 Kaggle训练代码
├─ kaggle_v21/               ← V2.1优化训练代码
├─ results/                  ← V2结果
└─ results_v21/              ← V2.1结果
```

如果你只是想知道“现在成绩怎么样”，优先看：

`小白说明书.md` → `reports/v2_1_latest.md` → `results_v21/metrics.json`

## 09｜如果你想自己运行一次

最简单的方法不是在电脑装一大堆环境，而是去 GitHub：

```text
Actions
→ 选择 Kaggle Auto Scientist V2 或 V2.1
→ Run workflow
```

需要满足两个条件：

1. GitHub Secrets 里已经配置 `KAGGLE_API_TOKEN`。
2. 第一次参加某场 Kaggle 比赛时，你本人已经在 Kaggle 页面点过 Join / Accept Rules。

后面的下载数据、训练、生成 submission、提交和写回 GitHub 都由自动流程处理。

## 10｜常见报错怎么处理

### RULES_REQUIRED

意思：你还没有加入这场比赛。

处理：打开 Kaggle 比赛页面，点 `Join Competition`，接受 Rules，再重跑。

### Missing KAGGLE_API_TOKEN

意思：GitHub 没拿到 Kaggle 认证信息。

处理：检查 GitHub Repository Secrets 里的 `KAGGLE_API_TOKEN`。

### Kernel Failed

意思：Kaggle 云端代码运行失败。

处理：先看项目里的 `kaggle/logs.txt` 或 `kaggle_v21/logs.txt`。系统也会把错误写回状态文件，后续可以继续自动修。

### CV 很高，但 LB 很低

意思：本地模拟考试和 Kaggle 真正考试差距太大。

常见原因：数据泄漏、验证切分不合理、过拟合、ID/时间字段处理错误。

### submission 格式错误

意思：答题卡列名、行数或顺序不符合 Kaggle 要求。

处理：对照 `sample_submission.csv`，保证列结构一致。

## 11｜下一步准备怎么提升

自动优化通常会按这个顺序继续：

```text
可靠 baseline
→ CatBoost / LightGBM / XGBoost
→ 交叉验证
→ 特征工程
→ 多随机种子
→ 模型融合
→ 比较 CV 与 LB
→ 只保留真正有效的改动
```

目标不是“模型越复杂越好”，而是：**每次修改都能解释、能复现、最好还能让线上分数真的变高。**

## 12｜一句话总结

**你可以把这个项目当成一个会自己做 Kaggle 作业的 AI：它自己读题、自己训练、自己交卷、自己看成绩；这份《小白说明书》负责让你随时知道它到底在干什么。**
'''


def main() -> None:
    v2 = load_json(ROOT / "state" / "v2_status.json", {}) or {}
    slugs: list[str] = []
    if v2.get("competition"):
        slugs.append(str(v2["competition"]))
    competitions = ROOT / "competitions"
    if competitions.exists():
        for p in competitions.iterdir():
            if p.is_dir() and p.name not in slugs:
                slugs.append(p.name)

    generated = []
    for slug in slugs:
        project = competitions / slug
        if not project.exists():
            continue
        guide = render_guide(slug)
        path = project / "小白说明书.md"
        path.write_text(guide, encoding="utf-8")
        generated.append(str(path.relative_to(ROOT)))

    print(json.dumps({"generated": generated}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
