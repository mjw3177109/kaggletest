"""Append an easy-to-understand V2.2 chapter after the existing beginner guide."""
from __future__ import annotations

from pathlib import Path
from v2_1_pipeline import ROOT, load_json

START = "<!-- V2.2 START -->"
END = "<!-- V2.2 END -->"


def stamp(x):
    return chr(96) + str(x) + chr(96)


def update(slug):
    project = ROOT / "competitions" / slug
    guide = project / "小白说明书.md"
    if not guide.exists():
        raise FileNotFoundError(f"Beginner guide missing: {guide}")
    status = load_json(ROOT / "state" / "v2_2_status.json", {}) or {}
    result = load_json(project / "results_v22" / "metrics.json", {}) or {}
    importance = load_json(project / "results_v22" / "feature_importance.json", {}) or {}
    text = guide.read_text(encoding="utf-8")
    if START in text:
        text = text.split(START)[0].rstrip() + "\n"
    if status.get("competition") != slug:
        chapter = "\n".join([
            "## V2.2｜自动调参（待运行）", "",
            "下一版会使用 Optuna 自动寻找 LightGBM 参数，并自动判断是否值得提交。",
        ])
    else:
        states = {
            "kernel_running": "Kaggle 正在训练", "kernel_complete": "训练结束，正在下载结果",
            "complete": "训练与线上评分都已完成", "skipped_no_cv_gain": "新模型没有超过原冠军，自动跳过提交",
            "submission_pending": "提交成功，等待 Kaggle 评分", "kernel_failed": "Kaggle 内部训练失败",
            "kernel_access_error": "Kaggle Notebook 权限或名称错误",
        }
        chapter = "\n".join([
            "## V2.2｜Optuna 自动优化（新手版）", "",
            "这一阶段像帮模型安排模拟考试：先在一部分数据上试不同参数，再把最好的参数放到完整数据上做 3 折验证。",
            "只有验证分数比之前的模型更高，才可能向 Kaggle 提交新的答卷。", "",
            "当前状态：" + states.get(status.get("stage"), str(status.get("stage"))),
            "旧冠军本地 CV：" + stamp(status.get("incumbent_cv")),
            "旧冠军 Public LB：" + stamp(status.get("incumbent_lb")), "",
        ])
        if result:
            chapter += "\n".join([
                "本轮 Optuna 搜索次数：" + stamp(result.get("tuning", {}).get("trials_completed")), "",
                "| 训练方案 | 3-Fold CV ROC-AUC |", "|---|---:|"
            ])
            for name, row in (result.get("candidates") or {}).items():
                chapter += "\n| " + name + " | " + str(round(float(row["cv_auc"]), 8)) + " |"
            gate = status.get("cv_gate") or {}
            chapter += "\n\n" + "\n".join([
                "本轮选出的模型：" + stamp(result.get("best_candidate")),
                "验证分数提升：" + stamp(round(float(gate.get("delta", 0)), 8)),
                "是否通过提交门槛：" + ("是" if gate.get("passed") else "否"),
                "是否已提交 Kaggle：" + ("是" if status.get("submitted") else "否"),
                "目前线上保留的冠军：" + stamp(status.get("champion", "V2.1 ensemble")),
                "",
                "### 特征重要性（前十名）", "",
                "它表示模型依赖哪些字段，但不代表这些字段在现实生活中一定具有因果关系。", "",
                "| 特征 | Gain |", "|---|---:|",
            ])
            top = (importance.get("candidates") or {}).get(result.get("best_candidate"), [])[:10]
            for row in top:
                chapter += "\n| " + str(row["feature"]) + " | " + str(row["gain"]) + " |"
        chapter += "\n\n" + "\n".join([
            "### 小白怎样重跑？", "",
            "1. 进入 GitHub 仓库 Actions。",
            "2. 点击 Kaggle Auto Scientist V2.2。",
            "3. 选择 Run workflow。",
            "4. 看 reports/v2_2_latest.md 里的成绩和实验分析。",
            "已完成的相同实验会自动跳过，防止重复训练和重复提交。",
            "",
            "如果显示 skipped_no_cv_gain，不是故障：说明新方案没有在本地验证中赢过旧冠军。",
        ])
    guide.write_text(text.rstrip() + "\n\n" + START + "\n" + chapter + "\n" + END + "\n", encoding="utf-8")
    print("Updated beginner guide:", guide)


if __name__ == "__main__":
    state = load_json(ROOT / "state" / "v2_status.json", {}) or {}
    if state.get("competition"):
        update(str(state["competition"]))
