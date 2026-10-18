"""Append V2.3 TPU chapter to each competition's Chinese beginner manual.

Run after beginner_guide.py and v22_guide.py so regenerating earlier sections
will never hide the latest TPU stage.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- V2.3 TPU START -->"
END = "<!-- V2.3 TPU END -->"


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def chapter(slug):
    state = read(ROOT / "state" / "v2_3_status.json")
    if state.get("competition") != slug:
        return "## V2.3 TPU 训练线\n\n暂未启动。这一阶段使用独立 JAX MLP，原有 V2.1 树模型完全保留。\n"
    hw = state.get("hardware") or {}
    metrics = state.get("metrics") or {}
    gate = state.get("cv_gate") or {}
    mapping = {
        "pushing_tpu_kernel": "正在请求 Kaggle TPU 资源",
        "kernel_running": "Notebook 正在 Kaggle 实际运行",
        "kernel_queued": "Kaggle 仍在排队等待算力（未开始训练）",
        "kernel_complete": "神经网络已执行，正在检验结果",
        "tpu_unavailable": "Kaggle 没有分配真正的 TPU，已安全停止",
        "tpu_push_failed": "无法创建 Kaggle TPU Notebook",
        "kernel_failed": "TPU Notebook 训练失败",
        "kernel_access_error": "无法访问 Notebook，需检查用户名或权限",
        "skipped_no_cv_gain": "神经网络的 CV 没赢过旧冠军，没有提交",
        "complete": "TPU 训练及线上对比完成",
        "submission_pending": "已提交，等待线上评分",
        "awaiting_submission_approval": "验证通过，但自动提交目前被关闭",
    }
    lines = [
        "## V2.3｜新增 TPU 神经网络实验（零基础版）", "",
        "V2.1 的 LightGBM / CatBoost / XGBoost 像擅长处理表格的三个选手。",
        "V2.3 新增另一位选手：JAX MLP 神经网络，它可以尝试发现不同的规律。",
        "TPU 是专门用于加速部分神经网络计算的硬件，不是所有算法都能用。", "",
        "我们**没有替换旧冠军**：旧分数在，旧模型也在。新选手先做模拟考试，证明自己更强才允许提交。", "",
        f"当前阶段：**{mapping.get(state.get('stage'), state.get('stage', '尚未开始'))}**。",
        f"Kaggle Notebook：{state.get('kernel_url', '尚未创建')}",
        f"旧冠军 V2.1 Public LB：{state.get('incumbent_lb')}",
        f"旧冠军 3-Fold CV：{state.get('incumbent_cv')}",
        f"实际检测到的训练硬件：{hw.get('actual', '尚未确认')}",
        f"TPU 确认成功：{hw.get('ok', False)}",
    ]
    if metrics:
        lines += [
            f"本次 JAX TPU 3-Fold CV：{metrics.get('cv_auc')}",
            f"比旧冠军的 CV 变化：{gate.get('delta')}",
            f"是否通过提交门槛：{'是' if gate.get('passed') else '否'}",
            f"是否在线上提交：{'是' if state.get('submitted') else '否'}",
            f"最终线上保留的冠军：{state.get('champion')}",
        ]
    lines += [
        "", "### 我需要做什么？", "",
        "一般不用自己装 TPU：打开 GitHub Actions，选择 Kaggle Auto Scientist V2.3 TPU。",
        "点击 Run workflow（默认 mode=run），GitHub 会把独立模型送到 Kaggle。", "",
        "如果提示 tpu_unavailable：不是旧模型坏了，而是 Kaggle API 没提供真正 TPU。",
        "打开上方 Notebook 链接，在网页 Settings → Accelerator 选择 TPU v5e-8，",
        "点击 Save & Run All，再回 GitHub Actions 选择 mode=recover_existing，",
        "它只读取已有结果，不重复上传或训练。", "",
        "### 为什么新模型不一定被提交？", "",
        "本地 CV 是一次自己组织的模拟考试。只有同一 3 折标准下明显更好才值得提交。",
        "因为上一轮没有存储树模型逐行 OOF，我们暂时不声称已经比较过 TPU + 树模型融合。",
        "完整运行记录：reports/v2_3_latest.md。", "",
    ]
    return "\n".join(lines)


def main():
    v2 = read(ROOT / "state" / "v2_status.json")
    folders = [p for p in (ROOT / "competitions").iterdir() if p.is_dir()]
    for project in folders:
        path = project / "小白说明书.md"
        if not path.exists():
            continue
        src = path.read_text(encoding="utf-8")
        if START in src:
            src = src.split(START)[0].rstrip() + "\n"
        path.write_text(src.rstrip() + "\n\n" + START + "\n" + chapter(project.name) + "\n" + END + "\n", encoding="utf-8")
        print("Refreshed", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
