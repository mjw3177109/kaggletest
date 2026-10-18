"""Add an optional V2.3 GPU section to each Chinese beginner manual."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- V2.3 GPU START -->"
END = "<!-- V2.3 GPU END -->"


def load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main():
    state = load(ROOT / "state" / "v2_3_gpu_status.json")
    for path in (ROOT / "competitions").glob("*/小白说明书.md"):
        text = path.read_text(encoding="utf-8")
        if START in text:
            text = text.split(START)[0].rstrip() + "\n"
        if state.get("competition") != path.parent.name:
            section = "## V2.3 GPU 独立训练线\n\n该比赛尚未启动 GPU 实验；原有 CPU 冠军不变。"
        else:
            m = state.get("metrics") or {}
            gate = state.get("cv_gate") or {}
            names = {
                "kernel_queued": "Kaggle GPU 排队中（尚未训练）",
                "kernel_running": "GPU Notebook 正在执行或等待中",
                "skipped_no_cv_gain": "本地成绩未超过旧冠军，自动跳过提交",
                "gpu_unavailable": "没分配真实的 T4 GPU，已安全停止",
                "complete": "GPU 本地训练和线上验收完成",
            }
            section = "\n".join([
                "## V2.3 GPU｜NVIDIA T4 独立训练（小白版）", "",
                "这条线使用 CUDA XGBoost，与 TPU 神经网络、原有 CPU 树模型互不覆盖。",
                "必须识别真实 T4 和 XGBoost CUDA 设备；用相同 3-Fold CV 评估。",
                "只有新 CV 明显更高才会考虑提交。", "",
                "当前状态：" + names.get(state.get("stage"), str(state.get("stage"))),
                "旧冠军 Public LB：" + str(state.get("incumbent_lb")),
                "旧冠军 OOF CV：" + str(state.get("incumbent_cv")),
                "GPU 新 CV：" + str(m.get("best_cv_auc", "暂无")),
                "CV 变化：" + str(gate.get("delta", "暂无")),
                "GPU 新提交：" + ("有" if state.get("submitted") else "没有"),
                "保留的线上冠军：" + str(state.get("champion", "V2.1 ensemble")), "",
                "Notebook：" + str(state.get("kernel_url", "暂无")), "",
                "### 怎么看？", "",
                "打开 reports/v2_3_gpu_latest.md 看全量对比；",
                "如果 Kernel 状态是 QUEUED，表示 Kaggle 仍在等算力。",
                "使用 GPU 不代表一定更强：成绩必须由同样的交叉验证来证明。",
                "Kaggle T4 ×2 中，当前 XGBoost 只使用其中一个 CUDA 设备。",
            ])
        path.write_text(text.rstrip() + "\n\n" + START + "\n" + section + "\n" + END + "\n",
                        encoding="utf-8")
        print("GPU beginner chapter:", path)


if __name__ == "__main__":
    main()
