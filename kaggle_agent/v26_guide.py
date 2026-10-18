"""V2.6 beginner guide: show experiment result, winner, score AND official rank."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
START="<!-- V2.6 GPU TE START -->"
END="<!-- V2.6 GPU TE END -->"


def get(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def section(slug):
    state=get(ROOT/"state"/"v2_6_status.json")
    rank=get(ROOT/"state"/"score_rank_latest.json")
    parts=[
        "## V2.6｜交叉拟合频率特征的 GPU 优化（小白版）", "",
        "问题：V2.5 已经从 0.95819 提升到 0.96035，但仍需降低 AUC 排序误差。",
        "V2.6 用严格交叉拟合的客群频率作为新增信号，验证其是否能进一步提升线上表现。", "",
        "**V2.6 改进三点：**",
        "1. 保留 V2.5 已验证的全 21 字段交叉拟合 Target Encoding，并严格防止标签泄漏。",
        "2. 新增 12 组客群/航程/评分的频率特征，在每个外层训练折内部再做 5 折频率交叉拟合。",
        "3. 同时保留航线画像，比较 LightGBM、GPU XGBoost 和固定权重融合的 3 折 ROC-AUC。",
        "",
        "提醒：GPU XGBoost 只占用 Kaggle T4 的一个 CUDA 设备，",
        "LightGBM 在同一 Notebook 的 CPU 端训练，没有假冒 GPU 加速。",
        "",
    ]
    if state.get("competition")==slug:
        metrics=state.get("metrics") or {}
        submission=state.get("latest_submission") or {}
        parts+= [
            "- 实验状态："+str(state.get("stage")),
            "- V2.3 原冠军 Public LB："+str(state.get("incumbent_lb")),
            "- V2.3 原冠军 3 折 CV："+str(state.get("incumbent_cv")),
            "- V2.6 最佳 CV："+str(metrics.get("best_cv_auc","尚未完成")),
            "- V2.6 最佳方法："+str(metrics.get("best_model","尚未完成")),
            "- CV 超越门槛："+str((state.get("cv_gate") or {}).get("threshold","0.00010")),
            "- V2.6 实际提交："+("是" if state.get("submitted") else "否"),
            "- V2.6 新 Public LB："+str(submission.get("publicScore","尚未评分")),
            "- 最后保留冠军："+str(state.get("champion")),
            "- Kaggle GPU Notebook："+str(state.get("kernel_url")),
        ]
    else:
        parts+=["该比赛尚未启动 V2.6 实验。"]
    if rank.get("competition")==slug:
        public=rank.get("rank") if rank.get("rank_verified") else None
        count=rank.get("ranked_teams")
        parts+=["", "### 现在的官方 Public 排名", "",
                 "- 当前最佳分数："+str((rank.get("score") or {}).get("public_score","未核实")),
                 "- 当前排名："+(f"第 {public} 名 / {count or '未知'} 支队伍" if public else "暂未核实"),
                 "- 排名核验时间："+str(rank.get("checked_at","暂无")),
                 "Public 榜不是最终 Private 榜，切记不要把临时排名当做领奖资格。"]
    parts+=["", "详细研究资料：competitions/"+slug+"/v26_peer_research.md",
             "V2.6 实验报告：reports/v2_6_latest.md。"]
    return "\n".join(parts)


def main():
    root=ROOT/"competitions"
    for guide in root.glob("*/小白说明书.md"):
        old=guide.read_text(encoding="utf-8")
        if START in old:
            old=old.split(START)[0].rstrip()
        guide.write_text(old.rstrip()+"\n\n"+START+"\n"+section(guide.parent.name)
                         +"\n"+END+"\n",encoding="utf-8")
        print("Updated V2.6 beginner guide:",guide)


if __name__=="__main__":
    main()
