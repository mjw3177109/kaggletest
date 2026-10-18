"""V2.8 beginner guide: show experiment result, winner, score AND official rank."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
START="<!-- V2.8 GPU TE START -->"
END="<!-- V2.8 GPU TE END -->"


def get(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def section(slug):
    state=get(ROOT/"state"/"v2_8_status.json")
    rank=get(ROOT/"state"/"score_rank_latest.json")
    parts=[
        "## V2.8｜CatBoost GPU 多样性实验（小白版）", "",
        "问题：V2.5 已经从 0.95819 提升到 0.96035，但仍需降低 AUC 排序误差。",
        "V2.8 用严格交叉拟合的客群频率作为新增信号，验证其是否能进一步提升线上表现。", "",
        "**V2.8 改进三点：**",
        "1. 保留 V2.5 已验证的全 21 字段交叉拟合 Target Encoding，并严格防止标签泄漏。",
        "2. 保留 V2.5 成功的航线画像和客群交叉特征，不复用 V2.6 没有带来收益的频率编码。",
        "3. 同时保留航线画像，比较 LightGBM、GPU CatBoost 和固定权重融合的 3 折 ROC-AUC。",
        "",
        "提醒：GPU CatBoost 只占用 Kaggle T4 的一个 CUDA 设备，",
        "LightGBM 在同一 Notebook 的 CPU 端训练，没有假冒 GPU 加速。",
        "",
    ]
    if state.get("competition")==slug:
        metrics=state.get("metrics") or {}
        submission=state.get("latest_submission") or {}
        parts+= [
            "- 实验状态："+str(state.get("stage")),
            "- 正式旧冠军 Public LB："+str(state.get("incumbent_lb")),
            "- 正式旧冠军 3 折 CV："+str(state.get("incumbent_cv")),
            "- V2.8 最佳 CV："+str(metrics.get("best_cv_auc","尚未完成")),
            "- V2.8 最佳方法："+str(metrics.get("best_model","尚未完成")),
            "- CV 超越门槛："+str((state.get("cv_gate") or {}).get("threshold","0.00010")),
            "- V2.8 实际提交："+("是" if state.get("submitted") else "否"),
            "- V2.8 新 Public LB："+str(submission.get("publicScore","尚未评分")),
            "- 最后保留冠军："+str(state.get("champion")),
            "- Kaggle GPU Notebook："+str(state.get("kernel_url")),
        ]
    else:
        parts+=["该比赛尚未启动 V2.8 实验。"]
    if rank.get("competition")==slug:
        public=rank.get("rank") if rank.get("rank_verified") else None
        count=rank.get("ranked_teams")
        parts+=["", "### 现在的官方 Public 排名", "",
                 "- 当前最佳分数："+str((rank.get("score") or {}).get("public_score","未核实")),
                 "- 当前排名："+(f"第 {public} 名 / {count or '未知'} 支队伍" if public else "暂未核实"),
                 "- 排名核验时间："+str(rank.get("checked_at","暂无")),
                 "Public 榜不是最终 Private 榜，切记不要把临时排名当做领奖资格。"]
    parts+=["", "详细研究资料：competitions/"+slug+"/v25_peer_research.md",
             "V2.8 实验报告：reports/v2_8_latest.md。"]
    return "\n".join(parts)


def main():
    root=ROOT/"competitions"
    for guide in root.glob("*/小白说明书.md"):
        old=guide.read_text(encoding="utf-8")
        if START in old:
            old=old.split(START)[0].rstrip()
        guide.write_text(old.rstrip()+"\n\n"+START+"\n"+section(guide.parent.name)
                         +"\n"+END+"\n",encoding="utf-8")
        print("Updated V2.8 beginner guide:",guide)


if __name__=="__main__":
    main()
