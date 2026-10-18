"""V2.7 Chinese guide: paired OOF ensemble, saved GPU time, verified rank."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
START="<!-- V2.7 OOF FUSION START -->"
END="<!-- V2.7 OOF FUSION END -->"


def get(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main():
    status=get(ROOT/"state"/"v2_7_status.json")
    rank=get(ROOT/"state"/"score_rank_latest.json")
    for guide in (ROOT/"competitions").glob("*/小白说明书.md"):
        slug=guide.parent.name
        original=guide.read_text(encoding="utf-8")
        if START in original:
            original=original.split(START)[0].rstrip()
        lines=[
            "## V2.7｜不占 GPU 配额的 OOF 双模型融合（小白版）", "",
            "这一版不是重新训练新模型，而是把我们已经得到的 V2.5 和 V2.6 折外预测进行固定 50/50 排名融合。",
            "只会读取自己 Kaggle Notebook 保存的 OOF 和提交预测；不使用别人的答案、私榜标签或测试集真实标签。",
            "两版必须使用相同 3 折分组、相同训练标签和完全一致的测试 ID 顺序，任何不一致立即停止。",
            "V2.7 的 CV 必须超过已验证最佳 CV 0.00010 才可能提交，且会先确认今天的 Kaggle 提交限额。",
            "",
        ]
        if status.get("competition")==slug:
            m=status.get("metrics") or {}
            sub=status.get("latest_submission") or {}
            lines.extend([
                "- 当前阶段："+str(status.get("stage","待检查")),
                "- V2.5 OOF："+str(m.get("v25_oof","待完成")),
                "- V2.6 OOF："+str(m.get("v26_oof","待完成")),
                "- 固定融合 OOF："+str(m.get("blended_cv_auc","待完成")),
                "- 晋级门槛："+str(status.get("required_cv","待验证")),
                "- 已提交："+str(status.get("submitted",False)),
                "- V2.7 新 LB："+str(sub.get("publicScore","尚未正式评分")),
            ])
        else:
            lines.append("当前比赛尚未执行 V2.7，不影响旧冠军和 TPU 分支。")
        if rank.get("competition")==slug:
            r=rank.get("rank") if rank.get("rank_verified") else None
            n=rank.get("ranked_teams")
            lines.extend(["", "### 官方认证成绩和排名", "",
                "- 当前最佳 Public LB："+str((rank.get("score") or {}).get("public_score","未知")),
                "- 当前 Public 排名："+(f"第 {r} 名 / {n or '未知'} 支队伍" if r else "未核实"),
                "- 排名核验 UTC："+str(rank.get("checked_at","未知"))])
        lines.extend(["", "完整结果：reports/v2_7_latest.md",
                      "Public 排名只能证明当前公开榜位置，不能代替 Private 最终榜或获奖确认。"])
        guide.write_text(original.rstrip()+"\n\n"+START+"\n"+"\n".join(lines)+"\n"+END+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
