"""Add a V2.4 contest-switching chapter to each competition beginner manual."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- V2.4 LADDER START -->"
END = "<!-- V2.4 LADDER END -->"


def read(path):
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def ascode(value):
    return chr(96) + str(value) + chr(96)


def explanation(data, slug):
    reasons = data.get("handoff_reasons") or []
    labels = {
        "public_top3": "已经确认公开榜名次在前 3",
        "submission_deadline_window": "距离最终提交截止已进入 48 小时以内",
        "competition_ended": "比赛的最终提交截止已经过去",
    }
    nxt = data.get("next_competition") or {}
    lines = [
        "## V2.4｜GPU + TPU 双线自动换赛（新手说明）", "",
        "比赛由 GPU XGBoost 和 TPU JAX 各自训练；原先 CPU Ensemble 成绩保留。",
        "自动换赛触发条件只要满足其一即可：**公开榜前三、最终提交截止倒计时 48 小时、最终提交截止已过**。",
        "只有 Kaggle 官方接口可证明的名次才算前三；Public 榜前三不代表最终领奖资格。", "",
        f"- 这份说明书所属比赛：{ascode(slug)}",
        f"- 当前聚焦比赛：{ascode(data.get('active_competition', '未选择'))}",
        f"- 当前比赛官方截止：{ascode(data.get('active_deadline', '未知'))}",
        f"- 距截止还有（小时）：{ascode(data.get('hours_to_deadline', '未知'))}",
        f"- 排名：{ascode(data.get('rank', '未核实'))}",
        f"- 换赛阶段：{ascode(data.get('status', '待检查'))}",
        f"- 已触发条件：{', '.join(labels.get(x, x) for x in reasons) or '尚未触发'}",
        "", "### 触发之后会发生什么？", "",
        "系统会从 Kaggle 官方列表挑选至少还剩 72 小时的合适比赛。",
        "优先挑已经加入的比赛；若已加入，就自动切换选赛焦点。",
        "如果未加入，会留下待接受规则提示，**不会代替你同意比赛条款**。",
        "切换后必须核对新比赛的标签类型、官方评分指标、提交格式，",
        "不能把旧比赛的二分类模型直接当成已经适配新比赛的训练器。",
        f"- 下一场候选：{ascode(nxt.get('competition', '暂无'))}",
        f"- 新赛是否已报名：{ascode(nxt.get('entered', '未知'))}",
        f"- GPU 最近状态：{ascode(data.get('gpu_training', '未知'))}",
        f"- TPU 最近状态：{ascode(data.get('tpu_training', '未知'))}",
        "", "### 去哪里查看？", "",
        "GitHub Actions → Kaggle Dual GPU-TPU Top3 or Deadline Ladder。",
        "完整自动决策报告：reports/dual_ladder_latest.md。",
        "运行记录：state/dual_ladder.json。",
        "领奖需要最终 Private 榜和官方确认；公开榜前三只触发选赛，不等于领奖成功。",
    ]
    return "\n".join(lines)


def main():
    status=read(ROOT / "state" / "dual_ladder.json")
    if not status:
        print("No ladder state yet; skip manual updates.")
        return
    base=read(ROOT / "state" / "v2_1_status.json")
    next_state=status.get("next_competition") or {}
    slugs={x for x in (base.get("competition"),status.get("active_competition"),next_state.get("competition")) if x}
    root=ROOT / "competitions"
    for slug in sorted(slugs):
        project=root / slug
        project.mkdir(parents=True, exist_ok=True)
        guide=project / "小白说明书.md"
        source=guide.read_text(encoding="utf-8") if guide.exists() else (
            f"# Kaggle 竞赛 {slug}｜小白说明书\n\n"
            "本次由自动换赛模块发现，尚未完成正式数据评估和 GPU/TPU 训练。\n"
            "后续在确认参赛规则、数据结构和官方评分指标后才允许训练与提交。\n"
        )
        if START in source:
            source=source.split(START)[0].rstrip()
        guide.write_text(source.rstrip() + "\n\n" + START + "\n"
                         + explanation(status,slug) + "\n" + END + "\n", encoding="utf-8")
        print("V2.4 guide:", guide.relative_to(ROOT))


if __name__ == "__main__":
    main()
