"""Make every scored model report show the matching verified official Kaggle rank.

Only substitute a rank when the currently verified official rank belongs to the
same competition and the *exact same* Public LB. If a new submission isn't
scored yet or scores changed, preserve 'rank awaiting verification'.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from v24_rank_tracker import ROOT, STATE as RANK_STATE, score_rank_lines

REPORTS = [
    ("state/v2_7_status.json", "reports/v2_7_latest.md",
     "## 当前冠军分数对应的官方名次"),
    ("state/v2_6_status.json", "reports/v2_6_latest.md",
     "## 当前已评分冠军对应的官方排名"),
    ("state/v2_5_status.json", "reports/v2_5_latest.md",
     "## 当前已评分冠军对应的官方排名"),
    ("state/v2_3_gpu_status.json", "reports/v2_3_gpu_latest.md",
     "## 当前 Kaggle Public 榜名次（与成绩一起报告）"),
    ("state/v2_3_status.json", "reports/v2_3_latest.md",
     "## Kaggle 当前排名（和 Public LB 一起报告）"),
]


def refresh(content, header, lines):
    # Match section until the next markdown ## heading or EOF.
    replacement = header + "\n\n" + "\n".join(lines) + "\n\n"
    pattern = re.compile(r"(?m)^" + re.escape(header) + r"\s*\n.*?(?=^## |\Z)", re.S)
    if pattern.search(content):
        return pattern.sub(lambda _:replacement,content,count=1)
    # Older model reports predate the rank requirement.
    return content.rstrip() + "\n\n" + replacement


def sync():
    rank = json.loads(RANK_STATE.read_text(encoding="utf-8")) if RANK_STATE.exists() else {}
    if not rank:
        return []
    changed=[]
    for status_file, report_file, header in REPORTS:
        state_path=ROOT/status_file
        report_path=ROOT/report_file
        if not state_path.exists() or not report_path.exists():
            continue
        status=json.loads(state_path.read_text(encoding="utf-8"))
        submission=status.get("latest_submission") or {}
        score=submission.get("publicScore")
        if score in (None, ""):
            # Pending candidate: show verified incumbent rank only if current
            # rank tracker is tied to its same score.
            score=status.get("incumbent_lb")
        lines=score_rank_lines(status.get("competition"),score)
        # Keep the user-facing report explicit about which score the rank is tied to.
        lines=["- 此名次关联的 Public LB："+str(score or "暂无")]+lines
        old=report_path.read_text(encoding="utf-8")
        new=refresh(old,header,lines)
        if new!=old:
            report_path.write_text(new,encoding="utf-8")
            changed.append(str(report_path.relative_to(ROOT)))
    return changed


if __name__=="__main__":
    print("Verified rank reports updated:",sync())
