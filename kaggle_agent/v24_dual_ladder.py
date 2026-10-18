"""V2.4 dual GPU/TPU competition ladder: verify *public* rank and choose the next eligible contest.

This module never signs rules, registers for a competition, claims rewards,
uploads a kernel or submits predictions. It only updates deterministic state.
Existing GPU and TPU recoveries are separate workflows and never overwritten.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "state" / "dual_ladder.json"
REPORT = ROOT / "reports" / "dual_ladder_latest.md"
BASELINE = ROOT / "state" / "v2_1_status.json"
LIMIT = 3
MIN_HOURS_LEFT = 48
ELIGIBLE_CATEGORIES = {"playground", "gettingstarted"}


def utc():
    return datetime.now(timezone.utc)


def load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def parse_json_rows(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("items", "results", "competitions", "data"):
            if isinstance(value.get(key), list):
                return value[key]
        return [value]
    return []


def norm_slug(row):
    ref = str(row.get("ref") or row.get("slug") or "").rstrip("/")
    return ref.split("/")[-1]


def parse_deadline(raw):
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def is_entered(row):
    value = row.get("userHasEntered", False)
    return value is True or str(value).strip().lower() == "true"


def safe_rank(row):
    raw = row.get("userRank")
    # Kaggle ranks 1,2,3 are legitimate. Zero / None / unknown never means top 3.
    if isinstance(raw, bool):
        return None
    try:
        if isinstance(raw, float) and not raw.is_integer():
            return None
        rank = int(raw)
        return rank if rank > 0 else None
    except (TypeError, ValueError):
        return None


def cli_list(*args):
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise RuntimeError("KAGGLE_API_TOKEN not available")
    cmd = ["kaggle", "competitions", "list", *args, "--format", "json"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
    if proc.returncode:
        raise RuntimeError(f"Kaggle list failed: {(proc.stderr or proc.stdout)[-1200:]}")
    return [x for x in parse_json_rows(json.loads(proc.stdout)) if isinstance(x, dict)]


def next_candidates(rows, current, history, when):
    choices = []
    for item in rows:
        slug = norm_slug(item)
        category = str(item.get("category") or "").replace(" ", "").lower()
        deadline = parse_deadline(item.get("deadline"))
        if (not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,}", slug or "")
                or slug == current or slug in history
                or category not in ELIGIBLE_CATEGORIES
                or deadline is None):
            continue
        hours_left = (deadline - when).total_seconds() / 3600
        if hours_left < MIN_HOURS_LEFT:
            continue
        choices.append({
            "competition": slug,
            "url": f"https://www.kaggle.com/competitions/{slug}",
            "category": str(item.get("category")),
            "deadline": deadline.isoformat(),
            "hours_left": round(hours_left, 1),
            "entered": is_entered(item),
            "observed_user_rank": safe_rank(item),
            "reward": item.get("reward"),
        })
    # Prefer contests the user has ALREADY legally joined; no invented rules consent.
    choices.sort(key=lambda d: (not d["entered"], -d["hours_left"], d["competition"]))
    return choices


def decide(previous, current, rank_row, candidates, now):
    data = dict(previous)
    data["version"] = "V2.4 ladder"
    data["active_competition"] = data.get("active_competition") or current
    data["updated_at"] = now.isoformat()
    data["public_rank_verified"] = False
    data["status"] = "monitoring_public_leaderboard"
    data["warning"] = ""
    data["gpu_training"] = (load(ROOT / "state" / "v2_3_gpu_status.json").get("stage") or "unknown")
    data["tpu_training"] = (load(ROOT / "state" / "v2_3_status.json").get("stage") or "unknown")
    data["promotion_policy"] = {
        "public_top3": "Trigger change of focus, NOT an official prize",
        "final_top3": "Must separately verify final private leaderboard and organizer eligibility",
        "join_next": "Never accept Kaggle rules without the user's explicit acceptance",
        "training_next": "Only use verified compatible task and scoring metric",
    }

    if rank_row is None or not is_entered(rank_row):
        data["rank"] = None
        data["status"] = "rank_unverified"
        data["warning"] = "Kaggle did not confirm entered participation and rank; no promotion."
        return data
    rank = safe_rank(rank_row)
    data["rank"] = rank
    data["public_rank_verified"] = rank is not None
    if rank is None:
        data["status"] = "rank_unverified"
        data["warning"] = "userRank is 0/missing/unreadable; refuse to guess placement."
        return data
    if rank > LIMIT:
        data["status"] = "working_toward_top3"
        return data

    # A rank <= 3 is LIVE/PUBLIC evidence only; it cannot prove final award.
    data["public_top3_achieved"] = {
        "competition": current, "rank": rank,
        "observed_at": now.isoformat(), "basis": "kaggle competitions list userRank (PUBLIC)",
        "official_prize_confirmed": False,
    }
    history = set(previous.get("rotated_from") or [])
    choices = next_candidates(candidates, current, history, now)
    data["rotated_from"] = sorted(history | {current})
    if not choices:
        data["status"] = "top3_no_eligible_contest"
        data["next_competition"] = None
        data["warning"] = "No eligible open Playground/gettingStarted competition has >=48h left."
        return data
    choice = choices[0]
    data["next_competition"] = choice
    if choice["entered"]:
        data["status"] = "next_entered_requires_task_check"
        data["warning"] = (
            "Next contest is already joined; verify target type and official evaluation metric "
            "before attempting an independent GPU+TPU training run. "
            "Existing S6E10 model scores are NOT comparable across contests."
        )
    else:
        data["status"] = "next_awaits_rule_acceptance"
        data["warning"] = (
            "Next contest chosen but not joined. Do not autoaccept its rules or auto-submit; "
            "user must join on Kaggle and accept the specific competition's rules first."
        )
    return data


def report(data):
    gpu = load(ROOT / "state" / "v2_3_gpu_status.json")
    tpu = load(ROOT / "state" / "v2_3_status.json")
    rank = data.get("rank")
    nxt = data.get("next_competition") or {}
    lines = [
        "# Kaggle Auto Scientist｜双线冲榜与前三自动换赛监控", "",
        f"- 检查时间：{data.get('updated_at')}",
        f"- 当前比赛：{data.get('active_competition')}",
        f"- Kaggle 官方公开榜名次：{rank if rank is not None else '暂时无法确认'}",
        f"- 是否有有效排名证据：{data.get('public_rank_verified')}",
        f"- GPU：{gpu.get('stage', 'unknown')}；{gpu.get('kernel_url', '')}",
        f"- TPU：{tpu.get('stage', 'unknown')}；{tpu.get('kernel_url', '')}",
        f"- 换赛阶段：{data.get('status')}",
        "", "## 转换规则", "",
        "Public 榜 ≤3 名仅表示实时冲榜达到阶段目标，**不等于最终 Private 榜前三，也不代表已经获得奖品**。",
        "公开榜达到前三后自动检索下一场仍开放、至少剩余 48 小时的 Playground/Get Started 比赛。",
        "只对已报名且任务类型、评分指标兼容的下一场执行训练；没有报名时必须本人同意并接受规则，不能自动替用户签署。",
        "GPU 与 TPU 训练状态分别保留，不会重复创建已经排队的 Notebook。",
        "",
        "## 下一场比赛", "",
        f"- 候选：{nxt.get('competition', '暂无符合资格的候选')}",
        f"- 链接：{nxt.get('url', '暂无')}",
        f"- 是否已报名：{nxt.get('entered', '未知')}",
        f"- 截止时间：{nxt.get('deadline', '暂无')}",
        "", "## 说明", "",
        data.get("warning") or "继续提升当前比赛的验证和公开榜排名。",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    base = load(BASELINE)
    if base.get("stage") != "complete":
        raise RuntimeError("Cannot monitor ladder without a verified completed prior submission")
    current = str(load(STATE).get("active_competition") or base["competition"])
    old = load(STATE)
    # Only KAGGLE's authenticated entered list can establish a rank.
    entered = cli_list("--group", "entered", "--page-size", "100")
    rank_row = next((x for x in entered if norm_slug(x) == current), None)
    candidates = []
    for category in ("playground", "gettingStarted"):
        try:
            candidates.extend(cli_list("--category", category, "--page-size", "100"))
        except Exception as e:
            print(f"Candidate query failed for {category}: {e}", flush=True)
    # Deduplicate; prefer entered evidence when the category list is incomplete.
    merged = {norm_slug(c): c for c in candidates}
    for row in entered:
        k = norm_slug(row)
        merged[k] = {**merged.get(k, {}), **row}
    result = decide(old, current, rank_row, list(merged.values()), utc())
    write(STATE, result)
    report(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
