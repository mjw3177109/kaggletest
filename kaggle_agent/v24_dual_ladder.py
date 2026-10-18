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
MIN_HOURS_LEFT = 72  # Leave enough time to start a NEW competition before its 48h handoff.
HANDOFF_HOURS = 48
# Official verified final-submission deadline for the existing S6E10 competition.
# Future contests MUST use a deadline supplied by Kaggle itself or archived at transition.
KNOWN_DEADLINES = {"playground-series-s6e10": "2026-10-31T23:59:00+00:00"}
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


def current_deadline(previous, current, rank_row, candidates):
    """Never guess deadlines. Prefer fresh Kaggle CLI; keep previously verified fallback."""
    rows = ([rank_row] if rank_row is not None else []) + candidates
    for row in rows:
        if row and norm_slug(row) == current:
            deadline = parse_deadline(row.get("deadline"))
            if deadline is not None:
                return deadline, "Kaggle competitions list"
    # A deadline inherited from a previously selected Kaggle competition remains usable
    # if the competition no longer appears in the default open-only list.
    if previous.get("active_competition") == current:
        archived = parse_deadline(previous.get("active_deadline"))
        if archived is not None:
            return archived, "archived Kaggle competition deadline"
    official = parse_deadline(KNOWN_DEADLINES.get(current))
    if official is not None:
        return official, "verified official competition timeline"
    return None, "unavailable"


def handoff_reasons(rank_row, deadline, now):
    reasons = []
    rank = safe_rank(rank_row) if rank_row and is_entered(rank_row) else None
    if rank is not None and rank <= LIMIT:
        reasons.append("public_top3")
    if deadline is not None:
        hours_left = (deadline - now).total_seconds() / 3600
        if hours_left <= 0:
            reasons.append("competition_ended")
        elif hours_left <= HANDOFF_HOURS:
            reasons.append("submission_deadline_window")
    return reasons


def decide(previous, current, rank_row, candidates, now):
    data = dict(previous)
    data["version"] = "V2.4 ladder"
    data["active_competition"] = current
    data["updated_at"] = now.isoformat()
    data["warning"] = ""
    data["status"] = "monitoring_public_leaderboard"
    data["gpu_training"] = (load(ROOT / "state" / "v2_3_gpu_status.json").get("stage") or "unknown")
    data["tpu_training"] = (load(ROOT / "state" / "v2_3_status.json").get("stage") or "unknown")
    rank = safe_rank(rank_row) if rank_row is not None and is_entered(rank_row) else None
    data["rank"] = rank
    data["public_rank_verified"] = rank is not None
    deadline, source = current_deadline(previous, current, rank_row, candidates)
    data["active_deadline"] = deadline.isoformat() if deadline else None
    data["deadline_source"] = source
    data["hours_to_deadline"] = round((deadline - now).total_seconds() / 3600, 2) if deadline else None
    data["promotion_policy"] = {
        "public_top3": "A verified Kaggle PUBLIC rank <= 3 triggers handoff; not a prize",
        "submission_deadline_window": "Trigger handoff inside 48h before the verified final submission cutoff",
        "competition_ended": "Trigger handoff after the verified final submission cutoff",
        "final_top3": "Requires separate final private leaderboard and organizer verification",
        "join_next": "Never accept Kaggle rules without explicit user consent",
        "training_next": "Only after target type and scoring metric preflight; existing GPU/TPU kernels remain scoped to S6E10",
    }
    reasons = handoff_reasons(rank_row, deadline, now)
    data["handoff_reasons"] = reasons
    data["next_competition"] = None
    data["next_training_enabled"] = False

    if not reasons:
        if rank is None and deadline is None:
            data["status"] = "rank_and_deadline_unverified"
            data["warning"] = "Rank and deadline are both unverified; do not switch."
        elif rank is None:
            data["status"] = "monitoring_deadline"
            data["warning"] = "Rank unknown; independently monitoring the verified final submission deadline."
        else:
            data["status"] = "working_toward_top3"
        return data

    if "public_top3" in reasons:
        data["public_top3_achieved"] = {
            "competition": current, "rank": rank,
            "observed_at": now.isoformat(), "basis": "kaggle competitions list userRank (PUBLIC)",
            "official_prize_confirmed": False,
        }

    history = set(previous.get("rotated_from") or [])
    choices = next_candidates(candidates, current, history, now)
    data["last_handoff"] = {
        "from": current, "reasons": reasons, "rank": rank,
        "deadline": data["active_deadline"], "detected_at": now.isoformat(),
    }
    if not choices:
        data["status"] = "handoff_no_eligible_contest"
        data["warning"] = "No qualifying open next contest with at least 72h before its deadline."
        return data

    choice = choices[0]
    data["next_competition"] = choice
    if not choice["entered"]:
        # User hasn't joined this next contest; no automatic terms/rules acceptance.
        data["status"] = "next_awaits_rule_acceptance"
        data["warning"] = "Suggested next contest is not entered. Await user acceptance of its competition rules."
        return data

    # Actually move the *selection focus* when the next contest is already joined.
    # DO NOT assume S6E10 GPU/TPU trainer is compatible or auto-submit predictions.
    data["rotated_from"] = sorted(history | {current})
    data["active_competition"] = choice["competition"]
    data["active_deadline"] = choice["deadline"]
    data["deadline_source"] = "selected Kaggle competitions list"
    data["rank"] = choice["observed_user_rank"]
    data["public_rank_verified"] = choice["observed_user_rank"] is not None
    data["hours_to_deadline"] = choice["hours_left"]
    data["status"] = "rotated_waiting_task_preflight"
    data["warning"] = (
        "Automatically switched selection to a joined new contest. GPU and TPU S6E10 kernels "
        "are not cross-competition trainers; first verify dataset, target, metric, competition "
        "rules and submission format before training the new contest."
    )
    return data


def report(data):
    gpu = load(ROOT / "state" / "v2_3_gpu_status.json")
    tpu = load(ROOT / "state" / "v2_3_status.json")
    rank = data.get("rank")
    nxt = data.get("next_competition") or {}
    lines = [
        "# Kaggle Auto Scientist｜GPU+TPU 双线冲榜与截止期自动换赛", "",
        f"- 检查时间：{data.get('updated_at')}",
        f"- 当前比赛：{data.get('active_competition')}",
        f"- Kaggle 官方公开榜名次：{rank if rank is not None else '暂时无法确认'}",
        f"- 是否有有效排名证据：{data.get('public_rank_verified')}",
        f"- GPU：{gpu.get('stage', 'unknown')}；{gpu.get('kernel_url', '')}",
        f"- TPU：{tpu.get('stage', 'unknown')}；{tpu.get('kernel_url', '')}",
        f"- 换赛阶段：{data.get('status')}",
        f"- 当前比赛最终截止：{data.get('active_deadline')}",
        f"- 截止时间证据：{data.get('deadline_source')}",
        f"- 距离截止（小时）：{data.get('hours_to_deadline')}",
        f"- 已触发换赛条件：{data.get('handoff_reasons')}",
        "", "## 转换规则", "",
        "Public 榜 ≤3 名仅表示实时冲榜达到阶段目标，**不等于最终 Private 榜前三，也不代表已经获得奖品**。",
        "以下任一条件触发选赛：认证公开榜前三、最终提交截止剩余 48 小时以内、最终提交截止已过。",
        "挑选剩余至少 72 小时且已报名的 Playground/Get Started 比赛优先。",
        "若下一场已报名，自动切换选赛焦点；训练需额外通过任务类型、评分指标和提交格式自检。",
        "若下一场未报名，只提供候选和待接受规则状态，不会替用户注册或同意规则。",
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
    # Only KAGGLE's authenticated entered list can establish a public rank.
    entered = cli_list("--group", "entered", "--page-size", "100")
    rank_row = next((x for x in entered if norm_slug(x) == current), None)
    # A completed contest may disappear from the entered/open list. The archived
    # competition deadline still allows closed-competition handoff safely.
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
        merged[k] = {**merged.get(k, {}), **{p: v for p, v in row.items() if v is not None}}
    result = decide(old, current, rank_row, list(merged.values()), utc())
    write(STATE, result)
    report(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
