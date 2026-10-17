from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "agent.yaml"


def run(cmd: list[str]) -> str:
    env = os.environ.copy()
    if not env.get("KAGGLE_API_TOKEN"):
        raise RuntimeError("Missing KAGGLE_API_TOKEN environment variable")
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc.stdout


def parse_competitions(raw: str) -> list[dict]:
    rows: list[dict] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("ref") or set(stripped) <= {"-"}:
            continue

        # Kaggle CLI 2.x renders fixed-width columns separated by 2+ spaces.
        # Expected fields:
        # ref | deadline | category | reward | teamCount | userHasEntered | userRank
        parts = re.split(r"\s{2,}", stripped)
        if len(parts) < 7:
            continue

        ref, deadline, category, reward, team_count, user_has_entered, user_rank = parts[:7]
        slug = ref.rstrip("/").split("/")[-1]
        rows.append({
            "ref": ref,
            "slug": slug,
            "deadline": deadline,
            "category": category,
            "reward": reward,
            "teamCount": int(team_count) if team_count.isdigit() else team_count,
            "userHasEntered": user_has_entered.lower() == "true",
            "userRank": int(user_rank) if user_rank.isdigit() else user_rank,
        })
    return rows


def is_open(deadline: str, now: datetime) -> bool:
    try:
        dt = datetime.strptime(deadline, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return dt > now
    except ValueError:
        return False


def days_remaining(deadline: str, now: datetime) -> int:
    try:
        dt = datetime.strptime(deadline, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return max(0, (dt - now).days)
    except ValueError:
        return 0


def score(row: dict, prefer_categories: list[str], now: datetime) -> int:
    s = 0
    category = str(row.get("category", ""))
    if category in prefer_categories:
        s += 35 - prefer_categories.index(category) * 5
    elif category.lower() in {"featured", "research"}:
        s += 10

    slug = str(row.get("slug", "")).lower()
    if any(k in slug for k in ["playground", "prediction", "classification", "regression", "tabular"]):
        s += 30
    if any(k in slug for k in ["image", "audio", "video", "segmentation", "agi", "llm"]):
        s -= 10

    remaining = days_remaining(str(row.get("deadline", "")), now)
    if remaining >= 7:
        s += 15
    elif remaining >= 2:
        s += 8

    team_count = row.get("teamCount")
    if isinstance(team_count, int) and team_count <= 5000:
        s += 5
    return s


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    scout_cfg = cfg["scout"]
    raw = run([
        "kaggle",
        "competitions",
        "list",
        "--page-size",
        str(scout_cfg.get("page_size", 100)),
    ])

    now = datetime.now(timezone.utc)
    comps = parse_competitions(raw)
    min_days = int(scout_cfg.get("min_days_remaining", 2))
    comps = [
        c for c in comps
        if is_open(c["deadline"], now) and days_remaining(c["deadline"], now) >= min_days
    ]

    prefer = scout_cfg.get("prefer_categories", [])
    for c in comps:
        c["daysRemaining"] = days_remaining(c["deadline"], now)
        c["score"] = score(c, prefer, now)

    comps.sort(key=lambda x: (x.get("score", 0), x.get("daysRemaining", 0)), reverse=True)
    comps = comps[: int(scout_cfg.get("max_candidates", 20))]

    now_iso = now.isoformat()
    payload = {
        "generated_at": now_iso,
        "agent": cfg["agent"],
        "candidate_count": len(comps),
        "competitions": comps,
    }

    json_path = ROOT / cfg["output"]["competitions_json"]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = ROOT / cfg["output"]["latest_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Kaggle Auto Scientist — Scout Report",
        "",
        f"Generated: `{now_iso}`",
        "",
        f"Open candidates: **{len(comps)}**",
        "",
        "| Rank | Competition | Category | Deadline | Days left | Teams | Entered | Score |",
        "|---:|---|---|---|---:|---:|---|---:|",
    ]
    for i, c in enumerate(comps, start=1):
        lines.append(
            f"| {i} | `{c['slug']}` | {c['category']} | {c['deadline']} | {c['daysRemaining']} | {c['teamCount']} | {c['userHasEntered']} | {c['score']} |"
        )
    lines.extend([
        "",
        "## Safety",
        "",
        "- This V1 only discovers and ranks competitions.",
        "- It does **not** accept competition rules.",
        "- It does **not** submit predictions.",
        "- Auto-training and auto-submission will be added only after this authentication/scouting loop is verified.",
        "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {json_path.relative_to(ROOT)}")
    print(f"Wrote {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
