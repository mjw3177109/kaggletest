from __future__ import annotations

import json
import os
import re
import subprocess
import time
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
        raise RuntimeError(f"Kaggle CLI failed with exit code {proc.returncode}; credentials and raw stderr withheld")
    return proc.stdout


def parse_competitions(raw: str) -> list[dict]:
    """Accept the official CLI JSON output; retain legacy table compatibility."""
    raw = raw.strip()
    if raw.startswith(("[", "{")):
        payload = json.loads(raw)
        if isinstance(payload, dict):
            payload = next((payload[k] for k in ("competitions", "items", "results", "data")
                            if isinstance(payload.get(k), list)), [])
        if not isinstance(payload, list):
            raise ValueError("Unexpected Kaggle competitions JSON structure")
        return [
            dict(item, slug=str(item.get("ref") or item.get("slug") or "").rstrip("/").split("/")[-1])
            for item in payload if isinstance(item, dict)
        ]
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


def parse_deadline(deadline: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(deadline).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def is_open(deadline: str, now: datetime) -> bool:
    dt = parse_deadline(deadline)
    return dt is not None and dt > now


def days_remaining(deadline: str, now: datetime) -> int:
    dt = parse_deadline(deadline)
    return max(0, (dt - now).days) if dt else 0


def score(row: dict, prefer_categories: list[str], now: datetime) -> int:
    s = 0
    category = str(row.get("category", "")).replace(" ", "").lower()
    preferred = [str(x).replace(" ", "").lower() for x in prefer_categories]
    if category in preferred:
        s += 35 - preferred.index(category) * 5
    elif category in {"featured", "research"}:
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


def discover_competitions(page_size: int, max_pages: int = 2) -> list[dict]:
    """Bounded, category-specific read-only discovery; entered list verifies membership."""
    merged: dict[str, dict] = {}
    scopes = [
        ("general", "playground"),
        ("general", "gettingStarted"),
        ("general", "featured"),
        ("general", "research"),
        ("entered", "all"),
    ]
    first_request = True
    for group, category in scopes:
        for page in range(1, max_pages + 1):
            if not first_request:
                time.sleep(3)  # Avoid Kaggle API bursts.
            first_request = False
            rows = parse_competitions(run([
                "kaggle", "competitions", "list",
                "--group", group, "--category", category,
                "--sort-by", "recentlyCreated",
                "--page-size", str(page_size), "--page", str(page),
                "--format", "json",
            ]))
            for item in rows:
                slug = str(item.get("slug") or "").strip()
                if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,}", slug):
                    continue
                if slug not in merged:
                    merged[slug] = dict(item)
                if group == "entered" or item.get("userHasEntered") is True:
                    merged[slug]["userHasEntered"] = True
                    if item.get("userRank") not in (None, 0, "0"):
                        merged[slug]["userRank"] = item["userRank"]
            if len(rows) < page_size:
                break
    return list(merged.values())


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    scout_cfg = cfg["scout"]
    now = datetime.now(timezone.utc)
    comps = discover_competitions(int(scout_cfg.get("page_size", 100)))
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
