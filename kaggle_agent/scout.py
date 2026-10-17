from __future__ import annotations

import csv
import json
import os
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
    lines = [line for line in raw.splitlines() if line.strip()]
    if not lines:
        return []

    # Kaggle CLI table output is whitespace-separated. We only need stable columns.
    rows: list[dict] = []
    for line in lines:
        if line.startswith("ref") or set(line.strip()) <= {"-", " "}:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        ref = parts[0]
        deadline = parts[1]
        category = parts[-2]
        reward = parts[-1]
        title = " ".join(parts[2:-2])
        rows.append({
            "ref": ref,
            "deadline": deadline,
            "category": category,
            "reward": reward,
            "title": title,
        })
    return rows


def score(row: dict, prefer_categories: list[str]) -> int:
    s = 0
    category = str(row.get("category", ""))
    if category in prefer_categories:
        s += 30 - prefer_categories.index(category) * 5
    title = str(row.get("title", "")).lower()
    if any(k in title for k in ["playground", "prediction", "classification", "regression", "tabular"]):
        s += 25
    if any(k in title for k in ["image", "audio", "video", "llm", "segmentation"]):
        s -= 10
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

    comps = parse_competitions(raw)
    prefer = scout_cfg.get("prefer_categories", [])
    for c in comps:
        c["score"] = score(c, prefer)

    comps.sort(key=lambda x: x.get("score", 0), reverse=True)
    comps = comps[: int(scout_cfg.get("max_candidates", 20))]

    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "generated_at": now,
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
        f"Generated: `{now}`",
        "",
        f"Candidates: **{len(comps)}**",
        "",
        "| Rank | Competition | Category | Deadline | Score |",
        "|---:|---|---|---|---:|",
    ]
    for i, c in enumerate(comps, start=1):
        lines.append(
            f"| {i} | `{c['ref']}` — {c['title']} | {c['category']} | {c['deadline']} | {c['score']} |"
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
