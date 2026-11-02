"""Read-only Kaggle S6E11 enrollment gate. Never accepts rules, trains or submits.

Two independent authenticated CLI views must agree before membership is trusted.
Only a material status change is committed; hourly proof remains in Actions logs.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from kaggle_agent import scout

ROOT = Path(__file__).resolve().parents[1]
SLUG = "playground-series-s6e11"
STATE = ROOT / "state" / "s6e11_entry_gate.json"
REPORT = ROOT / "reports" / "s6e11_entry_gate_latest.md"


def rows_for(group: str, category: str, max_pages: int) -> tuple[list[dict], bool]:
    rows: list[dict] = []
    for page in range(1, max_pages + 1):
        if group != "general" or page > 1:
            time.sleep(3)  # Kaggle rate-limit protection across sequential calls.
        result = scout.parse_competitions(scout.run([
            "kaggle", "competitions", "list",
            "--group", group, "--category", category,
            "--sort-by", "recentlyCreated",
            "--page-size", "100", "--page", str(page), "--format", "json",
        ]))
        rows.extend(result)
        if len(result) < 100:
            return rows, True
    return rows, False


def exact(rows: list[dict]) -> dict | None:
    matches = [r for r in rows if r.get("slug") == SLUG]
    if len(matches) > 1:
        raise RuntimeError("Duplicate target competition in authenticated CLI result")
    return matches[0] if matches else None


def verify(general: list[dict], entered: list[dict], entered_exhaustive: bool) -> dict:
    target = exact(general)
    if target is None:
        raise RuntimeError("Target missing from fresh Kaggle Playground listing")
    enrolled = exact(entered)
    general_flag = target.get("userHasEntered")
    if general_flag is not True and general_flag is not False:
        raise RuntimeError("Kaggle membership flag missing or non-boolean")
    if enrolled is not None:
        if general_flag is not True or enrolled.get("userHasEntered") is not True:
            raise RuntimeError("Kaggle authenticated membership sources disagree")
        joined = True
    else:
        if not entered_exhaustive or general_flag is True:
            raise RuntimeError("Kaggle membership not independently verifiable")
        joined = False
    deadline = scout.parse_deadline(target.get("deadline"))
    if deadline is None:
        raise RuntimeError("Official competition deadline missing")
    now = datetime.now(timezone.utc)
    if now >= deadline:
        stage = "closed_no_submission"
    elif joined:
        stage = "joined_pending_official_rules_review"
    else:
        stage = "blocked_awaiting_user_rule_acceptance"
    return {
        "competition": SLUG,
        "stage": stage,
        "userHasEntered": joined,
        "deadline": deadline.isoformat(),
        "source": "authenticated Kaggle CLI general+entered",
        "training_allowed": False,
        "submission_allowed": False,
    }


def main() -> None:
    general, _ = rows_for("general", "playground", 3)
    entered, exhaustive = rows_for("entered", "all", 5)
    result = verify(general, entered, exhaustive)
    now = datetime.now(timezone.utc).isoformat()
    print(f"S6E11 authenticated entry check at {now}: {result['stage']}")
    print("No competition rules accepted, data downloaded, model trained, or prediction submitted.")

    old = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    if all(old.get(k) == v for k, v in result.items()):
        print("No material gate change; preserving existing durable state.")
        return
    result["changed_at"] = now
    STATE.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT.write_text(
        "# Kaggle S6E11 enrollment gate\n\n"
        f"- Checked: {now}\n"
        f"- Stage: `{result['stage']}`\n"
        f"- Authenticated joined: `{result['userHasEntered']}`\n"
        f"- Official deadline: `{result['deadline']}`\n"
        "- Sources: Kaggle CLI general and entered groups, both authenticated.\n"
        "- No rules acceptance, data download, training, or submission performed.\n"
        "- Joined status alone does not authorize training until official rules and data schema are reviewed.\n",
        encoding="utf-8",
    )
    print("Durable entry gate state changed and recorded.")


if __name__ == "__main__":
    main()
