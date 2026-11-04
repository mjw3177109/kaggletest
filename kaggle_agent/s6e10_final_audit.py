"""Only read Kaggle S6E10 final results. Never submit, train or accept rules.

Official Kaggle CLI 2.2.5 SDK: get_submission(), list_submissions(),
get_leaderboard() with page_token, private leaderboard by default.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

SLUG = "playground-series-s6e10"
SUBMISSION_ID = 57446286
REPORT = Path("reports/2026-11-02_S6E10_赛后归纳.md")
STATE = Path("state/s6e10_final_audit.json")
LATEST = Path("reports/s6e10_final_audit_latest.md")
SECTION = "## 八、S6E10 官方最终成绩认证补证（自动核验）"
AUDIT_STAGE = "init"


def number(value):
    if value is None or not str(value).strip():
        return None
    try:
        n = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    return str(n) if n.is_finite() and Decimal(0) <= n <= Decimal(1) else None


def normalize(value):
    return " ".join(str(value or "").casefold().split())


def status_name(value):
    return str(getattr(value, "name", value or "")).upper().split(".")[-1]


def resolve(owned_refs, detail, rows, complete=True):
    """Pure conservative decision: authenticated submission + final-score + unique team."""
    result = {
        "competition": SLUG,
        "submission_id": SUBMISSION_ID,
        "status": "unverified",
        "private_score_submission": None,
        "final_rank": None,
        "final_team_score": None,
        "team_id": None,
        "rank_source": None,
        "official_award_verified": False,
        "counted_submission_id_verified": False,
    }
    if SUBMISSION_ID not in {int(x) for x in owned_refs}:
        result["status"] = "unverified_submission_not_in_authenticated_account"
        return result
    if int(detail.get("ref") or 0) != SUBMISSION_ID or status_name(detail.get("status")) != "COMPLETE":
        result["status"] = "unverified_submission_not_complete"
        return result
    private = number(detail.get("private_score"))
    result["private_score_submission"] = private
    if private is None:
        result["status"] = "unverified_private_score_not_released"
        return result
    result["status"] = "partial_private_score_verified"
    team = normalize(detail.get("team_name"))
    if not team or not complete:
        result["status"] = "unverified_team_name_or_leaderboard_incomplete"
        return result
    hits = [(i + 1, x) for i, x in enumerate(rows) if normalize(x.get("team_name")) == team]
    if len(hits) != 1:
        result["status"] = "unverified_final_team_ambiguous_or_absent"
        return result
    rank, item = hits[0]
    score = number(item.get("score"))
    team_id = item.get("team_id")
    if not score or not isinstance(team_id, int) or team_id <= 0:
        result["status"] = "unverified_final_row_invalid"
        return result
    result.update(
        status="final_rank_verified",
        final_rank=rank,
        final_team_score=score,
        team_id=team_id,
        rank_source="Kaggle authenticated final leaderboard ordered row; all pages read",
        counted_submission_id_verified=(Decimal(score) == Decimal(private)),
    )
    # An equal numeric score does not prove selection of a specific submission ID.
    result["counted_submission_id_verified"] = False
    return result


def read_authenticated():
    global AUDIT_STAGE
    AUDIT_STAGE = "import_kaggle"
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.competitions.types.competition_api_service import ApiGetLeaderboardRequest

    AUDIT_STAGE = "authenticate"
    api = KaggleApi()
    api.authenticate()
    # Verify the known ref is visible in the *authenticated user's own* history.
    AUDIT_STAGE = "list_account_submissions"
    mine = api.competition_submissions(SLUG, page_size=200) or []
    own_refs = [int(getattr(v, "ref", -1)) for v in mine if v is not None]
    AUDIT_STAGE = "get_historical_submission"
    detail = api.get_submission(SUBMISSION_ID)
    info = {
        "ref": getattr(detail, "ref", None),
        "status": getattr(detail, "status", ""),
        "private_score": getattr(detail, "private_score", ""),
        "team_name": getattr(detail, "team_name", ""),
    }
    if SUBMISSION_ID not in own_refs or number(info["private_score"]) is None:
        return resolve(own_refs, info, [])
    rows = []
    token = ""
    seen = set()
    for _ in range(60):  # bounded: 12,000 rows max; no infinite retry
        AUDIT_STAGE = "get_final_leaderboard_page"
        if token in seen:
            raise RuntimeError("repeated_leaderboard_page_token")
        seen.add(token)
        req = ApiGetLeaderboardRequest()
        req.competition_name = SLUG
        req.override_public = False
        req.page_size = 200
        if token:
            req.page_token = token
        with api.build_kaggle_client() as client:
            page = client.competitions.competition_api_client.get_leaderboard(req)
        batch = list(page.submissions or [])
        if not batch and getattr(page, "next_page_token", ""):
            raise RuntimeError("empty_page_with_cursor")
        for r in batch:
            if r is None:
                raise RuntimeError("null_leaderboard_row")
            rows.append({
                "team_id": int(getattr(r, "team_id", 0)),
                "team_name": getattr(r, "team_name", ""),
                "score": getattr(r, "score", ""),
            })
        token = str(getattr(page, "next_page_token", "") or "")
        if not token:
            return resolve(own_refs, info, rows)
    raise RuntimeError("leaderboard_page_limit_exceeded")


def make_report(evidence):
    rank = evidence.get("final_rank")
    score = evidence.get("private_score_submission")
    team_score = evidence.get("final_team_score")
    verified = evidence.get("status") == "final_rank_verified"
    values = [
        "# S6E10 Kaggle 账户认证赛后补证",
        "",
        "- 竞赛：" + SLUG,
        "- 已知历史提交 ID：" + str(SUBMISSION_ID),
        "- 查询状态：" + evidence["status"],
        "- 已认证提交 Private Score：" + (score if score is not None else "未核实"),
        "- 最终榜团队 Private Score：" + (team_score if team_score is not None else "未核实"),
        "- 最终榜名次：" + (str(rank) if verified else "未核实"),
        "- 最终榜队伍 ID：" + (str(evidence.get("team_id")) if verified else "未核实"),
        "- 名次证据口径：Kaggle 账户提交 + 分页完整最终榜；以官方榜顺序推导行次",
        "- 该 submission ID 是否为最终计分选中版本：未核实",
        "- 是否获得奖品及领取资格：未核实，须另有 Kaggle/主办方确认",
        "- 官网：https://www.kaggle.com/competitions/playground-series-s6e10/leaderboard",
        "- Kaggle 官方 SDK：https://github.com/Kaggle/kaggle-cli/blob/main/src/kaggle/api/kaggle_api_extended.py",
        "",
        "> 注：公开榜 Public 分数及历史排名不能充当最终 Private 名次；失败或缺失维持未核实，不能自动领奖。",
        "",
    ]
    return "\n".join(values)


def update_main_report(evidence, audit_md):
    if evidence["status"] != "final_rank_verified":
        return False
    if not REPORT.exists():
        return False
    old = REPORT.read_text(encoding="utf-8")
    section = SECTION + "\n\n" + audit_md
    if SECTION in old:
        old = old[:old.index(SECTION)].rstrip()
    new = old.rstrip() + "\n\n" + section
    if new == REPORT.read_text(encoding="utf-8"):
        return False
    REPORT.write_text(new, encoding="utf-8")
    return True


def offline_test():
    fake = {"ref": SUBMISSION_ID, "status": "COMPLETE", "private_score": "0.95042", "team_name": "Our Team"}
    rows = [
        {"team_id": 1, "team_name": "First", "score": "0.99"},
        {"team_id": 2, "team_name": "Our Team", "score": "0.95042"},
    ]
    ok = resolve([SUBMISSION_ID], fake, rows)
    assert ok["status"] == "final_rank_verified" and ok["final_rank"] == 2
    assert resolve([], fake, rows)["status"] == "unverified_submission_not_in_authenticated_account"
    assert resolve([SUBMISSION_ID], fake, rows + [rows[1]])["status"] == "unverified_final_team_ambiguous_or_absent"
    assert resolve([SUBMISSION_ID], {**fake, "private_score": ""}, rows)["final_rank"] is None
    assert resolve([SUBMISSION_ID], fake, rows, complete=False)["final_rank"] is None
    print("S6E10 offline evidence tests passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-test", action="store_true")
    args = parser.parse_args()
    if args.offline_test:
        offline_test()
        return
    if not os.environ.get("KAGGLE_API_TOKEN"):
        evidence = {"status": "unverified_missing_secret"}
    else:
        try:
            evidence = read_authenticated()
        except Exception as exc:
            # Never print raw exception strings: network errors can include secrets.
            evidence = {"status": "unverified_authenticated_query_failed",
                        "failed_stage": AUDIT_STAGE, "error_class": type(exc).__name__,
                        "missing_module": getattr(exc, "name", None) if isinstance(exc, ModuleNotFoundError) else None}
    evidence.setdefault("competition", SLUG)
    evidence.setdefault("submission_id", SUBMISSION_ID)
    # Do not churn repo state for timestamp-only polling; GitHub Actions run itself is timestamped.
    output = json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    STATE.parent.mkdir(parents=True, exist_ok=True)
    LATEST.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(output, encoding="utf-8")
    report = make_report(evidence)
    LATEST.write_text(report, encoding="utf-8")
    changed = update_main_report(evidence, report)
    print("S6E10 status:", evidence["status"])
    print("S6E10 authenticated private score available:", bool(evidence.get("private_score_submission")))
    print("S6E10 final rank verified:", evidence.get("final_rank") is not None)
    print("S6E10 main report updated:", changed)


if __name__ == "__main__":
    main()
