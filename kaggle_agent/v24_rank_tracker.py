"""Verified Kaggle score + public leaderboard position for each report.

Use authenticated competitions list userRank first. As a fallback inspect Kaggle's
full leaderboard: an exact *member username* match may verify a row, while merely
matching TeamName is only a hint, not proof of ownership.
Never calculate an official rank from a score or confuse Public and Private ranks.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "state" / "score_rank_latest.json"
REPORT = ROOT / "reports" / "score_rank_latest.md"


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def candidates_for(slug):
    """Choose the best actual scored submission, not an unsubmitted CV candidate."""
    found = []
    for path in ("v2_status.json", "v2_1_status.json", "v2_3_gpu_status.json", "v2_3_status.json", "v2_5_status.json", "v2_6_status.json", "v2_7_status.json", "v2_8_status.json"):
        obj = read(ROOT / "state" / path)
        if obj.get("competition") != slug:
            continue
        submission = obj.get("latest_submission") or {}
        score = submission.get("publicScore")
        if score in (None, ""):
            continue
        try:
            numeric = float(score)
        except (ValueError, TypeError):
            continue
        if not 0 <= numeric <= 1:
            continue
        found.append({
            "source": path,
            "model": obj.get("champion") or obj.get("version"),
            "public_score": numeric,
            "submission_id": submission.get("ref"),
            "submitted_at": submission.get("date"),
        })
    # Current V2.9 submissions have dedicated, durable one-shot status files.
    # Only an authenticated, fully scored Kaggle submission is eligible; a
    # successful CV candidate or a reserved/pending Kaggle request is NOT.
    if slug == "playground-series-s6e10":
        # A future V2915 public score is eligible only when tied to a
        # durably reserved SHA and an independently qualified 3fold OOF.
        # A leaderboard result cannot be inferred from Kaggle CV alone.
        newer = read(ROOT / "state" / "v2_9_15_submission.json")
        proposal = read(ROOT / "state" / "v2_9_15_status.json")
        entry15 = newer.get("submission") or {}
        sha15 = str(proposal.get("candidate_test_hash") or "")
        note15 = "KAS V2915 5E " + sha15[:16]
        comp15 = proposal.get("comparison") or {}
        if (newer.get("competition") == slug
            and newer.get("stage") == "complete"
            and newer.get("submitted") is True
            and newer.get("submission_attempted") is True
            and proposal.get("stage") == "candidate_ready_unsubmitted"
            and proposal.get("submitted") is False
            and len(sha15) == 64
            and newer.get("prediction_hash") == sha15
            and newer.get("submission_note") == note15
            and float(comp15.get("gain", 0)) >= .00008
            and float(comp15.get("candidate_auc", 0)) >= .96125635576833
            and entry15.get("status") == "SubmissionStatus.COMPLETE"
            and entry15.get("description") == note15
            and entry15.get("publicScore") not in (None, "")):
            try:
                ref = int(entry15.get("ref"))
                score15 = float(entry15["publicScore"])
                if (ref > 0 and 0 <= score15 <= 1
                    and abs(score15-float(newer.get("official_lb"))) < 1e-9):
                    found.append({
                        "source":"v2_9_15_submission.json",
                        "model":"V2.9.15 RealMLP 5epoch blended",
                        "public_score":score15,
                        "submission_id":ref,
                        "submitted_at":entry15.get("date"),
                    })
            except (ValueError, TypeError):
                pass
        # Newly completed V2913 official Kaggle submission (not CV alone).
        v2913 = read(ROOT / "state" / "v2_9_13_submission.json")
        experiment = read(ROOT / "state" / "v2_9_13_status.json")
        official = v2913.get("submission") or {}
        if (v2913.get("competition") == slug
            and v2913.get("stage") == "complete"
            and v2913.get("submitted") is True
            and experiment.get("stage") == "candidate_ready_unsubmitted"
            and float((experiment.get("comparison") or {}).get("candidate_auc", 0)) >= .96117635576833 - .000002
            and official.get("status") == "SubmissionStatus.COMPLETE"
            and str(official.get("ref")) == "57423879"
            and official.get("description") == v2913.get("submission_note")
            and official.get("publicScore") not in (None, "")):
            try:
                public = float(official["publicScore"])
                if 0 <= public <= 1 and abs(public-float(v2913.get("official_lb"))) < 1e-9:
                    found.append({
                        "source":"v2_9_13_submission.json",
                        "model":"V2.9.13 RealMLP Ens8 fusion",
                        "public_score":public,
                        "submission_id":official.get("ref"),
                        "submitted_at":official.get("date"),
                    })
            except (ValueError, TypeError):
                pass
        recent = read(ROOT / "state" / "v2_9_9_submission.json")
        candidate = read(ROOT / "state" / "v2_9_9_status.json")
        entry = recent.get("submission") or {}
        if (recent.get("competition") == slug
            and recent.get("stage") == "complete"
            and recent.get("submitted") is True
            and candidate.get("stage") == "candidate_ready_unsubmitted"
            and float(candidate.get("candidate_cv", 0)) >= .9610561936366048 - .000002
            and entry.get("status") == "SubmissionStatus.COMPLETE"
            and str(entry.get("ref")) == "57421208"
            and entry.get("description") == recent.get("submission_note")
            and entry.get("publicScore") not in (None, "")):
            try:
                official = float(entry["publicScore"])
                if 0 <= official <= 1 and abs(official-float(recent.get("official_lb"))) < 1e-9:
                    found.append({
                        "source":"v2_9_9_submission.json",
                        "model":"V2.9.9 Four-model fusion",
                        "public_score":official,
                        "submission_id":entry.get("ref"),
                        "submitted_at":entry.get("date"),
                    })
            except (ValueError, TypeError):
                pass
        v296 = read(ROOT / "state" / "v2_9_6_status.json")
        old = v296.get("latest_submission") or {}
        if (v296.get("competition") == slug
            and v296.get("stage") == "complete"
            and v296.get("submitted") is True
            and old.get("status") == "SubmissionStatus.COMPLETE"
            and str(old.get("ref")) == "57408736"
            and old.get("publicScore") not in (None,"")):
            try:
                score = float(old["publicScore"])
                if 0 <= score <= 1:
                    found.append({
                        "source":"v2_9_6_status.json",
                        "model":"V2.9.6 RealMLP",
                        "public_score":score,
                        "submission_id":old.get("ref"),
                        "submitted_at":old.get("date"),
                    })
            except (ValueError, TypeError):
                pass
    # ROC-AUC for the existing S6E10 contest: higher is better.
    return max(found, key=lambda x: x["public_score"]) if found else None


def kaggle_json(args):
    p = subprocess.run(["kaggle", "competitions", "list", *args, "--format", "json"],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    if p.returncode:
        raise RuntimeError((p.stderr or p.stdout)[-1400:])
    payload = json.loads(p.stdout)
    if isinstance(payload, list):
        return payload
    for key in ("results", "items", "competitions", "data"):
        if isinstance(payload.get(key), list):
            return payload[key]
    return []


def norm_slug(row):
    return str(row.get("ref") or row.get("slug") or "").rstrip("/").split("/")[-1]


def valid_rank(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    try:
        rank = int(value)
        return rank if rank > 0 else None
    except (TypeError, ValueError):
        return None


def valid_team_count(value):
    n = valid_rank(value)
    return n


def direct_rank(rows, slug):
    """Authenticated userRank is account-bound and must be preferred."""
    for row in rows:
        if not isinstance(row, dict) or norm_slug(row) != slug:
            continue
        r = valid_rank(row.get("userRank"))
        n = valid_team_count(row.get("teamCount"))
        if r is not None and str(row.get("userHasEntered", "")).lower() == "true":
            if n is not None and r > n:
                # Inconsistent upstream data: no proven place.
                return {"rank": None, "teams": n, "source": "inconsistent_rank"}
            return {"rank": r, "teams": n, "source": "kaggle_authenticated_userRank"}
        return {"rank": None, "teams": n, "source": "no_public_rank"}
    return {"rank": None, "teams": None, "source": "entry_not_returned"}


def column(row, *names):
    remap={re.sub(r"[^a-z0-9]", "", str(k).lower()):v for k,v in row.items()}
    for n in names:
        key=re.sub(r"[^a-z0-9]", "", n.lower())
        if key in remap:
            return remap[key]
    return None


def exact_members(raw, username):
    if not raw or not username:
        return False
    members = [x.casefold() for x in re.split(r"[,;|/]", str(raw).strip()) if x.strip()]
    return username.casefold() in [s.strip() for s in members]


def read_leaderboard_csv(csv_text, username):
    try:
        rows=list(csv.DictReader(io.StringIO(csv_text)))
    except csv.Error:
        return {"rank": None, "teams": None, "source": "leaderboard_invalid_csv"}
    if not rows:
        return {"rank": None, "teams": 0, "source": "leaderboard_empty"}
    count=len(rows)
    exact=[]
    team_name_only=[]
    for position, row in enumerate(rows, 1):
        rank=valid_rank(column(row, "Rank", "Place", "Position"))
        if rank is None:
            rank=position
        member=column(row, "TeamMembers", "TeamMemberNames", "Usernames", "MemberUsernames")
        if exact_members(member, username):
            exact.append(rank)
        elif str(column(row,"TeamName","Team") or "").casefold() == username.casefold():
            team_name_only.append(rank)
    if len(exact)==1:
        return {"rank":exact[0],"teams":count,"source":"leaderboard_verified_member_username"}
    if team_name_only:
        return {"rank":None,"teams":count,"source":"team_name_match_not_verified",
                "unverified_candidate_rank":team_name_only[0]}
    return {"rank":None,"teams":count,"source":"username_not_located_on_leaderboard"}


def leaderboard_lookup(slug, username):
    """Full raw CSV is read in temp storage and NEVER committed to GitHub."""
    with tempfile.TemporaryDirectory(prefix="kaggle-lb-") as folder:
        p = subprocess.run(
            ["kaggle", "competitions", "leaderboard", slug, "--download",
             "--path", folder, "--quiet"],
            cwd=ROOT, capture_output=True, text=True, timeout=180)
        if p.returncode:
            raise RuntimeError((p.stderr or p.stdout)[-1200:])
        files=sorted(Path(folder).glob("*.csv"))
        if not files:
            raise RuntimeError("No full leaderboard CSV downloaded from Kaggle")
        # Refuse to assume which CSV is the user or private leaderboard.
        public=[f for f in files if "public" in f.name.lower()]
        if not public and len(files)==1 and "private" not in files[0].name.lower():
            public=files
        if len(public)!=1:
            raise RuntimeError("Cannot positively identify exactly one Public leaderboard CSV")
        return read_leaderboard_csv(public[0].read_text(encoding="utf-8-sig"),username)


def best_rank(slug, username):
    warnings=[]
    rows=[]
    for args in (["--group","entered","--page-size","100"],
                 ["--search",slug,"--page-size","100"]):
        try:
            rows.extend(kaggle_json(args))
        except (RuntimeError, ValueError, json.JSONDecodeError) as err:
            warnings.append(str(err)[-350:])
    rank=direct_rank(rows,slug)
    if rank["rank"] is not None:
        return rank,warnings
    try:
        other=leaderboard_lookup(slug,username)
        if other["teams"]:
            rank["teams"]=other["teams"]
        if other["rank"] is not None:
            rank=other
        else:
            rank["source"] += "; "+other["source"]
            if other.get("unverified_candidate_rank"):
                warnings.append("Only same-name team found, membership not authenticated; omitted rank")
    except (OSError, RuntimeError, ValueError, UnicodeError, subprocess.TimeoutExpired) as err:
        warnings.append("Public leaderboard lookup unavailable: "+str(err)[-500:])
    return rank,warnings


def create_snapshot(slug, username, when=None):
    checked=(when or datetime.now(timezone.utc)).isoformat()
    rank,warnings=best_rank(slug,username)
    model=candidates_for(slug)
    return {
        "competition":slug,
        "checked_at":checked,
        "ranking_type":"Public",
        "rank":rank["rank"],
        "rank_verified":rank["rank"] is not None,
        "rank_source":rank["source"],
        "ranked_teams":rank["teams"],
        "score":model,
        "warnings":warnings,
        "leaderboard_url":f"https://www.kaggle.com/competitions/{slug}/leaderboard",
        "note":"Public rank is not final Private leaderboard rank or prize entitlement."
    }


def render(snapshot):
    score=snapshot.get("score") or {}
    rank=snapshot.get("rank")
    teams=snapshot.get("ranked_teams")
    place=(f"第 {rank} 名 / 共 {teams} 支队伍" if rank is not None and teams
           else f"第 {rank} 名（参赛队伍总数未核实）" if rank is not None
           else "官方排名暂未核实")
    return "\n".join([
        "# Kaggle 自动成绩 + 当前排名",
        "",
        f"- 比赛：{snapshot['competition']}",
        f"- 当前最佳 Public LB：{score.get('public_score','暂无')}",
        f"- 提交编号：{score.get('submission_id','暂无')}",
        f"- 最佳模型：{score.get('model','暂无')}",
        f"- **当前 Public 排名：{place}**",
        f"- 已识别排行榜队伍数：{teams if teams is not None else '未知'}",
        f"- 排名数据源：{snapshot['rank_source']}",
        f"- 最近核验 UTC：{snapshot['checked_at']}",
        f"- 官方榜：{snapshot['leaderboard_url']}",
        "",
        "注意：公开榜名次会随着其他人提交而变化。比赛结束后需另行确认 Private 榜最终排名和领奖资格。",
        "找不到可信的账户绑定名次时绝不按分数推算排名。",
        "",
    ])



def score_rank_lines(competition, public_score):
    """Reusable verified-rank lines for GPU and TPU reports.

    A prior ranking is only eligible when tied to the SAME currently scored
    submission. A changed score always requires a fresh Kaggle rank query.
    """
    snapshot=read(STATE)
    model=snapshot.get("score") or {}
    same=snapshot.get("competition")==competition
    try:
        same_score=(public_score not in (None,"") and
                    model.get("public_score") not in (None,"") and
                    abs(float(public_score)-float(model["public_score"])) < 1e-10)
    except (TypeError,ValueError):
        same_score=False
    if not same or not same_score:
        return [
            "- 当前 Public 排名：**未核实（需按最新成绩重新查询）**",
            "- 官方排名查询：reports/score_rank_latest.md",
        ]
    rank=snapshot.get("rank") if snapshot.get("rank_verified") else None
    teams=snapshot.get("ranked_teams")
    label=(f"**第 {rank} 名 / {teams} 支队伍**" if rank and teams
           else f"**第 {rank} 名**" if rank else "**未核实**")
    return [
        "- 当前 Public 排名："+label,
        "- 排名核验 UTC："+str(snapshot.get("checked_at", "未知")),
        "- 排名证据："+str(snapshot.get("rank_source", "未知")),
        "- 官方排行榜："+str(snapshot.get("leaderboard_url", "未知")),
    ]


def main():
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise RuntimeError("Missing Kaggle auth token")
    ladder=read(ROOT / "state" / "dual_ladder.json")
    base=read(ROOT / "state" / "v2_1_status.json")
    slug=ladder.get("active_competition") or base.get("competition")
    if not slug:
        raise RuntimeError("No current competition")
    username=os.environ.get("KAGGLE_USERNAME") or str(base.get("kernel") or "").split("/")[0]
    if not username:
        raise RuntimeError("No verified Kaggle account username")
    snapshot=create_snapshot(slug,username)
    save(STATE,snapshot)
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(render(snapshot),encoding="utf-8")
    print(render(snapshot),flush=True)


if __name__ == "__main__":
    main()
