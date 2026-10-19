"""V2.9.3 Kaggle CPU OOF fusion controller with crash-safe one-shot submission.

Modes:
  --evaluate : authenticate to Kaggle, download OWN prior outputs, validate, score
  --reserve  : mark unique submission attempt for git persistence BEFORE submit
  --submit   : submit at most once after reserved state committed by workflow
  --recover  : read already-submitted Kaggle entry; NEVER call submit

This is a new verified CPU experiment using real Kaggle model predictions.
It does not claim to train a new GPU model.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from v291_ledger import read, write, record_experiment
from v293_blend_math import VALIDATION, N_TEST, MIN_CV_GAIN, evaluate, verify_sources

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "state/v2_9_3_status.json"
LEDGER = ROOT / "state/v2_9_experiments.json"
REPORT = ROOT / "reports/v2_9_3_latest.md"
COMP = "playground-series-s6e10"
PREFIX = "KAS V293 BLEND"
TERMINAL = {"skipped_no_cv_gain", "complete", "submission_unverified",
            "submission_failed", "evaluation_failed", "submission_ambiguous"}
SUBMISSION_DAY_LIMIT = 5


def now():
    return datetime.now(timezone.utc)


def cli(cmd, timeout=240):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(ROOT))
    if p.returncode:
        raise RuntimeError(f"Kaggle CLI failed ({cmd[:3]}): " + (p.stderr or p.stdout)[-1200:])
    return p.stdout


def records():
    content = cli(["kaggle", "competitions", "submissions", COMP,
                   "--page-size", "100", "--format", "json"], timeout=120)
    rows = json.loads(content)
    if isinstance(rows, dict):
        rows = rows.get("submissions", rows.get("items", rows.get("results")))
    if not isinstance(rows, list):
        raise RuntimeError("Kaggle submission listing could not be authenticated / decoded")
    return rows


def matched(rows, note):
    found = [row for row in rows if row.get("description") == note]
    if len(found) > 1:
        raise RuntimeError("Multiple matching submissions found; manual reconciliation")
    return found[0] if found else None


def count_today(rows, day):
    number = 0
    for row in rows:
        value = row.get("date")
        if not value:
            raise RuntimeError("Kaggle date unknown; submission quota cannot be checked")
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        number += (dt.astimezone(timezone.utc).date() == day)
    return number


def allowed_competition():
    state = read(ROOT/"state/dual_ladder.json")
    if state.get("active_competition") != COMP:
        raise RuntimeError("Active competition changed; refuse old-contest experiment")
    deadline = state.get("active_deadline")
    if not deadline:
        raise RuntimeError("Verified competition deadline is missing")
    end = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
    if end.tzinfo is None or now() >= end:
        raise RuntimeError("Competition submission deadline passed / timezone unverified")


def baseline():
    s25 = read(ROOT/"state/v2_5_status.json")
    s28 = read(ROOT/"state/v2_8_status.json")
    if s25.get("stage") != "complete" or not s25.get("submitted"):
        raise RuntimeError("Real V2.5 submission required")
    if s25.get("latest_submission", {}).get("ref") != 57346852:
        raise RuntimeError("Baseline V2.5 submission reference changed; manual verification required")
    lb = float(s25["latest_submission"]["publicScore"])
    cv = max(float(s25["metrics"]["best_cv_auc"]),float(s28["metrics"]["best_cv_auc"]))
    for v in ("v2_6","v2_7"):
        source=read(ROOT/f"state/{v}_status.json")
        m=source.get("metrics") or {}
        if source.get("competition")==COMP and m.get("validation")==VALIDATION:
            score=m.get("best_cv_auc",m.get("blended_cv_auc"))
            if isinstance(score,(int,float)):
                cv=max(cv,float(score))
    return s25,s28,cv,lb


def only(directory, filename):
    hits=list(directory.rglob(filename))
    if len(hits)!=1:
        raise RuntimeError(f"Expected ONE {filename}, found {len(hits)}")
    return hits[0]


def download(handle, location):
    if not isinstance(handle,str) or "/" not in handle or not handle.startswith("mjw3177109/"):
        raise RuntimeError("Kernel handle does not belong to the authorized Kaggle account")
    location.mkdir(parents=True, exist_ok=True)
    cli(["kaggle","kernels","output",handle,"-p",str(location),"-o","-q"],timeout=600)


def trial_entry(result,status):
    name = result.get("candidate") or "v293_no_cv_gain"
    score = result.get("cv_auc")
    if score is None:
        scored = [c for c in result.get("candidates",[]) if "cv_auc" in c]
        score = max((c["cv_auc"] for c in scored),default=None)
    return {"competition":COMP,"candidate":"v293_"+name,
            "validation":VALIDATION,"seed":20261018,
            "parameters":{"sources":["v25","v28"],"candidate":name,
                          "weights":result.get("weights",{}),"min_gain":MIN_CV_GAIN},
            "status":status,"cv":score,
            "gpu_seconds":0,"cpu_seconds":result.get("cpu_seconds"),
            "completed_at":now().isoformat() if status in ("completed","skipped_no_cv_gain") else None,
            "kernel_sources":["mjw3177109/kas-v25-gpu-te-playground-series-s6e10",
                              "mjw3177109/kas-v28-catboost-playground-series-s6e10"]}


def save(state):
    state["updated_at"]=now().isoformat()
    write(STATE,state)
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    m=state.get("result") or {}
    tested=[x for x in m.get("candidates",[]) if isinstance(x.get("cv_auc"),(float,int))]
    best=max(tested,key=lambda x:x["cv_auc"]) if tested else {}
    measured=m.get("cv_auc",best.get("cv_auc"))
    measured_delta=m.get("delta",measured-float(state["benchmark_cv"]) if measured is not None and state.get("benchmark_cv") is not None else None)
    measured_name=m.get("candidate",best.get("candidate"))
    lines=[
        "# V2.9.3｜Kaggle 真实 OOF 模型融合实验","",
        f"- 检查 UTC：{state['updated_at']}",
        f"- 阶段：{state.get('stage')}",
        f"- 竞赛：{COMP}",
        "- 训练来源：本人 V2.5 和 V2.8 Kaggle Notebook 的既有 GPU OOF；本轮不重复 GPU 训练。",
        f"- CV 最强历史基准：{state.get('benchmark_cv')}",
        f"- 本轮最佳 CV：{measured}",
        f"- CV 提升：{measured_delta}",
        f"- 最低提升门槛：{MIN_CV_GAIN}",
        f"- 候选：{measured_name}",
        f"- 是否已发送新的 Kaggle 提交：{state.get('submitted',False)}",
        f"- 新 Public LB：{(state.get('latest_submission') or {}).get('publicScore','未评分/未提交')}",
        f"- 旧 Public LB：{state.get('incumbent_lb')}",
        "- 官方名次需独立通过认证排行追踪器核验，本报告不按分数推测。",
        "- Public 榜前三不等于最终 Private 榜前三或获奖。",
        "",
        "## 预注册候选 CV（相同三折、相同训练行顺序）","",
        "| 方法 | 3-fold OOF AUC | vs 最强 CV | 状态 |","|---|---:|---:|---|"
    ]
    for row in m.get("candidates",[]):
        lines.append(f"| {row.get('candidate')} | {row.get('cv_auc','—')} | {row.get('delta_vs_best','—')} | {row.get('status','evaluated')} |")
    lines.extend(["", f"- 唯一提交标识：{state.get('submission_note','无')}",
                  f"- Kaggle notebook V2.5：https://www.kaggle.com/code/mjw3177109/kas-v25-gpu-te-playground-series-s6e10",
                  f"- Kaggle notebook V2.8：https://www.kaggle.com/code/mjw3177109/kas-v28-catboost-playground-series-s6e10"])
    if state.get("error"):
        lines.extend(["","## 故障与保护",str(state["error"])[:1500]])
    REPORT.write_text("\n".join(lines)+"\n",encoding="utf-8")


def do_evaluate(directory):
    old=read(STATE)
    lacks_fold_diagnostics=(old.get("stage")=="skipped_no_cv_gain" and not any("fold_gain_vs_v28" in x for x in ((old.get("result") or {}).get("candidates") or [])))
    if (old.get("stage") in TERMINAL and not lacks_fold_diagnostics) or old.get("submission_attempted"):
        print("Existing terminal/submission-attempted V2.9.3 state; no repeat evaluation", old.get("stage"))
        return
    allowed_competition()
    s25,s28,benchmark,lb=baseline()
    started=time.monotonic()
    status={"version":"V2.9.3","competition":COMP,"stage":"evaluating",
            "benchmark_cv":benchmark,"incumbent_lb":lb,
            "submitted":False,"submission_attempted":False}
    save(status)
    try:
        path25=directory/"v25";path28=directory/"v28"
        download(s25["kernel"],path25)
        download(s28["kernel"],path28)
        csv25=pd.read_csv(only(path25,"candidate_submission.csv"))
        csv28=pd.read_csv(only(path28,"candidate_submission.csv"))
        with np.load(only(path25,"v25_oof.npz")) as a, np.load(only(path28,"v28_oof.npz")) as b:
            y, oofs, tests=verify_sources(a,b,s25,s28,csv25,csv28)
        result, prediction=evaluate(y,oofs,tests,benchmark)
        result["cpu_seconds"]=round(time.monotonic()-started,3)
        status["result"]=result
        status["stage"]=result["stage"]
        if prediction is not None:
            sub=csv25.copy()
            sub["satisfaction"]=prediction
            if list(sub.columns)!=["id","satisfaction"] or len(sub)!=N_TEST:
                raise RuntimeError("Kaggle submission schema changed")
            location=directory/"v293_submission.csv"
            sub.to_csv(location,index=False)
            status["candidate_file_sha256"]=__import__("hashlib").sha256(location.read_bytes()).hexdigest()
            status["submission_note"]=f"{PREFIX} {result['signature']}"
        save(status)
        ledger_status="planned" if result["stage"]=="candidate_ready" else "skipped_no_cv_gain"
        record_experiment(LEDGER,trial_entry(result,ledger_status))
        print(json.dumps({"stage":status["stage"],"result":result},ensure_ascii=False,default=str))
    except Exception as exc:
        status.update({"stage":"evaluation_failed","error":str(exc)[-1400:]})
        save(status)
        raise


def do_reserve(directory):
    status=read(STATE)
    if status.get("stage")!="candidate_ready" or status.get("submission_attempted"):
        raise RuntimeError("No fresh CV-qualified candidate to reserve")
    allowed_competition()
    _,_,benchmark,lb=baseline()
    result=status["result"]
    if result.get("cv_auc",0)<benchmark+MIN_CV_GAIN:
        raise RuntimeError("Recheck of CV gate failed")
    sub=directory/"v293_submission.csv"
    if not sub.is_file() or __import__("hashlib").sha256(sub.read_bytes()).hexdigest()!=status.get("candidate_file_sha256"):
        raise RuntimeError("Submission file is missing or differs from CV-verified candidate")
    rows=records()  # Must establish authenticated Kaggle quota before reserving.
    if matched(rows,status["submission_note"]):
        raise RuntimeError("Exact candidate already exists on Kaggle; use recovery")
    if count_today(rows,now().date())>=SUBMISSION_DAY_LIMIT:
        status["stage"]="quota_wait";save(status)
        raise RuntimeError("Kaggle daily submission limit reached")
    status["stage"]="submission_reserved"
    status["submission_attempted"]=True
    status["reserved_at"]=now().isoformat()
    save(status)
    print("Reserved exactly one submission attempt; MUST commit this state to GitHub before --submit.")


def finish(status,listing):
    status["latest_submission"]=listing
    status["submitted"]=True
    score=listing.get("publicScore")
    status["stage"]="submission_pending" if score in (None,"") else "complete"
    if score not in (None,""):
        status["public_lb_delta"]=round(float(score)-float(status["incumbent_lb"]),8)
        status["champion"]="V2.9.3 "+status["result"]["candidate"] if float(score)>float(status["incumbent_lb"]) else "V2.5 blend_rank"
    save(status)
    result=status["result"]
    payload=trial_entry(result,"completed" if status["stage"]=="complete" else "running")
    if listing.get("ref") is not None:
        payload["submission_id"]=listing["ref"]
    if score not in (None,""):
        payload["public_lb"]=score
    record_experiment(LEDGER,payload)


def do_submit(directory,recover_only=False):
    status=read(STATE)
    if not status.get("submission_note") or not status.get("submission_attempted"):
        raise RuntimeError("No persisted reservation; refusing unreserved submission")
    allowed_competition()
    rows=records()
    found=matched(rows,status["submission_note"])
    if found:
        finish(status,found)
        return
    if recover_only:
        print("No matching Kaggle submission found; recovery does not submit")
        return
    if status.get("stage")!="submission_reserved":
        raise RuntimeError("Ambiguous prior submission attempt; refuse retry")
    # A separate GitHub commit must have persisted this reservation before calling here.
    if os.environ.get("V293_RESERVED_COMMITTED")!="true":
        raise RuntimeError("Uncommitted reservation would risk duplicate submission")
    file=directory/"v293_submission.csv"
    if not file.is_file() or __import__("hashlib").sha256(file.read_bytes()).hexdigest()!=status.get("candidate_file_sha256"):
        raise RuntimeError("CV-qualified submission was modified")
    if count_today(rows,now().date())>=SUBMISSION_DAY_LIMIT:
        status["stage"]="quota_wait";save(status)
        return
    status["stage"]="submitting"
    save(status)
    try:
        # This call happens only once per persisted reservation, even after a crash.
        cli(["kaggle","competitions","submit",COMP,"-f",str(file),"-m",
             status["submission_note"],"--wait","600","--poll-interval","30"],timeout=900)
        found=matched(records(),status["submission_note"])
        if not found:
            status["stage"]="submission_unverified"
            status["error"]="CLI returned without authenticated matching submission"
            save(status)
            return
        finish(status,found)
    except Exception as exc:
        status["stage"]="submission_ambiguous"
        status["error"]=str(exc)[-1400:]+". Do NOT automatically retry this submission."
        save(status)
        raise


def main():
    cli_parser=argparse.ArgumentParser()
    flags=cli_parser.add_mutually_exclusive_group(required=True)
    flags.add_argument("--evaluate",action="store_true")
    flags.add_argument("--reserve",action="store_true")
    flags.add_argument("--submit",action="store_true")
    flags.add_argument("--recover",action="store_true")
    cli_parser.add_argument("--workdir",default=os.getenv("RUNNER_TEMP") or tempfile.gettempdir())
    args=cli_parser.parse_args()
    directory=Path(args.workdir)/"kas_v293"
    directory.mkdir(parents=True,exist_ok=True)
    if args.evaluate:do_evaluate(directory)
    elif args.reserve:do_reserve(directory)
    elif args.submit:do_submit(directory)
    else:do_submit(directory,recover_only=True)

if __name__=="__main__":
    main()
