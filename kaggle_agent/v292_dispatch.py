"""V2.9.2 bounded fallback dispatcher. Safe default is dry-run.
Only preapproved, existing competition-specific workflows can be dispatched.
"""
import argparse
import json
import os
import subprocess
from datetime import datetime,timezone
from pathlib import Path
from v291_ledger import read,write,queue_age,daily_metrics,record_experiment

ROOT=Path(__file__).resolve().parents[1]
LEDGER=ROOT/"state/v2_9_experiments.json"
CONTROL=ROOT/"state/v2_9_dispatch_control.json"
LOCK=ROOT/"state/v2_9_dispatch_lock.json"
QUEUE=ROOT/"state/v2_9_queue_memory.json"
ALLOWLIST={"gpu":"kaggle_v28_catboost.yml","cpu":"kaggle_v26_gpu.yml"}
MAX_PER_DAY=1
MAX_ATTEMPTS=2
MAX_GPU_HOURS=2.0
VALIDATION="StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)"

def classify_failure(row):
    text=str(row.get("error") or row.get("failure_reason") or "").lower()
    if "out of memory" in text or "cuda oom" in text:
        return "memory"
    if "quota" in text or "resource" in text or "queued" in text:
        return "capacity"
    if "leak" in text or "validation" in text:
        return "validation"
    if "timeout" in text:
        return "timeout"
    if row.get("status")=="skipped_no_cv_gain":
        return "no_gain"
    return "other"

def choose(history,candidates):
    """Prioritize unexplored methods, penalize recurring failures and weak CV."""
    scored=[]
    for item in candidates:
        name=item["candidate"]
        prior=[x for x in history if x.get("candidate")==name]
        if any(x.get("status") in ("completed","skipped_no_cv_gain") for x in prior):
            continue
        failures=[classify_failure(x) for x in prior if x.get("status")=="failed"]
        if len(failures)>=MAX_ATTEMPTS or "validation" in failures:
            continue
        kind=item.get("kind","feature")
        # Encourage diverse methods rather than fixed ordering.
        novelty=5 if not prior else 1
        diversity={"model":3,"feature":2,"blend":1}.get(kind,0)
        penalty=3*len(failures)+2*failures.count("memory")
        scored.append((novelty+diversity-penalty,name,item))
    return max(scored,key=lambda x:(x[0],x[1]))[2] if scored else None

def eligible(status,now,control,ledger):
    day=now.date().isoformat()
    if status.get("active_competition")!="playground-series-s6e10":
        return False,"competition_not_allowlisted"
    deadline=str(status.get("active_deadline") or "")
    if deadline:
        try:
            if now>=datetime.fromisoformat(deadline.replace("Z","+00:00")):
                return False,"deadline_passed"
        except ValueError:
            return False,"deadline_invalid"
    else:
        return False,"deadline_unverified"
    if control.get("date_utc")==day and control.get("dispatches",0)>=MAX_PER_DAY:
        return False,"daily_dispatch_budget"
    used=daily_metrics(LEDGER,day,"playground-series-s6e10")
    if used["measured_gpu_hours"]>=MAX_GPU_HOURS:
        return False,"gpu_budget"
    if any(x.get("status") in ("planned","running") for x in ledger if x.get("competition")=="playground-series-s6e10"):
        return False,"unfinished_experiment"
    return True,"ok"

def run(now=None,dispatch=False):
    now=now or datetime.now(timezone.utc)
    ladder=read(ROOT/"state/dual_ladder.json")
    tpu=read(ROOT/"state/v2_3_status.json")
    q=queue_age(QUEUE,tpu.get("kernel"),tpu.get("stage")=="kernel_queued",now)
    history=read(LEDGER).get("experiments",[])
    choices=[
      {"kind":"model","candidate":"catboost_gpu_recovery","backend":"gpu"},
      {"kind":"feature","candidate":"frequency_feature_recovery","backend":"cpu"},
    ]
    selected=choose(history,choices)
    control=read(CONTROL)
    ok,reason=eligible(ladder,now,control,history)
    if not q["tripped"]:
        ok,reason=False,"tpu_breaker_not_tripped"
    if not selected:
        ok,reason=False,"no_unused_approved_candidate"
    workflow=ALLOWLIST.get(selected["backend"]) if selected else None
    # These are recovery-only workflows; they must never push a fresh Kaggle kernel.
    result={"at":now.isoformat(),"queue":q,"selected":selected,"workflow":workflow,
            "eligible":ok,"reason":reason,"dispatched":False,"dry_run":not dispatch}
    if not dispatch or not ok:
        return result
    if os.getenv("V29_ALLOW_DISPATCH")!="true":
        result["reason"]="explicit_dispatch_flag_missing"
        return result
    # Workflow-dispatch requests need a scoped token. Never echo its value.
    token=os.getenv("GH_TOKEN")
    if not token:
        result["reason"]="github_token_missing"
        return result
    # A lock is local to this job; GitHub concurrency and persisted control
    # provide additional guards against duplicate dispatches.
    if LOCK.exists():
        result["reason"]="dispatch_lock_exists"
        return result
    write(LOCK,{"at":now.isoformat(),"candidate":selected["candidate"]})
    try:
        command=["gh","workflow","run",workflow,"--repo","mjw3177109/kaggletest",
                 "--ref","main","-f","mode=recover_existing"]
        proc=subprocess.run(command,env={**os.environ,"GH_TOKEN":token},capture_output=True,text=True,timeout=60)
        if proc.returncode:
            result["reason"]="workflow_dispatch_failed"
            return result
        control={"date_utc":now.date().isoformat(),
                 "dispatches":(control.get("dispatches",0)+1 if control.get("date_utc")==now.date().isoformat() else 1),
                 "last_workflow":workflow,"last_at":now.isoformat()}
        write(CONTROL,control)
        result["dispatched"]=True
        result["reason"]="recovery_workflow_dispatched_not_training_verified"
        return result
    finally:
        LOCK.unlink(missing_ok=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dispatch",action="store_true")
    args=p.parse_args()
    result=run(dispatch=args.dispatch)
    write(ROOT/"state/v2_9_dispatch_latest.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
