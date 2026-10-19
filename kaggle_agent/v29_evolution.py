"""V2.9 conservative experiment planner, TPU queue circuit breaker, and efficiency ledger.
No Kaggle push, training, or submission happens here. Only verified experiments
are eligible for promotion by an existing competition-specific controller.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/"state"
REPORTS=ROOT/"reports"
QUEUE_HOURS=6
CANDIDATES=[
 ("feature","cross_fitted_segment_interactions"),
 ("feature","fold_safe_route_frequency"),
 ("model","extra_trees_diversity"),
 ("model","hist_gradient_boosting"),
 ("blend","oof_weight_search_20_80"),
 ("blend","rank_blend_30_70"),
]
FINISHED={"complete","skipped_no_cv_gain","kernel_failed","artifacts_missing","gpu_unavailable"}

def read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError,ValueError):
        return {}

def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def stamp(value):
    try:
        dt=datetime.fromisoformat(str(value).replace("Z","+00:00"))
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    except (ValueError,TypeError):
        return None

def evaluate(now=None):
    now=now or datetime.now(timezone.utc)
    states={v:read(STATE/f"{v}_status.json") for v in ("v2_3","v2_5","v2_6","v2_7","v2_8")}
    ladder=read(STATE/"dual_ladder.json")
    competition=ladder.get("active_competition","playground-series-s6e10")
    tpu=states["v2_3"]
    queued=tpu.get("stage")=="kernel_queued" and tpu.get("competition")==competition
    first=stamp(tpu.get("queued_since") or tpu.get("created_at"))
    # updated_at reflects polling, not queue entry; do not use it to claim age.
    age=(now-first).total_seconds()/3600 if queued and first else None
    tripped=bool(queued and age is not None and age>=QUEUE_HOURS)
    fallback="gpu_or_cpu_candidate_planning" if tripped else "none"
    # Fail closed when the queue start is unknown; surface operator action.
    queue_state="timed_out" if tripped else ("queued_age_unknown" if queued and age is None else ("queued" if queued else tpu.get("stage","unknown")))
    seen=set()
    records=[]
    for version,s in states.items():
        metrics=s.get("metrics") or {}
        cv=metrics.get("best_cv_auc",metrics.get("blended_cv_auc"))
        signature=hashlib.sha256(json.dumps({"version":version,"model":metrics.get("best_model"),"cv":cv,"validation":metrics.get("validation")},sort_keys=True).encode()).hexdigest()[:16]
        if cv is not None and s.get("competition")==competition:
            seen.add(signature)
            records.append({"version":version,"cv":float(cv),"signature":signature,"submitted":bool(s.get("submitted")),"gpu_seconds":metrics.get("elapsed_seconds") if (s.get("hardware") or {}).get("ok") else None})
    used=set()
    history=read(STATE/"v2_9_experiments.json")
    for x in history.get("experiments",[]):
        if x.get("competition")==competition:
            used.add(x.get("candidate"))
    choice=next(((kind,name) for kind,name in CANDIDATES if name not in used),None)
    proposal={"kind":choice[0],"candidate":choice[1]} if choice else None
    # Do not conflate model CV across different validation protocols.
    baseline=states["v2_5"].get("metrics",{}).get("best_cv_auc")
    comparable=[r["cv"] for r in records if r["version"] in ("v2_5","v2_6","v2_7","v2_8")]
    best=max(comparable) if comparable else None
    day=now.date().isoformat()
    today=[r for r in records if (stamp(states[r["version"]].get("updated_at")) or datetime.min.replace(tzinfo=timezone.utc)).date().isoformat()==day]
    result={
      "checked_at":now.isoformat(),"competition":competition,
      "mode":"proposal_only_no_training_or_submission",
      "next_experiment":proposal,
      "tpu":{"stage":queue_state,"queued_hours":round(age,2) if age is not None else None,"breaker_tripped":tripped,"fallback":fallback,"requires_queued_since":queued and age is None},
      "efficiency":{"date_utc":day,"new_distinct_completed_experiments":len(today),"known_distinct_experiments":len(seen),
        "best_known_cv":best,"cv_delta_vs_v25":round(best-float(baseline),10) if best is not None and baseline is not None else None,
        "measured_gpu_hours":round(sum(float(x["gpu_seconds"]) for x in today if x["gpu_seconds"] is not None)/3600,4),
        "gpu_usage_coverage":"partial_only_reported_elapsed_seconds","new_scored_submissions":sum(bool(x["submitted"]) for x in today)},
      "safety":{"no_auto_rules_acceptance":True,"no_duplicate_kernel_push":True,"no_automatic_submission":True,"cross_validation_comparability_required":True}
    }
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args()
    result=evaluate()
    if not args.dry_run:
        write(STATE/"v2_9_evolution.json",result)
        REPORTS.mkdir(exist_ok=True)
        (REPORTS/"v2_9_evolution_latest.md").write_text(
            "# V2.9 自动精进监控（安全提案模式）\n\n"
            "注意：此流程只生成下一实验提案和资源熔断建议，不会伪称已经训练或提交。\n\n"
            +"\x60\x60\x60json\n"+json.dumps(result,ensure_ascii=False,indent=2)+"\n\x60\x60\x60\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
