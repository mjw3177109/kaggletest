"""V2.9.1 persistent event ledger and queue-age tracking; no remote execution."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

def read(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError,ValueError):
        return {}

def write(path,obj):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def identity(competition, candidate, validation, seed, parameters):
    payload=[competition,candidate,validation,seed,parameters]
    return hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def record_experiment(path, entry):
    """Idempotent: preserve initial outcome; a conflicting replay is rejected."""
    required=("competition","candidate","validation","seed","parameters","status")
    if any(k not in entry for k in required):
        raise ValueError("Incomplete experiment provenance")
    if entry["status"] not in ("planned","running","completed","failed","skipped_no_cv_gain"):
        raise ValueError("Invalid experiment state")
    key=identity(*(entry[k] for k in required[:5]))
    ledger=read(path)
    rows=ledger.get("experiments",[])
    for old in rows:
        if old.get("experiment_id")==key:
            if old.get("status") in ("completed","failed","skipped_no_cv_gain"):
                return old,False
            # Allow state progression without changing experiment identity.
            old.update({k:v for k,v in entry.items() if k not in required[:5]})
            write(path,{"experiments":rows})
            return old,True
    item=dict(entry,experiment_id=key,created_at=datetime.now(timezone.utc).isoformat())
    rows.append(item)
    write(path,{"experiments":rows})
    return item,True

def queue_age(path, kernel, is_queued, now=None, threshold_hours=6):
    now=now or datetime.now(timezone.utc)
    previous=read(path)
    if not is_queued or not kernel:
        write(path,{"kernel":kernel,"queued":False,"last_observed_at":now.isoformat()})
        return {"queued":False,"hours":0,"tripped":False}
    first=previous.get("first_seen_at") if previous.get("kernel")==kernel and previous.get("queued") else None
    try:
        start=datetime.fromisoformat(first) if first else now
        if start.tzinfo is None:
            start=start.replace(tzinfo=timezone.utc)
    except ValueError:
        start=now
    hours=max(0,(now-start).total_seconds()/3600)
    write(path,{"kernel":kernel,"queued":True,"first_seen_at":start.isoformat(),"last_observed_at":now.isoformat()})
    return {"queued":True,"hours":round(hours,3),"tripped":hours>=threshold_hours,
            "fallback":"recommend_cpu_or_gpu_experiment" if hours>=threshold_hours else None}

def daily_metrics(path,day,competition):
    rows=read(path).get("experiments",[])
    today=[x for x in rows if x.get("competition")==competition and
           str(x.get("completed_at",""))[:10]==day and x.get("status") in ("completed","skipped_no_cv_gain")]
    unique={x["experiment_id"]:x for x in today if x.get("experiment_id")}
    measured=[float(x["gpu_seconds"]) for x in unique.values() if isinstance(x.get("gpu_seconds"),(int,float)) and x["gpu_seconds"]>=0]
    cvs=[float(x["cv"]) for x in unique.values() if isinstance(x.get("cv"),(int,float))]
    return {"date_utc":day,"new_independent_experiments":len(unique),
            "best_cv_today":max(cvs) if cvs else None,
            "measured_gpu_hours":round(sum(measured)/3600,4),
            "gpu_usage_coverage":"partial" if len(measured)<len(unique) else "reported_all",
            "new_submission_count":len({str(x["submission_id"]) for x in unique.values() if x.get("submission_id") is not None})}
