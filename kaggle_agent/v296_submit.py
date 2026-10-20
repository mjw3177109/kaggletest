"""V2.9.6 verified RealMLP submission: persisted one-shot reservation.

Use only after complete 3-fold candidate passed gate. No Kaggle submission
is triggered by this file on import; explicit --reserve and --submit required.
"""
import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime,timezone
from pathlib import Path

from v291_ledger import read,write

ROOT=Path(__file__).resolve().parents[1]
STATUS=ROOT/"state/v2_9_6_status.json"
LEDGER=ROOT/"state/v2_9_experiments.json"
SLUG="playground-series-s6e10"
SUBMISSION_LIMIT=5


def now():
    return datetime.now(timezone.utc)


def competition_open():
    ladder=read(ROOT/"state/dual_ladder.json")
    deadline=ladder.get("active_deadline")
    if not deadline or ladder.get("active_competition")!=SLUG:
        raise RuntimeError("Competition not active or deadline missing")
    cutoff=datetime.fromisoformat(deadline.replace("Z","+00:00"))
    if cutoff.tzinfo is None or now()>=cutoff:
        raise RuntimeError("Contest deadline passed")


def cli(args,timeout=120):
    out=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                       text=True,timeout=timeout,cwd=str(ROOT))
    if out.returncode:
        raise RuntimeError("Kaggle CLI failed: "+(out.stderr or out.stdout)[-1100:])
    return out.stdout


def history():
    data=json.loads(cli(["kaggle","competitions","submissions",SLUG,
                         "--page-size","100","--format","json"]))
    if isinstance(data,dict):
        data=data.get("submissions",data.get("items",data.get("results",[])))
    if not isinstance(data,list):
        raise RuntimeError("Cannot verify authenticated Kaggle submissions")
    return data


def matching(rows,note):
    matches=[r for r in rows if r.get("description")==note]
    if len(matches)>1:
        raise RuntimeError("Duplicate submission note; cannot reconcile")
    return matches[0] if matches else None


def quota(rows):
    day=now().date()
    used=0
    for row in rows:
        stamp=row.get("date")
        if not stamp:
            raise RuntimeError("Submission date missing, quota unverified")
        dt=datetime.fromisoformat(str(stamp).replace("Z","+00:00"))
        if dt.tzinfo is None:
            dt=dt.replace(tzinfo=timezone.utc)
        used+=dt.astimezone(timezone.utc).date()==day
    return used < SUBMISSION_LIMIT


def candidate_file(status):
    file=ROOT/"competitions"/SLUG/"results_v296"/"v296_submission.csv"
    if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=status.get("candidate_file_sha256"):
        raise RuntimeError("CV-verified prediction CSV missing or tampered")
    return file


def verify_gate(status):
    competition_open()
    comp=status.get("comparison") or {}
    weights=comp.get("pre_registered_weights")
    if (status.get("stage")!="candidate_ready_unsubmitted"
        or status.get("submitted") or status.get("submission_attempted")
        or weights!={"v293":.8,"realmlp":.2}
        or float(comp.get("delta",0))<0.00006
        or int(comp.get("positive_folds",0))<2
        or min(comp.get("fold_gain") or [-1])<=-0.00005
        or not (status.get("hardware") or {}).get("ok")
        or not (status.get("metrics") or {}).get("gpu_verified")):
        raise RuntimeError("RealMLP CV gate is not approved")


def reserve():
    s=read(STATUS)
    verify_gate(s)
    candidate_file(s)
    rows=history()
    if matching(rows,s["submission_note"]):
        raise RuntimeError("Candidate already submitted; use --recover")
    if not quota(rows):
        raise RuntimeError("Kaggle daily submission budget exhausted")
    s["stage"]="submission_reserved"
    s["submission_attempted"]=True
    s["reserved_at"]=now().isoformat()
    write(STATUS,s)
    print("Reserved uniquely. STOP until GitHub has durably committed this state.")


def annotate(s,submission):
    ledger=read(LEDGER)
    rows=ledger.get("experiments") or []
    for row in rows:
        if row.get("candidate")!="v296_realmlp_full_threefold":
            continue
        existing=row.get("submission_id")
        new_id=submission.get("ref")
        if new_id is not None:
            if existing is not None and str(existing)!=str(new_id):
                raise RuntimeError("Ledger already contains a different submission ref")
            row["submission_id"]=new_id
        score=submission.get("publicScore")
        if score not in (None,""):
            row["public_lb"]=score
        write(LEDGER,{"experiments":rows})
        return
    raise RuntimeError("Missing independent experiment ledger record")


def finish(s,entry):
    s["latest_submission"]=entry
    s["submitted"]=True
    score=entry.get("publicScore")
    s["stage"]="submission_pending" if score in (None,"") else "complete"
    if score not in (None,""):
        s["public_lb_delta"]=round(float(score)-float(s.get("incumbent_lb",.96035)),8)
        s["public_champion"]="V2.9.6 RealMLP" if float(score)>float(s.get("incumbent_lb",.96035)) else "V2.5"
    write(STATUS,s)
    annotate(s,entry)


def submit(recover_only=False):
    s=read(STATUS)
    if not s.get("submission_note") or not s.get("submission_attempted"):
        raise RuntimeError("No previous persisted submission reservation")
    competition_open()
    records=history()
    item=matching(records,s["submission_note"])
    if item:
        finish(s,item)
        return
    if recover_only:
        print("No known scored submission yet; recovery is read-only.")
        return
    if s.get("stage")!="submission_reserved":
        raise RuntimeError("Submission state is ambiguous; refuse a second attempt")
    if os.getenv("V296_RESERVED_COMMITTED")!="true":
        raise RuntimeError("Reservation has not been durably committed")
    file=candidate_file(s)
    if not quota(records):
        raise RuntimeError("Daily submission quota unavailable")
    s["stage"]="submitting"
    write(STATUS,s)
    try:
        cli(["kaggle","competitions","submit",SLUG,
             "-f",str(file),"-m",s["submission_note"],
             "--wait","600","--poll-interval","30"],timeout=900)
        verified=matching(history(),s["submission_note"])
        if verified is None:
            s.update({"stage":"submission_unverified","error":"Cannot authenticate matching result"})
            write(STATUS,s)
            return
        finish(s,verified)
    except Exception as exc:
        s.update({"stage":"submission_ambiguous",
                  "error":str(exc)[-1300:]+". DO NOT AUTO-RETRY"})
        write(STATUS,s)
        raise


def main():
    p=argparse.ArgumentParser()
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--reserve",action="store_true")
    group.add_argument("--submit",action="store_true")
    group.add_argument("--recover",action="store_true")
    args=p.parse_args()
    if args.reserve:reserve()
    elif args.submit:submit()
    else:submit(recover_only=True)


if __name__=="__main__":
    main()
