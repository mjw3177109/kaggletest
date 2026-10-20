"""V2.9.9 official Kaggle submission: archival reconstruction + one-shot reservation.

Phases are explicit: --reserve (read-only Kaggle + create local predictions),
git commit/push reserved status, --submit (one write), --recover (read-only).
A repeated GitHub job must NEVER send another submission.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import v298_fusion as base
import v299_teacher_fusion as teacher
import v296_submit as protection
from v291_ledger import read,write
from v293_blend_math import verify_sources,N_TEST,VALIDATION

ROOT=base.ROOT
SLUG=base.SLUG
CV=ROOT/"state/v2_9_9_status.json"
STATE=ROOT/"state/v2_9_9_submission.json"
LEDGER=ROOT/"state/v2_9_experiments.json"
REPORT=ROOT/"reports/v2_9_9_submission.md"
OUTPUT=ROOT/"competitions"/SLUG/"results_v299"/"v299_submission.csv"
EXPECTED_AUC=.9610561936366048
PREVIOUS_OFFICIAL=.9609541991872741
NOTE_PREFIX="KAS V299 FIXED4 "

def now():
    return datetime.now(timezone.utc).isoformat()

def write_report(data):
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text("# V2.9.9 Official Kaggle Submission\n\n"+
       "- UTC: "+str(data.get("updated_at",""))+"\n"+
       "- Stage: "+str(data.get("stage",""))+"\n"+
       "- Fixed four-model CV: "+str(data.get("cv_auc",EXPECTED_AUC))+"\n"+
       "- Candidate test hash: "+str(data.get("prediction_hash",""))+"\n"+
       "- Submission note: "+str(data.get("submission_note",""))+"\n"+
       "- Submission ID: "+str((data.get("submission") or {}).get("ref","not yet available"))+"\n"+
       "- Scored Public LB: "+str((data.get("submission") or {}).get("publicScore","not yet available"))+"\n"+
       "- Never retry an ambiguous submission. CV is not Public LB.\n",encoding="utf-8")

def persist(record):
    record["updated_at"]=now()
    write(STATE,record)
    write_report(record)

def verified_cv():
    base.check_competition()
    status=read(CV)
    official=read(ROOT/"state/v2_9_6_status.json")
    if (status.get("competition")!=SLUG
       or status.get("validation")!=VALIDATION
       or status.get("stage")!="candidate_ready_unsubmitted"
       or status.get("submitted") is not False
       or status.get("weights")!=teacher.WEIGHTS
       or abs(float(status.get("candidate_cv",0))-EXPECTED_AUC)>0.000002
       or abs(float(status.get("incumbent_v296_cv",0))-PREVIOUS_OFFICIAL)>0.000002
       or float(status.get("gain",0))<.00010
       or int(status.get("positive_folds",0))<3
       or len(status.get("fold_gain") or [])!=3
       or min(status["fold_gain"])<=-.00005
       or len(str(status.get("candidate_test_hash","")))!=64
       or len(status.get("source_artifacts") or {})!=5
       or official.get("stage")!="complete" or official.get("submitted") is not True
       or str((official.get("latest_submission") or {}).get("ref"))!="57408736"
       or float((official.get("latest_submission") or {}).get("publicScore",0))!=.96059):
       raise RuntimeError("Strict V299 full threefold/source/official LB safety gate failed")
    # Do not submit an older candidate if a substantially stronger and eligible
    # pre-registered full-OOF challenger has overtaken it.
    v2912=read(ROOT/"state/v2_9_12_status.json")
    newer=v2912.get("comparison") or {}
    if (v2912.get("stage")=="candidate_ready_unsubmitted"
        and float(newer.get("candidate_auc",0))>=EXPECTED_AUC+.00006):
        raise RuntimeError("Newer eligible V2912 CV winner exists: do not submit V299")
    return status

def compare_artifacts(actual,expected):
    if set(actual)!=set(expected):
        raise RuntimeError("Archived input provenance set changed")
    for key in expected:
        for field,value in expected[key].items():
            if actual[key].get(field)!=value:
                raise RuntimeError("Archived source SHA/Notebook mismatch: "+key+" "+field)

def prepare_file():
    registered=verified_cv()
    names={"v25":"state/v2_5_status.json","v28":"state/v2_8_status.json",
           "v296":"state/v2_9_6_status.json","v297":"state/v2_9_7_status.json"}
    states={k:read(ROOT/p) for k,p in names.items()}
    base.check_provenance(states)
    with tempfile.TemporaryDirectory(prefix="kas_v299_submit_") as scratch:
        train,test,archive,artifacts=base.download_inputs(scratch,states)
        train_id,test_id,y=base.validate_official(train,test)
        base.evaluate(train,test,archive,states,audit_legacy_order=True)
        tutor,tutor_test,tutor_info=teacher.get_teacher(
              scratch,read(ROOT/"state/v2_9_4_status.json"),train_id,test_id,y)
        artifacts["v294"]=tutor_info
        compare_artifacts(artifacts,registered["source_artifacts"])
        _,oofs,tests=verify_sources(archive["v25"],archive["v28"],
            states["v25"],states["v28"],archive["v25_submission"],
            archive["v28_submission"])
        hist=.35*base.rank(oofs["v25"])+.65*base.rank(oofs["v28"])
        real=base.align(archive["v296"]["oof_realmlp"],
           archive["v296"]["train_id"],train_id,"RealMLP")
        cond=base.align(archive["v297"]["oof_conditional"],
           archive["v297"]["train_id"],train_id,"Conditional")
        checked=teacher.fuse_fixed(y,hist,real,cond,tutor,base.EXPECTED_V296)
        if checked["stage"]!="candidate_ready_unsubmitted":
            raise RuntimeError("Replayed four-model threefold gate no longer passes")
        for fld in ("candidate_cv","gain","incumbent_v296_cv"):
            if abs(checked[fld]-float(registered[fld]))>0.000002:
                raise RuntimeError("Replayed CV differs from preregistered result: "+fld)
        for got,expected in zip(checked["fold_gain"],registered["fold_gain"]):
            if abs(got-float(expected))>0.000002:
                raise RuntimeError("Replayed fold-level gain changed")
        pred_v25=.35*tests["v25"]+.65*tests["v28"]
        pred_real=base.align(archive["v296"]["pred_realmlp"],
                    archive["v296"]["test_id"],test_id,"RealMLP test")
        pred_cond=base.align(archive["v297"]["test_conditional"],
                    archive["v297"]["test_id"],test_id,"Conditional test")
        predictions=(.60*base.rank(pred_v25)+.20*base.rank(pred_real)
                    +.15*base.rank(pred_cond)+.05*base.rank(tutor_test))
        if len(predictions)!=N_TEST or not np.isfinite(predictions).all():
            raise RuntimeError("Incomplete/nonfinite candidate test prediction")
        digest=hashlib.sha256(predictions.astype("<f8").tobytes()).hexdigest()
        if digest!=registered["candidate_test_hash"]:
            raise RuntimeError("Exact preregistered V299 candidate-test hash mismatch")
        result=pd.DataFrame({"id":test_id,"satisfaction":predictions})
        if result["id"].duplicated().any() or not result["satisfaction"].between(0,1).all():
            raise RuntimeError("Submission schema/ID/probability validation failed")
        OUTPUT.parent.mkdir(parents=True,exist_ok=True)
        result.to_csv(OUTPUT,index=False,float_format="%.17g")
        reread=pd.read_csv(OUTPUT)
        if (list(reread.columns)!=["id","satisfaction"] or len(reread)!=N_TEST
            or not np.array_equal(reread["id"].to_numpy(),test_id)
            or not np.allclose(reread["satisfaction"].to_numpy(),predictions,atol=1e-15,rtol=0)):
            raise RuntimeError("Roundtrip candidate CSV lost precision or ID order")
    return digest,hashlib.sha256(OUTPUT.read_bytes()).hexdigest()

def candidate(record):
    if not OUTPUT.is_file():
        raise RuntimeError("No locally generated candidate file; cannot submit")
    expected=record.get("candidate_file_sha256")
    if hashlib.sha256(OUTPUT.read_bytes()).hexdigest()!=expected:
        raise RuntimeError("Candidate CSV does not match durably reserved SHA")
    return OUTPUT

def note_hash(digest):
    return NOTE_PREFIX+digest[:16]

def authenticated_history():
    # Reuse proven competition-specific authentication/quota routine.
    records=protection.history()
    if not isinstance(records,list):
        raise RuntimeError("Cannot authenticate Kaggle submission history")
    return records

def reserve():
    if read(STATE):
        raise RuntimeError("Existing submission state already present: recover only, no second reservation")
    digest,file_sha=prepare_file()
    rows=authenticated_history()
    note=note_hash(digest)
    if protection.matching(rows,note) is not None:
        raise RuntimeError("Identical V299 prediction note already exists on Kaggle")
    if not protection.quota(rows):
        raise RuntimeError("Account Kaggle daily submission limit exhausted")
    data={"competition":SLUG,"stage":"submission_reserved",
          "submission_attempted":True,"submitted":False,
          "reserved_at":now(),"submission_note":note,
          "prediction_hash":digest,"candidate_file_sha256":file_sha,
          "cv_auc":EXPECTED_AUC,"verified_folds":3,"submission":None}
    persist(data)
    print("Reserved V299 one-shot with verified candidate and Kaggle quota. COMMIT before submit.")

def record_entry(s,entry):
    s["submission"]=entry
    s["submitted"]=True
    score=entry.get("publicScore")
    s["stage"]="submission_pending" if score in (None,"") else "complete"
    if score not in (None,""):
        s["official_lb"]=float(score)
        s["official_lb_delta_vs_v296"]=float(score)-.96059
    persist(s)
    ledger=read(LEDGER)
    found=False
    for row in ledger.get("experiments",[]):
        if row.get("candidate")!="v299_archived_teacher_fixed_four_model":
            continue
        old_id=row.get("submission_id")
        if old_id is not None and str(old_id)!=str(entry.get("ref")):
            raise RuntimeError("V299 ledger has different committed submission ID")
        row["submission_id"]=entry.get("ref")
        if score not in (None,""):
            row["public_lb"]=score
        found=True
        break
    if not found:
        raise RuntimeError("V299 archived experiment missing from independent ledger")
    write(LEDGER,ledger)

def submit():
    s=read(STATE)
    if s.get("stage")!="submission_reserved" or not s.get("submission_attempted"):
        raise RuntimeError("Missing exact one-shot V299 reservation; never resubmit")
    if os.environ.get("V299_RESERVED_COMMITTED")!="true":
        raise RuntimeError("GitHub has not durably committed V299 reservation")
    verified_cv()
    f=candidate(s)
    rows=authenticated_history()
    existing=protection.matching(rows,s["submission_note"])
    if existing is not None:
        record_entry(s,existing)
        return
    if not protection.quota(rows):
        raise RuntimeError("Daily Kaggle submission quota exhausted after reservation")
    # Commit-before-attempt is handled by GitHub workflow. From here retry is FORBIDDEN.
    s["stage"]="submitting"
    persist(s)
    try:
        protection.cli(["kaggle","competitions","submit",SLUG,
              "-f",str(f),"-m",s["submission_note"],
              "--wait","600","--poll-interval","30"],timeout=900)
        entry=protection.matching(authenticated_history(),s["submission_note"])
        if entry is None:
            s["stage"]="submission_unverified"
            s["error"]="No matching authenticated Kaggle submission yet; read-only recover"
            persist(s)
            return
        record_entry(s,entry)
    except Exception as exc:
        s["stage"]="submission_ambiguous"
        s["error"]=str(exc)[-1500:]+"; DO NOT RETRY SUBMISSION"
        persist(s)
        raise

def recover():
    s=read(STATE)
    if not s.get("submission_attempted") or not s.get("submission_note"):
        print("No V299 submission reservation; nothing to recover")
        return
    rows=authenticated_history()
    row=protection.matching(rows,s["submission_note"])
    if row:
        record_entry(s,row)
    else:
        print("No official V299 submission found; recovery performed READ-ONLY.")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--reserve",action="store_true")
    group.add_argument("--submit",action="store_true")
    group.add_argument("--recover",action="store_true")
    args=parser.parse_args()
    if args.reserve:reserve()
    elif args.submit:submit()
    else:recover()
