"""V2915 Kaggle official submission: reproducible, conditional, single shot.

This file NEVER trains or pushes a Notebook. It only submits one immutable
CSV from an independent completed OOF challenger, with a committed reservation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import v298_fusion as old
import v299_teacher_fusion as teacher
import v2913_pipeline as official
import v2914_pipeline as dual
import v2915_pipeline as five
import v296_submit as guard
from v291_ledger import read,write
from v293_blend_math import verify_sources,N_TEST,VALIDATION

ROOT=old.ROOT
SLUG=old.SLUG
CV=five.STATE
STATE=ROOT/"state/v2_9_15_submission.json"
REPORT=ROOT/"reports/v2_9_15_submission.md"
OUTPUT=ROOT/"competitions"/SLUG/"results_v2915"/"v2915_submission.csv"
LEDGER=ROOT/"state/v2_9_experiments.json"
PREVIOUS_AUC=.96117635576833
PREVIOUS_LB=.96083
MIN_GAIN=.00008
NOTE_PREFIX="KAS V2915 5E "

def now():
    return old.timestamp()

def save(status):
    status["updated_at"]=now()
    write(STATE,status)
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    item=status.get("submission") or {}
    REPORT.write_text("# V2.9.15 single official Kaggle submission\n\n"+
        f"- UTC: {status['updated_at']}\n"+
        f"- Stage: {status.get('stage')}\n"+
        f"- Full 3fold CV: {status.get('cv_auc','unavailable')}\n"+
        f"- Fixed weights: 80% V2913, 10% V2914, 10% V2915\n"+
        f"- Candidate-test SHA: {status.get('prediction_hash','')}\n"+
        f"- Submission note: {status.get('submission_note','')}\n"+
        f"- Official Kaggle submission ID: {item.get('ref','pending')}\n"+
        f"- Public LB: {item.get('publicScore','pending')}\n"+
        "- An ambiguous Kaggle write must NEVER be retried; read-only recovery only.\n",
        encoding="utf-8")

def qualified():
    old.check_competition()
    state=read(CV)
    previous=read(ROOT/"state/v2_9_13_submission.json")
    comparison=state.get("comparison") or {}
    model=state.get("metrics") or {}
    expected={"v2913_rank":.80,"v2914_dual_rank":.10,
              "v2915_five_epoch_rank":.10}
    if (state.get("competition")!=SLUG or state.get("kernel")!=five.KERNEL
        or state.get("stage")!="candidate_ready_unsubmitted"
        or state.get("submitted") is not False
        or model.get("competition")!=SLUG or model.get("validation")!=VALIDATION
        or model.get("stage")!="three_folds_complete" or model.get("folds")!=3
        or len(model.get("fold_results") or [])!=3
        or model.get("train_rows")!=699635 or model.get("test_rows")!=N_TEST
        or model.get("gpu_verified") is not True
        or (model.get("hardware") or {}).get("ok") is not True
        or model.get("model_seed")!=20261021 or model.get("epochs")!=5
        or model.get("leakage_safe_inner_cross_fit") is not True
        or model.get("outer_validation_used_in_training") is not False
        or comparison.get("fixed_weights")!=expected
        or abs(float(comparison.get("baseline_auc",0))-PREVIOUS_AUC)>.000002
        or float(comparison.get("gain",0))<MIN_GAIN
        or float(comparison.get("candidate_auc",0))<PREVIOUS_AUC+MIN_GAIN
        or int(comparison.get("positive_folds",0))<2
        or len(comparison.get("fold_gain") or [])!=3
        or min(comparison["fold_gain"])<=-.00005
        or len(state.get("source_artifacts") or {})!=8
        or len(str(state.get("candidate_test_hash","")))!=64
        or previous.get("stage")!="complete" or previous.get("submitted") is not True
        or str((previous.get("submission") or {}).get("ref"))!="57423879"
        or abs(float(previous.get("official_lb",0))-PREVIOUS_LB)>1e-9):
        raise RuntimeError("V2915 strict full-OOF/model-hash/official-incumbent gate failed")
    return state

def check_provenance(actual,expected):
    if set(actual)!=set(expected):
        raise RuntimeError("Historical Kaggle artifacts set mismatch")
    for key,record in expected.items():
        for field,value in record.items():
            if actual[key].get(field)!=value:
                raise RuntimeError("Kaggle artifact SHA or notebook changed: "+key+" "+field)

def score_file():
    signed=qualified()
    histories={k:read(ROOT/path) for k,path in {
        "v25":"state/v2_5_status.json","v28":"state/v2_8_status.json",
        "v296":"state/v2_9_6_status.json","v297":"state/v2_9_7_status.json"}.items()}
    old.check_provenance(histories)
    with tempfile.TemporaryDirectory(prefix="kas_v2915_submission_") as temp:
        train,test,data,files=old.download_inputs(temp,histories)
        train_id,test_id,y=old.validate_official(train,test)
        old_audit=old.evaluate(train,test,data,histories,audit_legacy_order=True)
        if abs(old_audit["candidate_cv"]-.9610183558267038)>.000002:
            raise RuntimeError("Original V298 OOF benchmark cannot be reproduced")
        t,t_test,t_info=teacher.get_teacher(temp,
              read(ROOT/"state/v2_9_4_status.json"),train_id,test_id,y)
        files["v294"]=t_info
        oof={}
        rawtest={}
        for label,kernel in (("v2913",official.KERNEL),
                             ("v2914",dual.KERNEL),("v2915",five.KERNEL)):
            destination=Path(temp)/label
            destination.mkdir(parents=True,exist_ok=True)
            old.checked_command(["kaggle","kernels","output",kernel,
                       "-p",str(destination),"-o","-q"])
            if label=="v2915":
                f,arrays=five.read_new(destination)
                train_key,test_key="oof_realmlp_5epoch","pred_realmlp_5epoch"
            else:
                f,arrays,train_key,test_key=dual.read_oof(destination,label)
            files[label]={"notebook":kernel,"sha256":old.sha256(f)}
            oof[label],rawtest[label]=dual.align_new(
                   arrays,train_key,test_key,train_id,test_id,y,label)
        check_provenance(files,signed["source_artifacts"])
        for label,status_file in (("v2913",official.STATE),("v2914",dual.STATE)):
            model=read(status_file)
            actual=float(roc_auc_score(y,oof[label]))
            expected=float(model["metrics"]["full_oof_auc"])
            if abs(actual-expected)>.000002:
                raise RuntimeError(label+" standalone OOF metric changed")
        actual_five=float(roc_auc_score(y,oof["v2915"]))
        if abs(actual_five-float(signed["metrics"]["full_oof_auc"]))>.000002:
            raise RuntimeError("New five epoch OOF score mismatch")
        official.evaluate_fold_safety(y,oof["v2915"],signed["metrics"])
        _,hist_oof,hist_test=verify_sources(
              data["v25"],data["v28"],histories["v25"],histories["v28"],
              data["v25_submission"],data["v28_submission"])
        v293=.35*old.rank(hist_oof["v25"])+.65*old.rank(hist_oof["v28"])
        real=old.align(data["v296"]["oof_realmlp"],
                       data["v296"]["train_id"],train_id,"RealMLP V296")
        conditional=old.align(data["v297"]["oof_conditional"],
                       data["v297"]["train_id"],train_id,"Conditional V297")
        baseline=teacher.fuse_fixed(y,v293,real,conditional,t,old.EXPECTED_V296)
        if abs(baseline["candidate_cv"]-.9610561936366048)>.000002:
            raise RuntimeError("V299 original blend differs")
        v299=.60*old.rank(v293)+.20*old.rank(real)+.15*old.rank(conditional)+.05*old.rank(t)
        v2913=.85*old.rank(v299)+.15*old.rank(oof["v2913"])
        compared=five.evaluate_blend(y,v2913,oof["v2914"],oof["v2915"])
        for name in ("baseline_auc","candidate_auc","gain"):
            if abs(compared[name]-float(signed["comparison"][name]))>.000002:
                raise RuntimeError("V2915 reread CV changed: "+name)
        if (compared["gain"]<MIN_GAIN or compared["positive_folds"]<2
            or min(compared["fold_gain"])<=-.00005):
            raise RuntimeError("V2915 promotion safety gate no longer passes")
        for actual,expected in zip(compared["fold_gain"],
                                    signed["comparison"]["fold_gain"]):
            if abs(actual-float(expected))>.000002:
                raise RuntimeError("Historical fold improvement changed")
        from_test=.35*hist_test["v25"]+.65*hist_test["v28"]
        real_test=old.align(data["v296"]["pred_realmlp"],
                      data["v296"]["test_id"],test_id,"RealMLP V296 test")
        conditional_test=old.align(data["v297"]["test_conditional"],
                      data["v297"]["test_id"],test_id,"Conditional V297 test")
        v299_test=(.60*old.rank(from_test)+.20*old.rank(real_test)
                   +.15*old.rank(conditional_test)+.05*old.rank(t_test))
        v2913_test=.85*old.rank(v299_test)+.15*old.rank(rawtest["v2913"])
        if hashlib.sha256(v2913_test.astype("<f8").tobytes()).hexdigest()!=read(official.STATE).get("candidate_test_hash"):
            raise RuntimeError("Scored V2913 test prediction SHA mismatch")
        result=(.80*old.rank(v2913_test)+.10*old.rank(rawtest["v2914"])
                +.10*old.rank(rawtest["v2915"]))
        if len(result)!=N_TEST or not np.isfinite(result).all() or result.min()<0 or result.max()>1:
            raise RuntimeError("Invalid V2915 prediction vector")
        digest=hashlib.sha256(result.astype("<f8").tobytes()).hexdigest()
        if digest!=signed.get("candidate_test_hash"):
            raise RuntimeError("Exact preregistered V2915 test hash differs")
        submission=pd.DataFrame({"id":test_id,"satisfaction":result})
        if submission["id"].duplicated().any():
            raise RuntimeError("Duplicate official Kaggle test IDs")
        OUTPUT.parent.mkdir(parents=True,exist_ok=True)
        submission.to_csv(OUTPUT,index=False,float_format="%.17g")
        back=pd.read_csv(OUTPUT)
        if (list(back.columns)!=["id","satisfaction"] or len(back)!=N_TEST
            or not np.array_equal(back["id"].to_numpy(),test_id)
            or not np.allclose(back["satisfaction"].to_numpy(),result,
                               atol=1e-15,rtol=0)):
            raise RuntimeError("Kaggle submission CSV roundtrip failed")
    return digest,hashlib.sha256(OUTPUT.read_bytes()).hexdigest()

def signed_csv(record):
    if not OUTPUT.is_file():
        raise RuntimeError("No V2915 signed candidate file on this runner")
    if hashlib.sha256(OUTPUT.read_bytes()).hexdigest()!=record.get("candidate_file_sha256"):
        raise RuntimeError("V2915 reserved CSV SHA256 mismatch")
    return OUTPUT

def note(digest):
    return NOTE_PREFIX+digest[:16]

def reserve():
    if read(STATE):
        raise RuntimeError("V2915 reservation exists: no duplicate creation")
    digest,checksum=score_file()
    rows=guard.history()
    message=note(digest)
    if guard.matching(rows,message):
        raise RuntimeError("V2915 duplicate official Kaggle submission note")
    if not guard.quota(rows):
        raise RuntimeError("Kaggle submission quota exhausted")
    s={"competition":SLUG,"stage":"submission_reserved",
       "submitted":False,"submission_attempted":True,"reserved_at":now(),
       "submission_note":message,"prediction_hash":digest,
       "candidate_file_sha256":checksum,
       "cv_auc":float(read(CV)["comparison"]["candidate_auc"]),
       "threefold_verified":True,"submission":None}
    save(s)
    print("V2915 candidate rebuilt, Kaggle quota checked, unique SHA reserved.")

def mark_official(s,entry):
    s["submission"]=entry
    s["submitted"]=True
    public=entry.get("publicScore")
    s["stage"]="submission_pending" if public in (None,"") else "complete"
    if public not in (None,""):
        s["official_lb"]=float(public)
        s["public_lb_delta_vs_v2913"]=float(public)-PREVIOUS_LB
    save(s)
    ledger=read(LEDGER)
    found=False
    for item in ledger.get("experiments",[]):
        if item.get("candidate")!="v2915_five_epoch_realmlp_ens8":continue
        previous=item.get("submission_id")
        if previous is not None and str(previous)!=str(entry.get("ref")):
            raise RuntimeError("Experiment ledger has contradictory Kaggle submission ID")
        item["submission_id"]=entry.get("ref")
        if public not in (None,""):item["public_lb"]=public
        found=True
    if not found:
        raise RuntimeError("No verified V2915 ledger entry")
    write(LEDGER,ledger)

def submit_once():
    s=read(STATE)
    if (s.get("stage")!="submission_reserved" or not s.get("submission_attempted")
        or s.get("submitted") is True):
        raise RuntimeError("A committed exact one-shot reservation is required")
    if os.getenv("V2915_RESERVED_COMMITTED")!="true":
        raise RuntimeError("GitHub immutable submission reservation was not persisted")
    qualified()
    payload=signed_csv(s)
    history=guard.history()
    original=guard.matching(history,s["submission_note"])
    if original:
        mark_official(s,original)
        return
    if not guard.quota(history):
        raise RuntimeError("Kaggle daily submission quota unavailable after reservation")
    s["stage"]="submitting"
    save(s)
    try:
        guard.cli(["kaggle","competitions","submit",SLUG,
                 "-f",str(payload),"-m",s["submission_note"],
                 "--wait","600","--poll-interval","30"],timeout=900)
        row=guard.matching(guard.history(),s["submission_note"])
        if row is None:
            s["stage"]="submission_unverified"
            s["error"]="No authoritative Kaggle ID/score yet. Read-only recovery only."
            save(s)
            return
        mark_official(s,row)
    except Exception as exc:
        s["stage"]="submission_ambiguous"
        s["error"]=str(exc)[-1500:]+"; NEVER RETRY KAGGLE SUBMIT"
        save(s)
        raise

def recover():
    s=read(STATE)
    if not s.get("submission_attempted") or not s.get("submission_note"):
        print("No reserved V2915 submission exists; read-only recovery ended.")
        return
    row=guard.matching(guard.history(),s["submission_note"])
    if row:
        mark_official(s,row)
    else:
        print("Kaggle submission not confirmed; no write or retry was performed.")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--reserve",action="store_true")
    g.add_argument("--submit",action="store_true")
    g.add_argument("--recover",action="store_true")
    a=parser.parse_args()
    if a.reserve:reserve()
    elif a.submit:submit_once()
    else:recover()
