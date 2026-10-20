"""V2.9.13 authenticated Kaggle submission with immutable single-attempt reservation.

Every historical OOF/test file is downloaded from its original Kaggle kernel,
SHA-verified, canonical sample-aligned, and the precommitted candidate-test
hash is reconstructed. Kaggle receives only one explicitly reserved CSV.
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

import v298_fusion as audit
import v299_teacher_fusion as teacher
import v2913_pipeline as ens8
import v296_submit as kaggle_guard
from v291_ledger import read,write
from v293_blend_math import verify_sources,N_TEST,VALIDATION

ROOT=audit.ROOT
SLUG=audit.SLUG
CV=ROOT/"state/v2_9_13_status.json"
STATE=ROOT/"state/v2_9_13_submission.json"
LEDGER=ROOT/"state/v2_9_experiments.json"
REPORT=ROOT/"reports/v2_9_13_submission.md"
OUTPUT=ROOT/"competitions"/SLUG/"results_v2913"/"v2913_submission.csv"
EXPECTED_CV=.96117635576833
CURRENT_V299=.9610561936366048
OFFICIAL_V299_LB=.96070
NOTE_PREFIX="KAS V2913 ENS8 "
MIN_CV_GAIN=.00006

def now():
    return audit.timestamp()

def report(s):
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    official=s.get("submission") or {}
    REPORT.write_text("# V2.9.13 official Kaggle submit (once)\n\n"+
        f"- Updated UTC: {s.get('updated_at','')}\n"+
        f"- Stage: {s.get('stage','')}\n"+
        f"- Full threefold CV: {s.get('cv_auc',EXPECTED_CV)}\n"+
        f"- Candidate aligned test SHA256: {s.get('prediction_hash','')}\n"+
        f"- Submission note: {s.get('submission_note','')}\n"+
        f"- Kaggle submission ID: {official.get('ref','not yet scored')}\n"+
        f"- Official Public LB: {official.get('publicScore','not yet scored')}\n"+
        "- An ambiguous/failed submission may only be recovered read-only.\n",
        encoding="utf-8")

def persist(s):
    s["updated_at"]=now()
    write(STATE,s)
    report(s)

def verified_cv():
    audit.check_competition()
    s=read(CV)
    previous=read(ROOT/"state/v2_9_9_submission.json")
    v299=read(ROOT/"state/v2_9_9_status.json")
    metric=s.get("metrics") or {}
    comp=s.get("comparison") or {}
    hardware=metric.get("hardware") or {}
    if (s.get("competition")!=SLUG or s.get("kernel")!=ens8.KERNEL
        or s.get("stage")!="candidate_ready_unsubmitted"
        or s.get("submitted") is not False
        or metric.get("competition")!=SLUG
        or metric.get("validation")!=VALIDATION
        or metric.get("stage")!="three_folds_complete"
        or metric.get("folds")!=3 or len(metric.get("fold_results") or [])!=3
        or metric.get("train_rows")!=699635 or metric.get("test_rows")!=N_TEST
        or metric.get("gpu_verified") is not True or hardware.get("ok") is not True
        or not any("T4" in str(x) for x in hardware.get("devices",[]))
        or metric.get("leakage_safe_inner_cross_fit") is not True
        or metric.get("outer_validation_used_in_training") is not False
        or not str(metric.get("model","")).startswith("RealMLP-TD n_ens=8")
        or comp.get("fixed_weights")!={"v299_rank":.85,"realmlp_ens8_rank":.15}
        or abs(float(comp.get("baseline_auc",0))-CURRENT_V299)>.000002
        or abs(float(comp.get("candidate_auc",0))-EXPECTED_CV)>.000002
        or float(comp.get("gain",0))<MIN_CV_GAIN
        or int(comp.get("positive_folds",0))<2
        or len(comp.get("fold_gain") or [])!=3
        or min(comp["fold_gain"])<=-.00005
        or len(str(s.get("candidate_test_hash","")))!=64
        or len(s.get("source_artifacts") or {})!=6
        or v299.get("stage")!="candidate_ready_unsubmitted"
        or abs(float(v299.get("candidate_cv",0))-CURRENT_V299)>.000002
        or previous.get("stage")!="complete"
        or previous.get("submitted") is not True
        or str((previous.get("submission") or {}).get("ref"))!="57421208"
        or abs(float(previous.get("official_lb",0))-OFFICIAL_V299_LB)>1e-9):
        raise RuntimeError("V2913 exact T4 ensemble8/threefold/champion safety gate failed")
    return s

def same_sources(retrieved,recorded):
    if set(retrieved)!=set(recorded):
        raise RuntimeError("Historical artifact provenance set differs")
    for key,expected in recorded.items():
        for name,value in expected.items():
            if retrieved[key].get(name)!=value:
                raise RuntimeError("Kaggle archived artifact checksum mismatch: "+key+" "+name)

def replay_file():
    registered=verified_cv()
    names={"v25":"state/v2_5_status.json","v28":"state/v2_8_status.json",
           "v296":"state/v2_9_6_status.json","v297":"state/v2_9_7_status.json"}
    statuses={key:read(ROOT/path) for key,path in names.items()}
    audit.check_provenance(statuses)
    with tempfile.TemporaryDirectory(prefix="kas_v2913_submit_") as scratch:
        train,test,archive,files=audit.download_inputs(scratch,statuses)
        train_id,test_id,y=audit.validate_official(train,test)
        baseline=audit.evaluate(train,test,archive,statuses,audit_legacy_order=True)
        if abs(baseline["candidate_cv"]-.9610183558267038)>.000002:
            raise RuntimeError("V298 source score no longer matches")
        t_oof,t_test,t_meta=teacher.get_teacher(scratch,
            read(ROOT/"state/v2_9_4_status.json"),train_id,test_id,y)
        files["v294"]=t_meta
        destination=Path(scratch)/"v2913"
        destination.mkdir(parents=True,exist_ok=True)
        audit.checked_command(["kaggle","kernels","output",ens8.KERNEL,
                               "-p",str(destination),"-o","-q"])
        f=audit.unique_file(destination,"v2913_realmlp_oof.npz")
        with np.load(f,allow_pickle=False) as ds:
            required={"train_id","target","test_id","oof_realmlp_ens8","pred_realmlp_ens8"}
            if not required.issubset(ds.files):
                raise RuntimeError("Neural OOF archive missing canonical IDs or predictions")
            src_id=np.asarray(ds["train_id"]).copy()
            src_y=np.asarray(ds["target"]).copy()
            oof=np.asarray(ds["oof_realmlp_ens8"]).copy()
            test_src=np.asarray(ds["test_id"]).copy()
            test_pred=np.asarray(ds["pred_realmlp_ens8"]).copy()
        files["v2913"]={"notebook":ens8.KERNEL,"sha256":audit.sha256(f)}
        same_sources(files,registered["source_artifacts"])
        ordered_y=pd.Series(src_y,index=audit.ids(src_id,len(train),"V2913 IDs")).reindex(train_id).to_numpy()
        if not np.array_equal(ordered_y,y):
            raise RuntimeError("V2913 target ID/label mismatch")
        verified_new=audit.align(oof,src_id,train_id,"V2913 OOF")
        if abs(roc_auc_score(y,verified_new)-float(registered["metrics"]["full_oof_auc"]))>.000002:
            raise RuntimeError("Neural threefold score changed")
        ens8.evaluate_fold_safety(y,verified_new,registered["metrics"])
        _,oofs,tests=verify_sources(archive["v25"],archive["v28"],
                                   statuses["v25"],statuses["v28"],
                                   archive["v25_submission"],archive["v28_submission"])
        old_oof=.35*audit.rank(oofs["v25"])+.65*audit.rank(oofs["v28"])
        r_oof=audit.align(archive["v296"]["oof_realmlp"],
                     archive["v296"]["train_id"],train_id,"V296 RealMLP")
        c_oof=audit.align(archive["v297"]["oof_conditional"],
                     archive["v297"]["train_id"],train_id,"V297 conditional")
        v299=teacher.fuse_fixed(y,old_oof,r_oof,c_oof,t_oof,audit.EXPECTED_V296)
        if abs(v299["candidate_cv"]-CURRENT_V299)>.000002:
            raise RuntimeError("Reconstructed old winner changed")
        comp=ens8.compare(y,old_oof,r_oof,c_oof,t_oof,verified_new)
        for name in ("baseline_auc","candidate_auc","gain"):
            if abs(float(comp[name])-float(registered["comparison"][name]))>.000002:
                raise RuntimeError("V2913 OOF promotion metric mismatch: "+name)
        if (int(comp["positive_folds"])<2
            or min(comp["fold_gain"])<=-.00005
            or float(comp["gain"])<MIN_CV_GAIN):
            raise RuntimeError("V2913 OOF promotion gate no longer passes")
        for a,b in zip(comp["fold_gain"],registered["comparison"]["fold_gain"]):
            if abs(a-b)>.000002:
                raise RuntimeError("V2913 fold deltas changed")
        old_test=.35*tests["v25"]+.65*tests["v28"]
        r_test=audit.align(archive["v296"]["pred_realmlp"],
                          archive["v296"]["test_id"],test_id,"V296 test")
        c_test=audit.align(archive["v297"]["test_conditional"],
                          archive["v297"]["test_id"],test_id,"V297 test")
        v299_test=(.60*audit.rank(old_test)+.20*audit.rank(r_test)
                   +.15*audit.rank(c_test)+.05*audit.rank(t_test))
        sha_old=hashlib.sha256(v299_test.astype("<f8").tobytes()).hexdigest()
        if sha_old!=read(ROOT/"state/v2_9_9_status.json").get("candidate_test_hash"):
            raise RuntimeError("V299 scored prediction hash has changed")
        current_test=audit.align(test_pred,test_src,test_id,"V2913 test")
        result=.85*audit.rank(v299_test)+.15*audit.rank(current_test)
        if len(result)!=N_TEST or not np.isfinite(result).all() or result.min()<0 or result.max()>1:
            raise RuntimeError("V2913 test vectors have invalid values")
        sha=hashlib.sha256(result.astype("<f8").tobytes()).hexdigest()
        if sha!=registered.get("candidate_test_hash"):
            raise RuntimeError("V2913 candidate full test prediction hash mismatch")
        output=pd.DataFrame({"id":test_id,"satisfaction":result})
        if output["id"].duplicated().any():
            raise RuntimeError("Duplicate Kaggle test IDs")
        OUTPUT.parent.mkdir(parents=True,exist_ok=True)
        output.to_csv(OUTPUT,index=False,float_format="%.17g")
        reread=pd.read_csv(OUTPUT)
        if (len(reread)!=N_TEST or list(reread.columns)!=["id","satisfaction"]
            or not np.array_equal(reread["id"].to_numpy(),test_id)
            or not np.allclose(reread["satisfaction"].to_numpy(),result,atol=1e-15,rtol=0)):
            raise RuntimeError("V2913 CSV roundtrip altered exact test predictions")
    return sha,hashlib.sha256(OUTPUT.read_bytes()).hexdigest()

def note(sha):
    return NOTE_PREFIX+sha[:16]

def locked_candidate(s):
    if not OUTPUT.is_file():
        raise RuntimeError("Reserved V2913 CSV not found on this runner")
    if hashlib.sha256(OUTPUT.read_bytes()).hexdigest()!=s.get("candidate_file_sha256"):
        raise RuntimeError("Reserved V2913 CSV SHA mismatch")
    return OUTPUT

def reserve():
    if read(STATE):
        raise RuntimeError("V2913 reservation already exists; recovery only")
    digest,file_sha=replay_file()
    rows=kaggle_guard.history()
    submission_note=note(digest)
    if kaggle_guard.matching(rows,submission_note):
        raise RuntimeError("This exact V2913 candidate already scored/submitted")
    if not kaggle_guard.quota(rows):
        raise RuntimeError("Kaggle submission quota unavailable; no reservation made")
    s={"competition":SLUG,"stage":"submission_reserved","submitted":False,
       "submission_attempted":True,"reserved_at":now(),"submission_note":submission_note,
       "candidate_file_sha256":file_sha,"prediction_hash":digest,
       "cv_auc":EXPECTED_CV,"folds":3,"submission":None}
    persist(s)
    print("V2913 candidate rebuilt and slot reserved: COMMIT reservation before Kaggle submit.")

def annotate(s,entry):
    s["submission"]=entry
    s["submitted"]=True
    score=entry.get("publicScore")
    s["stage"]="submission_pending" if score in (None,"") else "complete"
    if score not in (None,""):
        s["official_lb"]=float(score)
        s["public_lb_delta_vs_v299"]=float(score)-OFFICIAL_V299_LB
    persist(s)
    ledger=read(LEDGER)
    changed=False
    for line in ledger.get("experiments",[]):
        if line.get("candidate")!="v2913_realmlp_ens8_threefold":
            continue
        former=line.get("submission_id")
        if former is not None and str(former)!=str(entry.get("ref")):
            raise RuntimeError("Independent ledger already has conflicting official ID")
        line["submission_id"]=entry.get("ref")
        if score not in (None,""):line["public_lb"]=score
        changed=True
    if not changed:
        raise RuntimeError("Cannot locate V2913 independent ledger record")
    write(LEDGER,ledger)

def submit():
    s=read(STATE)
    if (s.get("stage")!="submission_reserved"
        or s.get("submitted") or not s.get("submission_attempted")):
        raise RuntimeError("Missing exact one-shot V2913 reservation")
    if os.environ.get("V2913_RESERVED_COMMITTED")!="true":
        raise RuntimeError("Kaggle reservation not durably persisted to GitHub")
    verified_cv()
    path=locked_candidate(s)
    records=kaggle_guard.history()
    existing=kaggle_guard.matching(records,s["submission_note"])
    if existing:
        annotate(s,existing)
        return
    if not kaggle_guard.quota(records):
        raise RuntimeError("Kaggle daily quota exhausted after reservation")
    s["stage"]="submitting"
    persist(s)
    try:
        kaggle_guard.cli(["kaggle","competitions","submit",SLUG,
                 "-f",str(path),"-m",s["submission_note"],
                 "--wait","600","--poll-interval","30"],timeout=900)
        entry=kaggle_guard.matching(kaggle_guard.history(),s["submission_note"])
        if not entry:
            s["stage"]="submission_unverified"
            s["error"]="No matching official Kaggle submission; READ ONLY recovery"
            persist(s)
            return
        annotate(s,entry)
    except Exception as exc:
        s["stage"]="submission_ambiguous"
        s["error"]=str(exc)[-1500:]+" DO NOT RESUBMIT"
        persist(s)
        raise

def recover():
    s=read(STATE)
    if not s.get("submission_attempted") or not s.get("submission_note"):
        print("No V2913 reservation to recover");return
    row=kaggle_guard.matching(kaggle_guard.history(),s["submission_note"])
    if row:
        annotate(s,row)
    else:
        print("No matching official Kaggle submission found; read-only check.")

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
