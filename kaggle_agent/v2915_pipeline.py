"""V2.9.15 independent 5-epoch RealMLP-TD full 3fold OOF evaluator.

Rebuilds V2913 official champion and V2914 exploratory dual model exactly.
One fixed 80:10:10 blend; no weight search, GPU kernel deduplicated by owner ID.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import v298_fusion as old
import v299_teacher_fusion as teacher
import v2913_pipeline as champion
import v2914_pipeline as dual
from v291_ledger import read,write,record_experiment
from v293_blend_math import N_TRAIN,N_TEST,VALIDATION,verify_sources
from v23_tpu_pipeline import poll_kernel

ROOT=old.ROOT
SLUG=old.SLUG
KERNEL="mjw3177109/kas-v2915-realmlp-5epoch-t4-s6e10"
STATE=ROOT/"state/v2_9_15_status.json"
REPORT=ROOT/"reports/v2_9_15_realmlp_5epoch.md"
WORKDIR=ROOT/"competitions"/SLUG/"kaggle_gpu_v2915"
OUTDIR=ROOT/"competitions"/SLUG/"results_v2915"
LEDGER=ROOT/"state/v2_9_experiments.json"
OFFICIAL_BASELINE=.96117635576833
MIN_GAIN=.00008
TERMINAL={"candidate_ready_unsubmitted","skipped_no_cv_gain",
          "kernel_failed","output_failed","validation_mismatch","push_failed","kernel_access_error"}

def now():
    return datetime.now(timezone.utc).isoformat()

def cli(cmd,timeout=180,check=True):
    response=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=timeout)
    if check and response.returncode:
        raise RuntimeError("Kaggle CLI failed: "+(response.stderr or response.stdout)[-1200:])
    return response

def gate():
    old.check_competition()
    dualstatus=read(dual.STATE)
    official=read(ROOT/"state/v2_9_13_submission.json")
    incumbent=read(champion.STATE)
    if (dualstatus.get("stage")!="skipped_no_cv_gain"
        or abs(float((dualstatus.get("comparison") or {}).get("candidate_auc",0))
              -.9612261589523977)>.000002
        or incumbent.get("stage")!="candidate_ready_unsubmitted"
        or abs(float((incumbent.get("comparison") or {}).get("candidate_auc",0))
              -OFFICIAL_BASELINE)>.000002
        or official.get("stage")!="complete" or official.get("submitted") is not True
        or str((official.get("submission") or {}).get("ref"))!="57423879"
        or abs(float(official.get("official_lb",0))-.96083)>1e-9):
        raise RuntimeError("Cannot prove V2914 terminal OOF and scored V2913 official champion")

def persist(status):
    status["updated_at"]=now()
    write(STATE,status)
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    m=status.get("metrics") or {}
    c=status.get("comparison") or {}
    REPORT.write_text("# V2.9.15 5-epoch RealMLP Ens8 independent audit\n\n"
         +f"- UTC: {status['updated_at']}\n"
         +f"- Stage: {status.get('stage')}\n"
         +f"- Kaggle Notebook: https://www.kaggle.com/code/{status.get('kernel') or KERNEL}\n"
         +f"- Independent model 3fold CV: {m.get('full_oof_auc','pending')}\n"
         +f"- Official V2913 Public LB: 0.96083 (submission 57423879)\n"
         +f"- Historical V2913 3fold CV: {c.get('baseline_auc',OFFICIAL_BASELINE)}\n"
         +f"- Fixed 80/10/10 3fold CV: {c.get('candidate_auc','pending')}\n"
         +f"- Threefold paired AUC gains: {c.get('fold_gain','pending')}\n"
         +f"- Stricter OOF improvement gate: >= {MIN_GAIN} due selected V2914 reuse.\n"
         +"- No Kaggle competition submission from this evaluator.\n"
         +(f"- Error: {status.get('error')}\n" if status.get("error") else ""),
         encoding="utf-8")

def unique(path,name):
    matches=list(Path(path).rglob(name))
    if len(matches)!=1:raise ValueError(f"Expected exactly one {name}, got {len(matches)}")
    return matches[0]

def existing_or_absent():
    lookup=cli(["kaggle","kernels","status",KERNEL],timeout=90,check=False)
    if lookup.returncode==0:return True
    listing=cli(["kaggle","kernels","list","--mine",
                 "--search",KERNEL.split("/",1)[1],"--page-size","50","-v"],
                timeout=100,check=False)
    if listing.returncode:
        raise RuntimeError("Kaggle owner-authenticated list failed; refuse duplicate push")
    if KERNEL.lower() in listing.stdout.lower():
        return True
    if listing.stdout.strip().lower()!="not found":
        raise RuntimeError("Ambiguous Kaggle owner listing; refuse kernel push")
    witness=cli(["kaggle","kernels","status",dual.KERNEL],timeout=90,check=False)
    if witness.returncode!=0 or "KernelWorkerStatus." not in witness.stdout:
        raise RuntimeError("Cannot authenticate known-existing owner kernel; refuse push")
    return False

def validate_metrics(m):
    hardware=m.get("hardware") or {}
    if (m.get("competition")!=SLUG or m.get("stage")!="three_folds_complete"
        or m.get("validation")!=VALIDATION or m.get("folds")!=3
        or len(m.get("fold_results") or [])!=3 or m.get("train_rows")!=N_TRAIN
        or m.get("test_rows")!=N_TEST
        or m.get("model")!="RealMLP-TD n_ens=8 5 epochs independent seeded"
        or m.get("model_seed")!=20261021 or m.get("epochs")!=5
        or m.get("gpu_verified") is not True or hardware.get("ok") is not True
        or not any("T4" in str(x) for x in hardware.get("devices",[]))
        or m.get("outer_validation_used_in_training") is not False
        or m.get("leakage_safe_inner_cross_fit") is not True):
        raise ValueError("V2915 GPU, 5-epoch, threefold or leakage evidence mismatch")

def read_new(path):
    file=unique(path,"v2915_realmlp_oof.npz")
    keys=("train_id","target","oof_realmlp_5epoch",
          "test_id","pred_realmlp_5epoch")
    with np.load(file,allow_pickle=False) as f:
        if not set(keys)<=set(f.files):
            raise ValueError("V2915 missing complete canonical OOF/test arrays")
        return file,{k:np.asarray(f[k]).copy() for k in keys}

def evaluate_blend(y,baseline,dual_prediction,five_epoch):
    rank=old.rank
    challenger=(.80*rank(baseline)+.10*rank(dual_prediction)
                +.10*rank(five_epoch))
    base_auc=float(roc_auc_score(y,baseline))
    if abs(base_auc-OFFICIAL_BASELINE)>.000002:
        raise ValueError("Cannot reproduce V2913 official champion CV")
    candidate_auc=float(roc_auc_score(y,challenger))
    differences=[]
    for _,valid in StratifiedKFold(3,shuffle=True,random_state=20261018).split(
            np.zeros(len(y),dtype=np.uint8),y):
        differences.append(float(roc_auc_score(y[valid],challenger[valid])
                                 -roc_auc_score(y[valid],baseline[valid])))
    return {"baseline_auc":base_auc,
            "dual_standalone_auc":float(roc_auc_score(y,dual_prediction)),
            "new_standalone_auc":float(roc_auc_score(y,five_epoch)),
            "candidate_auc":candidate_auc,"gain":candidate_auc-base_auc,
            "fold_gain":differences,"positive_folds":sum(x>0 for x in differences),
            "fixed_weights":{"v2913_rank":.80,"v2914_dual_rank":.10,
                             "v2915_five_epoch_rank":.10}}

def validate(raw,status):
    m=json.loads(unique(raw,"v2915_metrics.json").read_text(encoding="utf-8"))
    validate_metrics(m)
    newfile,new=read_new(raw)
    oldstates={k:read(ROOT/path) for k,path in {
        "v25":"state/v2_5_status.json","v28":"state/v2_8_status.json",
        "v296":"state/v2_9_6_status.json","v297":"state/v2_9_7_status.json"}.items()}
    old.check_provenance(oldstates)
    prior=read(dual.STATE)
    official_model=read(champion.STATE)
    with tempfile.TemporaryDirectory(prefix="kas_v2915_full_") as folder:
        train,test,archives,files=old.download_inputs(folder,oldstates)
        train_id,test_id,y=old.validate_official(train,test)
        base=old.evaluate(train,test,archives,oldstates,audit_legacy_order=True)
        if abs(base["candidate_cv"]-.9610183558267038)>.000002:
            raise ValueError("V298 original source benchmark mismatched")
        t,t_test,t_meta=teacher.get_teacher(folder,
              read(ROOT/"state/v2_9_4_status.json"),train_id,test_id,y)
        files["v294"]=t_meta
        prior_data={}
        for version,handle in (("v2913",champion.KERNEL),("v2914",dual.KERNEL)):
            output=Path(folder)/version
            output.mkdir(parents=True,exist_ok=True)
            old.checked_command(["kaggle","kernels","output",handle,
                                 "-p",str(output),"-o","-q"])
            file,data,train_key,test_key=dual.read_oof(output,version)
            files[version]={"notebook":handle,"sha256":old.sha256(file)}
            prior_data[version]=(data,train_key,test_key)
        # Verify every historical notebook and SHA against an independently
        # completed, immutably registered V2914 OOF evaluation.
        for key,prescribed in (prior.get("source_artifacts") or {}).items():
            if key not in files:raise ValueError("Missing historical OOF artifact: "+key)
            for field,val in prescribed.items():
                if files[key].get(field)!=val:
                    raise ValueError("Historical source SHA/Notebook changed: "+key+" "+field)
        files["v2915"]={"notebook":status["kernel"],"sha256":old.sha256(newfile)}
        prev_oof,prev_test=dual.align_new(
              *prior_data["v2913"],train_id,test_id,y,"V2913")
        dual_oof,dual_test=dual.align_new(
              *prior_data["v2914"],train_id,test_id,y,"V2914")
        new_oof,new_test=dual.align_new(
              new,"oof_realmlp_5epoch","pred_realmlp_5epoch",
              train_id,test_id,y,"V2915")
        if abs(float(roc_auc_score(y,prev_oof))
               -float(official_model["metrics"]["full_oof_auc"]))>.000002:
            raise ValueError("Original RealMLP output differs from V2913")
        if abs(float(roc_auc_score(y,dual_oof))
               -float(prior["metrics"]["full_oof_auc"]))>.000002:
            raise ValueError("Dual RealMLP OOF changed")
        if abs(float(roc_auc_score(y,new_oof))-float(m["full_oof_auc"]))>.000002:
            raise ValueError("Five-epoch RealMLP standalone OOF mismatch")
        champion.evaluate_fold_safety(y,new_oof,m)
        _,oofs,tests=verify_sources(archives["v25"],archives["v28"],
                     oldstates["v25"],oldstates["v28"],
                     archives["v25_submission"],archives["v28_submission"])
        v293=.35*old.rank(oofs["v25"])+.65*old.rank(oofs["v28"])
        real=old.align(archives["v296"]["oof_realmlp"],
                      archives["v296"]["train_id"],train_id,"V296")
        conditional=old.align(archives["v297"]["oof_conditional"],
                      archives["v297"]["train_id"],train_id,"V297")
        oldblend=teacher.fuse_fixed(y,v293,real,conditional,t,old.EXPECTED_V296)
        if abs(oldblend["candidate_cv"]-.9610561936366048)>.000002:
            raise ValueError("V299 historical score changed")
        v299=.60*old.rank(v293)+.20*old.rank(real)+.15*old.rank(conditional)+.05*old.rank(t)
        v2913=.85*old.rank(v299)+.15*old.rank(prev_oof)
        result=evaluate_blend(y,v2913,dual_oof,new_oof)
        if result["fixed_weights"]!={"v2913_rank":.80,
             "v2914_dual_rank":.10,"v2915_five_epoch_rank":.10}:
            raise ValueError("Weights unexpectedly changed")
        # Exact official V2913 test vector and previous dual OOF reproduce.
        source_test=.35*tests["v25"]+.65*tests["v28"]
        real_test=old.align(archives["v296"]["pred_realmlp"],
                            archives["v296"]["test_id"],test_id,"V296 test")
        conditional_test=old.align(archives["v297"]["test_conditional"],
                            archives["v297"]["test_id"],test_id,"V297 test")
        v299_test=.60*old.rank(source_test)+.20*old.rank(real_test)+.15*old.rank(conditional_test)+.05*old.rank(t_test)
        v2913_test=.85*old.rank(v299_test)+.15*old.rank(prev_test)
        if hashlib.sha256(v2913_test.astype("<f8").tobytes()).hexdigest()!=official_model.get("candidate_test_hash"):
            raise ValueError("Official V2913 test prediction SHA changed")
        promoted=(result["gain"]>=MIN_GAIN and result["positive_folds"]>=2
                  and min(result["fold_gain"])>-.00005)
        status.update({"metrics":m,"comparison":result,"source_artifacts":files,
                 "stage":"candidate_ready_unsubmitted" if promoted else "skipped_no_cv_gain"})
        if promoted:
            # Full predictions stay in Kaggle workspace, not in GitHub source control.
            combination=.80*old.rank(v2913_test)+.10*old.rank(dual_test)+.10*old.rank(new_test)
            status["candidate_test_hash"]=hashlib.sha256(
                combination.astype("<f8").tobytes()).hexdigest()
    record_experiment(LEDGER,{"competition":SLUG,
        "candidate":"v2915_five_epoch_realmlp_ens8",
        "validation":VALIDATION,"seed":20261018,
        "parameters":{"model_seed":20261021,"epochs":5,"n_ens":8,"lr":.038,
            "blend":{"v2913":.80,"v2914":.10,"v2915":.10}},
        "status":"completed" if promoted else "skipped_no_cv_gain",
        "completed_at":now(),"cv":result["candidate_auc"],
        "gpu_seconds":m.get("runtime_seconds"),"kernel_sources":[KERNEL]})

def run(recover_only=False):
    previous=read(STATE)
    if previous.get("stage") in TERMINAL:
        print("V2915 terminal; never repeat unique T4 Notebook")
        return previous
    gate()
    state=previous or {"competition":SLUG,"stage":"preflight",
                       "kernel":None,"submitted":False}
    if not state.get("kernel"):
        if recover_only:
            print("V2915 recovery refuses creation without stored Notebook ID")
            return state
        try:
            existing=existing_or_absent()
        except RuntimeError as exc:
            state.update({"stage":"kernel_access_error","error":str(exc)[-800:]})
            persist(state);return state
        if existing:
            state.update({"stage":"kernel_queued","kernel":KERNEL,
                          "preexisting_kernel_recovered":True})
            persist(state)
    if not state.get("kernel"):
        WORKDIR.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/"kaggle_agent/v2915_realmlp_5epoch_train.py",
                     WORKDIR/"main.py")
        write(WORKDIR/"kernel-metadata.json",{
            "id":KERNEL,"title":KERNEL.split("/",1)[1],
            "code_file":"main.py","language":"python","kernel_type":"script",
            "is_private":True,"enable_gpu":True,"machine_shape":"NvidiaTeslaT4",
            "enable_internet":True,"dataset_sources":[],
            "competition_sources":[SLUG],"kernel_sources":[],"model_sources":[]})
        state.update({"stage":"pushing_kernel","kernel":KERNEL})
        persist(state)
        result=cli(["kaggle","kernels","push","-p",str(WORKDIR),
                    "-t","3600","--accelerator","NvidiaTeslaT4"],
                   timeout=300,check=False)
        if result.returncode:
            state.update({"stage":"push_failed",
                         "error":(result.stderr or result.stdout)[-1500:]})
            persist(state);return state
        proof=re.search(r"https?://(?:www\.)?kaggle\.com/code/([A-Za-z0-9_-]+)/([a-z0-9-]+)",
            (result.stdout or "")+"\n"+(result.stderr or ""))
        if not proof or proof.group(1).lower()!="mjw3177109":
            state.update({"stage":"kernel_access_error",
                  "error":"Cannot prove canonical Kaggle Notebook owner after push"})
            persist(state);return state
        state["kernel"]="/".join(proof.groups())
        if state["kernel"]!=KERNEL:
            state["requested_kernel"]=KERNEL
            state["canonical_note"]="Recover actual returned Notebook; never push requested slug again"
        persist(state)
    handle=state.get("kernel")
    if not isinstance(handle,str) or not handle.startswith("mjw3177109/"):
        raise RuntimeError("Untrusted Kaggle canonical Notebook reference")
    stage,message=poll_kernel(handle,90 if recover_only else 300,20)
    state["kernel_state"]=stage
    if stage in ("kernel_running","kernel_queued","kernel_access_error"):
        state.update({"stage":stage,"kernel_message":str(message)[-900:]})
        persist(state);return state
    raw=OUTDIR/"raw";raw.mkdir(parents=True,exist_ok=True)
    response=cli(["kaggle","kernels","output",handle,"-p",str(raw),"-o","-q"],
            timeout=600,check=False)
    if stage!="kernel_complete" or response.returncode:
        errs=list(raw.rglob("v2915_error.json"))
        state.update({"stage":"kernel_failed" if stage!="kernel_complete" else "output_failed",
          "error":errs[0].read_text(encoding="utf-8")[-2200:] if errs else
                   str(response.stderr or message)[-2200:]})
        persist(state);return state
    try:
        validate(raw,state)
    except Exception as exc:
        state.update({"stage":"validation_mismatch","error":str(exc)[-2200:]})
        persist(state)
        raise
    persist(state)
    return state

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--recover-only",action="store_true")
    args=parser.parse_args()
    result=run(recover_only=args.recover_only)
    print(json.dumps({"stage":result.get("stage"),"kernel":result.get("kernel"),
          "comparison":result.get("comparison"),"error":result.get("error")},
          ensure_ascii=False,indent=2))
