"""V2.9.12 CPU controller: one unique Kaggle kernel, strict 3fold OOF and NO submit."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from v291_ledger import read,write,record_experiment
from v293_blend_math import VALIDATION,N_TRAIN,N_TEST,verify_sources
from v23_tpu_pipeline import poll_kernel
import v298_fusion as audit
import v299_teacher_fusion as teacher

ROOT=Path(__file__).resolve().parents[1]
SLUG="playground-series-s6e10"
KERNEL="mjw3177109/kas-v2912-native-cat-cpu-s6e10"
STATE=ROOT/"state/v2_9_12_status.json"
LEDGER=ROOT/"state/v2_9_experiments.json"
REPORT=ROOT/"reports/v2_9_12_native_cat_cpu.md"
WORKDIR=ROOT/"competitions"/SLUG/"kaggle_cpu_v2912"
OUTDIR=ROOT/"competitions"/SLUG/"results_v2912"
BASELINE=0.9610561936366048
MIN_GAIN=.00006
TERMINAL={"candidate_ready_unsubmitted","skipped_no_cv_gain","kernel_failed",
          "output_failed","validation_mismatch","push_failed","kernel_access_error"}

def now():
    return datetime.now(timezone.utc)

def kaggle(cmd,timeout=240,check=True):
    proc=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=timeout)
    if check and proc.returncode:
        raise RuntimeError("Kaggle command failed: "+(proc.stderr or proc.stdout)[-1200:])
    return proc

def eligible():
    audit.check_competition()
    previous=read(ROOT/"state/v2_9_10_robustness_status.json")
    best=read(ROOT/"state/v2_9_9_status.json")
    older=read(ROOT/"state/v2_9_7_status.json")
    prior_cpu=read(ROOT/"state/v2_9_11_status.json")
    if (previous.get("stage")!="robustness_complete"
        or best.get("stage")!="candidate_ready_unsubmitted"
        or abs(float(best.get("candidate_cv",0))-BASELINE)>0.000002
        or older.get("stage")!="candidate_ready_unsubmitted"
        or prior_cpu.get("stage") not in ("skipped_no_cv_gain","candidate_ready_unsubmitted")):
        raise RuntimeError("Existing champion or prior CPU experiment not terminal")

def persist(s):
    s["updated_at"]=now().isoformat()
    write(STATE,s)
    cmp=s.get("comparison") or {}
    m=s.get("metrics") or {}
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    text=["# V2.9.12 independent native categorical CatBoost CPU","",
          f"- UTC: {s['updated_at']}",
          f"- Stage: {s.get('stage')}",
          f"- Kaggle Notebook: https://www.kaggle.com/code/{s.get('kernel') or KERNEL}",
          f"- Real full threefold new-model CV: {m.get('full_cv_auc','not yet complete')}",
          f"- V299 incumbent CV: {cmp.get('baseline_auc',BASELINE)}",
          f"- Fixed five-model CV: {cmp.get('candidate_auc','not yet complete')}",
          f"- Paired improvement: {cmp.get('gain','not yet computed')}",
          f"- Fold deltas: {cmp.get('fold_gain','not yet computed')}",
          "- Kaggle submission: NONE. New notebook CPU-only, single unique ID.",
          "- Native CatBoost categorical features; NO external target encoding.",
          ""]
    if s.get("error"):text.extend(["## Error",str(s["error"])[-2000:]])
    REPORT.write_text("\n".join(text)+"\n",encoding="utf-8")

def only(folder,name):
    files=list(Path(folder).rglob(name))
    if len(files)!=1:raise ValueError(f"Need exactly one {name}; got {len(files)}")
    return files[0]

def combine(y,base,real,conditional,t,pnew):
    """Single pre-registered five-model fusion; no label-fitted weight search."""
    y=np.asarray(y,dtype=np.int8)
    base_r=audit.rank(base);r=audit.rank(real)
    c=audit.rank(conditional);teacher_r=audit.rank(t);new_r=audit.rank(pnew)
    incumbent=.60*base_r+.20*r+.15*c+.05*teacher_r
    challenger=.50*base_r+.20*r+.15*c+.05*teacher_r+.10*new_r
    old_auc=float(roc_auc_score(y,incumbent))
    if abs(old_auc-BASELINE)>0.000002:
        raise ValueError(f"Cannot reconstruct V299 champion, found {old_auc}")
    candidate=float(roc_auc_score(y,challenger))
    gains=[]
    for _,iv in StratifiedKFold(3,shuffle=True,random_state=20261018).split(
            np.zeros(len(y),dtype=np.uint8),y):
        gains.append(float(roc_auc_score(y[iv],challenger[iv])-
                           roc_auc_score(y[iv],incumbent[iv])))
    return {"baseline_auc":old_auc,"standalone_auc":float(roc_auc_score(y,new_r)),
            "candidate_auc":candidate,"gain":candidate-old_auc,
            "fold_gain":gains,"positive_folds":sum(g>0 for g in gains),
            "fixed_weights":{"v293":.50,"realmlp":.20,"conditional":.15,
                             "teacher":.05,"native_cat":.10}}

def verify_outputs(raw,s):
    m=json.loads(only(raw,"v2912_metrics.json").read_text(encoding="utf-8"))
    params=m.get("parameters") or {}
    if (m.get("stage")!="full_threefold_complete" or m.get("competition")!=SLUG
        or m.get("validation")!=VALIDATION or m.get("folds")!=3
        or len(m.get("fold_results",[]))!=3 or m.get("train_rows")!=N_TRAIN
        or m.get("test_rows")!=N_TEST or m.get("native_ordered_categorical") is not True
        or m.get("external_teacher") is not False or m.get("source")!="competition_train_only"
        or m.get("task_type")!="CPU" or m.get("submitted") is not False
        or m.get("outer_validation_used_for_early_stopping") is not False
        or int(m.get("cat_features",0))!=9
        or int(params.get("iterations",0))!=750
        or int(params.get("depth",0))!=6
        or str(params.get("task_type"))!="CPU"):
        raise ValueError("Native CatBoost CPU training metadata/provenance mismatch")
    with np.load(only(raw,"v2912_oof.npz"),allow_pickle=False) as f:
        new_ids=np.asarray(f["train_id"]).copy()
        new_y=np.asarray(f["target"]).copy()
        pred=np.asarray(f["oof_native_cat"],dtype=np.float64).copy()
        test_ids=np.asarray(f["test_id"]).copy()
        test_pred=np.asarray(f["test_native_cat"],dtype=np.float64).copy()
    # The historical OOF auditor loads and validates the exact source notebooks.
    paths={"v25":"state/v2_5_status.json","v28":"state/v2_8_status.json",
           "v296":"state/v2_9_6_status.json","v297":"state/v2_9_7_status.json"}
    statuses={k:read(ROOT/name) for k,name in paths.items()}
    audit.check_provenance(statuses)
    destination=OUTDIR/"existing";destination.mkdir(parents=True,exist_ok=True)
    train,test,archive,files=audit.download_inputs(destination,statuses)
    train_id,canonical_test,y=audit.validate_official(train,test)
    old=audit.evaluate(train,test,archive,statuses,audit_legacy_order=True)
    if abs(old["candidate_cv"]-0.9610183558267038)>0.000002:
        raise ValueError("Historical V298 score mismatch")
    new=audit.align(pred,new_ids,train_id,"V2912 OOF")
    aligned_labels=pd.Series(new_y,index=audit.ids(new_ids,N_TRAIN,"new train")).reindex(train_id).to_numpy()
    if not np.array_equal(aligned_labels,y):
        raise ValueError("V2912 target IDs do not match canonical competition rows")
    if not np.array_equal(np.sort(test_ids),np.sort(canonical_test)):
        raise ValueError("V2912 test ID mismatch")
    if abs(float(roc_auc_score(y,new))-float(m["full_cv_auc"]))>0.000002:
        raise ValueError("V2912 reported OOF CV cannot be reproduced")
    splits=StratifiedKFold(3,shuffle=True,random_state=20261018)
    for fold,(_,iv) in enumerate(splits.split(np.zeros(len(y),dtype=np.uint8),y)):
        if abs(roc_auc_score(y[iv],new[iv])-float(m["fold_results"][fold]["auc"]))>0.000002:
            raise ValueError("New CPU fold model score mismatch")
    _,oofs,tests=verify_sources(archive["v25"],archive["v28"],statuses["v25"],statuses["v28"],
                               archive["v25_submission"],archive["v28_submission"])
    base=.35*audit.rank(oofs["v25"])+.65*audit.rank(oofs["v28"])
    real=audit.align(archive["v296"]["oof_realmlp"],archive["v296"]["train_id"],train_id,"RealMLP")
    conditional=audit.align(archive["v297"]["oof_conditional"],archive["v297"]["train_id"],train_id,"Conditional")
    tt,teacher_test,teacher_artifact=teacher.get_teacher(destination,
            read(ROOT/"state/v2_9_4_status.json"),train_id,canonical_test,y)
    comparison=combine(y,base,real,conditional,tt,new)
    gains=comparison["fold_gain"]
    qualified=(comparison["gain"]>=MIN_GAIN and comparison["positive_folds"]>=2
               and min(gains)>-.00005)
    s.update({"metrics":m,"comparison":comparison,
              "stage":"candidate_ready_unsubmitted" if qualified else "skipped_no_cv_gain",
              "source_artifacts":{**files,"v294":teacher_artifact,
                "v2912":{"notebook":s["kernel"],
                         "sha256":hashlib.sha256(only(raw,"v2912_oof.npz").read_bytes()).hexdigest()}}})
    if qualified:
        test_base=.35*tests["v25"]+.65*tests["v28"]
        real_test=audit.align(archive["v296"]["pred_realmlp"],
                              archive["v296"]["test_id"],canonical_test,"RealMLP test")
        conditional_test=audit.align(archive["v297"]["test_conditional"],
                                     archive["v297"]["test_id"],canonical_test,"Conditional test")
        newer_test=audit.align(test_pred,test_ids,canonical_test,"V2912 test")
        blend=(.50*audit.rank(test_base)+.20*audit.rank(real_test)
               +.15*audit.rank(conditional_test)+.05*audit.rank(teacher_test)
               +.10*audit.rank(newer_test))
        s["candidate_hash"]=hashlib.sha256(blend.astype("<f8").tobytes()).hexdigest()
    record_experiment(LEDGER,{"competition":SLUG,"candidate":"v2912_segment_contrast_cpu",
       "validation":VALIDATION,"seed":20261018,
       "parameters":{"algorithm":"CatBoostCPU750D6","new_features":"native_catboost_ordered_ctr",
                     "weights":comparison["fixed_weights"]},
       "status":"completed" if qualified else "skipped_no_cv_gain",
       "completed_at":now().isoformat(),"cv":comparison["candidate_auc"],
       "cpu_seconds":m.get("elapsed_seconds"),"gpu_seconds":0,
       "kernel_sources":[s.get("kernel")]})
    return s

def run(recover=False):
    old=read(STATE)
    if old.get("stage") in TERMINAL:
        print("V2912 terminal; never launch duplicate notebook");return old
    eligible()
    s=old if old else {"competition":SLUG,"stage":"preflight",
                       "kernel":None,"submitted":False}
    if not s.get("kernel"):
        if recover:
            print("No existing CPU Notebook; recovery must never create a kernel");return s
        # Authoritative preflight: a previously created canonical ID must NEVER
        # be pushed again, even if GitHub status persistence was interrupted.
        probe=kaggle(["kaggle","kernels","status",KERNEL],timeout=60,check=False)
        if probe.returncode==0:
            s.update({"kernel":KERNEL,"stage":"kernel_queued",
                      "preexisting_kernel_recovered":True})
            persist(s)
        else:
            reason=(probe.stderr or probe.stdout or "").lower()
            if not any(mark in reason for mark in ("404","not found","not_found")):
                s.update({"stage":"kernel_access_error",
                          "error":"Cannot establish canonical notebook absence; no push allowed"})
                persist(s)
                return s
    if not s.get("kernel"):
        WORKDIR.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/"kaggle_agent/v2912_native_cat_cpu_train.py",WORKDIR/"main.py")
        write(WORKDIR/"kernel-metadata.json",{
            "id":KERNEL,"title":"kas-v2912-native-cat-cpu-s6e10",
            "code_file":"main.py","language":"python","kernel_type":"script",
            "is_private":True,"enable_gpu":False,"enable_internet":False,
            "dataset_sources":[],"competition_sources":[SLUG],
            "kernel_sources":[],"model_sources":[]})
        s.update({"stage":"pushing_kernel","kernel":KERNEL})
        persist(s)
        response=kaggle(["kaggle","kernels","push","-p",str(WORKDIR),"-t","3600"],
                        timeout=300,check=False)
        if response.returncode:
            s.update({"stage":"push_failed","error":(response.stderr or response.stdout)[-1200:]})
            persist(s);return s
        match=re.search(r"https?://(?:www\.)?kaggle\.com/code/([A-Za-z0-9_-]+)/([a-z0-9-]+)",
                        (response.stdout or "")+"\n"+(response.stderr or ""))
        if not match or match.group(1).lower()!="mjw3177109":
            s.update({"stage":"kernel_access_error","error":"No owner-authenticated Kaggle canonical URL"})
            persist(s);return s
        s["kernel"]="/".join(match.groups());persist(s)
    handle=s.get("kernel")
    if not isinstance(handle,str) or not handle.startswith("mjw3177109/"):
        raise RuntimeError("Invalid existing CPU kernel owner")
    stage,message=poll_kernel(handle,90 if recover else 300,20)
    s["stage"]=stage;s["kernel_state"]=stage
    if stage in ("kernel_running","kernel_queued","kernel_access_error"):
        s["kernel_message"]=message[-800:];persist(s);return s
    raw=OUTDIR/"raw";raw.mkdir(parents=True,exist_ok=True)
    outcome=kaggle(["kaggle","kernels","output",handle,"-p",str(raw),"-o","-q"],
                   timeout=600,check=False)
    if stage!="kernel_complete" or outcome.returncode:
        s["stage"]="kernel_failed" if stage!="kernel_complete" else "output_failed"
        errors=list(raw.rglob("v2912_error.json"))
        s["error"]=errors[0].read_text(encoding="utf-8")[-2200:] if errors else (
            outcome.stderr or message or outcome.stdout)[-2000:]
        persist(s);return s
    try:
        verify_outputs(raw,s)
    except Exception as exc:
        s.update({"stage":"validation_mismatch","error":str(exc)[-2200:]})
        persist(s)
        raise
    persist(s)
    return s

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--recover-only",action="store_true")
    args=parser.parse_args()
    result=run(recover=args.recover_only)
    print(json.dumps({"stage":result.get("stage"),"kernel":result.get("kernel"),
                      "comparison":result.get("comparison"),
                      "error":result.get("error")},ensure_ascii=False,indent=2))
