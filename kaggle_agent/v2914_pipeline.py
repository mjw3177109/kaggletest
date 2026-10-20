"""V2914 Dual-Representation Ens8: one T4 kernel + full-OOF incumbent audit.

A Kaggle Notebook may be created only after authenticated exact owner-ID
absence check. Once status contains a kernel ID, every run is recovery-only.
No Kaggle competition submission code is present in this file.
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

import v298_fusion as audit
import v299_teacher_fusion as teacher
import v2913_pipeline as champion
from v23_tpu_pipeline import poll_kernel
from v291_ledger import read,write,record_experiment
from v293_blend_math import N_TRAIN,N_TEST,VALIDATION,verify_sources

ROOT=audit.ROOT
SLUG=audit.SLUG
KERNEL="mjw3177109/kas-v2914-dual-numeric-realmlp-t4-s6e10"
STATE=ROOT/"state/v2_9_14_status.json"
REPORT=ROOT/"reports/v2_9_14_dual_realmlp.md"
WORKDIR=ROOT/"competitions"/SLUG/"kaggle_gpu_v2914"
OUTDIR=ROOT/"competitions"/SLUG/"results_v2914"
LEDGER=ROOT/"state/v2_9_experiments.json"
BASELINE=.96117635576833
MIN_GAIN=.00006
TERMINAL={"candidate_ready_unsubmitted","skipped_no_cv_gain","kernel_failed",
 "output_failed","validation_mismatch","push_failed","kernel_access_error"}

def now():
    return datetime.now(timezone.utc).isoformat()

def call(argv,timeout=250,check=True):
    p=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True,timeout=timeout)
    if p.returncode and check:
        raise RuntimeError("Kaggle command failed: "+(p.stderr or p.stdout)[-1000:])
    return p

def preflight():
    audit.check_competition()
    old=read(ROOT/"state/v2_9_13_status.json")
    scored=read(ROOT/"state/v2_9_13_submission.json")
    if (old.get("stage")!="candidate_ready_unsubmitted"
        or abs(float((old.get("comparison") or {}).get("candidate_auc",0))-BASELINE)>.000002
        or scored.get("stage")!="complete" or scored.get("submitted") is not True
        or abs(float(scored.get("official_lb",0))-.96083)>1e-9
        or str((scored.get("submission") or {}).get("ref"))!="57423879"):
        raise RuntimeError("V2913 authentic scored incumbent missing, stop V2914")

def persist(s):
    s["updated_at"]=now()
    write(STATE,s)
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    m=s.get("metrics") or {}
    c=s.get("comparison") or {}
    REPORT.write_text("# V2.9.14 dual numeric RealMLP Ens8\n\n"
       +f"- Updated UTC: {s['updated_at']}\n"
       +f"- Stage: {s.get('stage')}\n"
       +f"- Notebook: https://www.kaggle.com/code/{s.get('kernel') or KERNEL}\n"
       +f"- Independent 3fold CV: {m.get('full_oof_auc','pending')}\n"
       +f"- Official incumbent V2913 Public LB: 0.96083 (submission 57423879)\n"
       +f"- Verified V2913 full 3fold CV: {c.get('baseline_auc',BASELINE)}\n"
       +f"- Frozen 90/10 challenger CV: {c.get('candidate_auc','pending')}\n"
       +f"- Fold gains: {c.get('fold_gain','pending')}\n"
       +f"- No Kaggle competition submit performed here.\n"
       +(f"- Error: {s.get('error')}\n" if s.get("error") else ""),
       encoding="utf-8")

def unique(path,name):
    hits=list(Path(path).rglob(name))
    if len(hits)!=1:
        raise ValueError(f"Expected exactly one {name}; found {len(hits)}")
    return hits[0]

def existing_or_absent():
    status=call(["kaggle","kernels","status",KERNEL],timeout=75,check=False)
    if status.returncode==0:
        return True
    mine=call(["kaggle","kernels","list","--mine","--search",KERNEL.split("/")[1],
               "--page-size","50","-v"],timeout=90,check=False)
    if mine.returncode!=0:
        raise RuntimeError("Kaggle owner-owned Notebook list unavailable; refuse push")
    if KERNEL.lower() in mine.stdout.lower():
        return True
    if mine.stdout.strip().lower()!="not found":
        raise RuntimeError("Ambiguous Notebook existence; no push")
    existing=call(["kaggle","kernels","status",champion.KERNEL],timeout=75,check=False)
    if existing.returncode!=0 or "KernelWorkerStatus." not in existing.stdout:
        raise RuntimeError("Known original Notebook positive control failed; refuse push")
    return False

def metrics_check(metrics):
    hardware=metrics.get("hardware") or {}
    if (metrics.get("stage")!="three_folds_complete"
        or metrics.get("competition")!=SLUG
        or metrics.get("validation")!=VALIDATION
        or metrics.get("folds")!=3
        or len(metrics.get("fold_results") or [])!=3
        or metrics.get("train_rows")!=N_TRAIN
        or metrics.get("test_rows")!=N_TEST
        or not str(metrics.get("model","")).startswith("RealMLP-TD n_ens=8 dual-numeric")
        or metrics.get("leakage_safe_inner_cross_fit") is not True
        or metrics.get("outer_validation_used_in_training") is not False
        or metrics.get("gpu_verified") is not True
        or hardware.get("ok") is not True
        or not any("T4" in str(v) for v in hardware.get("devices",[]))):
        raise ValueError("V2914 threefold GPU / leakage / row metadata mismatch")

def read_oof(folder,prefix):
    path=unique(folder,prefix+"_realmlp_oof.npz")
    with np.load(path,allow_pickle=False) as data:
        required={"train_id","target","test_id"}
        if prefix=="v2913":
            required|={"oof_realmlp_ens8","pred_realmlp_ens8"}
            train_key,test_key="oof_realmlp_ens8","pred_realmlp_ens8"
        else:
            required|={"oof_dual_realmlp_ens8","pred_dual_realmlp_ens8"}
            train_key,test_key="oof_dual_realmlp_ens8","pred_dual_realmlp_ens8"
        if not required.issubset(data.files):
            raise ValueError("OOF archive lacks explicit train/test IDs and target")
        return path,{k:np.asarray(data[k]).copy() for k in required},train_key,test_key

def align_new(data,train_key,test_key,train_ids,test_ids,y,who):
    source_id=audit.ids(data["train_id"],len(y),who+" IDs")
    labels=pd.Series(data["target"],index=source_id).reindex(train_ids).to_numpy()
    if not np.array_equal(labels,y):
        raise ValueError(who+" target/ID alignment mismatch")
    oof=audit.align(data[train_key],source_id,train_ids,who+" OOF")
    test=audit.align(data[test_key],data["test_id"],test_ids,who+" test")
    return oof,test

def frozen_blend(y,v299,older,new):
    r=audit.rank
    current=.85*r(v299)+.15*r(older)
    incumbent_auc=float(roc_auc_score(y,current))
    if abs(incumbent_auc-BASELINE)>.000002:
        raise ValueError("V2913 official champion CV changed")
    candidate=.90*r(current)+.10*r(new)
    auc=float(roc_auc_score(y,candidate))
    fold_gain=[]
    for _,valid in StratifiedKFold(3,shuffle=True,random_state=20261018).split(
            np.zeros(len(y),dtype=np.uint8),y):
        fold_gain.append(float(roc_auc_score(y[valid],candidate[valid])-
                               roc_auc_score(y[valid],current[valid])))
    return {"baseline_auc":incumbent_auc,
        "standalone_auc":float(roc_auc_score(y,new)),
        "candidate_auc":auc,"gain":auc-incumbent_auc,
        "fold_gain":fold_gain,"positive_folds":sum(z>0 for z in fold_gain),
        "fixed_weights":{"v2913_rank":.90,"dual_ens8_rank":.10}}

def validate_outputs(raw,s):
    metrics=json.loads(unique(raw,"v2914_metrics.json").read_text(encoding="utf-8"))
    metrics_check(metrics)
    new_path,new_data,new_train_key,new_test_key=read_oof(raw,"v2914")
    baseline_status=read(ROOT/"state/v2_9_13_status.json")
    historical={key:read(ROOT/file) for key,file in {
        "v25":"state/v2_5_status.json","v28":"state/v2_8_status.json",
        "v296":"state/v2_9_6_status.json","v297":"state/v2_9_7_status.json"}.items()}
    audit.check_provenance(historical)
    with tempfile.TemporaryDirectory(prefix="kas_v2914_") as folder:
        train,test,source,proof=audit.download_inputs(folder,historical)
        train_id,test_id,y=audit.validate_official(train,test)
        z=audit.evaluate(train,test,source,historical,audit_legacy_order=True)
        if abs(z["candidate_cv"]-.9610183558267038)>.000002:
            raise ValueError("V298 source OOF CV changed")
        old_t,old_ttest,t_info=teacher.get_teacher(folder,
            read(ROOT/"state/v2_9_4_status.json"),train_id,test_id,y)
        proof["v294"]=t_info
        neural_folder=Path(folder)/"v2913"
        neural_folder.mkdir(parents=True,exist_ok=True)
        audit.checked_command(["kaggle","kernels","output",champion.KERNEL,
                        "-p",str(neural_folder),"-o","-q"])
        older_path,older_data,older_tr,older_test=read_oof(neural_folder,"v2913")
        proof["v2913"]={"notebook":champion.KERNEL,"sha256":audit.sha256(older_path)}
        proof["v2914"]={"notebook":s["kernel"],"sha256":audit.sha256(new_path)}
        expected=baseline_status.get("source_artifacts") or {}
        for key,record in expected.items():
            if key not in proof:
                raise ValueError("V2913 source artifact not loaded: "+key)
            for field,val in record.items():
                if proof[key].get(field)!=val:
                    raise ValueError("V2913 prior SHA mismatch: "+key+" "+field)
        former,former_test=align_new(older_data,older_tr,older_test,
                                    train_id,test_id,y,"V2913")
        if abs(roc_auc_score(y,former)-float(baseline_status["metrics"]["full_oof_auc"]))>.000002:
            raise ValueError("Prior RealMLP OOF has changed")
        newcomer,new_test=align_new(new_data,new_train_key,new_test_key,
                                    train_id,test_id,y,"V2914")
        if abs(roc_auc_score(y,newcomer)-float(metrics["full_oof_auc"]))>.000002:
            raise ValueError("New RealMLP OOF metric mismatch")
        champion.evaluate_fold_safety(y,newcomer,metrics)
        _,oofs,tests=verify_sources(source["v25"],source["v28"],
            historical["v25"],historical["v28"],
            source["v25_submission"],source["v28_submission"])
        v293=.35*audit.rank(oofs["v25"])+.65*audit.rank(oofs["v28"])
        v296=audit.align(source["v296"]["oof_realmlp"],
                         source["v296"]["train_id"],train_id,"V296")
        v297=audit.align(source["v297"]["oof_conditional"],
                         source["v297"]["train_id"],train_id,"V297")
        historical_comp=teacher.fuse_fixed(y,v293,v296,v297,old_t,audit.EXPECTED_V296)
        if abs(historical_comp["candidate_cv"]-.9610561936366048)>.000002:
            raise ValueError("V299 historical baseline changed")
        v299=.60*audit.rank(v293)+.20*audit.rank(v296)+.15*audit.rank(v297)+.05*audit.rank(old_t)
        result=frozen_blend(y,v299,former,newcomer)
        if result["candidate_auc"]<=result["baseline_auc"]:
            print("V2914 found no AUC gain; this is valid negative evidence.")
        previous=.35*tests["v25"]+.65*tests["v28"]
        test296=audit.align(source["v296"]["pred_realmlp"],
                            source["v296"]["test_id"],test_id,"V296 test")
        test297=audit.align(source["v297"]["test_conditional"],
                            source["v297"]["test_id"],test_id,"V297 test")
        v299_test=(.60*audit.rank(previous)+.20*audit.rank(test296)
                   +.15*audit.rank(test297)+.05*audit.rank(old_ttest))
        v2913_test=.85*audit.rank(v299_test)+.15*audit.rank(former_test)
        champion_digest=hashlib.sha256(v2913_test.astype("<f8").tobytes()).hexdigest()
        if champion_digest!=baseline_status.get("candidate_test_hash"):
            raise ValueError("Official V2913 candidate SHA changed")
        qualified=(result["gain"]>=MIN_GAIN
                   and result["positive_folds"]>=2
                   and min(result["fold_gain"])>-.00005)
        s.update({"metrics":metrics,"comparison":result,
                  "stage":"candidate_ready_unsubmitted" if qualified else "skipped_no_cv_gain",
                  "source_artifacts":proof})
        if qualified:
            mix=.90*audit.rank(v2913_test)+.10*audit.rank(new_test)
            s["candidate_test_hash"]=hashlib.sha256(mix.astype("<f8").tobytes()).hexdigest()
    record_experiment(LEDGER,{"competition":SLUG,"candidate":"v2914_dual_realmlp_ens8",
        "validation":VALIDATION,"seed":20261018,
        "parameters":{"model":"RealMLP Ens8 dual all numeric","weights":{ "v2913":.90,"v2914":.10}},
        "status":"completed" if qualified else "skipped_no_cv_gain",
        "completed_at":now(),"cv":result["candidate_auc"],
        "gpu_seconds":metrics.get("runtime_seconds"),"kernel_sources":[KERNEL]})

def run(recover_only=False):
    prior=read(STATE)
    if prior.get("stage") in TERMINAL:
        print("V2914 already terminal; never run duplicate Kaggle notebook")
        return prior
    preflight()
    s=prior or {"competition":SLUG,"stage":"preflight","kernel":None,"submitted":False}
    if not s.get("kernel"):
        if recover_only:
            print("Recovery requires existing kernel; refuses create")
            return s
        try:
            exists=existing_or_absent()
        except RuntimeError as exc:
            s.update({"stage":"kernel_access_error","error":str(exc)[-800:]})
            persist(s);return s
        if exists:
            s.update({"stage":"kernel_queued","kernel":KERNEL,
                      "preexisting_kernel_recovered":True})
            persist(s)
    if not s.get("kernel"):
        WORKDIR.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/"kaggle_agent/v2914_dual_realmlp_train.py",
                     WORKDIR/"main.py")
        write(WORKDIR/"kernel-metadata.json",{
            "id":KERNEL,"title":KERNEL.split("/",1)[1],
            "code_file":"main.py","language":"python","kernel_type":"script",
            "is_private":True,"enable_gpu":True,"machine_shape":"NvidiaTeslaT4",
            "enable_internet":True,"dataset_sources":[],
            "competition_sources":[SLUG],"kernel_sources":[],"model_sources":[]})
        s.update({"stage":"pushing_kernel","kernel":KERNEL})
        persist(s)
        pushed=call(["kaggle","kernels","push","-p",str(WORKDIR),"-t","3600",
                     "--accelerator","NvidiaTeslaT4"],timeout=300,check=False)
        if pushed.returncode:
            s.update({"stage":"push_failed",
                      "error":(pushed.stderr or pushed.stdout)[-1200:]})
            persist(s);return s
        match=re.search(r"https?://(?:www\.)?kaggle\.com/code/([A-Za-z0-9_-]+)/([a-z0-9-]+)",
                        (pushed.stdout or "")+"\n"+(pushed.stderr or ""))
        if not match or match.group(1).lower()!="mjw3177109":
            s.update({"stage":"kernel_access_error",
                      "error":"No owner-authenticated Notebook URL from Kaggle"})
            persist(s);return s
        s["kernel"]="/".join(match.groups())
        if s["kernel"]!=KERNEL:
            s["requested_id"]=KERNEL
            s["canonical_note"]="Kaggle canonical slug changed: recover actual ID only"
        persist(s)
    handle=s.get("kernel")
    if not isinstance(handle,str) or not handle.startswith("mjw3177109/"):
        raise RuntimeError("Untrusted Kaggle kernel ID")
    stage,message=poll_kernel(handle,90 if recover_only else 300,20)
    s["kernel_state"]=stage
    if stage in ("kernel_queued","kernel_running","kernel_access_error"):
        s.update({"stage":stage,"kernel_message":str(message)[-800:]})
        persist(s);return s
    target=OUTDIR/"raw";target.mkdir(parents=True,exist_ok=True)
    outcome=call(["kaggle","kernels","output",handle,"-p",str(target),"-o","-q"],
                 timeout=600,check=False)
    if stage!="kernel_complete" or outcome.returncode:
        failures=list(target.rglob("v2914_error.json"))
        s.update({"stage":"kernel_failed" if stage!="kernel_complete" else "output_failed",
                  "error":failures[0].read_text()[-2200:] if failures
                          else str(outcome.stderr or message)[-2200:]})
        persist(s);return s
    try:
        validate_outputs(target,s)
    except Exception as exc:
        s.update({"stage":"validation_mismatch","error":str(exc)[-2200:]})
        persist(s)
        raise
    persist(s)
    return s

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--recover-only",action="store_true")
    args=p.parse_args()
    result=run(recover_only=args.recover_only)
    print(json.dumps({"stage":result.get("stage"),"kernel":result.get("kernel"),
        "comparison":result.get("comparison"),"error":result.get("error")},
        indent=2,ensure_ascii=False))
