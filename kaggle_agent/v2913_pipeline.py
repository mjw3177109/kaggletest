"""V2.9.13: one T4 RealMLP ensemble8 Kaggle kernel, full 3fold CV, NO submit.

A notebook is created only after authenticated owner-based dedupe. Once its
canonical ID is stored, every later action is --recover-only.
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

from v23_tpu_pipeline import poll_kernel
from v291_ledger import read,write,record_experiment
from v293_blend_math import VALIDATION,N_TRAIN,N_TEST,verify_sources
import v298_fusion as audit
import v299_teacher_fusion as teacher

ROOT=audit.ROOT
SLUG=audit.SLUG
KERNEL="mjw3177109/kas-v2913-realmlp-ens8-t4-s6e10"
STATE=ROOT/"state/v2_9_13_status.json"
REPORT=ROOT/"reports/v2_9_13_realmlp_ens8.md"
WORKDIR=ROOT/"competitions"/SLUG/"kaggle_gpu_v2913"
OUTDIR=ROOT/"competitions"/SLUG/"results_v2913"
LEDGER=ROOT/"state/v2_9_experiments.json"
BASELINE=0.9610561936366048
MIN_GAIN=.00006
TERMINAL={"candidate_ready_unsubmitted","skipped_no_cv_gain","kernel_failed",
          "output_failed","validation_mismatch","push_failed","kernel_access_error"}

def now():
    return datetime.now(timezone.utc).isoformat()

def cmd(args,timeout=240,check=True):
    r=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=timeout)
    if check and r.returncode:
        raise RuntimeError("Kaggle failed: "+(r.stderr or r.stdout)[-1400:])
    return r

def preflight():
    audit.check_competition()
    prior=read(ROOT/"state/v2_9_12_status.json")
    best=read(ROOT/"state/v2_9_9_status.json")
    official=read(ROOT/"state/v2_9_9_submission.json")
    if (prior.get("stage") not in ("skipped_no_cv_gain","candidate_ready_unsubmitted")
        or best.get("stage")!="candidate_ready_unsubmitted"
        or abs(float(best.get("candidate_cv",0))-BASELINE)>.000002
        or official.get("stage")!="complete" or official.get("submitted") is not True
        or abs(float(official.get("official_lb",0))-.96070)>.0000001
        or str((official.get("submission") or {}).get("ref"))!="57421208"):
        raise RuntimeError("V299 verified champion or V2912 terminal state missing")

def persist(s):
    s["updated_at"]=now()
    write(STATE,s)
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    m=s.get("metrics") or {}
    cp=s.get("comparison") or {}
    REPORT.write_text("# V2.9.13 RealMLP ensemble8 T4 audit\n\n"+
        "- UTC: "+s["updated_at"]+"\n"+
        "- Stage: "+str(s.get("stage"))+"\n"+
        "- Kaggle Notebook: https://www.kaggle.com/code/"+str(s.get("kernel") or KERNEL)+"\n"+
        "- Complete 3fold new-model CV: "+str(m.get("full_oof_auc","pending"))+"\n"+
        "- V299 official submission #57421208 Public LB: 0.96070\n"+
        "- Pre-registered V299 OOF baseline: "+str(cp.get("baseline_auc",BASELINE))+"\n"+
        "- Fixed ensemble CV: "+str(cp.get("candidate_auc","pending"))+"\n"+
        "- Paired fold deltas: "+str(cp.get("fold_gain","pending"))+"\n"+
        "- Submitted to Kaggle leaderboard: NO (not supported by this workflow)\n"+
        ("- Error: "+str(s["error"])[-2000:]+"\n" if s.get("error") else ""),
        encoding="utf-8")

def only(folder,name):
    files=list(Path(folder).rglob(name))
    if len(files)!=1:
        raise ValueError(f"Expected unique {name}, got {len(files)}")
    return files[0]

def existing_or_absent():
    attempt=cmd(["kaggle","kernels","status",KERNEL],timeout=90,check=False)
    if attempt.returncode==0:
        return True
    slug=KERNEL.split("/",1)[1]
    mine=cmd(["kaggle","kernels","list","--mine","--search",slug,
              "--page-size","50","-v"],timeout=120,check=False)
    if mine.returncode!=0:
        raise RuntimeError("Cannot verify owned Kaggle Notebook list; refuse duplicate push")
    if KERNEL.lower() in mine.stdout.lower():
        return True
    if mine.stdout.strip().lower()!="not found":
        raise RuntimeError("Ambiguous Notebook-list result; refuse push")
    # Positive control verifies authorization for an existing owned kernel.
    known=cmd(["kaggle","kernels","status",
               "mjw3177109/kas-v2912-native-cat-cpu-s6e10"],timeout=90,check=False)
    if known.returncode!=0 or "KernelWorkerStatus." not in known.stdout:
        raise RuntimeError("Existing-kernel authenticated positive control failed")
    return False

def evaluate_fold_safety(y,pred,metrics):
    splitter=StratifiedKFold(3,shuffle=True,random_state=20261018)
    actual=[]
    for fold,(_,iv) in enumerate(splitter.split(np.zeros(len(y),dtype=np.uint8),y)):
        auc=float(roc_auc_score(y[iv],pred[iv]))
        if abs(auc-float(metrics["fold_results"][fold]["auc"]))>.000002:
            raise ValueError("V2913 folded model AUC discrepancy")
        actual.append(auc)
    return actual

def compare(y,historical,real,conditional,teacher_oof,new):
    r=audit.rank
    old=.60*r(historical)+.20*r(real)+.15*r(conditional)+.05*r(teacher_oof)
    old_score=float(roc_auc_score(y,old))
    if abs(old_score-BASELINE)>.000002:
        raise ValueError("Current official V299 full threefold not reproduced")
    new_score=.85*r(old)+.15*r(new)
    candidate=float(roc_auc_score(y,new_score))
    fold_gain=[]
    for _,iv in StratifiedKFold(3,shuffle=True,random_state=20261018).split(
            np.zeros(len(y),dtype=np.uint8),y):
        fold_gain.append(float(roc_auc_score(y[iv],new_score[iv])-
                               roc_auc_score(y[iv],old[iv])))
    return {"baseline_auc":old_score,"standalone_auc":float(roc_auc_score(y,new)),
        "candidate_auc":candidate,"gain":candidate-old_score,"fold_gain":fold_gain,
        "positive_folds":sum(g>0 for g in fold_gain),
        "fixed_weights":{"v299_rank":.85,"realmlp_ens8_rank":.15}}

def validate_output(raw,s):
    metrics=json.loads(only(raw,"v2913_metrics.json").read_text(encoding="utf-8"))
    h=metrics.get("hardware") or {}
    if (metrics.get("competition")!=SLUG or metrics.get("stage")!="three_folds_complete"
        or metrics.get("validation")!=VALIDATION or metrics.get("folds")!=3
        or len(metrics.get("fold_results") or [])!=3
        or metrics.get("train_rows")!=N_TRAIN or metrics.get("test_rows")!=N_TEST
        or metrics.get("leakage_safe_inner_cross_fit") is not True
        or metrics.get("outer_validation_used_in_training") is not False
        or metrics.get("gpu_verified") is not True or h.get("ok") is not True
        or not any("T4" in str(device) for device in h.get("devices",[]))
        or not str(metrics.get("model","")).startswith("RealMLP-TD n_ens=8")):
        raise ValueError("RealMLP ensemble8 GPU/CV provenance mismatch")
    file=only(raw,"v2913_realmlp_oof.npz")
    with np.load(file,allow_pickle=False) as f:
        required={"train_id","target","oof_realmlp_ens8","test_id","pred_realmlp_ens8"}
        if not required.issubset(f.files):
            raise ValueError("V2913 missing required OOF/test arrays")
        train_id=np.asarray(f["train_id"]).copy()
        target=np.asarray(f["target"]).copy()
        pred=np.asarray(f["oof_realmlp_ens8"]).copy()
        test_id=np.asarray(f["test_id"]).copy()
        test_pred=np.asarray(f["pred_realmlp_ens8"]).copy()
    states={key:read(ROOT/p) for key,p in {
          "v25":"state/v2_5_status.json","v28":"state/v2_8_status.json",
          "v296":"state/v2_9_6_status.json","v297":"state/v2_9_7_status.json"}.items()}
    audit.check_provenance(states)
    with tempfile.TemporaryDirectory(prefix="kas_v2913_") as work:
        train,test,archive,source=audit.download_inputs(work,states)
        canonical_ids,canonical_test,y=audit.validate_official(train,test)
        replay=audit.evaluate(train,test,archive,states,audit_legacy_order=True)
        if abs(replay["candidate_cv"]-.9610183558267038)>.000002:
            raise ValueError("V298 replay unexpectedly changed")
        newcomer=audit.align(pred,train_id,canonical_ids,"V2913 OOF")
        yy=pd.Series(target,index=audit.ids(train_id,N_TRAIN,"v2913 ids")).reindex(canonical_ids).to_numpy()
        if not np.array_equal(yy,y):
            raise ValueError("V2913 canonical target order differs")
        if not np.array_equal(np.sort(test_id),np.sort(canonical_test)):
            raise ValueError("V2913 test ID set differs")
        if abs(float(roc_auc_score(y,newcomer))-float(metrics["full_oof_auc"]))>.000002:
            raise ValueError("New standalone OOF score differs")
        evaluate_fold_safety(y,newcomer,metrics)
        _,oofs,tests=verify_sources(archive["v25"],archive["v28"],
              states["v25"],states["v28"],archive["v25_submission"],
              archive["v28_submission"])
        history=.35*audit.rank(oofs["v25"])+.65*audit.rank(oofs["v28"])
        neural=audit.align(archive["v296"]["oof_realmlp"],
            archive["v296"]["train_id"],canonical_ids,"RealMLP original")
        condition=audit.align(archive["v297"]["oof_conditional"],
            archive["v297"]["train_id"],canonical_ids,"Conditional")
        tt,tt_test,tt_meta=teacher.get_teacher(work,
            read(ROOT/"state/v2_9_4_status.json"),canonical_ids,canonical_test,y)
        old_comp=teacher.fuse_fixed(y,history,neural,condition,tt,audit.EXPECTED_V296)
        if abs(old_comp["candidate_cv"]-BASELINE)>.000002:
            raise ValueError("V299 champion fold alignment changed")
        comp=compare(y,history,neural,condition,tt,newcomer)
        qualified=(comp["gain"]>=MIN_GAIN and comp["positive_folds"]>=2
                   and min(comp["fold_gain"])>-.00005)
        s.update({"metrics":metrics,"comparison":comp,
                  "stage":"candidate_ready_unsubmitted" if qualified else "skipped_no_cv_gain",
                  "source_artifacts":{**source,"v294":tt_meta,
                    "v2913":{"notebook":s["kernel"],"sha256":hashlib.sha256(file.read_bytes()).hexdigest()}}})
        if qualified:
            old_test=.35*tests["v25"]+.65*tests["v28"]
            neural_test=audit.align(archive["v296"]["pred_realmlp"],
                archive["v296"]["test_id"],canonical_test,"V296 test")
            condition_test=audit.align(archive["v297"]["test_conditional"],
                archive["v297"]["test_id"],canonical_test,"V297 test")
            new_test=audit.align(test_pred,test_id,canonical_test,"V2913 test")
            v299_test=(.60*audit.rank(old_test)+.20*audit.rank(neural_test)
                       +.15*audit.rank(condition_test)+.05*audit.rank(tt_test))
            v299_sha=hashlib.sha256(v299_test.astype("<f8").tobytes()).hexdigest()
            if v299_sha!=read(ROOT/"state/v2_9_9_status.json").get("candidate_test_hash"):
                raise ValueError("Official V299 candidate predictions changed")
            candidate_test=.85*audit.rank(v299_test)+.15*audit.rank(new_test)
            s["candidate_test_hash"]=hashlib.sha256(
                   candidate_test.astype("<f8").tobytes()).hexdigest()
    record_experiment(LEDGER,{"competition":SLUG,"candidate":"v2913_realmlp_ens8_threefold",
        "validation":VALIDATION,"seed":20261018,
        "parameters":{"n_ens":8,"n_epochs":3,"batch_size":256,
            "scheduler":"flat_anneal/cos_log_15","blend":{ "v299":.85,"realmlp":.15}},
        "status":"completed" if qualified else "skipped_no_cv_gain",
        "completed_at":now(),"cv":comp["candidate_auc"],
        "gpu_seconds":metrics.get("runtime_seconds"),"kernel_sources":[KERNEL]})

def run(recover_only=False):
    prior=read(STATE)
    if prior.get("stage") in TERMINAL:
        print("V2913 already terminal; never train or submit again")
        return prior
    preflight()
    s=prior or {"competition":SLUG,"stage":"preflight","kernel":None,"submitted":False}
    if not s.get("kernel"):
        if recover_only:
            print("Recovery refuses creating missing Kaggle Notebook")
            return s
        try:
            found=existing_or_absent()
        except RuntimeError as exc:
            s.update({"stage":"kernel_access_error","error":str(exc)[-800:]})
            persist(s)
            return s
        if found:
            s.update({"stage":"kernel_queued","kernel":KERNEL,
                      "preexisting_kernel_recovered":True})
            persist(s)
    if not s.get("kernel"):
        WORKDIR.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/"kaggle_agent/v2913_realmlp_ens8_train.py",
                     WORKDIR/"main.py")
        write(WORKDIR/"kernel-metadata.json",{
            "id":KERNEL,"title":KERNEL.split("/",1)[1],
            "code_file":"main.py","language":"python","kernel_type":"script",
            "is_private":True,"enable_gpu":True,
            "machine_shape":"NvidiaTeslaT4","enable_internet":True,
            "dataset_sources":[],"competition_sources":[SLUG],
            "kernel_sources":[],"model_sources":[]})
        s.update({"stage":"pushing_kernel","kernel":KERNEL})
        persist(s)
        push=cmd(["kaggle","kernels","push","-p",str(WORKDIR),
                  "-t","3600","--accelerator","NvidiaTeslaT4"],timeout=300,check=False)
        if push.returncode:
            s.update({"stage":"push_failed","error":(push.stderr or push.stdout)[-1300:]})
            persist(s)
            return s
        match=re.search(r"https?://(?:www\.)?kaggle\.com/code/([A-Za-z0-9_-]+)/([a-z0-9-]+)",
                        (push.stdout or "")+"\n"+(push.stderr or ""))
        if not match or match.group(1).lower()!="mjw3177109":
            s.update({"stage":"kernel_access_error",
                      "error":"Kaggle did not return an owner-authenticated Notebook URL"})
            persist(s);return s
        s["kernel"]="/".join(match.groups())
        if s["kernel"]!=KERNEL:
            s["requested_id"]=KERNEL
            s["canonical_note"]="Kaggle returned a distinct canonical slug; recover it, never repush"
        persist(s)
    handle=s["kernel"]
    if not isinstance(handle,str) or not handle.startswith("mjw3177109/"):
        raise RuntimeError("Kaggle owner mismatch")
    stage,message=poll_kernel(handle,90 if recover_only else 300,20)
    s["kernel_state"]=stage
    if stage in ("kernel_queued","kernel_running","kernel_access_error"):
        s["stage"]=stage;s["kernel_message"]=str(message)[-800:]
        persist(s)
        return s
    raw=OUTDIR/"raw";raw.mkdir(parents=True,exist_ok=True)
    out=cmd(["kaggle","kernels","output",handle,"-p",str(raw),"-o","-q"],timeout=600,check=False)
    if stage!="kernel_complete" or out.returncode:
        errs=list(raw.rglob("v2913_error.json"))
        s.update({"stage":"kernel_failed" if stage!="kernel_complete" else "output_failed",
                  "error":errs[0].read_text()[-2300:] if errs else str(out.stderr or message)[-2300:]})
        persist(s);return s
    try:
        validate_output(raw,s)
    except Exception as exc:
        s.update({"stage":"validation_mismatch","error":str(exc)[-2300:]})
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
           "comparison":result.get("comparison"),
           "error":result.get("error")},ensure_ascii=False,indent=2))
