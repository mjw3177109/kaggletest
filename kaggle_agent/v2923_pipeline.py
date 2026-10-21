"""V2.9.23 LightGBM with preregistered 2x Personal Travel sample weighting; no submission.

All outputs are reconstructed against frozen V2915 OOF/test predictions.
Only one pre-registered 85/15 rank blend is evaluated for promotion.
A new Kaggle Notebook can only be pushed once after owner-based deduplication.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import v298_fusion as audit
import v299_teacher_fusion as teacher
import v2913_pipeline as v13
import v2914_pipeline as v14
import v2915_pipeline as v15
from v291_ledger import read, write, record_experiment
from v293_blend_math import VALIDATION, N_TRAIN, N_TEST, verify_sources
from v23_tpu_pipeline import poll_kernel

ROOT=audit.ROOT
SLUG=audit.SLUG
KERNEL="mjw3177109/kas-v2923-personal-weight2-lightgbm-cpu-s6e10"
STATE=ROOT/"state/v2_9_23_status.json"
REPORT=ROOT/"reports/v2_9_23_personal_weight_lgbm.md"
WORKDIR=ROOT/"competitions"/SLUG/"kaggle_cpu_v2923_personal_weight"
OUTDIR=ROOT/"competitions"/SLUG/"results_v2923_personal_weight"
LEDGER=ROOT/"state/v2_9_experiments.json"
BASELINE=.9612850399332336
MIN_GAIN=.00005
WEIGHTS={"v2915_rank":.85, "v2923_rank":.15}
MODEL_SEED=20261030
TERMINAL={"candidate_ready_unsubmitted","skipped_no_cv_gain",
          "kernel_failed","output_failed","validation_mismatch",
          "push_failed","kernel_access_error"}

def now():
    return datetime.now(timezone.utc).isoformat()

def cmd(parts, timeout=180, check=True):
    p=subprocess.run(parts,cwd=ROOT,capture_output=True,text=True,timeout=timeout)
    if check and p.returncode:
        raise RuntimeError("Kaggle CLI failed: "+(p.stderr or p.stdout)[-1500:])
    return p

def preflight():
    audit.check_competition()
    prior=read(v15.STATE)
    submitted=read(ROOT/"state/v2_9_15_submission.json")
    rank=read(ROOT/"state/score_rank_latest.json")
    if (prior.get("stage")!="candidate_ready_unsubmitted"
        or prior.get("submitted") is not False
        or (prior.get("metrics") or {}).get("stage")!="three_folds_complete"
        or (prior.get("metrics") or {}).get("gpu_verified") is not True
        or abs(float((prior.get("comparison") or {}).get("candidate_auc",0))-BASELINE)>.000002
        or len(prior.get("source_artifacts") or {})!=8
        or len(str(prior.get("candidate_test_hash") or ""))!=64
        or submitted.get("stage")!="complete" or submitted.get("submitted") is not True
        or submitted.get("submission_attempted") is not True
        or str((submitted.get("submission") or {}).get("ref"))!="57446286"
        or (submitted.get("submission") or {}).get("status")!="SubmissionStatus.COMPLETE"
        or abs(float(submitted.get("official_lb",0))-.96099)>1e-9
        or submitted.get("prediction_hash")!=prior.get("candidate_test_hash")
        or str((submitted.get("submission") or {}).get("description"))!=submitted.get("submission_note")
        or str((rank.get("score") or {}).get("submission_id"))!="57446286"
        or abs(float((rank.get("score") or {}).get("public_score",0))-.96099)>1e-9):
        raise RuntimeError("V2923 cannot authenticate completed V2915 official champion/CV/SHA")
    v16=read(ROOT/"state/v2_9_16_status.json")
    prev_result=v16.get("comparison") or {}
    if (v16.get("competition")!=SLUG or v16.get("stage")!="skipped_no_cv_gain"
        or v16.get("submitted") is not False
        or (v16.get("metrics") or {}).get("stage")!="three_folds_complete"
        or abs(float(prev_result.get("candidate_auc",0))-.9613243650285448)>.000002
        or float(prev_result.get("gain",1))>=.00005):
        raise RuntimeError("V2923 needs authentic terminal V2916 failed-promotion evidence")
    v21=read(ROOT/"state/v2_9_21_status.json")
    c21=v21.get("comparison") or {}
    if (v21.get("stage")!="skipped_no_cv_gain"
        or v21.get("submitted") is not False
        or abs(float(c21.get("candidate_auc",0))-.9612213583225888)>.000002
        or float(c21.get("gain",1))>=0):
        raise RuntimeError("V2923 requires authenticated terminal negative CatBoost OOF")
    diagnostic=ROOT/"reports/v2_9_22_subgroup_stability.md"
    if not diagnostic.exists():
        raise RuntimeError("V2923 requires completed common OOF subgroup diagnostic")
    diag_text=diagnostic.read_text(encoding="utf-8")
    for token in ("V2915 RealMLP champion","Personal Travel","0.830127","V2921 CatBoost"):
        if token not in diag_text:
            raise RuntimeError("V2923 subgroup diagnostic missing "+token)
    return prior

def persist(s):
    s["updated_at"]=now()
    write(STATE,s)
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    m=s.get("metrics") or {}
    c=s.get("comparison") or {}
    REPORT.write_text(
        "# V2.9.23 Personal-Travel-weighted LightGBM CPU threefold audit\n\n"
        +f"- UTC: {s['updated_at']}\n"
        +f"- Stage: {s.get('stage')}\n"
        +f"- Notebook: https://www.kaggle.com/code/{s.get('kernel') or KERNEL}\n"
        +f"- Frozen V2915 OOF baseline: {BASELINE}\n"
        +f"- Verified V2915 official Public LB: 0.96099 (submission 57446286)\n"
        +f"- New independent 3fold OOF AUC: {m.get('full_oof_auc','pending')}\n"
        +f"- Frozen 85/15 blend 3fold AUC: {c.get('candidate_auc','pending')}\n"
        +f"- Paired fold deltas: {c.get('fold_gain','pending')}\n"
        +f"- Promotion rule: gain >= {MIN_GAIN}, >=2 positive folds, worst fold > -0.00005\n"
        +"- Kaggle competition submission: NEVER performed by this evaluator\n"
        +(f"- Error: {s['error']}\n" if s.get("error") else ""),
        encoding="utf-8")

def unique(folder,name):
    matches=list(Path(folder).rglob(name))
    if len(matches)!=1:
        raise ValueError(f"Expected exactly one {name}; found {len(matches)}")
    return matches[0]

def existing_or_absent():
    status=cmd(["kaggle","kernels","status",KERNEL],timeout=90,check=False)
    if status.returncode==0:
        return True
    mine=cmd(["kaggle","kernels","list","--mine","--search",
              KERNEL.split("/",1)[1],"--page-size","50","-v"],
             timeout=120,check=False)
    if mine.returncode:
        raise RuntimeError("Authenticated owner list unavailable: no Notebook push")
    if KERNEL.lower() in mine.stdout.lower():
        return True
    if mine.stdout.strip().lower()!="not found":
        raise RuntimeError("Ambiguous owner listing: no Notebook push")
    witness=cmd(["kaggle","kernels","status",v15.KERNEL],timeout=90,check=False)
    if witness.returncode or "KernelWorkerStatus." not in witness.stdout:
        raise RuntimeError("Known owner Notebook control failed: no Notebook push")
    return False

def validate_metrics(m):
    if (m.get("competition") != SLUG
        or m.get("stage") != "three_folds_complete"
        or m.get("model") != "LightGBM personal-travel-weight2 route-profile TE23 3fold seeded"
        or m.get("validation") != VALIDATION
        or m.get("folds") != 3 or len(m.get("fold_results") or []) != 3
        or m.get("train_rows") != N_TRAIN or m.get("test_rows") != N_TEST
        or m.get("model_seed") != MODEL_SEED
        or m.get("feature_family") != "exact_distance_profile_te23_personal_weight2_v1"
        or m.get("route_profile_feature_count") != 12
        or m.get("target_encoding_columns") != 23
        or m.get("inner_target_folds") != 5
        or m.get("sample_weight_policy") != {"Personal Travel":2.0,"all_other_rows":1.0}
        or m.get("weighting_source") != "Type of Travel feature only; no validation labels"
        or m.get("cpu_only") is not True or m.get("gpu_verified") is not False
        or m.get("data_source") != "competition_train_only"
        or m.get("external_labels") is not False
        or m.get("submitted") is not False
        or m.get("leakage_safe_inner_cross_fit") is not True
        or m.get("outer_validation_used_in_training") is not False):
        raise ValueError("V2923 CPU/route-profile/TE23/3fold provenance mismatch")


def read_new(folder):
    file=unique(folder,"v2923_personal_weight_lgbm_oof.npz")
    keys=("train_id","target","oof_personal_weight_lgbm",
          "test_id","pred_personal_weight_lgbm")
    with np.load(file,allow_pickle=False) as f:
        if not set(keys).issubset(f.files):
            raise ValueError("V2923 missing canonical OOF/test arrays")
        return file,{k:np.asarray(f[k]).copy() for k in keys}

def blend(y,prior,new):
    if not (len(y)==len(prior)==len(new)):
        raise ValueError("V2923 baseline/new OOF lengths differ")
    if abs(float(roc_auc_score(y,prior))-BASELINE)>.000002:
        raise ValueError("V2915 frozen baseline OOF AUC did not reproduce")
    result=.85*audit.rank(prior)+.15*audit.rank(new)
    score=float(roc_auc_score(y,result))
    folds=[]
    for _,idx in StratifiedKFold(3,shuffle=True,random_state=20261018).split(
            np.zeros(len(y),dtype=np.uint8),y):
        folds.append(float(roc_auc_score(y[idx],result[idx])-
                           roc_auc_score(y[idx],prior[idx])))
    return {"baseline_auc":float(roc_auc_score(y,prior)),
            "standalone_auc":float(roc_auc_score(y,new)),
            "candidate_auc":score,"gain":score-BASELINE,
            "fold_gain":folds,"positive_folds":sum(x>0 for x in folds),
            "fixed_weights":dict(WEIGHTS)}

def validate(raw,s):
    metrics=json.loads(unique(raw,"v2923_metrics.json").read_text(encoding="utf-8"))
    validate_metrics(metrics)
    new_file,new=read_new(raw)
    source=preflight()
    older={k:read(ROOT/p) for k,p in {
       "v25":"state/v2_5_status.json","v28":"state/v2_8_status.json",
       "v296":"state/v2_9_6_status.json","v297":"state/v2_9_7_status.json"}.items()}
    audit.check_provenance(older)
    with tempfile.TemporaryDirectory(prefix="kas_v2923_verify_") as temporary:
        train,test,data,files=audit.download_inputs(temporary,older)
        train_id,test_id,y=audit.validate_official(train,test)
        replay=audit.evaluate(train,test,data,older,audit_legacy_order=True)
        if abs(replay["candidate_cv"]-.9610183558267038)>.000002:
            raise ValueError("V298 archived benchmark changed")
        t,t_test,t_meta=teacher.get_teacher(temporary,
              read(ROOT/"state/v2_9_4_status.json"),train_id,test_id,y)
        files["v294"]=t_meta
        outputs={}
        for version,kernel in (("v2913",v13.KERNEL),("v2914",v14.KERNEL),
                               ("v2915",v15.KERNEL)):
            folder=Path(temporary)/version
            folder.mkdir(parents=True,exist_ok=True)
            audit.checked_command(["kaggle","kernels","output",kernel,
                                   "-p",str(folder),"-o","-q"])
            if version=="v2915":
                file,arrays=v15.read_new(folder)
                train_key,test_key="oof_realmlp_5epoch","pred_realmlp_5epoch"
            else:
                file,arrays,train_key,test_key=v14.read_oof(folder,version)
            files[version]={"notebook":kernel,"sha256":audit.sha256(file)}
            outputs[version]=v14.align_new(arrays,train_key,test_key,
                                          train_id,test_id,y,version)
        prior_files=source["source_artifacts"]
        if set(files)!=set(prior_files):
            raise ValueError("V2915 archived source set changed")
        for key,fields in prior_files.items():
            for field,value in fields.items():
                if files[key].get(field)!=value:
                    raise ValueError("Historical Kaggle source SHA changed: "+key+"/"+field)
        files["v2923"]={"notebook":s["kernel"],"sha256":audit.sha256(new_file)}
        new_oof,new_test=v14.align_new(
            new,"oof_personal_weight_lgbm","pred_personal_weight_lgbm",
            train_id,test_id,y,"V2923")
        v2913_oof,v2913_test=outputs["v2913"]
        v2914_oof,v2914_test=outputs["v2914"]
        v2915_oof,v2915_test=outputs["v2915"]
        for version,pred,expected in (
            ("v2913",v2913_oof,read(v13.STATE)["metrics"]["full_oof_auc"]),
            ("v2914",v2914_oof,read(v14.STATE)["metrics"]["full_oof_auc"]),
            ("v2915",v2915_oof,source["metrics"]["full_oof_auc"]),
            ("v2923",new_oof,metrics["full_oof_auc"])):
            if abs(float(roc_auc_score(y,pred))-float(expected))>.000002:
                raise ValueError(version+" full OOF AUC reproduction failed")
        v13.evaluate_fold_safety(y,new_oof,metrics)
        _,old_oof,old_test=verify_sources(
            data["v25"],data["v28"],older["v25"],older["v28"],
            data["v25_submission"],data["v28_submission"])
        prior_history=.35*audit.rank(old_oof["v25"])+.65*audit.rank(old_oof["v28"])
        real=audit.align(data["v296"]["oof_realmlp"],
                         data["v296"]["train_id"],train_id,"V296")
        conditional=audit.align(data["v297"]["oof_conditional"],
                                data["v297"]["train_id"],train_id,"V297")
        historical=teacher.fuse_fixed(y,prior_history,real,conditional,t,
                                      audit.EXPECTED_V296)
        if abs(historical["candidate_cv"]-.9610561936366048)>.000002:
            raise ValueError("V299 historical OOF changed")
        v299=(.60*audit.rank(prior_history)+.20*audit.rank(real)+
              .15*audit.rank(conditional)+.05*audit.rank(t))
        v2913=.85*audit.rank(v299)+.15*audit.rank(v2913_oof)
        v15_result=v15.evaluate_blend(y,v2913,v2914_oof,v2915_oof)
        if abs(v15_result["candidate_auc"]-BASELINE)>.000002:
            raise ValueError("V2915 official OOF baseline changed")
        v2915=(.80*audit.rank(v2913)+.10*audit.rank(v2914_oof)+
               .10*audit.rank(v2915_oof))
        result=blend(y,v2915,new_oof)
        promoted=(result["gain"]>=MIN_GAIN and result["positive_folds"]>=2
                  and min(result["fold_gain"])>-.00005)
        s.update({"metrics":metrics,"comparison":result,"source_artifacts":files,
                  "stage":"candidate_ready_unsubmitted" if promoted else "skipped_no_cv_gain",
                  "submitted":False})
        if promoted:
            old_test_blend=.35*old_test["v25"]+.65*old_test["v28"]
            real_test=audit.align(data["v296"]["pred_realmlp"],
                                 data["v296"]["test_id"],test_id,"V296 test")
            conditional_test=audit.align(data["v297"]["test_conditional"],
                                        data["v297"]["test_id"],test_id,"V297 test")
            v299_test=(.60*audit.rank(old_test_blend)+.20*audit.rank(real_test)+
                       .15*audit.rank(conditional_test)+.05*audit.rank(t_test))
            v2913_test=.85*audit.rank(v299_test)+.15*audit.rank(v2913_test)
            if hashlib.sha256(v2913_test.astype("<f8").tobytes()).hexdigest()!=read(v13.STATE).get("candidate_test_hash"):
                raise ValueError("V2913 official test hash differs")
            v2915_test=(.80*audit.rank(v2913_test)+.10*audit.rank(v2914_test)+
                        .10*audit.rank(v2915_test))
            digest=hashlib.sha256(v2915_test.astype("<f8").tobytes()).hexdigest()
            if digest!=source.get("candidate_test_hash"):
                raise ValueError("V2915 official test hash differs")
            if digest!=read(ROOT/"state/v2_9_15_submission.json").get("prediction_hash"):
                raise ValueError("V2915 committed reservation SHA differs")
            candidate=.85*audit.rank(v2915_test)+.15*audit.rank(new_test)
            if len(candidate)!=N_TEST or not np.isfinite(candidate).all():
                raise ValueError("V2923 candidate test vector invalid")
            s["candidate_test_hash"]=hashlib.sha256(
                candidate.astype("<f8").tobytes()).hexdigest()
    record_experiment(LEDGER,{"competition":SLUG,
        "candidate":"v2923_personal_travel_weight2_lightgbm",
        "validation":VALIDATION,"seed":20261018,
        "parameters":{"model_seed":MODEL_SEED,"model":"LightGBM_sample_weighted",
                      "feature_family":"exact_distance_profile_te23_personal_weight2_v1",
                      "n_estimators":900,"sample_weight_personal_travel":2.0,"inner_target_folds":5,
                      "blend":dict(WEIGHTS),"min_gain":MIN_GAIN},
        "status":"completed" if promoted else "skipped_no_cv_gain",
        "completed_at":now(),"cv":result["candidate_auc"],
        "cpu_seconds":metrics.get("runtime_seconds"),"kernel_sources":[KERNEL]})

def run(recover_only=False):
    previous=read(STATE)
    if previous.get("stage") in TERMINAL:
        print("V2923 terminal; never create another Notebook or submission")
        return previous
    preflight()
    s=previous or {"competition":SLUG,"stage":"preflight",
                   "kernel":None,"submitted":False}
    if not s.get("kernel"):
        if recover_only:
            print("Recovery forbids creating an unrecorded Notebook")
            return s
        try:
            exists=existing_or_absent()
        except Exception as exc:
            s.update({"stage":"kernel_access_error","error":str(exc)[-1000:]})
            persist(s);return s
        if exists:
            s.update({"stage":"kernel_queued","kernel":KERNEL,
                      "preexisting_kernel_recovered":True})
            persist(s)
    if not s.get("kernel"):
        WORKDIR.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/"kaggle_agent/v2923_personal_weight_lgbm_train.py",
                     WORKDIR/"main.py")
        write(WORKDIR/"kernel-metadata.json",{
            "id":KERNEL,"title":KERNEL.split("/",1)[1],"code_file":"main.py",
            "language":"python","kernel_type":"script","is_private":True,
            "enable_gpu":False,
            "enable_internet":True,"dataset_sources":[],
            "competition_sources":[SLUG],"kernel_sources":[],"model_sources":[]})
        s.update({"stage":"pushing_kernel","kernel":KERNEL})
        persist(s)
        push=cmd(["kaggle","kernels","push","-p",str(WORKDIR),"-t","3600"],timeout=300,check=False)
        if push.returncode:
            s.update({"stage":"push_failed","error":(push.stderr or push.stdout)[-1500:]})
            persist(s);return s
        match=re.search(
            r"https?://(?:www\.)?kaggle\.com/code/([A-Za-z0-9_-]+)/([a-z0-9-]+)",
            (push.stdout or "")+"\n"+(push.stderr or ""))
        if not match or match.group(1).lower()!="mjw3177109":
            s.update({"stage":"kernel_access_error",
                      "error":"Kaggle push did not return a canonical owned Notebook URL"})
            persist(s);return s
        s["kernel"]="/".join(match.groups())
        if s["kernel"]!=KERNEL:
            s["canonical_note"]="Kaggle changed slug; recover actual ID and never repush"
        persist(s)
    handle=s.get("kernel")
    if not isinstance(handle,str) or not handle.startswith("mjw3177109/"):
        raise RuntimeError("Untrusted Kaggle Notebook ownership")
    status,message=poll_kernel(handle,90 if recover_only else 300,20)
    s["kernel_state"]=status
    if status in ("kernel_queued","kernel_running","kernel_access_error"):
        s.update({"stage":status,"kernel_message":str(message)[-900:]})
        persist(s);return s
    raw=OUTDIR/"raw"
    raw.mkdir(parents=True,exist_ok=True)
    response=cmd(["kaggle","kernels","output",handle,"-p",str(raw),"-o","-q"],
                 timeout=600,check=False)
    if status!="kernel_complete" or response.returncode:
        err=list(raw.rglob("v2923_error.json"))
        s.update({"stage":"kernel_failed" if status!="kernel_complete" else "output_failed",
                  "error":err[0].read_text(encoding="utf-8")[-2000:] if err else
                          str(response.stderr or message)[-2000:]})
        persist(s);return s
    try:
        validate(raw,s)
    except Exception as exc:
        s.update({"stage":"validation_mismatch","error":str(exc)[-2000:]})
        persist(s)
        raise
    persist(s)
    return s

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--recover-only",action="store_true")
    args=parser.parse_args()
    result=run(recover_only=args.recover_only)
    print(json.dumps({"stage":result.get("stage"),"kernel":result.get("kernel"),
         "comparison":result.get("comparison"),"error":result.get("error")},
         ensure_ascii=False,indent=2))
