"""V2932 fail-closed single-Notebook evaluator. Never submit competition predictions."""
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import v2924_pipeline as v24

ROOT=Path(__file__).resolve().parents[1]
SLUG="playground-series-s6e10"
BRANCH="feature/v29-autonomous-evolution"
KERNEL="mjw3177109/kas-v2932-attention-baseline37-t4-s6e10"
STATUS=ROOT/"state/v2_9_32_status.json"
REPORT=ROOT/"reports/v2_9_32_result.md"
WORKDIR=ROOT/"competitions"/SLUG/"kaggle_gpu_v2932_attention"
OUTDIR=ROOT/"competitions"/SLUG/"results_v2932_attention"
LEDGER=ROOT/"state/v2_9_experiments.json"
BASELINE=0.9612850399332336
FLOOR=0.96125635576833
MIN_GAIN=0.00005
SEED=20261018
MODEL_SEED=20261101
NEW_WEIGHT=0.15
FEATURE_FAMILY="v2915_original_37_features_with_3_inner5fold_te_no_new_features"
CANONICAL=["Age","Flight Distance","Inflight wifi service","Departure/Arrival time convenient","Ease of Online booking","Gate location","Food and drink","Online boarding","Seat comfort","Inflight entertainment","On-board service","Leg room service","Baggage handling","Checkin service","Cleanliness","Departure Delay in Minutes","Arrival Delay in Minutes","Gender","Customer Type","Type of Travel","Class","dual_Age","dual_Flight Distance","dual_Inflight wifi service","dual_Online boarding","dual_Seat comfort","dual_Inflight entertainment","dual_Leg room service","combo_0","count_combo_0","combo_1","count_combo_1","combo_2","count_combo_2","te_0","te_1","te_2"]
CATEGORICAL=set(["Gender","Customer Type","Type of Travel","Class",
                 "dual_Age","dual_Flight Distance","dual_Inflight wifi service",
                 "dual_Online boarding","dual_Seat comfort",
                 "dual_Inflight entertainment","dual_Leg room service",
                 "combo_0","combo_1","combo_2"])
TERMINAL={"candidate_ready_unsubmitted","skipped_no_cv_gain","kernel_failed",
          "output_failed","validation_mismatch","push_failed","kernel_access_error"}
ACTIVE={"preregistered_not_started","pushing_kernel","kernel_queued","kernel_running"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(values):
    return hashlib.sha256(np.asarray(values,dtype="<f8").tobytes()).hexdigest()


def frozen_params():
    return {
        "baseline_features":37,"protected_target_encoded_features":3,
        "new_features":0,"selected_total_features":37,"outer_folds":3,
        "inner_target_folds":5,"seed":SEED,"model_seed":MODEL_SEED,
        "epochs":4,"batch_size":512,"optimizer":"AdamW",
        "learning_rate":0.001,"weight_decay":0.0001,
        "loss":"BCEWithLogitsLoss","early_stopping":False,"n_ens":1,
        "architecture":{"d_model":64,"n_layers":2,"n_heads":4,
                        "ffn_hidden":128,"dropout":0.1,
                        "cls_token":True,"pre_norm":True},
        "numeric_standardization":"outer_train_only_mean_std",
        "categorical_vocabulary":"outer_train_only_unknown_missing_id",
    }


def verify_prereg(s):
    frozen={
        "competition":SLUG,"version":"V2.9.32",
        "intended_kernel":KERNEL,
        "hypothesis":"feature_token_attention_original_champion_37_features",
        "model_family":"FT_Transformer_PyTorch_T4",
        "feature_family":FEATURE_FAMILY,
        "baseline_cv":BASELINE,"official_submission_id":57446286,
        "official_public_lb":0.96099,
        "blend":{"v2915_rank":0.85,"v2932_rank":NEW_WEIGHT},
        "promotion":{"min_cv":FLOOR,"min_gain_vs_v2915":MIN_GAIN,
                     "min_positive_folds":2,"worst_fold_strict_gt":-0.00005},
        "submission_policy":"evaluation_only_no_submission",
        **frozen_params(),
    }
    if any(s.get(k)!=v for k,v in frozen.items()):
        raise RuntimeError("V2932 immutable pre-registration mismatch")
    for k in ("submitted","submission_attempted","external_labels","pseudo_labels",
              "route_features","pca_features","ordinal_threshold_features",
              "travel_gated_experts","new_conditional_te_features","feature_pruning"):
        if s.get(k) is not False:
            raise RuntimeError("V2932 prohibited option enabled: "+k)
    if s.get("kernel") not in (None,"",KERNEL) or s.get("stage") not in ACTIVE|TERMINAL:
        raise RuntimeError("V2932 Notebook owner/stage mismatch")
    if s.get("stage")=="preregistered_not_started":
        if s.get("kernel") or s.get("notebook_created") is not False or s.get("training_started") is not False:
            raise RuntimeError("V2932 not-started state is inconsistent")
    elif s.get("stage") not in TERMINAL and s.get("kernel")!=KERNEL:
        raise RuntimeError("V2932 active stage without durable owner")
    return s.get("stage") not in TERMINAL


def preflight():
    s=read(STATUS)
    if not verify_prereg(s):
        return None
    champ=v24.preflight()
    prev=read(ROOT/"state/v2_9_31_status.json")
    if (prev.get("stage")!="skipped_no_cv_gain" or prev.get("submitted") is not False
        or prev.get("kernel_state")!="kernel_complete"
        or abs(float((prev.get("comparison") or {}).get("candidate_auc",0))
               -0.9612867596798029)>1e-12
        or (prev.get("source_artifacts") or {}).get("v2931",{}).get("sha256")
           !="9c13ae6515dd1700a05a7b9edccc10ad610884bf9228ef990c57fbc35bf05320"):
        raise RuntimeError("V2931 terminal evidence changed")
    if champ.get("candidate_test_hash")!="906185547dcd5f0cac3ac173de644fd9997f20ac863c6745bcb1fd3e9a89e5f7":
        raise RuntimeError("V2915 official champion prediction SHA changed")
    return s


def persist(s):
    s["updated_at"]=now()
    v24.dump(STATUS,s)
    c=s.get("comparison") or {}
    m=s.get("metrics") or {}
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(
        "# V2.9.32 frozen FT-Transformer attention 37-feature threefold OOF\n\n"
        +f"- UTC: {s['updated_at']}\n"
        +f"- Stage: {s.get('stage')}\n"
        +f"- Unique Kaggle Notebook: https://www.kaggle.com/code/{KERNEL}\n"
        +f"- Official V2915 champion CV: {BASELINE}\n"
        +f"- Independent attention threefold CV: {m.get('full_oof_auc','pending')}\n"
        +f"- Fixed 85/15 candidate CV: {c.get('candidate_auc','pending')}\n"
        +f"- Gain: {c.get('gain','pending')}\n"
        +f"- Fold gains: {c.get('fold_gain','pending')}\n"
        +"- Evaluation only: no Kaggle submission. CV is not Public LB.\n"
        +(f"- Error: {s['error']}\n" if s.get("error") else ""),
        encoding="utf-8")


def checked_owner_exists():
    q=v24.cmd(["kaggle","kernels","status",KERNEL],timeout=90,check=False)
    if q.returncode==0 and "KernelWorkerStatus." in (q.stdout or ""):
        return True
    mine=v24.cmd(["kaggle","kernels","list","--mine","--search",
                  KERNEL.split("/",1)[1],"--page-size","50","-v"],
                 timeout=120,check=False)
    if mine.returncode:
        raise RuntimeError("Kaggle owner listing unavailable: refuse push")
    if KERNEL.lower() in (mine.stdout or "").lower():
        return True
    if (mine.stdout or "").strip().lower()!="not found":
        raise RuntimeError("Kaggle Notebook listing ambiguous: refuse push")
    witness=v24.cmd(["kaggle","kernels","status",v24.v23.v15.KERNEL],
                    timeout=90,check=False)
    if witness.returncode or "KernelWorkerStatus.COMPLETE" not in (witness.stdout or ""):
        raise RuntimeError("Kaggle owner witness failed: refuse push")
    return False


def durably_reserve_unique_kernel(s):
    if s.get("kernel") or s.get("stage")!="preregistered_not_started":
        raise RuntimeError("V2932 reservation requires untouched preregistration")
    subprocess.run(["git","fetch","origin",BRANCH],cwd=ROOT,check=True,
                   timeout=90,capture_output=True)
    local=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,check=True,
                         timeout=20,capture_output=True,text=True).stdout.strip()
    remote=subprocess.run(["git","rev-parse","FETCH_HEAD"],cwd=ROOT,check=True,
                          timeout=20,capture_output=True,text=True).stdout.strip()
    if local!=remote:
        raise RuntimeError("Branch changed after preflight: refuse push")
    s.update({"kernel":KERNEL,"stage":"pushing_kernel",
              "notebook_created":False,"training_started":False,
              "reservation_at":now()})
    persist(s)
    for cmd in (["git","config","user.name","kaggle-auto-scientist[bot]"],
                ["git","config","user.email","kaggle-auto-scientist[bot]@users.noreply.github.com"],
                ["git","add","--",str(STATUS.relative_to(ROOT)),str(REPORT.relative_to(ROOT))],
                ["git","commit","-m","bot: reserve V2932 sole attention Notebook before push"],
                ["git","push","origin",f"HEAD:{BRANCH}"]):
        subprocess.run(cmd,cwd=ROOT,check=True,timeout=90,capture_output=True)
    reserved=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,check=True,
                            timeout=20,capture_output=True,text=True).stdout.strip()
    remote_ref=subprocess.run(["git","ls-remote","origin",f"refs/heads/{BRANCH}"],
                              cwd=ROOT,check=True,timeout=30,
                              capture_output=True,text=True).stdout.strip()
    if not remote_ref.startswith(reserved+"\t"):
        raise RuntimeError("V2932 durable reservation not verified")
    return s


def verify_metrics(metrics,y,oof):
    source=hashlib.sha256((ROOT/"kaggle_agent/v2932_attention_train.py").read_bytes()).hexdigest()
    expected={
        "competition":SLUG,"stage":"three_folds_complete",
        "model":"FT-Transformer PyTorch original V2915 37-feature independent",
        "model_family":"FT_Transformer_PyTorch_T4",
        "feature_family":FEATURE_FAMILY,
        "validation":"StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)",
        "train_rows":699635,"test_rows":299844,"folds":3,
        "feature_count":37,"inner_target_folds":5,
        "leakage_safe_inner_cross_fit":True,
        "outer_validation_used_in_training":False,
        "external_labels":False,"pseudo_labels":False,"submitted":False,
        "gpu_verified":True,"model_seed":MODEL_SEED,
        "source_code_sha256":source,**frozen_params(),
    }
    for k,v in expected.items():
        if k in ("baseline_features","protected_target_encoded_features",
                 "new_features","selected_total_features","outer_folds","seed"):
            continue
        if metrics.get(k)!=v:
            raise ValueError("V2932 frozen metrics/source mismatch: "+k)
    hardware=metrics.get("hardware") or {}
    versions=metrics.get("environment_versions") or {}
    if (hardware.get("ok") is not True
        or "T4" not in str(hardware.get("device",""))
        or not any("T4" in str(d) for d in hardware.get("devices",[]))
        or not all(isinstance(versions.get(k),str) and versions[k].strip()
                   for k in ("python","torch","cuda","numpy","pandas","sklearn"))):
        raise ValueError("V2932 T4 hardware/environment provenance absent")
    results=metrics.get("fold_results") or []
    if len(results)!=3:
        raise ValueError("V2932 requires all three folds")
    for fold,(itr,iva) in enumerate(StratifiedKFold(3,shuffle=True,random_state=SEED).split(
            np.zeros(len(y),dtype="uint8"),y)):
        item=results[fold]
        expected_hash=hashlib.sha256(np.asarray(iva,dtype="<i8").tobytes()).hexdigest()
        if (item.get("fold")!=fold or item.get("train_rows")!=len(itr)
            or item.get("validation_rows")!=len(iva)
            or item.get("validation_index_sha256")!=expected_hash
            or item.get("model_seed")!=MODEL_SEED+fold
            or item.get("features")!=37 or item.get("epochs")!=4
            or item.get("feature_order")!=CANONICAL
            or item.get("token_types")!=[
                "categorical" if c in CATEGORICAL else "numeric" for c in CANONICAL]
            or item.get("categorical_columns")!=[c for c in CANONICAL if c in CATEGORICAL]
            or item.get("numeric_columns")!=[c for c in CANONICAL if c not in CATEGORICAL]
            or len(item.get("categorical_cardinalities") or [])!=len(CATEGORICAL)
            or any(type(n) is not int or n<2 for n in item.get("categorical_cardinalities",[]))):
            raise ValueError("V2932 fold membership/token order/vocabulary mismatch")
        if abs(float(roc_auc_score(y[iva],oof[iva]))-float(item.get("auc",-1)))>2e-6:
            raise ValueError("V2932 per-fold AUC replay mismatch")
    if abs(float(roc_auc_score(y,oof))-float(metrics.get("full_oof_auc",-1)))>2e-6:
        raise ValueError("V2932 full threefold AUC replay mismatch")


def validate(raw,s):
    file=v24.v23.unique(raw,"v2932_attention_oof.npz")
    metrics=read(v24.v23.unique(raw,"v2932_metrics.json"))
    with np.load(file,allow_pickle=False) as saved:
        keys={"train_id","target","oof_attention","test_id","pred_attention"}
        if not keys.issubset(saved.files):
            raise ValueError("V2932 numeric OOF/test/ID arrays missing")
        arrays={k:np.asarray(saved[k]).copy() for k in keys}
        for key,n in (("train_id",699635),("target",699635),
                      ("oof_attention",699635),("test_id",299844),
                      ("pred_attention",299844)):
            if arrays[key].shape!=(n,) or arrays[key].dtype.kind not in "iuf":
                raise ValueError("V2932 unsafe dtype/shape: "+key)
    champion=v24.preflight()
    with tempfile.TemporaryDirectory(prefix="kas_v2932_immutable_") as tmp:
        train,test,train_id,test_id,y,champion_oof,champion_test,sources=\
            v24.baseline_arrays(tmp,champion)
        new_oof,new_test=v24.v23.v14.align_new(
            arrays,"oof_attention","pred_attention",
            train_id,test_id,y,"V2932")
        verify_metrics(metrics,y,new_oof)
        source_sha=v24.v23.audit.sha256(file)
        rank=v24.v23.audit.rank
        candidate_oof=0.85*rank(champion_oof)+NEW_WEIGHT*rank(new_oof)
        candidate_test=0.85*rank(champion_test)+NEW_WEIGHT*rank(new_test)
        if not np.isfinite(candidate_oof).all() or not np.isfinite(candidate_test).all():
            raise ValueError("V2932 nonfinite candidate")
        if abs(float(roc_auc_score(y,champion_oof))-BASELINE)>1e-12:
            raise ValueError("V2915 champion OOF baseline mismatch")
        score=float(roc_auc_score(y,candidate_oof))
        fold_auc=[];fold_gain=[]
        for _,iva in StratifiedKFold(3,shuffle=True,random_state=SEED).split(
                np.zeros(len(y),dtype="uint8"),y):
            auc=float(roc_auc_score(y[iva],candidate_oof[iva]))
            fold_auc.append(auc)
            fold_gain.append(auc-float(roc_auc_score(y[iva],champion_oof[iva])))
        positive=sum(x>0 for x in fold_gain)
        promotable=bool(score>=FLOOR and score-BASELINE>=MIN_GAIN
                        and positive>=2 and min(fold_gain)>-0.00005)
        comparison={"baseline_auc":BASELINE,
                    "standalone_auc":float(roc_auc_score(y,new_oof)),
                    "candidate_auc":score,"gain":score-BASELINE,
                    "fold_auc":fold_auc,"fold_gain":fold_gain,
                    "positive_folds":positive,
                    "fixed_weights":{"v2915_rank":0.85,"v2932_rank":NEW_WEIGHT},
                    "promotable":promotable}
        s.update({
            "stage":"candidate_ready_unsubmitted" if promotable else "skipped_no_cv_gain",
            "metrics":metrics,"comparison":comparison,
            "source_artifacts":{**sources,"v2932":{
                "notebook":KERNEL,"sha256":source_sha,
                "code_sha256":metrics["source_code_sha256"]}},
            "candidate_oof_hash":digest(candidate_oof),
            "candidate_test_hash":digest(candidate_test),
            "submitted":False,"submission_attempted":False})
        v24.v23.record_experiment(LEDGER,{
            "competition":SLUG,"candidate":"v2932_attention_original37",
            "validation":metrics["validation"],"seed":SEED,
            "parameters":{"model_seed":MODEL_SEED,"model_parameters":frozen_params(),
                          "fusion_weights":dict(comparison["fixed_weights"]),
                          "min_gain":MIN_GAIN},
            "status":"completed" if promotable else "skipped_no_cv_gain",
            "cv":score,"completed_at":now(),
            "gpu_seconds":metrics.get("runtime_seconds"),
            "kernel_sources":[KERNEL]})
    return s


def run(recover_only=False):
    previous=read(STATUS)
    if previous.get("stage") in TERMINAL:
        print("V2932 terminal; never rerun or submit")
        return previous
    s=preflight()
    if s is None:
        return previous
    if not s.get("kernel"):
        if recover_only:
            print("RECOVER ONLY: no durable slug; refuse Notebook creation")
            return s
        try:
            exists=checked_owner_exists()
        except Exception as e:
            s.update({"stage":"kernel_access_error","error":str(e)[-1400:]})
            persist(s)
            return s
        if exists:
            s.update({"kernel":KERNEL,"stage":"kernel_queued",
                      "preexisting_kernel_recovered":True})
            persist(s)
    if not s.get("kernel"):
        WORKDIR.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/"kaggle_agent/v2932_attention_train.py",WORKDIR/"main.py")
        v24.dump(WORKDIR/"kernel-metadata.json",{
            "id":KERNEL,"title":KERNEL.split("/",1)[1],
            "code_file":"main.py","language":"python","kernel_type":"script",
            "is_private":True,"enable_gpu":True,"enable_internet":False,
            "dataset_sources":[],"competition_sources":[SLUG],
            "kernel_sources":[],"model_sources":[]})
        s=durably_reserve_unique_kernel(s)
        pushed=v24.cmd(["kaggle","kernels","push","-p",str(WORKDIR),"-t","3600"],
                       timeout=300,check=False)
        if pushed.returncode:
            s.update({"stage":"push_failed",
                      "error":(pushed.stderr or pushed.stdout)[-1800:]})
            persist(s)
            return s
        match=re.search(
            r"https?://(?:www\.)?kaggle\.com/code/([a-z0-9_-]+)/([a-z0-9-]+)",
            (pushed.stdout or "")+"\n"+(pushed.stderr or ""),re.I)
        if not match or "/".join(match.groups()).lower()!=KERNEL:
            s.update({"stage":"kernel_access_error",
                      "error":"Kaggle canonical owner URL unverified; refuse second push"})
            persist(s)
            return s
        s.update({"notebook_created":True,"training_started":True})
        persist(s)
    if s.get("kernel")!=KERNEL:
        raise RuntimeError("Saved Kaggle Notebook slug mismatch")
    stage,message=v24.poll_kernel_strict(KERNEL,90 if recover_only else 300,20)
    s["kernel_state"]=stage
    s["kernel_message"]=message[-1600:]
    if stage in {"kernel_running","kernel_queued","kernel_access_error"}:
        s["stage"]=stage
        persist(s)
        return s
    raw=OUTDIR/"raw"
    if raw.exists():
        shutil.rmtree(raw)
    raw.mkdir(parents=True,exist_ok=True)
    got=v24.cmd(["kaggle","kernels","output",KERNEL,"-p",str(raw),"-o","-q"],
                timeout=600,check=False)
    if stage!="kernel_complete" or got.returncode:
        errors=list(raw.rglob("v2932_error.json"))
        s.update({"stage":"kernel_failed" if stage!="kernel_complete" else "output_failed",
                  "error":errors[0].read_text(encoding="utf-8")[-2000:]
                  if errors else (got.stderr or message)[-1600:]})
        persist(s)
        return s
    try:
        validate(raw,s)
    except Exception as e:
        s.update({"stage":"validation_mismatch","error":str(e)[-1800:]})
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
