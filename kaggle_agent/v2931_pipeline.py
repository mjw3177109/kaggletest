"""V2.9.31 safe single-owner Notebook evaluator, frozen 29-feature mi_pruned crossfit TE RealMLP-TD T4.
Only Kaggle kernels may be created; this evaluator NEVER submits competition predictions.
Recover-only cannot create or push a Notebook. Always persist owner reservation first.
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import v2924_pipeline as v24

ROOT=Path(__file__).resolve().parents[1]
SLUG="playground-series-s6e10"
BRANCH="feature/v29-autonomous-evolution"
KERNEL="mjw3177109/kas-v2931-foldlocal-mi-pruned-realmlp-t4-s6e10"
STATUS=ROOT/"state/v2_9_30_status.json"
REPORT=ROOT/"reports/v2_9_30_result.md"
WORKDIR=ROOT/"competitions"/SLUG/"kaggle_gpu_v2931_mi_pruned"
OUTDIR=ROOT/"competitions"/SLUG/"results_v2931_mi_pruned"
LEDGER=ROOT/"state/v2_9_experiments.json"
BASELINE=0.9612850399332336
FLOOR=0.96125635576833
MIN_GAIN=0.00005
SEED=20261018
MODEL_SEED=20261031
NEW_WEIGHT=0.15
TERMINAL={"candidate_ready_unsubmitted","skipped_no_cv_gain","kernel_failed",
          "output_failed","validation_mismatch","push_failed","kernel_access_error"}
ACTIVE={"preregistered_not_started","pushing_kernel","kernel_queued","kernel_running"}
FEATURE_FAMILY="v2915_37_features_fold_local_mi_select_29"
def known_provenance_recovery(s,recover_only):
    return False

def read(path):
    return v24.read(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(values):
    return hashlib.sha256(np.asarray(values,dtype="<f8").tobytes()).hexdigest()


def frozen_params():
    return {"baseline_features":37,"protected_target_encoded_features":3,
            "selected_nonprotected_features":26,"selected_total_features":29,
            "removed_features":8,"feature_selection":"outer_train_only_mutual_info_classif",
            "mi_n_neighbors":3,"mi_seed":20261031,
            "mi_tie_break":"canonical_v2915_feature_order",
            "outer_folds":3,"seed":SEED,"model_seed":MODEL_SEED,
            "epochs":5,"n_ens":8}

def verify_prereg(s,allow_known_recovery=False):
    frozen=frozen_params()
    if (s.get("competition")!=SLUG or s.get("version")!="V2.9.31"
        or s.get("intended_kernel")!=KERNEL
        or s.get("hypothesis")!="fold_local_supervised_mutual_information_pruning_of_champion_features"
        or s.get("model_family")!="global_RealMLP_TD_T4"
        or s.get("feature_family")!=FEATURE_FAMILY
        or any(s.get(k)!=v for k,v in frozen.items())
        or any(s.get(k) is not False for k in (
            "external_labels","pseudo_labels","route_features","pca_features",
            "ordinal_threshold_features","travel_gated_experts","new_conditional_te_features"))
        or s.get("blend")!={"v2915_rank":0.85,"v2931_rank":NEW_WEIGHT}
        or s.get("baseline_cv")!=BASELINE
        or s.get("promotion")!={"min_cv":FLOOR,"min_gain_vs_v2915":MIN_GAIN,
                                 "min_positive_folds":2,"worst_fold_strict_gt":-0.00005}
        or s.get("submission_policy")!="evaluation_only_no_submission"
        or s.get("submitted") is not False or s.get("submission_attempted") is not False
        or s.get("kernel") not in ("",None,KERNEL)
        or (s.get("stage") not in ACTIVE and s.get("stage") not in TERMINAL)
        or (s.get("stage")!="preregistered_not_started" and s.get("stage") not in TERMINAL
            and s.get("kernel")!=KERNEL)
        or (s.get("stage")=="preregistered_not_started"
            and (s.get("kernel") or s.get("notebook_created") is not False
                 or s.get("training_started") is not False))):
        raise RuntimeError("V2931 immutable preregistration/owner/no-submit invariant failed")
    return s.get("stage") not in TERMINAL

def preflight(recover_only=False):
    s=read(STATUS)
    if not verify_prereg(s):
        return None
    if s.get("stage") not in ACTIVE:
        raise RuntimeError("V2931 unknown or unsafe recovery stage")
    champion=v24.preflight()
    previous=read(ROOT/"state/v2_9_30_status.json")
    if (previous.get("stage")!="skipped_no_cv_gain"
        or previous.get("submitted") is not False
        or previous.get("kernel_state")!="kernel_complete"
        or abs(float((previous.get("comparison") or {}).get("candidate_auc",0))
               -0.9613286615476094)>1e-12
        or (previous.get("source_artifacts") or {}).get("v2930",{}).get("sha256")
           !="45e96e595676ec91a1e6c8eea98f975e63c9c0762412b898fe549c0636107362"):
        raise RuntimeError("V2930 terminal evidence changed; refuse launch")
    if champion.get("candidate_test_hash")!="906185547dcd5f0cac3ac173de644fd9997f20ac863c6745bcb1fd3e9a89e5f7":
        raise RuntimeError("V2915 champion official prediction SHA mismatch")
    return s

def persist(s):
    s["updated_at"]=now()
    v24.dump(STATUS,s)
    c=s.get("comparison") or {}
    m=s.get("metrics") or {}
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(
        "# V2.9.31 mi_pruned crossfit TE RealMLP-TD threefold OOF\n\n"
        +f"- UTC: {s['updated_at']}\n"
        +f"- Stage: {s.get('stage')}\n"
        +f"- Unique Notebook: https://www.kaggle.com/code/{KERNEL}\n"
        +f"- Baseline V2915 threefold CV: {BASELINE}\n"
        +f"- Independent RealMLP threefold CV: {m.get('full_oof_auc','pending')}\n"
        +f"- Fixed 85/15 rank fusion CV: {c.get('candidate_auc','pending')}\n"
        +f"- Gain: {c.get('gain','pending')}\n"
        +f"- Three fold gains: {c.get('fold_gain','pending')}\n"
        +"- Evaluation only, no Kaggle submission. Public LB remains the last certified 0.96099.\n"
        +(f"- Error: {s['error']}\n" if s.get("error") else ""),
        encoding="utf-8",
    )


def checked_owner_exists():
    q=v24.cmd(["kaggle","kernels","status",KERNEL],timeout=90,check=False)
    if q.returncode==0 and "KernelWorkerStatus." in (q.stdout or ""):
        return True
    mine=v24.cmd(["kaggle","kernels","list","--mine","--search",
                  KERNEL.split("/",1)[1],"--page-size","50","-v"],
                 timeout=120,check=False)
    if mine.returncode:
        raise RuntimeError("Kaggle owner listing unavailable; refuse Notebook push")
    if KERNEL.lower() in (mine.stdout or "").lower():
        return True
    if (mine.stdout or "").strip().lower()!="not found":
        raise RuntimeError("Kaggle owner listing ambiguous; refuse Notebook push")
    witness=v24.cmd(["kaggle","kernels","status",v24.v23.v15.KERNEL],
                    timeout=90,check=False)
    if witness.returncode or "KernelWorkerStatus.COMPLETE" not in (witness.stdout or ""):
        raise RuntimeError("Kaggle owner identity witness failed; refuse Notebook push")
    return False


def durably_reserve_unique_kernel(s):
    """Remote reservation MUST precede the only possible Kaggle kernels push."""
    if s.get("kernel") or s.get("stage")!="preregistered_not_started":
        raise RuntimeError("V2931 reservation requires untouched preregistration")
    subprocess.run(["git","fetch","origin",BRANCH],cwd=ROOT,check=True,
                   timeout=90,capture_output=True)
    local=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,check=True,
                         timeout=20,capture_output=True,text=True).stdout.strip()
    remote=subprocess.run(["git","rev-parse","FETCH_HEAD"],cwd=ROOT,check=True,
                          timeout=20,capture_output=True,text=True).stdout.strip()
    if local!=remote:
        raise RuntimeError("GitHub branch changed after preflight; refuse Notebook push")
    s.update({"kernel":KERNEL,"stage":"pushing_kernel",
              "notebook_created":False,"training_started":False,"reservation_at":now()})
    persist(s)
    for command in (["git","config","user.name","kaggle-auto-scientist[bot]"],
                    ["git","config","user.email","kaggle-auto-scientist[bot]@users.noreply.github.com"],
                    ["git","add","--",str(STATUS.relative_to(ROOT)),str(REPORT.relative_to(ROOT))],
                    ["git","commit","-m","bot: reserve V2931 unique Kaggle Notebook before push"],
                    ["git","push","origin",f"HEAD:{BRANCH}"]):
        subprocess.run(command,cwd=ROOT,check=True,timeout=90,capture_output=True)
    reserved=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,check=True,
                            timeout=20,capture_output=True,text=True).stdout.strip()
    remote_ref=subprocess.run(["git","ls-remote","origin",f"refs/heads/{BRANCH}"],
                              cwd=ROOT,check=True,timeout=30,
                              capture_output=True,text=True).stdout.strip()
    if not remote_ref.startswith(reserved+"\t"):
        raise RuntimeError("Durable Notebook reservation not confirmed on GitHub")
    return s


def verify_metrics(metrics,y,oof):
    hardware=metrics.get("hardware") or {}
    versions=metrics.get("environment_versions") or {}
    if any(not isinstance(versions.get(k),str) or not versions[k].strip()
           for k in ("python","numpy","pandas","sklearn","pytabkit","torch")):
        raise ValueError("V2931 missing environment provenance")
    if versions["pytabkit"]!="1.7.3":
        raise ValueError("V2931 pytabkit version mismatch")
    source_sha=hashlib.sha256((ROOT/"kaggle_agent/v2931_mi_pruned_realmlp_train.py").read_bytes()).hexdigest()
    frozen={
        "competition":SLUG,"stage":"three_folds_complete",
        "validation":"StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)",
        "train_rows":699635,"test_rows":299844,"folds":3,
        "model_seed":MODEL_SEED,"baseline_features":37,
        "protected_target_encoded_features":3,
        "selected_nonprotected_features":26,"selected_total_features":29,
        "removed_features":8,"feature_count":29,
        "feature_selection":"outer_train_only_mutual_info_classif",
        "mi_n_neighbors":3,"mi_seed":20261031,
        "mi_tie_break":"canonical_v2915_feature_order",
        "original_target_encoding_columns":3,
        "feature_family":FEATURE_FAMILY,"source_code_sha256":source_sha,
        "data_source":"competition_train_only","epochs":5,
        "outer_train_only_supervised_statistics":True,
        "external_labels":False,"pseudo_labels":False,"submitted":False,
        "gpu_verified":True,"leakage_safe_inner_cross_fit":True,
        "outer_validation_used_in_training":False,
    }
    if any(metrics.get(k)!=v for k,v in frozen.items()):
        raise ValueError("V2931 frozen 29feature/3fold/MI/source provenance mismatch")
    selections=metrics.get("fold_selection_evidence") or []
    if (len(metrics.get("fold_results") or [])!=3 or len(selections)!=3
        or hardware.get("ok") is not True
        or not any("T4" in str(d) for d in hardware.get("devices",[]))):
        raise ValueError("V2931 hardware or 3fold selection evidence missing")
    original=None
    for fold,(itr,iva) in enumerate(StratifiedKFold(3,shuffle=True,random_state=SEED)
                                    .split(np.zeros(len(y),dtype=np.uint8),y)):
        item=metrics["fold_results"][fold]
        selection=selections[fold]
        canonical=selection.get("canonical_original_features") or []
        protected=selection.get("protected_target_encoded_features") or []
        eligible=selection.get("eligible_features") or []
        scores=selection.get("mi_scores") or []
        mask=selection.get("discrete_feature_mask") or []
        selected=selection.get("selected_features") or []
        removed=selection.get("removed_features") or []
        if original is None:
            original=canonical
        if (canonical!=original or len(canonical)!=37 or len(set(canonical))!=37
            or protected!=["te_0","te_1","te_2"]
            or eligible!=[c for c in canonical if c not in protected]
            or len(eligible)!=34 or len(scores)!=34 or len(mask)!=34
            or any(type(x) is not bool for x in mask)
            or not np.isfinite(np.asarray(scores,dtype=float)).all()
            or selection.get("mi_n_neighbors")!=3 or selection.get("mi_seed")!=20261031
            or selection.get("mi_tie_break")!="canonical_v2915_feature_order"
            or selection.get("selected_nonprotected_features")!=26
            or selection.get("selected_total_features")!=29):
            raise ValueError("V2931 fold-local MI ranking provenance mismatch")
        order=sorted(range(34),key=lambda i:(-float(scores[i]),i))
        expected_set=set(eligible[i] for i in order[:26])|set(protected)
        expected=[c for c in canonical if c in expected_set]
        if (selected!=expected or len(selected)!=29
            or removed!=[c for c in canonical if c not in expected_set]
            or item.get("selection")!=selection
            or item.get("fold")!=fold or item.get("validation_rows")!=len(iva)
            or item.get("train_rows")!=len(itr)
            or item.get("features")!=29 or item.get("model_seed")!=MODEL_SEED+fold):
            raise ValueError("V2931 fold membership/MI tie break/selection mismatch")
        if abs(float(roc_auc_score(y[iva],oof[iva]))-float(item.get("auc",-1)))>2e-6:
            raise ValueError("V2931 fold AUC replay mismatch")
    if abs(float(roc_auc_score(y,oof))-float(metrics.get("full_oof_auc",-1)))>2e-6:
        raise ValueError("V2931 full OOF AUC replay mismatch")

def validate(raw,s):
    file=v24.v23.unique(raw,"v2931_realmlp_oof.npz")
    metrics=read(v24.v23.unique(raw,"v2931_metrics.json"))
    with np.load(file,allow_pickle=False) as saved:
        keys={"train_id","target","oof_mi_realmlp",
              "test_id","pred_mi_realmlp"}
        if not keys.issubset(saved.files):
            raise ValueError("V2931 numeric OOF/test/ID arrays missing")
        arrays={key:np.asarray(saved[key]).copy() for key in keys}
        for key,n in (("train_id",699635),("target",699635),
                      ("oof_mi_realmlp",699635),
                      ("test_id",299844),("pred_mi_realmlp",299844)):
            if arrays[key].shape!=(n,) or arrays[key].dtype.kind not in "iuf":
                raise ValueError("V2931 invalid numeric prediction/ID shape or dtype")
    champion=v24.preflight()
    with tempfile.TemporaryDirectory(prefix="kas_v2931_immutable_") as tmp:
        train,test,train_id,test_id,y,champion_oof,champion_test,sources=\
            v24.baseline_arrays(tmp,champion)
        new_oof,new_test=v24.v23.v14.align_new(
            arrays,"oof_mi_realmlp","pred_mi_realmlp",
            train_id,test_id,y,"V2931")
        verify_metrics(metrics,y,new_oof)
        source_sha=v24.v23.audit.sha256(file)
        rank=v24.v23.audit.rank
        candidate_oof=0.85*rank(champion_oof)+NEW_WEIGHT*rank(new_oof)
        candidate_test=0.85*rank(champion_test)+NEW_WEIGHT*rank(new_test)
        if not (np.isfinite(candidate_oof).all() and np.isfinite(candidate_test).all()):
            raise ValueError("V2931 nonfinite candidate")
        if abs(float(roc_auc_score(y,champion_oof))-BASELINE)>1e-12:
            raise ValueError("Frozen champion full OOF AUC not reproduced")
        score=float(roc_auc_score(y,candidate_oof))
        fold_auc=[];fold_gain=[]
        for _,iva in StratifiedKFold(3,shuffle=True,random_state=SEED).split(
            np.zeros(len(y),dtype=np.uint8),y):
            auc=float(roc_auc_score(y[iva],candidate_oof[iva]))
            fold_auc.append(auc)
            fold_gain.append(auc-float(roc_auc_score(y[iva],champion_oof[iva])))
        positive=sum(g>0 for g in fold_gain)
        promotable=bool(score>=FLOOR and score-BASELINE>=MIN_GAIN
                        and positive>=2 and min(fold_gain)>-0.00005)
        comparison={
            "baseline_auc":BASELINE,
            "standalone_auc":float(roc_auc_score(y,new_oof)),
            "candidate_auc":score,"gain":score-BASELINE,
            "fold_auc":fold_auc,"fold_gain":fold_gain,
            "positive_folds":positive,
            "fixed_weights":{"v2915_rank":0.85,"v2931_rank":NEW_WEIGHT},
            "promotable":promotable,
        }
        s.update({
            "stage":"candidate_ready_unsubmitted" if promotable else "skipped_no_cv_gain",
            "metrics":metrics,"comparison":comparison,
            "source_artifacts":{**sources,"v2931":{"notebook":KERNEL,"sha256":source_sha,"code_sha256":metrics["source_code_sha256"]}},
            "candidate_oof_hash":digest(candidate_oof),
            "candidate_test_hash":digest(candidate_test),
            "submitted":False,"submission_attempted":False,
        })
        v24.v23.record_experiment(LEDGER,{
            "competition":SLUG,"candidate":"v2931_foldlocal_mi_pruned_realmlp",
            "validation":metrics["validation"],"seed":SEED,
            "parameters":{"model_seed":MODEL_SEED,"model_parameters":frozen_params(),
                           "mi_n_neighbors":3,"mi_seed":20261031,
                           "fold_selection_evidence":metrics["fold_selection_evidence"],
                           "fusion_weights":dict(comparison["fixed_weights"]),
                           "min_gain":MIN_GAIN},
            "status":"completed" if promotable else "skipped_no_cv_gain",
            "cv":score,"completed_at":now(),
            "gpu_seconds":metrics.get("runtime_seconds"),"kernel_sources":[KERNEL],
        })
    return s


def run(recover_only=False):
    previous=read(STATUS)
    if previous.get("stage") in TERMINAL:
        print("V2931 terminal; never create, recover or submit again")
        return previous
    s=preflight(recover_only=recover_only)
    if s is None:
        return previous
    if not s.get("kernel"):
        if recover_only:
            print("RECOVER ONLY: no stored slug, refusing Notebook creation")
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
        shutil.copy2(ROOT/"kaggle_agent/v2931_mi_pruned_realmlp_train.py",
                     WORKDIR/"main.py")
        v24.dump(WORKDIR/"kernel-metadata.json",{
            "id":KERNEL,"title":KERNEL.split("/",1)[1],
            "code_file":"main.py","language":"python","kernel_type":"script",
            "is_private":True,"enable_gpu":True,"enable_internet":True,
            "dataset_sources":[],"competition_sources":[SLUG],
            "kernel_sources":[],"model_sources":[],
        })
        s=durably_reserve_unique_kernel(s)
        pushed=v24.cmd(["kaggle","kernels","push","-p",str(WORKDIR),"-t","3600"],
                       timeout=300,check=False)
        if pushed.returncode:
            s.update({"stage":"push_failed","error":(pushed.stderr or pushed.stdout)[-1800:]})
            persist(s)
            return s
        match=re.search(
            r"https?://(?:www\.)?kaggle\.com/code/([a-z0-9_-]+)/([a-z0-9-]+)",
            (pushed.stdout or "")+"\n"+(pushed.stderr or ""),re.I)
        if not match or "/".join(match.groups()).lower()!=KERNEL:
            s.update({"stage":"kernel_access_error",
                      "error":"Canonical Kaggle Notebook owner URL not verified; refuse second push"})
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
        errors=list(raw.rglob("v2931_error.json"))
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
