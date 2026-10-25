"""V2940 independent strict OOF validator. NEVER submits or trains.
Rebuilds frozen V2915 OOF/test from authenticated Kaggle source artifacts.
"""
from __future__ import annotations
import contextlib, hashlib, json, os, subprocess, tempfile, time, traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import v2924_pipeline as old
import v2940_monotone_train as train
import v2940_actions_gate as gate

ROOT=Path(__file__).resolve().parents[1]
STATUS=ROOT/"state/v2_9_38_status.json"
REPORT=ROOT/"reports/v2_9_38_result.md"
BASELINE=0.9612850399332336
MIN_GAIN=0.00005
MIN_CV=0.96125635576833

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def vector_sha(arr):
    return hashlib.sha256(np.asarray(arr,dtype="<f8").tobytes()).hexdigest()

def rank(arr):
    arr=np.asarray(arr,dtype=np.float64)
    if arr.ndim!=1 or not np.isfinite(arr).all():
        raise RuntimeError("Invalid V2940 rank vector")
    return rankdata(arr,method="average")/len(arr)

@contextlib.contextmanager
def rate_limited_kaggle_subprocess():
    """Enforce >=3 seconds between Kaggle CLI calls even in historical readers."""
    original=subprocess.run
    last=[None]
    def guarded(parts,*args,**kwargs):
        if isinstance(parts,(list,tuple)) and parts and str(parts[0])=="kaggle":
            if last[0] is not None:
                remain=3.0-(time.monotonic()-last[0])
                if remain>0:time.sleep(remain)
            last[0]=time.monotonic()
        result=original(parts,*args,**kwargs)
        if isinstance(parts,(list,tuple)) and parts and str(parts[0])=="kaggle":
            diagnostic=str(getattr(result,"stderr",""))+"\n"+str(getattr(result,"stdout",""))
            if result.returncode and any(token in diagnostic.lower() for token in
                 ("429","too many requests","rate limit")):
                raise RuntimeError("Kaggle API rate limit; fail closed")
        return result
    subprocess.run=guarded
    try:yield
    finally:subprocess.run=original

def load_candidate(folder):
    folder=Path(folder)
    artifact=folder/"v2940_monotone_oof.npz"
    metrics_file=folder/"v2940_metrics.json"
    if not artifact.is_file() or not metrics_file.is_file():
        raise RuntimeError("V2940 complete NPZ/metrics absent")
    metrics=json.loads(metrics_file.read_text(encoding="utf-8"))
    with np.load(artifact,allow_pickle=False) as z:
        keys={"target","train_id","oof_monotone","test_id","pred_monotone"}
        if set(z.files)!=keys:raise RuntimeError("V2940 NPZ keys changed")
        data={k:np.asarray(z[k]).copy() for k in keys}
    return artifact,metrics,data

def validate(folder):
    s=gate.read("state/v2_9_38_status.json")
    if (s.get("stage")!="run_reserved" or str(s.get("run_reservation_id"))!=os.getenv("GITHUB_RUN_ID")
        or s.get("run_reservation_sha256")!=gate.fingerprint(s)):
        raise RuntimeError("V2940 one-shot reservation invalid at validation")
    artifact,metrics,c=load_candidate(folder)
    if (metrics.get("stage")!="three_folds_complete" or metrics.get("competition")!="playground-series-s6e10"
        or metrics.get("folds")!=3 or metrics.get("train_rows")!=699635
        or metrics.get("test_rows")!=299844 or metrics.get("feature_count")!=37
        or metrics.get("inner_target_folds")!=5 or metrics.get("model_seed")!=20261107
        or metrics.get("model_parameters")!=train.PARAMS
        or metrics.get("environment_versions")!=dict(train.VERSIONS,
            python=metrics.get("environment_versions",{}).get("python"),device="cpu",omp_num_threads=4)
        or metrics.get("leakage_safe_inner_cross_fit") is not True
        or metrics.get("outer_validation_used_in_training") is not False
        or metrics.get("cpu_only") is not True or metrics.get("submitted") is not False
        or metrics.get("monotonic_features")!=list(train.MONOTONE)
        or metrics.get("monotonic_vector")!=train.frozen_monotonic_vector(train.CANONICAL)
        or metrics.get("source_code_sha256")!=sha256(ROOT/"kaggle_agent/v2940_monotone_train.py")):
        raise RuntimeError("V2940 frozen training provenance mismatch")
    with tempfile.TemporaryDirectory(prefix="v2940_champion_audit_") as td:
        # Original historical pipeline checks eight archived sources, source NPZ SHAs,
        # official train/test IDs and the V2915 champion test prediction SHA.
        with rate_limited_kaggle_subprocess():
            champion=old.preflight()
            official_train,official_test,train_id,test_id,y,baseline_oof,baseline_test,files=old.baseline_arrays(td,champion)
    if (not np.array_equal(c["train_id"],train_id)
        or not np.array_equal(c["test_id"],test_id)
        or not np.array_equal(c["target"],y)):
        raise RuntimeError("V2940 train/test IDs or target differ from official Kaggle data")
    oof=np.asarray(c["oof_monotone"],dtype=np.float64)
    pred=np.asarray(c["pred_monotone"],dtype=np.float64)
    if (oof.shape!=(699635,) or pred.shape!=(299844,)
        or not np.isfinite(oof).all() or not np.isfinite(pred).all()
        or np.any((oof<0)|(oof>1)) or np.any((pred<0)|(pred>1))):
        raise RuntimeError("V2940 missing/invalid full OOF/test predictions")
    folds=list(StratifiedKFold(3,shuffle=True,random_state=20261018).split(
        np.zeros(len(y),dtype="uint8"),y))
    results=metrics.get("fold_results") or []
    if len(results)!=3:raise RuntimeError("V2940 missing three fold metrics")
    fold_auc=[]
    for fold,(itr,iva) in enumerate(folds):
        info=results[fold]
        digest=hashlib.sha256(np.asarray(iva,dtype="<i8").tobytes()).hexdigest()
        auc=float(roc_auc_score(y[iva],oof[iva]))
        if (info.get("fold")!=fold or info.get("train_rows")!=len(itr)
            or info.get("validation_rows")!=len(iva)
            or info.get("model_seed")!=20261107+fold
            or info.get("features")!=37 or info.get("feature_order")!=train.CANONICAL
            or info.get("validation_index_sha256")!=digest
            or len(str(info.get("category_vocabulary_sha256","")))!=64
            or info.get("monotonic_features")!=list(train.MONOTONE)
            or info.get("monotonic_vector")!=train.frozen_monotonic_vector(train.CANONICAL)
            or info.get("rounds")!=train.PARAMS["max_iter"]
            or abs(float(info.get("auc",0))-auc)>1e-8):
            raise RuntimeError("V2940 fold membership, AUC or feature provenance mismatch")
        fold_auc.append(auc)
    standalone=float(roc_auc_score(y,oof))
    if abs(standalone-float(metrics.get("full_oof_auc",0)))>1e-8:
        raise RuntimeError("V2940 full independent OOF AUC mismatch")
    baseline=float(roc_auc_score(y,baseline_oof))
    if abs(baseline-BASELINE)>2e-6:
        raise RuntimeError("Official V2915 champion OOF mismatch")
    blend=.85*rank(baseline_oof)+.15*rank(oof)
    full=float(roc_auc_score(y,blend))
    gains=[float(roc_auc_score(y[iva],blend[iva])-
                  roc_auc_score(y[iva],baseline_oof[iva])) for _,iva in folds]
    gain=full-BASELINE
    positive=sum(x>0 for x in gains)
    promotable=full>=MIN_CV and gain>=MIN_GAIN and positive>=2 and min(gains)>-0.00005
    candidate_test=.85*rank(baseline_test)+.15*rank(pred)
    if candidate_test.shape!=(299844,) or not np.isfinite(candidate_test).all():
        raise RuntimeError("V2940 candidate test vector invalid")
    s.update(stage="candidate_ready_unsubmitted" if promotable else "skipped_no_cv_gain",
        training_started=True,submitted=False,submission_attempted=False,
        full_cv=full,standalone_cv=standalone,gain_vs_champion=gain,
        fold_auc=fold_auc,fold_gain=gains,positive_folds=positive,promotable=promotable,
        source_npz_sha256=sha256(artifact),candidate_oof_sha256=vector_sha(blend),
        candidate_test_sha256=vector_sha(candidate_test),
        champion_prediction_sha256=gate.CHAMPION_SHA,
        verified_source_artifacts=files,
        environment_versions=metrics["environment_versions"],
        updated_at=datetime.now(timezone.utc).isoformat())
    # Only a candidate passing all frozen gates gets a test CSV, never submitted here.
    if promotable:
        pd.DataFrame({"id":test_id,"satisfaction":candidate_test}).to_csv(
            Path(folder)/"v2940_candidate_unsubmitted.csv",index=False)
    STATUS.write_text(json.dumps(s,indent=2,ensure_ascii=False,default=str)+"\n",encoding="utf-8")
    REPORT.write_text("# V2940 immutable CPU Monotone HGB OOF result\n\n"
        +f"- UTC: {s['updated_at']}\n- Stage: {s['stage']}\n"
        +f"- Independent full OOF CV: {standalone}\n- Fixed 85/15 CV: {full}\n"
        +f"- Gain vs champion: {gain}\n- Fold gains: {gains}\n"
        +f"- Positive folds: {positive}/3\n- Candidate OOF SHA: {s['candidate_oof_sha256']}\n"
        +f"- Candidate test SHA: {s['candidate_test_sha256']}\n"
        +"- Evaluator never submits. CV is not official Public LB.\n",encoding="utf-8")
    print(json.dumps({"stage":s["stage"],"cv":full,"gain":gain,"fold_gain":gains},indent=2))
    return s

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-directory",required=True)
    args=parser.parse_args()
    try:validate(args.output_directory)
    except Exception as exc:
        # Record only a clear failure, never a fabricated CV.
        s=gate.read("state/v2_9_38_status.json")
        if s.get("stage")=="run_reserved":
            s.update(stage="validation_mismatch",error=str(exc)[-2000:],
                     submitted=False,updated_at=datetime.now(timezone.utc).isoformat())
            STATUS.write_text(json.dumps(s,indent=2)+"\n",encoding="utf-8")
        raise
