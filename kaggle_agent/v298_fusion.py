"""V2.9.8: one pre-registered OOF fusion; read Kaggle outputs only; never submit.

Legacy v25/v28 archives have no training IDs: require matching canonical
competition train.csv labels/order. V296/v297 must align by explicit IDs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from v291_ledger import read, write, record_experiment
from v293_blend_math import N_TRAIN, N_TEST, VALIDATION, verify_sources

ROOT=Path(__file__).resolve().parents[1]
SLUG="playground-series-s6e10"
STATUS=ROOT/"state/v2_9_8_status.json"
LEDGER=ROOT/"state/v2_9_experiments.json"
REPORT=ROOT/"reports/v2_9_8_fusion.md"
WEIGHTS={"v293":.65,"realmlp":.20,"conditional":.15}
EXPECTED_V293=.9607535043719558
EXPECTED_V296=.9609541991872741
MIN_GAIN=.00006
FOLD_FLOOR=-.00005
TERMINAL={"candidate_ready_unsubmitted","skipped_no_cv_gain","validation_blocked"}

def timestamp():
    return datetime.now(timezone.utc).isoformat()

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def checked_command(argv,timeout=900):
    p=subprocess.run(argv,cwd=ROOT,text=True,capture_output=True,timeout=timeout)
    if p.returncode:
        raise RuntimeError("Read-only Kaggle fetch failed: "+(p.stderr or p.stdout)[-600:])

def unique_file(folder,name):
    hits=list(Path(folder).rglob(name))
    if len(hits)!=1:raise ValueError(f"Expected one {name}, got {len(hits)}")
    return hits[0]

def load_npz(path,required):
    with np.load(path,allow_pickle=False) as f:
        for key in required:
            if key not in f.files:raise ValueError(f"Missing {key} in {path.name}")
        return {k:np.asarray(f[k]).copy() for k in required}

def vector(arr,expected,name):
    x=np.asarray(arr,dtype=np.float64)
    if x.ndim!=1 or len(x)!=expected or not np.isfinite(x).all():
        raise ValueError(f"Invalid {name} vector")
    if x.min()<0 or x.max()>1:raise ValueError(f"{name} values outside [0,1]")
    return x

def ids(arr,expected,name):
    x=np.asarray(arr)
    if x.ndim!=1 or len(x)!=expected or pd.isna(x).any() or len(np.unique(x))!=expected:
        raise ValueError(f"Invalid {name} IDs")
    return x

def align(pred,source_ids,canonical_ids,name):
    original=ids(source_ids,len(canonical_ids),name)
    if set(original.tolist())!=set(canonical_ids.tolist()):
        raise ValueError(f"{name} ID set mismatch")
    p=vector(pred,len(canonical_ids),name)
    indexer=pd.Index(original).get_indexer(canonical_ids)
    if (indexer<0).any():raise ValueError(f"{name} missing canonical ID")
    return p[indexer]

def rank(values):
    return rankdata(np.asarray(values,dtype=np.float64),method="average")/len(values)

def validate_official(train,test):
    if len(train)!=N_TRAIN or len(test)!=N_TEST:raise ValueError("Competition row-count mismatch")
    tid=ids(train["id"].to_numpy(),N_TRAIN,"train")
    sid=ids(test["id"].to_numpy(),N_TEST,"test")
    target=train["satisfaction"]
    if target.isna().any() or target.nunique()!=2:raise ValueError("Bad target")
    ordered=sorted(target.astype(str).unique())
    y=target.astype(str).map({ordered[0]:0,ordered[1]:1}).to_numpy(dtype=np.int8)
    return tid,sid,y

def check_provenance(statuses):
    if any(s.get("competition")!=SLUG for s in statuses.values()):
        raise ValueError("Competition mismatch")
    for key in ("v25","v28","v296","v297"):
        m=statuses[key].get("metrics") or {}
        if (m.get("validation")!=VALIDATION or m.get("train_rows")!=N_TRAIN
            or m.get("test_rows")!=N_TEST):raise ValueError(f"{key} CV protocol mismatch")
    if statuses["v296"].get("stage")!="complete" or not statuses["v296"].get("submitted"):
        raise ValueError("Official V296 not completed")
    if statuses["v297"].get("stage")!="candidate_ready_unsubmitted":
        raise ValueError("V297 not fully evaluated")
    if any(not str(s.get("kernel","")).startswith("mjw3177109/") for s in statuses.values()):
        raise ValueError("Notebook owner mismatch")

def evaluate(train,test,archive,statuses,audit_legacy_order=False):
    check_provenance(statuses)
    train_id,test_id,y=validate_official(train,test)
    if not audit_legacy_order:
        raise ValueError("Legacy archives lack train_id; canonical-order audit required")
    old25,old28=archive["v25"],archive["v28"]
    for key,old in (("v25",old25),("v28",old28)):
        if "train_id" in old and not np.array_equal(ids(old["train_id"],N_TRAIN,key),train_id):
            raise ValueError(f"{key} legacy ID mismatch")
        if not np.array_equal(old["target"],y):
            raise ValueError(f"{key} canonical target order mismatch")
    submissions={k:archive[f"{k}_submission"] for k in ("v25","v28")}
    for key,sub in submissions.items():
        if not np.array_equal(sub["id"].to_numpy(),test_id):
            raise ValueError(f"{key} test ID order mismatch")
    labels,oofs,tests=verify_sources(old25,old28,statuses["v25"],statuses["v28"],
                                    submissions["v25"],submissions["v28"])
    if not np.array_equal(labels,y):raise ValueError("Historical targets mismatch")
    base=.35*rank(oofs["v25"])+.65*rank(oofs["v28"])
    if abs(roc_auc_score(y,base)-EXPECTED_V293)>0.000002:
        raise ValueError("Cannot reproduce V293 benchmark")
    neural,cond=archive["v296"],archive["v297"]
    for key,item in (("v296",neural),("v297",cond)):
        model_ids=ids(item["train_id"],N_TRAIN,key)
        aligned_y=pd.Series(item["target"],index=model_ids).reindex(train_id).to_numpy()
        if not np.array_equal(aligned_y,y):raise ValueError(f"{key} aligned targets mismatch")
    real=align(neural["oof_realmlp"],neural["train_id"],train_id,"RealMLP OOF")
    conditional=align(cond["oof_conditional"],cond["train_id"],train_id,"Conditional OOF")
    if abs(roc_auc_score(y,real)-statuses["v296"]["metrics"]["full_oof_auc"])>.000002:
        raise ValueError("V296 model score mismatch")
    if abs(roc_auc_score(y,conditional)-statuses["v297"]["metrics"]["full_cv_auc"])>.000002:
        raise ValueError("V297 model score mismatch")
    incumbent=.8*rank(base)+.2*rank(real)
    benchmark=float(roc_auc_score(y,incumbent))
    if abs(benchmark-EXPECTED_V296)>0.000002:
        raise ValueError("Cannot reconstruct V296 incumbent")
    candidate=.65*rank(base)+.20*rank(real)+.15*rank(conditional)
    auc=float(roc_auc_score(y,candidate))
    fold_auc=[];fold_gain=[]
    for _,iv in StratifiedKFold(3,shuffle=True,random_state=20261018).split(np.zeros(len(y)),y):
        fold_auc.append(float(roc_auc_score(y[iv],candidate[iv])))
        fold_gain.append(float(fold_auc[-1]-roc_auc_score(y[iv],incumbent[iv])))
    passed=(auc-benchmark>=MIN_GAIN and sum(g>0 for g in fold_gain)>=2
            and min(fold_gain)>FOLD_FLOOR)
    result={"competition":SLUG,"submitted":False,
            "stage":"candidate_ready_unsubmitted" if passed else "skipped_no_cv_gain",
            "validation":VALIDATION,"weights":WEIGHTS,
            "baseline_v293_cv":float(roc_auc_score(y,base)),
            "incumbent_v296_cv":benchmark,"candidate_cv":auc,
            "gain":auc-benchmark,"fold_auc":fold_auc,"fold_gain":fold_gain,
            "positive_folds":sum(g>0 for g in fold_gain),
            "min_gain":MIN_GAIN,"min_fold_gain_allowed":FOLD_FLOOR,
            "historical_alignment":"canonical order verified; legacy OOF has no train_id",
            "new_model_alignment":"explicit train/test ID joins"}
    if passed:
        test_base=.35*tests["v25"]+.65*tests["v28"]
        test_real=align(neural["pred_realmlp"],neural["test_id"],test_id,"RealMLP test")
        test_cond=align(cond["test_conditional"],cond["test_id"],test_id,"Conditional test")
        values=.65*rank(test_base)+.20*rank(test_real)+.15*rank(test_cond)
        result["candidate_test_hash"]=hashlib.sha256(values.astype("<f8").tobytes()).hexdigest()
    return result

def download_inputs(folder,statuses):
    root=Path(folder);data=root/"competition";data.mkdir(parents=True,exist_ok=True)
    checked_command(["kaggle","competitions","download","-c",SLUG,"-p",str(data),"-q"])
    for path in data.glob("*.zip"):
        with zipfile.ZipFile(path) as archive:
            if any(Path(m.filename).is_absolute() or ".." in Path(m.filename).parts
                   for m in archive.infolist()):
                raise ValueError("Unsafe downloaded archive")
            archive.extractall(data)
    train=pd.read_csv(unique_file(data,"train.csv"))
    test=pd.read_csv(unique_file(data,"test.csv"))
    archives={};files={}
    artifact_names={"v25":"v25_oof.npz","v28":"v28_oof.npz",
                    "v296":"v296_realmlp_oof.npz","v297":"v297_oof.npz"}
    for key,name in artifact_names.items():
        dest=root/key;dest.mkdir(parents=True,exist_ok=True)
        checked_command(["kaggle","kernels","output",statuses[key]["kernel"],
                         "-p",str(dest),"-o","-q"])
        path=unique_file(dest,name)
        files[key]={"notebook":statuses[key]["kernel"],"sha256":sha256(path)}
        if key=="v25":required=("target","oof_lgb","oof_xgb")
        elif key=="v28":required=("target","oof_lgb","oof_cat")
        elif key=="v296":required=("target","train_id","test_id","oof_realmlp","pred_realmlp")
        else:required=("target","train_id","test_id","oof_conditional","test_conditional")
        archives[key]=load_npz(path,required)
        if key in ("v25","v28"):
            sub=unique_file(dest,"candidate_submission.csv")
            archives[f"{key}_submission"]=pd.read_csv(sub)
            files[key]["submission_sha256"]=sha256(sub)
    return train,test,archives,files

def check_competition():
    ladder=read(ROOT/"state/dual_ladder.json")
    if ladder.get("active_competition")!=SLUG:raise ValueError("Contest focus changed")
    deadline=datetime.fromisoformat(str(ladder.get("active_deadline","")).replace("Z","+00:00"))
    if deadline.tzinfo is None or datetime.now(timezone.utc)>=deadline:
        raise ValueError("Competition deadline expired")

def run():
    if read(STATUS).get("stage") in TERMINAL:
        print("Already terminal; do not repeat V298");return
    check_competition()
    paths={"v25":"state/v2_5_status.json","v28":"state/v2_8_status.json",
           "v296":"state/v2_9_6_status.json","v297":"state/v2_9_7_status.json"}
    statuses={k:read(ROOT/p) for k,p in paths.items()}
    check_provenance(statuses)
    with tempfile.TemporaryDirectory(prefix="kas_v298_") as temp:
        train,test,archive,files=download_inputs(temp,statuses)
        result=evaluate(train,test,archive,statuses,audit_legacy_order=True)
    result["source_artifacts"]=files
    result["verified_at_utc"]=timestamp()
    write(STATUS,result)
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text("# V2.9.8 read-only OOF audit\n\n"+
          f"- UTC: {result['verified_at_utc']}\n"+
          f"- Stage: {result['stage']} (no Kaggle submit)\n"+
          f"- Fixed weights: {WEIGHTS}\n"+
          f"- V296 benchmark: {result['incumbent_v296_cv']}\n"+
          f"- Threefold candidate: {result['candidate_cv']}\n"+
          f"- CV gain: {result['gain']}\n"+
          f"- Fold gains: {result['fold_gain']}\n"+
          "- Provenance: old canonical source ordering, new model explicit ID joins.\n",
          encoding="utf-8")
    record_experiment(LEDGER,{"competition":SLUG,
        "candidate":"v298_precommitted_v293_65_realmlp_20_conditional_15",
        "validation":VALIDATION,"seed":20261018,
        "parameters":{"weights":WEIGHTS,"benchmark":EXPECTED_V296,"min_gain":MIN_GAIN},
        "status":"completed" if result["stage"]=="candidate_ready_unsubmitted"
                 else "skipped_no_cv_gain",
        "cv":result["candidate_cv"],"completed_at":result["verified_at_utc"],
        "gpu_seconds":0,"kernel_sources":[s["kernel"] for s in statuses.values()]})
    print(json.dumps({k:result[k] for k in ("stage","candidate_cv","gain","fold_gain")},indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-existing-only",action="store_true",default=True)
    parser.parse_args()
    run()
