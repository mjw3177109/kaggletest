"""V2936 immutable original-37 XGBoost 3.0.5 DART T4 experiment.
Exactly one frozen threefold OOF; no early stopping, parameter search or submission.
Self-contained Kaggle main.py. All preprocessing/TE fitted on outer train only.
"""
from __future__ import annotations
import hashlib
import json
import platform
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

INPUT=Path("/kaggle/input")
OUTPUT=Path("/kaggle/working")
SEED=20261018
MODEL_SEED=20261104
FOLDS=3
TRAIN_ROWS=699635
TEST_ROWS=299844
RUNTIME_BUDGET_SECONDS=30000
CATS=["Gender","Customer Type","Type of Travel","Class"]
DUAL=["Age","Flight Distance","Inflight wifi service","Online boarding",
      "Seat comfort","Inflight entertainment","Leg room service"]
PAIRS=[("Class","Type of Travel","Gender"),
       ("Inflight wifi service","Online boarding"),
       ("Class","Customer Type","Online boarding")]
FEATURE_FAMILY="v2915_original37_3_inner5fold_TE_no_extra_features"
CANONICAL=["Age","Flight Distance","Inflight wifi service","Departure/Arrival time convenient","Ease of Online booking","Gate location","Food and drink","Online boarding","Seat comfort","Inflight entertainment","On-board service","Leg room service","Baggage handling","Checkin service","Cleanliness","Departure Delay in Minutes","Arrival Delay in Minutes","Gender","Customer Type","Type of Travel","Class","dual_Age","dual_Flight Distance","dual_Inflight wifi service","dual_Online boarding","dual_Seat comfort","dual_Inflight entertainment","dual_Leg room service","combo_0","count_combo_0","combo_1","count_combo_1","combo_2","count_combo_2","te_0","te_1","te_2"]
CATEGORICAL=set(CATS+["dual_"+c for c in DUAL]+["combo_0","combo_1","combo_2"])
PARAMS={
    "booster":"dart","objective":"binary:logistic","eval_metric":"auc",
    "tree_method":"hist","device":"cuda","n_estimators":900,
    "learning_rate":0.035,"max_depth":4,"min_child_weight":10,
    "subsample":0.85,"colsample_bytree":0.85,"max_bin":256,
    "reg_lambda":5.0,"reg_alpha":0.0,"gamma":0.0,
    "rate_drop":0.05,"skip_drop":0.0,"sample_type":"uniform",
    "normalize_type":"tree","n_jobs":4,
}

def dump(filename, value):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/filename).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding="utf-8")

def only_competition_file(name):
    files=[p for p in INPUT.rglob(name) if "playground-series-s6e10" in str(p.parent)]
    if len(files)!=1:
        raise RuntimeError("Exactly one official contest file required: "+name)
    return pd.read_csv(files[0])

def prove_gpu():
    if xgb.__version__!="3.0.5":
        raise RuntimeError("Pinned xgboost==3.0.5 required; got "+xgb.__version__)
    probe=subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],
                         capture_output=True,text=True,timeout=20,check=True)
    devices=[v.strip() for v in probe.stdout.splitlines() if v.strip()]
    if not devices or "T4" not in devices[0]:
        raise RuntimeError("Primary NVIDIA T4 not verified: "+repr(devices))
    evidence={"ok":True,"device":devices[0],"devices":devices,"xgboost_version":xgb.__version__}
    dump("v2936_hardware.json",evidence)
    return evidence

def _categorical(train, validation, name, left, right):
    fit = left.astype("string").fillna("__NA__")
    val = right.astype("string").fillna("__NA__")
    categories = pd.Index(fit.unique()).astype(str).tolist()
    train[name] = pd.Categorical(fit.astype(str), categories=categories)
    validation[name] = pd.Categorical(val.astype(str), categories=categories)


def engineer(train, validation):
    """All encoders/maps are fitted only on the *outer training partition*."""
    features = [c for c in train.columns if c not in ("id", "satisfaction")]
    tr = pd.DataFrame(index=train.index)
    va = pd.DataFrame(index=validation.index)
    for col in features:
        if col in CATS:
            _categorical(tr, va, col, train[col], validation[col])
        else:
            a = pd.to_numeric(train[col], errors="coerce")
            b = pd.to_numeric(validation[col], errors="coerce")
            median = float(a.median())
            tr[col] = a.fillna(median).astype("float32")
            va[col] = b.fillna(median).astype("float32")
    for col in DUAL:
        _categorical(tr, va, "dual_" + col, train[col], validation[col])

    te_train = pd.DataFrame(index=train.index)
    te_val = pd.DataFrame(index=validation.index)
    for i, cols in enumerate(PAIRS):
        name = "combo_" + str(i)
        tr_key = train[list(cols)].astype("string").fillna("__NA__").agg("|".join, axis=1)
        va_key = validation[list(cols)].astype("string").fillna("__NA__").agg("|".join, axis=1)
        _categorical(tr, va, name, tr_key, va_key)
        count = tr_key.value_counts()
        tr["count_" + name] = np.log1p(tr_key.map(count).astype("float32")).to_numpy()
        va["count_" + name] = np.log1p(va_key.map(count).fillna(0).astype("float32")).to_numpy()
        te_train[name] = tr_key.astype(str)
        te_val[name] = va_key.astype(str)

    # Inner cross-fitting, including the training row itself only via out-of-fold
    # mapping. Never use outer validation labels when fitting this encoder.
    return tr, va, te_train, te_val




def fold_matrix(xa,xv,xt):
    """Frozen order, fold-local categories, unseen category sentinel -1, finite numeric."""
    columns=list(xa.columns)
    if columns!=CANONICAL or list(xv.columns)!=CANONICAL or list(xt.columns)!=CANONICAL:
        raise ValueError("V2936 frozen 37-feature order mismatch")
    if len(columns)!=37 or columns[-3:]!=["te_0","te_1","te_2"]:
        raise ValueError("V2936 protected TE columns mismatch")
    vocab={}
    frames=[]
    for col in columns:
        if col in CATEGORICAL:
            if not isinstance(xa[col].dtype,pd.CategoricalDtype):
                raise ValueError("Outer-train categorical vocabulary missing: "+col)
            vocab[col]=list(map(str,xa[col].cat.categories))
    for frame in (xa,xv,xt):
        out=pd.DataFrame(index=frame.index)
        for col in columns:
            if col in CATEGORICAL:
                if not isinstance(frame[col].dtype,pd.CategoricalDtype):
                    raise ValueError("Fold-local category type mismatch: "+col)
                if list(map(str,frame[col].cat.categories))!=vocab[col]:
                    raise ValueError("Validation/test category vocabulary differs: "+col)
                out[col]=frame[col].cat.codes.to_numpy(dtype="float32")
            else:
                arr=pd.to_numeric(frame[col],errors="raise").to_numpy(dtype="float32")
                if not np.isfinite(arr).all():
                    raise ValueError("Nonfinite numeric field: "+col)
                out[col]=arr
        frames.append(out)
    vocab_sha=hashlib.sha256(json.dumps(vocab,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    return frames,vocab_sha

def fit_fold(train,test,y,itr,iva,fold):
    started=time.monotonic()
    xa,xv,te_train,te_val=engineer(train.iloc[itr],train.iloc[iva])
    _,xt,_,te_test=engineer(train.iloc[itr],test)
    if list(te_train.columns)!=["combo_0","combo_1","combo_2"]:
        raise RuntimeError("Original three inner-fivefold TE sources changed")
    enc=TargetEncoder(target_type="binary",smooth="auto",cv=5,
                      shuffle=True,random_state=SEED+fold)
    te_cols=["te_0","te_1","te_2"]
    xa[te_cols]=enc.fit_transform(te_train,y[itr]).astype("float32")
    xv[te_cols]=enc.transform(te_val).astype("float32")
    xt[te_cols]=enc.transform(te_test).astype("float32")
    frames,vocab_sha=fold_matrix(xa,xv,xt)
    model=xgb.XGBClassifier(**PARAMS,random_state=MODEL_SEED+fold)
    model.fit(frames[0],y[itr],verbose=False)
    config=json.loads(model.get_booster().save_config())
    actual_device=str(config.get("learner",{}).get("generic_param",{}).get("device",""))
    if not actual_device.startswith("cuda"):
        raise RuntimeError("XGBoost silently fell back from CUDA: "+repr(actual_device))
    # DART stochastic dropout is forbidden at prediction; explicitly use every tree.
    pv=np.asarray(model.predict_proba(frames[1],iteration_range=(0,PARAMS["n_estimators"]))[:,1],dtype="float32")
    pt=np.asarray(model.predict_proba(frames[2],iteration_range=(0,PARAMS["n_estimators"]))[:,1],dtype="float32")
    if pv.shape!=(len(iva),) or pt.shape!=(len(test),):
        raise RuntimeError("Incomplete V2936 validation/test prediction")
    if not all(np.isfinite(a).all() and np.all((a>=0)&(a<=1)) for a in (pv,pt)):
        raise RuntimeError("Invalid V2936 probabilities")
    info={"fold":fold,"train_rows":len(itr),"validation_rows":len(iva),
          "model_seed":MODEL_SEED+fold,"features":len(CANONICAL),
          "feature_order":list(CANONICAL),"categorical_columns":[c for c in CANONICAL if c in CATEGORICAL],
          "numeric_columns":[c for c in CANONICAL if c not in CATEGORICAL],
          "category_vocabulary_sha256":vocab_sha,
          "validation_index_sha256":hashlib.sha256(np.asarray(iva,dtype="<i8").tobytes()).hexdigest(),
          "auc":float(roc_auc_score(y[iva],pv)),
          "rounds":PARAMS["n_estimators"],"inference_iteration_range":[0,PARAMS["n_estimators"]],
          "booster_device":actual_device,
          "elapsed_seconds":round(time.monotonic()-started,2)}
    del model
    return pv,pt,info

def main():
    started=time.monotonic()
    hardware=prove_gpu()
    train=only_competition_file("train.csv")
    test=only_competition_file("test.csv")
    sample=only_competition_file("sample_submission.csv")
    if len(train)!=TRAIN_ROWS or len(test)!=TEST_ROWS or len(sample)!=TEST_ROWS:
        raise RuntimeError("Official train/test/sample row counts changed")
    if list(sample.columns)!=["id","satisfaction"] or not np.array_equal(sample["id"].to_numpy(),test["id"].to_numpy()):
        raise RuntimeError("Official test IDs and sample submission differ")
    if "satisfaction" not in train or "satisfaction" in test or "id" not in train:
        raise RuntimeError("Official target/ID schema mismatch")
    if train["id"].duplicated().any() or test["id"].duplicated().any():
        raise RuntimeError("Duplicate competition IDs")
    y=pd.to_numeric(train["satisfaction"],errors="raise").to_numpy()
    if not np.isin(y,[0,1]).all():
        raise RuntimeError("Binary labels 0/1 required")
    folds=list(StratifiedKFold(3,shuffle=True,random_state=SEED).split(np.zeros(len(y),dtype="uint8"),y))
    oof=np.full(len(y),np.nan,dtype="float32")
    test_pred=np.zeros(len(test),dtype="float64")
    results=[]
    for fold,(itr,iva) in enumerate(folds):
        if time.monotonic()-started>RUNTIME_BUDGET_SECONDS:
            raise TimeoutError("V2936 exceeded runtime budget; refuse incomplete OOF")
        pv,pt,info=fit_fold(train,test,y,itr,iva,fold)
        oof[iva]=pv
        test_pred+=pt/FOLDS
        results.append(info)
        dump("v2936_progress.json",{"finished":fold+1,"complete":False,"fold_results":results})
    if len(results)!=3 or not np.isfinite(oof).all() or not np.isfinite(test_pred).all():
        raise RuntimeError("Incomplete V2936 threefold OOF")
    OUTPUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUTPUT/"v2936_dart_oof.npz",
        target=y.astype("int8"),train_id=train["id"].to_numpy(),
        oof_dart=oof,test_id=test["id"].to_numpy(),
        pred_dart=test_pred.astype("float32"))
    dump("v2936_metrics.json",{
        "competition":"playground-series-s6e10","stage":"three_folds_complete",
        "model":"XGBoost 3.0.5 DART original V2915 37-feature independent",
        "model_family":"XGBoost_DART_T4_original37",
        "feature_family":FEATURE_FAMILY,"train_rows":len(train),"test_rows":len(test),
        "validation":"StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)",
        "folds":3,"fold_results":results,
        "full_oof_auc":float(roc_auc_score(y,oof)),
        "gpu_verified":True,"hardware":hardware,
        "feature_count":37,"inner_target_folds":5,
        "leakage_safe_inner_cross_fit":True,"outer_validation_used_in_training":False,
        "external_labels":False,"pseudo_labels":False,"submitted":False,
        "model_seed":MODEL_SEED,"model_parameters":PARAMS,"xgboost_version":xgb.__version__,
        "early_stopping":False,"weight_search":False,"feature_search":False,
        "source_code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment_versions":{"python":platform.python_version(),
            "xgboost":xgb.__version__,"numpy":np.__version__,"pandas":pd.__version__,
            "sklearn":__import__("sklearn").__version__},
        "runtime_seconds":round(time.monotonic()-started,1),"ready_for_submission":False})
    print("V2936 independent threefold OOF AUC",roc_auc_score(y,oof),flush=True)

if __name__=="__main__":
    try:
        main()
    except Exception:
        dump("v2936_error.json",{"traceback":traceback.format_exc()[-8000:]})
        raise
