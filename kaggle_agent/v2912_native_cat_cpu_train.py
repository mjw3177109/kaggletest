"""V2.9.12 independent native categorical CatBoost CPU, fixed one-run protocol.

No target encoding outside native CatBoost ordered CTR; no validation labels
enter features or fitting, and no Kaggle competition submission occurs here.
"""
from __future__ import annotations
import json
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA=Path("/kaggle/input")
OUT=Path("/kaggle/working")
SEED=20261018
FOLDS=3
MAX_SECONDS=3200
CAT=("Gender","Customer Type","Type of Travel","Class")
SERVICES=("Inflight wifi service","Departure/Arrival time convenient",
    "Ease of Online booking","Gate location","Food and drink","Online boarding",
    "Seat comfort","Inflight entertainment","On-board service","Leg room service",
    "Baggage handling","Checkin service","Cleanliness")
PAIRS=(("Online boarding","Inflight wifi service","Type of Travel"),
       ("Inflight entertainment","Seat comfort","Class"),
       ("On-board service","Baggage handling","Customer Type"),
       ("Departure/Arrival time convenient","Ease of Online booking","Type of Travel"))
PARAMS={"iterations":750,"depth":6,"learning_rate":0.045,
        "l2_leaf_reg":7,"bootstrap_type":"Bernoulli","subsample":.85,
        "one_hot_max_size":8,"max_ctr_complexity":1,"thread_count":4,
        "task_type":"CPU"}

def dump(name,obj):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")

def file(name):
    matches=[p for p in DATA.rglob(name) if "playground-series-s6e10" in str(p.parent)]
    if len(matches)!=1:
        raise ValueError(f"Expected exactly one competition {name}; got {len(matches)}")
    return pd.read_csv(matches[0])

def features(frame):
    """All transformations are row-only; no target, global frequency, or labels."""
    required=set(CAT)|set(SERVICES)|{"Age","Flight Distance",
               "Departure Delay in Minutes","Arrival Delay in Minutes"}
    if not required<=set(frame):
        raise ValueError("Missing competition feature columns: "+str(sorted(required-set(frame))))
    numeric={}
    for col in frame.columns:
        if col in ("id","satisfaction") or col in CAT:
            continue
        numeric[col]=pd.to_numeric(frame[col],errors="coerce").fillna(-1).astype("float32")
    x=pd.DataFrame(numeric,index=frame.index)
    for col in CAT:
        x[col]=frame[col].astype("string").fillna("__NA__").astype(str)
    group=(x["Type of Travel"]+"|"+x["Customer Type"]+"|"+x["Class"])
    x["traveler_segment"]=group
    cat_features=list(CAT)+["traveler_segment"]
    for ix,(a,b,context) in enumerate(PAIRS):
        left=x[a];right=x[b]
        x[f"gap_{ix}"]=(left-right).astype("float32")
        col=f"paired_service_{ix}"
        x[col]=(frame[a].astype("string").fillna("__NA__")+"|"+
                frame[b].astype("string").fillna("__NA__")+"|"+
                x[context].astype("string")).astype(str)
        cat_features.append(col)
    ratings=x[list(SERVICES)]
    x["rating_mean"]=ratings.mean(axis=1).astype("float32")
    x["rating_std"]=ratings.std(axis=1).fillna(0).astype("float32")
    x["rating_range"]=(ratings.max(axis=1)-ratings.min(axis=1)).astype("float32")
    x["rating_low_count"]=(ratings<=2).sum(axis=1).astype("float32")
    x["rating_high_count"]=(ratings>=4).sum(axis=1).astype("float32")
    departure=x["Departure Delay in Minutes"].clip(lower=0)
    arrival=x["Arrival Delay in Minutes"].clip(lower=0)
    x["combined_delay"]=(departure+arrival).astype("float32")
    x["delay_gap"]=(arrival-departure).astype("float32")
    x["distance_age_interaction"]=(x["Flight Distance"]/(x["Age"]+1).clip(lower=1)).astype("float32")
    if x.columns.duplicated().any() or set(cat_features)-set(x):
        raise ValueError("Duplicate features or missing CatBoost categorical columns")
    other=x.drop(columns=cat_features)
    if not np.isfinite(other.to_numpy(dtype=np.float64)).all():
        raise ValueError("Nonfinite numeric feature values")
    return x,cat_features

def main():
    from catboost import CatBoostClassifier
    start=time.monotonic()
    train=file("train.csv")
    test=file("test.csv")
    sample=file("sample_submission.csv")
    if len(train)!=699635 or len(test)!=299844 or len(sample)!=299844:
        raise ValueError("Competition row count changed; abort")
    if list(sample.columns)!=["id","satisfaction"] or "satisfaction" not in train:
        raise ValueError("Unexpected competition target/sample columns")
    y=pd.to_numeric(train["satisfaction"],errors="raise").to_numpy(dtype=np.int8)
    if not np.isin(y,[0,1]).all() or len(np.unique(y))!=2:
        raise ValueError("Invalid binary target")
    if train["id"].duplicated().any() or test["id"].duplicated().any():
        raise ValueError("Duplicate canonical row IDs")
    if not np.array_equal(test["id"].to_numpy(),sample["id"].to_numpy()):
        raise ValueError("Official sample/test ID order mismatch")
    x,cats=features(train)
    xt,test_cats=features(test)
    if cats!=test_cats or list(x.columns)!=list(xt.columns):
        raise ValueError("Training and test feature schemas differ")
    splitter=StratifiedKFold(3,shuffle=True,random_state=SEED)
    oof=np.full(len(train),np.nan,dtype=np.float32)
    test_pred=np.zeros(len(test),dtype=np.float64)
    folds=[]
    for fold,(fit_idx,valid_idx) in enumerate(splitter.split(np.zeros(len(y)),y)):
        if time.monotonic()-start>=MAX_SECONDS:
            raise TimeoutError("CatBoost CPU quota guard; refusing partial OOF")
        model=CatBoostClassifier(loss_function="Logloss",eval_metric="AUC",
                random_seed=SEED+fold,allow_writing_files=False,
                verbose=False,**PARAMS)
        model.fit(x.iloc[fit_idx],y[fit_idx],cat_features=cats,verbose=False)
        if str(model.get_all_params().get("task_type",""))!="CPU":
            raise RuntimeError("CatBoost did not confirm CPU execution")
        val=model.predict_proba(x.iloc[valid_idx])[:,1]
        tst=model.predict_proba(xt)[:,1]
        if not np.isfinite(val).all() or not np.isfinite(tst).all() or (
                (val<0).any() or (val>1).any() or (tst<0).any() or (tst>1).any()):
            raise ValueError("Non-finite/out-of-range prediction")
        oof[valid_idx]=val.astype("float32")
        test_pred+=tst/FOLDS
        folds.append({"fold":fold,"validation_rows":len(valid_idx),
                      "auc":float(roc_auc_score(y[valid_idx],val)),
                      "trained_iterations":PARAMS["iterations"],
                      "cat_features":len(cats),"features":len(x.columns)})
        dump("v2912_progress.json",{"folds_finished":fold+1,"fold_results":folds})
        print("V2912 native CatBoost fold",fold,folds[-1]["auc"],flush=True)
        del model
    if not np.isfinite(oof).all() or time.monotonic()-start>=MAX_SECONDS:
        raise RuntimeError("CatBoost threefold not complete within CPU time budget")
    out=sample.copy()
    out["satisfaction"]=test_pred
    OUT.mkdir(parents=True,exist_ok=True)
    out.to_csv(OUT/"v2912_candidate_unsubmitted.csv",index=False)
    np.savez_compressed(OUT/"v2912_oof.npz",
                       target=y,train_id=train["id"].to_numpy(),
                       oof_native_cat=oof,test_id=test["id"].to_numpy(),
                       test_native_cat=test_pred.astype(np.float32))
    metrics={"competition":"playground-series-s6e10",
             "model":"native_categorical_catboost_cpu",
             "train_rows":len(y),"test_rows":len(test),
             "validation":"StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)",
             "folds":3,"fold_results":folds,"full_cv_auc":float(roc_auc_score(y,oof)),
             "native_ordered_categorical":True,
             "cat_features":len(cats),"features_per_fold":len(x.columns),
             "source":"competition_train_only","external_teacher":False,
             "outer_validation_used_for_early_stopping":False,
             "task_type":"CPU","parameters":PARAMS,"stage":"full_threefold_complete",
             "elapsed_seconds":time.monotonic()-start,"submitted":False}
    dump("v2912_metrics.json",metrics)
    print("V2912 full 3fold native CatBoost OOF",metrics["full_cv_auc"],flush=True)

if __name__=="__main__":
    try:main()
    except Exception:
        dump("v2912_error.json",{"traceback":traceback.format_exc()[-7000:]})
        raise
