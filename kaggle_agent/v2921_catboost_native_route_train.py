"""V2.9.21: native-categorical CatBoost + fold-fitted unlabeled route profiles.
Distinct from LightGBM/XGBoost: native categorical statistics and CatBoost boosting.
Three-fold OOF only; no leaderboard submission; no external labels.
"""
from __future__ import annotations
import json, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA=Path("/kaggle/input")
OUTPUT=Path("/kaggle/working")
SEED=20261018
MODEL_SEED=20261029
FOLDS=3
N_TRAIN=699635
N_TEST=299844
MAX_RUNTIME_SECONDS=3300
MODEL="CatBoost native-categorical route-profile 3fold seeded"
FEATURE_FAMILY="exact_distance_profile_native_cat_v1"
CATS = ("Gender", "Customer Type", "Type of Travel", "Class")
SERVICES = (
    "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink",
    "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
)
PROFILE_COLUMNS = (
    "rating_mean", "Inflight wifi service", "Online boarding",
    "Seat comfort", "Inflight entertainment", "business",
)
ROUTE_FEATURE_COUNT = 12

def competition_csv(name):
    found = [
        path for path in DATA.rglob(name)
        if "playground-series-s6e10" in str(path.parent)
    ]
    if len(found) != 1:
        raise RuntimeError("Expected unique competition " + name + ": " + str(len(found)))
    return pd.read_csv(found[0])




def feature_names(frame):
    return [c for c in frame.columns if c not in ("id","satisfaction")]


def _route_table(fit):
    """Fit route-level *unlabelled* statistics only on outer training rows."""
    route = pd.to_numeric(fit["Flight Distance"], errors="coerce").fillna(-1).astype("int32")
    ratings = fit[list(SERVICES)].apply(pd.to_numeric, errors="coerce")
    medians = ratings.median().fillna(0.0)
    ratings = ratings.fillna(medians)
    table = pd.DataFrame(index=fit.index)
    table["route"] = route
    table["rating_mean"] = ratings.mean(axis=1).astype("float32")
    for name in PROFILE_COLUMNS[1:-1]:
        table[name] = ratings[name].astype("float32")
    table["business"] = (fit["Class"].astype("string").fillna("__NA__") == "Business").astype("float32")
    return table


def route_features(fit, holdout, test):
    """Generate 12 unlabelled fold-fitted distance-group features for three partitions."""
    reference = _route_table(fit)
    grouped = reference.groupby("route", sort=False)
    means = grouped[list(PROFILE_COLUMNS)].mean()
    counts = grouped.size()
    global_means = reference[list(PROFILE_COLUMNS)].mean()
    result = []
    for frame in (fit, holdout, test):
        route = pd.to_numeric(frame["Flight Distance"], errors="coerce").fillna(-1).astype("int32")
        z = pd.DataFrame(index=frame.index)
        z["route_profile_count_log"] = np.log1p(route.map(counts).fillna(0).astype("float32"))
        local_ratings = frame[list(SERVICES)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        local_values = {"rating_mean":local_ratings.mean(axis=1)}
        for name in PROFILE_COLUMNS[1:-1]:
            local_values[name] = local_ratings[name]
        for name in PROFILE_COLUMNS:
            avg = route.map(means[name]).fillna(float(global_means[name])).astype("float32")
            safe = name.lower().replace(" ", "_")
            z["route_profile_" + safe + "_mean"] = avg
            if name in local_values:
                z["route_profile_" + safe + "_residual"] = (
                    local_values[name].astype("float32") - avg
                )
        if len(z.columns) != ROUTE_FEATURE_COUNT:
            raise RuntimeError("Route profile schema has unexpected feature count")
        if not np.isfinite(z.to_numpy(dtype="float32")).all():
            raise RuntimeError("Nonfinite unlabelled route-profile data")
        result.append(z.astype("float32"))
    if not (result[0].columns.equals(result[1].columns)
            and result[0].columns.equals(result[2].columns)):
        raise RuntimeError("Train/validation/test route feature columns differ")
    return result




def build_features(fit, holdout, test):
    """Raw inputs + 12 fold-fitted unlabeled route features; no target encoding."""
    names=feature_names(fit)
    if len(names)!=21:
        raise RuntimeError("Expected exactly 21 original predictors")
    route_parts=route_features(fit,holdout,test)
    medians={}
    for col in names:
        if col not in CATS:
            v=pd.to_numeric(fit[col],errors="coerce")
            medians[col]=float(v.median()) if v.notna().any() else 0.0
    out=[]
    for frame,route in zip((fit,holdout,test),route_parts):
        x=frame[names].copy()
        for col in CATS:
            x[col]=x[col].astype("string").fillna("__NA__").astype(str)
        for col in names:
            if col not in CATS:
                x[col]=pd.to_numeric(x[col],errors="coerce").fillna(medians[col]).astype("float32")
        x=x.reset_index(drop=True)
        x=pd.concat([x,route.reset_index(drop=True)],axis=1)
        if len(x.columns)!=33 or x.columns.duplicated().any():
            raise RuntimeError("CatBoost native-categorical schema must have 33 unique features")
        if not np.isfinite(x.drop(columns=list(CATS)).to_numpy(dtype="float32")).all():
            raise RuntimeError("CatBoost feature matrix contains nonfinite numeric values")
        out.append(x)
    if not out[0].columns.equals(out[1].columns) or not out[1].columns.equals(out[2].columns):
        raise RuntimeError("CatBoost train/validation/test feature schema mismatch")
    return out


def load_catboost():
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier
    except ImportError:
        proc=subprocess.run([sys.executable,"-m","pip","install","--quiet",
            "--disable-pip-version-check","catboost==1.2.8"],
            capture_output=True,text=True,timeout=300)
        if proc.returncode:
            raise RuntimeError("Pinned CatBoost installation failed: "+
                               (proc.stderr or proc.stdout)[-1500:])
        from catboost import CatBoostClassifier
        return CatBoostClassifier


def dump(name,value):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding="utf-8")


def fit_fold(train,test,y,itr,iva,fold):
    started=time.monotonic()
    xtr,xval,xtest=build_features(train.iloc[itr],train.iloc[iva],test)
    if list(xtr.columns)!=list(xval.columns) or list(xtr.columns)!=list(xtest.columns):
        raise RuntimeError("CatBoost feature order mismatch")
    Model=load_catboost()
    model=Model(
        iterations=1000,depth=7,learning_rate=0.04,
        loss_function="Logloss",eval_metric="AUC",
        random_seed=MODEL_SEED+fold,verbose=False,thread_count=4,
        allow_writing_files=False,l2_leaf_reg=7.0,random_strength=0.5,
        border_count=128,bootstrap_type="Bernoulli",subsample=0.85,
        boosting_type="Plain",task_type="CPU",
    )
    # Only the outer-training partition enters fit. CatBoost computes its native
    # categorical statistics from these labels; outer validation is prediction-only.
    model.fit(xtr,y[itr],cat_features=list(CATS))
    pv=np.asarray(model.predict_proba(xval)[:,1],dtype=np.float32)
    pt=np.asarray(model.predict_proba(xtest)[:,1],dtype=np.float32)
    if pv.shape!=(len(iva),) or pt.shape!=(len(test),):
        raise RuntimeError("CatBoost returned wrong probability vector length")
    if not (np.isfinite(pv).all() and np.isfinite(pt).all()
            and ((pv>=0)&(pv<=1)).all() and ((pt>=0)&(pt<=1)).all()):
        raise RuntimeError("CatBoost probabilities nonfinite or outside [0,1]")
    del model
    return pv,pt,round(time.monotonic()-started,2),len(xtr.columns)


def main():
    started=time.monotonic()
    train=competition_csv("train.csv")
    test=competition_csv("test.csv")
    sample=competition_csv("sample_submission.csv")
    if len(train)!=N_TRAIN or len(test)!=N_TEST or len(sample)!=len(test):
        raise RuntimeError("Original official competition row counts mismatch")
    if "satisfaction" not in train or "satisfaction" in test or list(sample.columns)!=["id","satisfaction"]:
        raise RuntimeError("Official target/sample schema mismatch")
    if not np.array_equal(sample["id"].to_numpy(),test["id"].to_numpy()):
        raise RuntimeError("Official test ID sequence mismatch")
    y=pd.to_numeric(train["satisfaction"],errors="raise").to_numpy(dtype=np.int8)
    if not np.isin(y,[0,1]).all():
        raise RuntimeError("Target must be binary")
    oof=np.full(N_TRAIN,np.nan,dtype=np.float32)
    test_pred=np.zeros(N_TEST,dtype=np.float64)
    results=[]
    for fold,(itr,iva) in enumerate(StratifiedKFold(
            FOLDS,shuffle=True,random_state=SEED).split(np.zeros(N_TRAIN,dtype=np.uint8),y)):
        if time.monotonic()-started>MAX_RUNTIME_SECONDS:
            raise TimeoutError("CatBoost exceeded CPU budget; partial OOF forbidden")
        pv,pt,elapsed,nfeatures=fit_fold(train,test,y,itr,iva,fold)
        oof[iva]=pv
        test_pred+=pt/FOLDS
        results.append({"fold":fold,"validation_rows":len(iva),
            "auc":float(roc_auc_score(y[iva],pv)),"elapsed_seconds":elapsed,
            "features":nfeatures,"native_categorical_columns":list(CATS)})
        dump("v2921_progress.json",{"folds_finished":fold+1,"fold_results":results,"complete":False})
        print("V2921 CatBoost native categorical fold",fold,results[-1],flush=True)
    if not (np.isfinite(oof).all() and np.isfinite(test_pred).all()):
        raise RuntimeError("CatBoost OOF/test predictions incomplete")
    OUTPUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUTPUT/"v2921_catboost_native_route_oof.npz",
        target=y,train_id=train["id"].to_numpy(),oof_catboost_route=oof,
        test_id=test["id"].to_numpy(),pred_catboost_route=test_pred.astype(np.float32))
    dump("v2921_metrics.json",{
        "competition":"playground-series-s6e10","stage":"three_folds_complete",
        "model":MODEL,"validation":"StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)",
        "folds":3,"fold_results":results,"train_rows":len(train),"test_rows":len(test),
        "model_seed":MODEL_SEED,"iterations":1000,"depth":7,"learning_rate":0.04,
        "feature_family":FEATURE_FAMILY,"route_profile_feature_count":12,
        "original_predictor_count":21,"total_feature_count":33,
        "native_categorical_columns":list(CATS),"native_catboost_ctr":True,
        "target_encoding_columns":0,"inner_target_folds":0,
        "leakage_safe_inner_cross_fit":False,"outer_validation_used_in_training":False,
        "cpu_only":True,"gpu_verified":False,"data_source":"competition_train_only",
        "external_labels":False,"submitted":False,"full_oof_auc":float(roc_auc_score(y,oof)),"runtime_seconds":round(time.monotonic()-started,1),
    })
    print("V2921 full 3fold CatBoost ROC-AUC",roc_auc_score(y,oof),flush=True)


if __name__=="__main__":
    try:main()
    except Exception:
        dump("v2921_error.json",{"traceback":traceback.format_exc()[-8000:]})
        raise
