"""V2.9.19R2: independent TabM route-profile 3fold GPU challenger.
TabM is NOT RealMLP. Fixed 8 epochs and 8 efficient ensemble members.
All 56 fold-local features are reused from V2918 feature engineering.
Official Kaggle test labels are never used; no competition submit call exists.
"""
from __future__ import annotations
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

DATA = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working")
SEED = 20261018
MODEL_SEED = 20261027
FOLDS = 3
TABM_K = 8
EPOCHS = 8
MAX_RUNTIME_SECONDS = 3250
N_TRAIN = 699635
N_TEST = 299844
MODEL = "TabM-D k=8 8 epochs route-profile TE23 independent seeded"
FEATURE_FAMILY = "exact_distance_profile_te23_v1"


# Self-contained feature pipeline: Kaggle executes main.py as /kaggle/src/script.py.
# Never import a separate helper module in Kaggle's single-script runtime.
INNER_FOLDS = 5
TE_COUNT = 23
ROUTE_FEATURE_COUNT = 12
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

def dump(name, value):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def prove_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: no silent CPU fallback for TabM")
    probe = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=20,
    )
    devices = [d.strip() for d in probe.stdout.splitlines() if d.strip()]
    if not any("T4" in d for d in devices):
        raise RuntimeError("Expected genuine Kaggle Tesla T4 hardware")
    evidence = {
        "ok":True,"devices":devices,"device":str(torch.cuda.get_device_name(0)),
        "cuda_available":True,"torch_version":torch.__version__,
    }
    dump("v2919r2_hardware.json",evidence)
    return evidence


def load_tabm():
    try:
        import pytabkit
    except ImportError:
        result = subprocess.run(
            [sys.executable,"-m","pip","install","--quiet",
             "--disable-pip-version-check","pytabkit==1.7.3"],
            capture_output=True,text=True,timeout=240,
        )
        if result.returncode:
            raise RuntimeError("Pinned pytabkit==1.7.3 install failed: " +
                               (result.stderr or result.stdout)[-1500:])
    from pytabkit import TabM_D_Classifier
    return TabM_D_Classifier


def competition_csv(name):
    found = [
        path for path in DATA.rglob(name)
        if "playground-series-s6e10" in str(path.parent)
    ]
    if len(found) != 1:
        raise RuntimeError("Expected unique competition " + name + ": " + str(len(found)))
    return pd.read_csv(found[0])


def feature_names(fit):
    return [column for column in fit.columns if column not in ("id", "satisfaction")]


def target_keys(frame, names):
    """Only original input fields; target and row id cannot enter any encoding key."""
    z = pd.DataFrame(index=frame.index)
    for name in names:
        z["raw_" + name] = frame[name].astype("string").fillna("__NA__").astype(str)
    part = [frame[c].astype("string").fillna("__NA__").astype(str) for c in (
        "Class", "Type of Travel", "Customer Type",
    )]
    distance = frame["Flight Distance"].astype("string").fillna("__NA__").astype(str)
    z["cross_traveler"] = part[0] + "|" + part[1] + "|" + part[2]
    z["cross_exact_route_class"] = distance + "|" + part[0]
    if len(z.columns) != TE_COUNT or z.isna().any().any():
        raise RuntimeError("Unexpected 23 leakage-safe TE columns")
    return z


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


def basic_features(fit, holdout, test, names):
    result = [pd.DataFrame(index=part.index) for part in (fit, holdout, test)]
    for col in names:
        if col in CATS:
            vocabulary = pd.Index(fit[col].astype("string").fillna("__NA__").astype(str).unique())
            for out, frame in zip(result, (fit, holdout, test)):
                vals = frame[col].astype("string").fillna("__NA__").astype(str)
                out[col] = pd.Categorical(vals, categories=vocabulary).codes.astype("float32")
        else:
            fit_vals = pd.to_numeric(fit[col], errors="coerce")
            median = float(fit_vals.median())
            if not np.isfinite(median):
                raise RuntimeError("All-missing numeric field: " + col)
            for out, frame in zip(result, (fit, holdout, test)):
                out[col] = pd.to_numeric(frame[col], errors="coerce").fillna(median).astype("float32")
    return result


def matrices(fit, holdout, test, labels, fold):
    names = feature_names(fit)
    if len(names) != 21 or not set(CATS + SERVICES + ("Flight Distance",)).issubset(names):
        raise ValueError("Official 21-feature schema not verified")
    key_train, key_val, key_test = (target_keys(frame, names) for frame in (fit, holdout, test))
    encoder = TargetEncoder(target_type="binary", smooth="auto", cv=INNER_FOLDS,
                            shuffle=True, random_state=SEED + fold)
    train_te = encoder.fit_transform(key_train, np.asarray(labels, dtype=np.int8))
    val_te = encoder.transform(key_val)
    test_te = encoder.transform(key_test)
    route = route_features(fit, holdout, test)
    basic = basic_features(fit, holdout, test, names)
    outputs = []
    for i, frame in enumerate((fit, holdout, test)):
        encoded = pd.DataFrame(
            (train_te, val_te, test_te)[i].astype("float32"),
            index=frame.index,
            columns=["te_" + str(j) for j in range(TE_COUNT)],
        )
        values = pd.concat((basic[i], encoded, route[i]), axis=1)
        if values.columns.duplicated().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError("Invalid or leaked fold-local feature matrix")
        outputs.append(values.astype("float32"))
    if not (outputs[0].columns.equals(outputs[1].columns)
            and outputs[0].columns.equals(outputs[2].columns)):
        raise ValueError("Feature matrix column mismatch")
    return outputs



def fit_one_fold(train, test, targets, itr, iva, fold):
    from time import monotonic
    started=monotonic()
    xtr, xval, xtest = matrices(
        train.iloc[itr], train.iloc[iva], test, targets[itr], fold,
    )
    if len(xtr.columns)!=56 or not xtr.columns.equals(xval.columns):
        raise ValueError("Exact route-profile 56-feature fold mismatch")
    Model=load_tabm()
    model=Model(
        device="cuda:0", random_state=MODEL_SEED+fold,
        n_cv=1, n_refit=0, val_fraction=0.10,
        tabm_k=TABM_K, num_emb_type="none",
        n_epochs=EPOCHS, patience=12, batch_size=1024,
        d_block=192, n_blocks=2, dropout=0.10,
        lr=0.002, weight_decay=0.0,
        gradient_clipping_norm=1.0,
        allow_amp=False, compile_model=False,
        tfms=["quantile_tabr"], n_threads=4, verbosity=1,
        val_metric_name="1-auc_ovr",
    )
    # The 10% internal holdout is taken only from the outer fold training rows.
    # Never supply outer validation labels/rows to model.fit or as early stopping.
    model.fit(xtr, pd.Series(targets[itr]))
    pv=np.asarray(model.predict_proba(xval)[:,1],dtype=np.float32)
    pt=np.asarray(model.predict_proba(xtest)[:,1],dtype=np.float32)
    if pv.shape!=(len(iva),) or pt.shape!=(len(test),):
        raise RuntimeError("TabM returned wrong OOF or test vector sizes")
    if not (np.isfinite(pv).all() and np.isfinite(pt).all()
            and (pv>=0).all() and (pv<=1).all()
            and (pt>=0).all() and (pt<=1).all()):
        raise RuntimeError("TabM returned nonfinite/out-of-range probability")
    del model
    import torch
    torch.cuda.empty_cache()
    return pv, pt, round(monotonic()-started,2),len(xtr.columns)


def main():
    started=time.monotonic()
    hw=prove_gpu()
    train=competition_csv("train.csv")
    test=competition_csv("test.csv")
    sample=competition_csv("sample_submission.csv")
    if len(train)!=N_TRAIN or len(test)!=N_TEST or len(sample)!=len(test):
        raise RuntimeError("Kaggle original competition row counts did not match")
    if "satisfaction" not in train or "satisfaction" in test or list(sample.columns)!=["id","satisfaction"]:
        raise RuntimeError("Official target/sample schema mismatch")
    if not np.array_equal(sample["id"].to_numpy(), test["id"].to_numpy()):
        raise RuntimeError("Official test ID sequence mismatch")
    y=pd.to_numeric(train["satisfaction"],errors="raise").to_numpy(dtype=np.int8)
    if not np.isin(y,[0,1]).all():raise RuntimeError("Nonbinary label")
    oof=np.full(N_TRAIN,np.nan,dtype=np.float32)
    test_pred=np.zeros(N_TEST,dtype=np.float64)
    results=[]
    for fold,(itr,iva) in enumerate(StratifiedKFold(
            FOLDS,shuffle=True,random_state=SEED).split(
                np.zeros(N_TRAIN,dtype=np.uint8),y)):
        if time.monotonic()-started>MAX_RUNTIME_SECONDS:
            raise TimeoutError("TabM exceeded GPU budget; refusing partial OOF")
        pv,pt,elapsed,features=fit_one_fold(train,test,y,itr,iva,fold)
        oof[iva]=pv
        test_pred+=pt/FOLDS
        results.append({"fold":fold,"validation_rows":len(iva),
                        "auc":float(roc_auc_score(y[iva],pv)),
                        "elapsed_seconds":elapsed,"features":features})
        dump("v2919r2_progress.json",{"folds_finished":fold+1,
             "fold_results":results,"complete":False})
        print("V2919R2 independent TabM T4 fold",fold,results[-1],flush=True)
    if not (np.isfinite(oof).all() and np.isfinite(test_pred).all()):
        raise RuntimeError("Incomplete TabM OOF or test vector")
    OUTPUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        OUTPUT/"v2919r2_tabm_oof.npz",
        target=y,train_id=train["id"].to_numpy(),
        oof_tabm_route=oof,
        test_id=test["id"].to_numpy(),
        pred_tabm_route=test_pred.astype(np.float32),
    )
    dump("v2919r2_metrics.json",{
        "competition":"playground-series-s6e10",
        "model":MODEL,"stage":"three_folds_complete",
        "validation":"StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)",
        "folds":FOLDS,"fold_results":results,
        "train_rows":len(train),"test_rows":len(test),
        "full_oof_auc":float(roc_auc_score(y,oof)),
        "model_seed":MODEL_SEED,"epochs":EPOCHS,"tabm_k":TABM_K,
        "feature_family":FEATURE_FAMILY,
        "route_profile_feature_count":12,
        "target_encoding_columns":23,"inner_target_folds":5,
        "gpu_verified":True,"hardware":hw,
        "internal_validation_fraction":0.10,
        "cpu_only":False,"external_labels":False,
        "data_source":"competition_train_only",
        "leakage_safe_inner_cross_fit":True,
        "outer_validation_used_in_training":False,
        "submitted":False,
        "runtime_seconds":round(time.monotonic()-started,1),
    })
    print("V2919R2 genuine full 3fold TabM ROC-AUC",roc_auc_score(y,oof),flush=True)


if __name__=="__main__":
    try:
        main()
    except Exception:
        dump("v2919r2_error.json",{"traceback":traceback.format_exc()[-8000:]})
        raise
