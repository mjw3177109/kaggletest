"""V2.6 S6E10 GPU challenger: fold-safe TargetEncoder + exact route profiles + cross-fit conditional frequencies.

Original synthesis, based on publicly discussed feature ideas. NO external labels,
leaderboard feedback, or target values from validation folds enter training features.
Keep the original V2.3 GPU and V2.1 CPU champions untouched.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score

DATA = Path("/kaggle/input")
OUT = Path("/kaggle/working")
SEED = 20261018
FOLDS = 3
MINUTES_LIMIT = 48
CAT = ["Gender","Customer Type","Type of Travel","Class"]
RATINGS = [
    "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink",
    "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
]
ROUTE_RATINGS = ["Online boarding", "Inflight wifi service", "Seat comfort",
                 "Inflight entertainment", "On-board service", "Cleanliness"]


def dump(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def check_gpu():
    probe = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30,
    )
    names = [s.strip() for s in probe.stdout.splitlines() if s.strip()]
    info = {"devices": names, "ok": probe.returncode == 0 and any("T4" in v for v in names)}
    dump("v26_hardware.json", info)
    if not info["ok"]:
        raise RuntimeError("V2.6 requires actual Kaggle NVIDIA T4; refusing CPU fallback")
    return info


def one_file(name):
    files = list(DATA.rglob(name))
    if len(files) != 1:
        raise RuntimeError(f"Expected exactly one {name}, found {len(files)}")
    return pd.read_csv(files[0])


def label(y):
    if y.isna().any() or y.nunique() != 2:
        raise ValueError("Only binary targets supported.")
    vals = sorted(y.astype(str).unique())
    return y.astype(str).map({vals[0]: 0, vals[1]: 1}).to_numpy(dtype=np.int8)


def keys(frame):
    """High-granularity categorical keys WITHOUT target information."""
    source = frame.drop(columns=["id", "satisfaction"], errors="ignore")
    z = pd.DataFrame(index=frame.index)
    for col in source.columns:
        z["exact_"+col] = source[col].astype("string").fillna("__NA__").astype(str)
    required = CAT + RATINGS + ["Flight Distance", "Age"]
    missing = [c for c in required if c not in source]
    if missing:
        raise ValueError(f"Dataset schema mismatch: {missing}")
    seg = (source["Type of Travel"].astype(str) + "|" +
           source["Customer Type"].astype(str) + "|" +
           source["Class"].astype(str))
    z["travel_segment"] = seg
    for col in RATINGS:
        z["rating_by_segment_"+col] = source[col].astype(str) + "|" + seg
    for a, b in [
        ("Inflight wifi service", "Online boarding"),
        ("Online boarding", "Seat comfort"),
        ("Seat comfort", "Inflight entertainment"),
        ("Inflight entertainment", "Cleanliness"),
    ]:
        z["pair_"+a+"__"+b] = source[a].astype(str) + "|" + source[b].astype(str)
    for col in ["Age","Flight Distance"]:
        for category in ("Type of Travel", "Customer Type", "Class"):
            z["numeric_by_group_"+col+"__"+category] = (
                source[col].astype(str) + "|" + source[category].astype(str)
            )
    # fixed input order (train, valid, test): critical for the encoder
    return z


def raw_numeric(frame, vocab):
    """Preserve real numeric values + non-target, globally-fitted category codes."""
    raw = frame.drop(columns=["id", "satisfaction"], errors="ignore")
    values = {}
    for col in raw:
        if col in CAT:
            values[col] = pd.Categorical(
                raw[col].astype("string").fillna("__NA__").astype(str),
                categories=vocab[col],
            ).codes.astype("float32")
        else:
            values[col] = pd.to_numeric(raw[col], errors="coerce").to_numpy(dtype=np.float32)
    nums = pd.DataFrame(values, index=frame.index)
    if all(c in nums for c in RATINGS):
        rating = nums[RATINGS]
        nums["__rating_mean"] = rating.mean(axis=1)
        nums["__rating_std"] = rating.std(axis=1)
        nums["__rating_min"] = rating.min(axis=1)
        nums["__rating_max"] = rating.max(axis=1)
    if {"Departure Delay in Minutes","Arrival Delay in Minutes"} <= set(nums):
        a = nums["Departure Delay in Minutes"]
        b = nums["Arrival Delay in Minutes"]
        nums["__total_delay"] = a.fillna(0) + b.fillna(0)
        nums["__arrival_delay_missing"] = b.isna().astype("float32")
    return nums


def route_profile(tr, va, te):
    """Only fit on OUTER fold train. No target labels; no validation-row fitting."""
    col = "Flight Distance"
    if col not in tr:
        raise ValueError("Missing Flight Distance")
    counts = tr[col].value_counts()
    result = [pd.DataFrame(index=f.index) for f in (tr, va, te)]
    for dst, frame in zip(result, (tr, va, te)):
        dst["route_count"] = frame[col].map(counts).fillna(0).to_numpy(dtype="float32")
        dst["route_log_count"] = np.log1p(dst["route_count"])
    for feature in ROUTE_RATINGS:
        if feature not in tr:
            continue
        grp = tr.groupby(col)[feature].agg(["mean", "std"])
        mean = tr[feature].mean()
        std = tr[feature].std()
        for dst, frame in zip(result, (tr, va, te)):
            mu = frame[col].map(grp["mean"]).fillna(mean)
            sig = frame[col].map(grp["std"]).fillna(std)
            prefix = "route_"+feature.lower().replace(" ","_").replace("-","_")
            dst[prefix+"_mean"] = mu.to_numpy(dtype="float32")
            dst[prefix+"_std"] = sig.to_numpy(dtype="float32")
            dst[prefix+"_residual"] = (frame[feature] - mu).to_numpy(dtype="float32")
    return result


FREQUENCY_KEYS = [
    "exact_Flight Distance", "exact_Age", "travel_segment",
    "numeric_by_group_Flight Distance__Class",
    "numeric_by_group_Flight Distance__Customer Type",
    "numeric_by_group_Flight Distance__Type of Travel",
    "numeric_by_group_Age__Class",
    "numeric_by_group_Age__Type of Travel",
    "rating_by_segment_Online boarding",
    "rating_by_segment_Inflight wifi service",
    "rating_by_segment_Seat comfort",
    "rating_by_segment_Inflight entertainment",
]


def crossfit_frequency(train_keys, valid_keys, test_keys, fold):
    """Return train/valid/test counts with five-fold *inner* crossfitting.

    Fit each inner training-frequency table without the held-out rows.
    Validation/test frequencies are fitted on OUTER training rows only.
    This avoids train-row self-count bias, fold leakage, and test-only fitting.
    """
    from sklearn.model_selection import KFold

    missing = [k for k in FREQUENCY_KEYS if k not in train_keys]
    if missing:
        raise ValueError("Missing crossfit frequency keys: " + str(missing))
    n = len(train_keys)
    inner = list(KFold(5, shuffle=True, random_state=SEED + 400 + fold).split(train_keys))
    names = ["freq_" + str(i) for i in range(len(FREQUENCY_KEYS))]
    train_out = np.zeros((n, len(names)), dtype=np.float32)
    valid_out = np.zeros((len(valid_keys), len(names)), dtype=np.float32)
    test_out = np.zeros((len(test_keys), len(names)), dtype=np.float32)
    for i, name in enumerate(FREQUENCY_KEYS):
        fit_source = train_keys[name]
        for fit_idx, apply_idx in inner:
            values = fit_source.iloc[fit_idx].value_counts(normalize=True, dropna=False)
            mapped = fit_source.iloc[apply_idx].map(values).fillna(0).to_numpy(dtype=np.float32)
            train_out[apply_idx, i] = np.log1p(1e6 * mapped)
        full_values = fit_source.value_counts(normalize=True, dropna=False)
        for output, target in ((valid_out, valid_keys), (test_out, test_keys)):
            mapped = target[name].map(full_values).fillna(0).to_numpy(dtype=np.float32)
            output[:, i] = np.log1p(1e6 * mapped)
    return [
        pd.DataFrame(arr, index=index, columns=names)
        for arr, index in zip((train_out,valid_out,test_out),
                              (train_keys.index,valid_keys.index,test_keys.index))
    ]


def fold_input(train, test, y, outer_train, outer_valid, keys_train, keys_test, nums_train, nums_test, fold):
    tr_key = keys_train.iloc[outer_train]
    va_key = keys_train.iloc[outer_valid]
    full = TargetEncoder(target_type="binary", smooth="auto", cv=5,
                         shuffle=True, random_state=SEED + fold)
    te_tr = full.fit_transform(tr_key, y[outer_train]).astype(np.float32)
    te_va = full.transform(va_key).astype(np.float32)
    te_ts = full.transform(keys_test).astype(np.float32)
    # IMPORTANT: fit_transform = inner 5-fold cross-fit. DO NOT use fit(...).transform(X_train).
    encoded_cols = ["te_"+str(i) for i in range(te_tr.shape[1])]
    raw = [
        nums_train.iloc[outer_train].copy(),
        nums_train.iloc[outer_valid].copy(),
        nums_test.copy(),
    ]
    profiles = route_profile(*raw)
    freqs = crossfit_frequency(tr_key, va_key, keys_test, fold)
    matrices = []
    for idx, matrix in enumerate(raw):
        block = pd.DataFrame((te_tr, te_va, te_ts)[idx],
                             index=matrix.index, columns=encoded_cols)
        matrices.append(pd.concat((matrix, profiles[idx], block, freqs[idx]), axis=1))
    medians = matrices[0].median(axis=0).fillna(0)
    final = []
    for matrix in matrices:
        x = matrix.replace([np.inf,-np.inf],np.nan).fillna(medians).fillna(0)
        x = x.astype("float32")
        if x.columns.duplicated().any() or not np.isfinite(x.to_numpy()).all():
            raise ValueError("Invalid features or nonfinite values")
        final.append(x)
    return final


def auc(y, p):
    return float(roc_auc_score(y, p))


def main():
    from scipy.stats import rankdata
    import lightgbm as lgb
    import xgboost as xgb

    hw = check_gpu()
    begin = time.monotonic()
    train, test, sample = one_file("train.csv"), one_file("test.csv"), one_file("sample_submission.csv")
    targets = [c for c in train if c not in test]
    if len(targets) != 1:
        raise ValueError("Expected exactly one target")
    target = targets[0]
    if target not in sample or len(sample) != len(test):
        raise ValueError("Invalid competition sample_submission")
    y = label(train[target])
    all_data = pd.concat((train.drop(columns=[target]),test),ignore_index=True)
    vocab = {c: sorted(all_data[c].astype("string").fillna("__NA__").astype(str).unique())
             for c in CAT}
    nums_train, nums_test = raw_numeric(train,vocab),raw_numeric(test,vocab)
    train_keys, test_keys = keys(train), keys(test)
    cv = StratifiedKFold(FOLDS,shuffle=True,random_state=SEED)
    oof = {"lightgbm":np.zeros(len(y)), "xgboost":np.zeros(len(y))}
    pred = {"lightgbm":np.zeros(len(test)), "xgboost":np.zeros(len(test))}
    folds = []
    for fold,(itr,iv) in enumerate(cv.split(nums_train,y),start=1):
        if time.monotonic() - begin > MINUTES_LIMIT*60:
            raise TimeoutError("V2.6 reached Kaggle runtime budget. No partial submission.")
        xtr,xva,xt = fold_input(train,test,y,itr,iv,train_keys,test_keys,
                               nums_train,nums_test,fold)
        ym, yv = y[itr],y[iv]
        m1 = lgb.LGBMClassifier(
            objective="binary", metric="auc", n_estimators=2400,
            learning_rate=0.035, num_leaves=63, max_depth=-1,
            subsample=0.85, subsample_freq=1, colsample_bytree=0.78,
            reg_lambda=4.0, reg_alpha=0.12, min_child_samples=48,
            n_jobs=4, verbosity=-1, random_state=SEED + fold,
        )
        m1.fit(xtr,ym,eval_set=[(xva,yv)],eval_metric="auc",
               callbacks=[lgb.early_stopping(120,verbose=False)])
        lval=m1.predict_proba(xva)[:,1]
        ltest=m1.predict_proba(xt)[:,1]
        oof["lightgbm"][iv]=lval
        pred["lightgbm"]+=ltest/FOLDS
        del m1

        m2=xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="auc", tree_method="hist",
            device="cuda", max_depth=8, n_estimators=2200,
            learning_rate=0.035, min_child_weight=10,
            subsample=0.9, colsample_bytree=0.82,reg_alpha=0.10,
            reg_lambda=5.0,early_stopping_rounds=120,
            n_jobs=4,random_state=SEED+fold,
        )
        m2.fit(xtr,ym,eval_set=[(xva,yv)],verbose=False)
        device=json.loads(m2.get_booster().save_config()).get("learner",{}).get("generic_param",{}).get("device","")
        if not str(device).startswith("cuda"):
            raise RuntimeError("XGBoost ran without CUDA: "+str(device))
        xval=m2.predict_proba(xva)[:,1]
        xtest=m2.predict_proba(xt)[:,1]
        oof["xgboost"][iv]=xval
        pred["xgboost"]+=xtest/FOLDS
        row={"fold":fold, "lgb_auc":auc(yv,lval), "xgb_auc":auc(yv,xval),
             "xgb_device":device,"features":len(xtr.columns)}
        folds.append(row)
        print("V2.6 fold",fold,":",row,flush=True)
        dump("v26_progress.json",{"folds_finished":fold,"rows":folds})
        del xtr,xva,xt,m2
    results={k:auc(y,v) for k,v in oof.items()}
    # Two fixed blend candidates (no hyperparam search on full OOF).
    raw_oof=0.5*oof["lightgbm"]+0.5*oof["xgboost"]
    rank_oof=0.5*rankdata(oof["lightgbm"])+0.5*rankdata(oof["xgboost"])
    results["blend_mean"]=auc(y,raw_oof)
    results["blend_rank"]=auc(y,rank_oof)
    chosen=max(results,key=results.get)
    if chosen=="lightgbm":
        submit_pred=pred["lightgbm"]
    elif chosen=="xgboost":
        submit_pred=pred["xgboost"]
    elif chosen=="blend_rank":
        submit_pred=0.5*rankdata(pred["lightgbm"])/len(test)+0.5*rankdata(pred["xgboost"])/len(test)
    else:
        submit_pred=0.5*pred["lightgbm"]+0.5*pred["xgboost"]
    result=sample.copy()
    result[target]=submit_pred
    if len(result)!=len(test) or not np.isfinite(submit_pred).all():
        raise ValueError("Invalid predicted probabilities")
    result.to_csv(OUT/"candidate_submission.csv",index=False)
    np.savez_compressed(OUT/"v26_oof.npz",target=y,
                        oof_lgb=oof["lightgbm"].astype(np.float32),
                        oof_xgb=oof["xgboost"].astype(np.float32))
    dump("v26_metrics.json",{
        "competition":"playground-series-s6e10","target":target,
        "gpu_verified":True,"hardware":hw,"actual_xgboost_device":"cuda:0",
        "train_rows":len(train),"test_rows":len(test),
        "validation":"StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)",
        "folds":FOLDS,"inner_encoding_folds":5,
        "leakage_safe_target_encoding":True,
        "features_per_fold":folds[0]["features"],"fold_results":folds,
        "models":results,"best_model":chosen,"best_cv_auc":results[chosen],
        "elapsed_seconds":round(time.monotonic()-begin,1),
        "new_methods":["all-original-columns exact-value 5-fold target encoding",
                       "flight distance training-fold-only profile",
                       "service ratings by traveler segment",
                       "numeric class-group target encoding",
                       "fixed rank and raw blending candidates",
                       "inner-5fold-crossfit frequency: exact age/distance, traveler and rating segments"],
    })
    print("V2.6 GPU strongest OOF:",chosen,results[chosen],flush=True)


if __name__=="__main__":
    try:
        main()
    except Exception:
        dump("v26_error.json",{"traceback":traceback.format_exc()})
        raise
