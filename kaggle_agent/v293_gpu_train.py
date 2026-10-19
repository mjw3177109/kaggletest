"""V2.9.3 isolated GPU challenger: new label-free service contrasts and depth-8 CatBoost.

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
MINUTES_LIMIT = 43
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
    dump("v293_gpu_hardware.json", info)
    if not info["ok"]:
        raise RuntimeError("V2.9.3 GPU requires actual Kaggle NVIDIA T4; refusing CPU fallback")
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
    # V2.9.3 strictly label-free service contrasts and normalized distance.
    # Computed per-row: no fitted parameters, no validation/test label leakage.
    if {"Inflight wifi service", "Ease of Online booking"} <= set(nums):
        nums["__wifi_booking_gap"] = nums["Inflight wifi service"] - nums["Ease of Online booking"]
    if {"Online boarding", "Inflight wifi service"} <= set(nums):
        nums["__boarding_wifi_gap"] = nums["Online boarding"] - nums["Inflight wifi service"]
    if {"Seat comfort", "Leg room service"} <= set(nums):
        nums["__seat_legroom_gap"] = nums["Seat comfort"] - nums["Leg room service"]
    if {"Flight Distance", "Age"} <= set(nums):
        nums["__distance_per_age"] = nums["Flight Distance"] / nums["Age"].clip(lower=1)
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
    matrices = []
    for idx, matrix in enumerate(raw):
        block = pd.DataFrame((te_tr, te_va, te_ts)[idx],
                             index=matrix.index, columns=encoded_cols)
        matrices.append(pd.concat((matrix, profiles[idx], block), axis=1))
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
    from catboost import CatBoostClassifier

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
    oof = {"lightgbm":np.zeros(len(y)), "catboost":np.zeros(len(y))}
    pred = {"lightgbm":np.zeros(len(test)), "catboost":np.zeros(len(test))}
    folds = []
    for fold,(itr,iv) in enumerate(cv.split(nums_train,y),start=1):
        if time.monotonic() - begin > MINUTES_LIMIT*60:
            raise TimeoutError("V2.9.3 GPU reached Kaggle runtime budget. No partial submission.")
        xtr,xva,xt = fold_input(train,test,y,itr,iv,train_keys,test_keys,
                               nums_train,nums_test,fold)
        ym, yv = y[itr],y[iv]
        m1 = lgb.LGBMClassifier(
            objective="binary", metric="auc", n_estimators=2600,
            learning_rate=0.032, num_leaves=47, max_depth=-1,
            subsample=0.85, subsample_freq=1, colsample_bytree=0.78,
            reg_lambda=5.0, reg_alpha=0.16, min_child_samples=60,
            n_jobs=4, verbosity=-1, random_state=SEED + fold,
        )
        m1.fit(xtr,ym,eval_set=[(xva,yv)],eval_metric="auc",
               callbacks=[lgb.early_stopping(120,verbose=False)])
        lval=m1.predict_proba(xva)[:,1]
        ltest=m1.predict_proba(xt)[:,1]
        oof["lightgbm"][iv]=lval
        pred["lightgbm"]+=ltest/FOLDS
        del m1

        # CatBoost is a genuinely different tree-growing algorithm from
        # LightGBM/XGBoost, using Kaggle T4 GPU (not CPU fallback).
        m2=CatBoostClassifier(
            iterations=2100, depth=8, learning_rate=0.038,
            loss_function="Logloss", eval_metric="AUC",
            task_type="GPU", devices="0", random_seed=SEED+131+fold,
            l2_leaf_reg=9, random_strength=0.8,
            bootstrap_type="Bernoulli", subsample=0.86,
            od_type="Iter", od_wait=140,
            gpu_ram_part=0.70, thread_count=4,
            allow_writing_files=False, verbose=False,
        )
        m2.fit(xtr,ym,eval_set=(xva,yv),verbose=False)
        device=str(m2.get_all_params().get("task_type",""))
        if device!="GPU":
            raise RuntimeError("CatBoost did not report GPU task_type: "+device)
        cval=m2.predict_proba(xva)[:,1]
        ctest=m2.predict_proba(xt)[:,1]
        oof["catboost"][iv]=cval
        pred["catboost"]+=ctest/FOLDS
        row={"fold":fold,"lgb_auc":auc(yv,lval),"cat_auc":auc(yv,cval),
             "cat_device":device,"features":len(xtr.columns)}
        folds.append(row)
        print("V2.9.3 GPU fold",fold,":",row,flush=True)
        dump("v293_gpu_progress.json",{"folds_finished":fold,"rows":folds})
        del xtr,xva,xt,m2
    results={k:auc(y,v) for k,v in oof.items()}
    # Two fixed blend candidates (no hyperparam search on full OOF).
    raw_oof=0.5*oof["lightgbm"]+0.5*oof["catboost"]
    rank_oof=0.5*rankdata(oof["lightgbm"])+0.5*rankdata(oof["catboost"])
    results["blend_mean"]=auc(y,raw_oof)
    results["blend_rank"]=auc(y,rank_oof)
    chosen=max(results,key=results.get)
    if chosen=="lightgbm":
        submit_pred=pred["lightgbm"]
    elif chosen=="catboost":
        submit_pred=pred["catboost"]
    elif chosen=="blend_rank":
        submit_pred=0.5*rankdata(pred["lightgbm"])/len(test)+0.5*rankdata(pred["catboost"])/len(test)
    else:
        submit_pred=0.5*pred["lightgbm"]+0.5*pred["catboost"]
    result=sample.copy()
    result[target]=submit_pred
    if len(result)!=len(test) or not np.isfinite(submit_pred).all():
        raise ValueError("Invalid predicted probabilities")
    result.to_csv(OUT/"candidate_submission.csv",index=False)
    np.savez_compressed(OUT/"v293_gpu_oof.npz",target=y,
                        oof_lgb=oof["lightgbm"].astype(np.float32),
                        oof_cat=oof["catboost"].astype(np.float32))
    dump("v293_gpu_metrics.json",{
        "competition":"playground-series-s6e10","target":target,
        "gpu_verified":True,"hardware":hw,"actual_catboost_device":"GPU:0",
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
                       "CatBoost ordered GPU gradient boosting as independent model",
                       "V2.9.3 fold-safe per-row service contrasts and distance/age",
                       "CatBoost depth-8 diverse seed and stronger regularization"],
    })
    print("V2.9.3 GPU GPU strongest OOF:",chosen,results[chosen],flush=True)


if __name__=="__main__":
    try:
        main()
    except Exception:
        dump("v293_gpu_error.json",{"traceback":traceback.format_exc()})
        raise
