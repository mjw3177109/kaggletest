"""V2.9.24: independent exact-flight-distance route-profile LightGBM.
No external labels, no validation-label preprocessing, no leaderboard submission.
Pre-registered: 3 outer folds, 5 inner TargetEncoder folds, one 900-tree model.
Inspired by public S6E10 route-profile research; independently implemented.
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
from sklearn.preprocessing import TargetEncoder

DATA = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working")
SEED = 20261018
MODEL_SEED = 20261032
FOLDS = 3
INNER_FOLDS = 5
N_TRAIN = 699635
N_TEST = 299844
MAX_RUNTIME_SECONDS = 3300
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
MODEL = "Personal-travel specialist LightGBM TE23 route-profile interactions 3fold seeded"
FEATURE_FAMILY = "personal_travel_te23_route_interactions_v1"
ROUTE_FEATURE_COUNT = 12
TE_COUNT = 23


def dump(name, value):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


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




def interaction_features(frame):
    """Ten fixed numeric interactions; never uses target labels or IDs."""
    ratings=frame[list(SERVICES)].apply(pd.to_numeric,errors="coerce").fillna(0.0)
    departure=pd.to_numeric(frame["Departure Delay in Minutes"],errors="coerce").fillna(0.0)
    arrival=pd.to_numeric(frame["Arrival Delay in Minutes"],errors="coerce").fillna(0.0)
    distance=pd.to_numeric(frame["Flight Distance"],errors="coerce").fillna(0.0)
    age=pd.to_numeric(frame["Age"],errors="coerce").fillna(0.0)
    comfort=frame[["Seat comfort","Inflight entertainment","Food and drink","On-board service"]].apply(
        pd.to_numeric,errors="coerce").fillna(0.0)
    out=pd.DataFrame(index=frame.index)
    out["interaction_service_mean"]=ratings.mean(axis=1)
    out["interaction_service_std"]=ratings.std(axis=1).fillna(0.0)
    out["interaction_service_min"]=ratings.min(axis=1)
    out["interaction_low_rating_count"]=(ratings<=2).sum(axis=1)
    out["interaction_comfort_bundle_mean"]=comfort.mean(axis=1)
    out["interaction_boarding_wifi_gap"]=(
        pd.to_numeric(frame["Online boarding"],errors="coerce").fillna(0.0)-
        pd.to_numeric(frame["Inflight wifi service"],errors="coerce").fillna(0.0))
    out["interaction_delay_gap"]=arrival-departure
    out["interaction_delay_sum_log"]=np.log1p(np.maximum(0.0,arrival)+np.maximum(0.0,departure))
    out["interaction_distance_log"]=np.log1p(np.maximum(0.0,distance))
    out["interaction_distance_per_age"]=distance/(age+1.0)
    if out.shape[1]!=10 or not np.isfinite(out.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Interaction feature schema or numeric validity failed")
    return out.astype("float32")


def augment_matrices(base,fit,holdout,test):
    result=[]
    for matrix,frame in zip(base,(fit,holdout,test)):
        inter=interaction_features(frame)
        combined=pd.concat([matrix.reset_index(drop=True),inter.reset_index(drop=True)],axis=1)
        if combined.shape[1]!=66 or combined.columns.duplicated().any():
            raise RuntimeError("Expected 56 base + 10 interaction features")
        if not np.isfinite(combined.to_numpy(dtype=np.float32)).all():
            raise RuntimeError("Nonfinite personal-specialist matrix")
        result.append(combined.astype("float32"))
    if not result[0].columns.equals(result[1].columns) or not result[1].columns.equals(result[2].columns):
        raise RuntimeError("Personal-specialist feature order mismatch")
    return result


def main():
    import lightgbm as lgb
    started=time.monotonic()
    train,test=competition_csv("train.csv"),competition_csv("test.csv")
    sample=competition_csv("sample_submission.csv")
    if len(train)!=N_TRAIN or len(test)!=N_TEST or len(sample)!=N_TEST:
        raise RuntimeError("Unexpected official competition row counts")
    if "satisfaction" not in train or "satisfaction" in test:
        raise RuntimeError("Target schema mismatch")
    if list(sample.columns)!=["id","satisfaction"] or not np.array_equal(sample["id"].to_numpy(),test["id"].to_numpy()):
        raise RuntimeError("Official Kaggle sample ID order mismatch")
    y=pd.to_numeric(train["satisfaction"],errors="raise").to_numpy(dtype=np.int8)
    if not np.isin(y,[0,1]).all():raise ValueError("Unexpected nonbinary target")
    train_mask=train["Type of Travel"].astype("string").fillna("__NA__").eq("Personal Travel").to_numpy()
    test_mask=test["Type of Travel"].astype("string").fillna("__NA__").eq("Personal Travel").to_numpy()
    if train_mask.sum()<150000 or test_mask.sum()<50000:
        raise RuntimeError("Personal Travel subgroup unexpectedly small")
    oof=np.full(len(train),np.nan,dtype=np.float32)
    test_group=np.zeros(int(test_mask.sum()),dtype=np.float64)
    fold_results=[]
    for fold,(itr,iva) in enumerate(StratifiedKFold(
            n_splits=FOLDS,shuffle=True,random_state=SEED).split(np.zeros(len(y),dtype=np.uint8),y)):
        if time.monotonic()-started>MAX_RUNTIME_SECONDS:
            raise TimeoutError("Personal specialist exceeded CPU budget; partial OOF forbidden")
        itr_group=itr[train_mask[itr]]
        iva_group=iva[train_mask[iva]]
        fit=train.iloc[itr_group]
        holdout=train.iloc[iva_group]
        test_group_frame=test.loc[test_mask]
        if len(np.unique(y[itr_group]))!=2 or len(np.unique(y[iva_group]))!=2:
            raise RuntimeError("Personal Travel training/validation fold lacks both target classes")
        base=matrices(fit,holdout,test_group_frame,y[itr_group],fold)
        xtr,xval,xtest=augment_matrices(base,fit,holdout,test_group_frame)
        model=lgb.LGBMClassifier(
            objective="binary",metric="auc",n_estimators=1000,
            learning_rate=0.03,num_leaves=31,min_child_samples=50,
            colsample_bytree=0.90,subsample=0.90,subsample_freq=1,
            reg_alpha=0.30,reg_lambda=8.0,n_jobs=4,verbosity=-1,
            random_state=MODEL_SEED+fold,
        )
        # Fit uses only Personal Travel rows inside the outer training fold.
        # Outer validation is prediction-only; no early stopping or validation labels.
        model.fit(xtr,y[itr_group])
        pv=np.asarray(model.predict_proba(xval)[:,1],dtype=np.float32)
        pt=np.asarray(model.predict_proba(xtest)[:,1],dtype=np.float32)
        if not (np.isfinite(pv).all() and np.isfinite(pt).all()
                and ((pv>=0)&(pv<=1)).all() and ((pt>=0)&(pt<=1)).all()):
            raise RuntimeError("Personal specialist probabilities invalid")
        oof[iva_group]=pv
        test_group+=pt/FOLDS
        fold_results.append({
            "fold":fold,"validation_rows":len(iva_group),
            "training_rows":len(itr_group),
            "auc":float(roc_auc_score(y[iva_group],pv)),
            "features":int(xtr.shape[1]),"group":"Personal Travel",
        })
        dump("v2924_progress.json",{"folds_finished":fold+1,"fold_results":fold_results,"complete":False})
        print("V2924 Personal Travel specialist fold",fold,fold_results[-1],flush=True)
        del model,xtr,xval,xtest,base
    if not np.array_equal(np.isfinite(oof),train_mask):
        raise RuntimeError("Personal specialist OOF mask does not exactly match subgroup")
    if not np.isfinite(test_group).all() or (test_group<0).any() or (test_group>1).any():
        raise RuntimeError("Personal specialist test predictions invalid")
    pred_test=np.full(len(test),np.nan,dtype=np.float32)
    pred_test[test_mask]=test_group.astype(np.float32)
    if not np.array_equal(np.isfinite(pred_test),test_mask):
        raise RuntimeError("Personal specialist test mask mismatch")
    OUTPUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUTPUT/"v2924_personal_travel_oof.npz",
        train_id=train["id"].to_numpy(),target=y,oof_personal=oof,train_mask=train_mask,
        test_id=test["id"].to_numpy(),pred_personal=pred_test,test_mask=test_mask)
    dump("v2924_metrics.json",{
        "competition":"playground-series-s6e10","stage":"three_folds_complete","model":MODEL,
        "validation":"StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)",
        "folds":FOLDS,"fold_results":fold_results,"train_rows":len(train),"test_rows":len(test),
        "model_seed":MODEL_SEED,"feature_family":FEATURE_FAMILY,
        "specialist_group":"Personal Travel","group_only_training":True,
        "personal_train_rows":int(train_mask.sum()),"personal_test_rows":int(test_mask.sum()),
        "personal_oof_auc":float(roc_auc_score(y[train_mask],oof[train_mask])),
        "interaction_feature_count":10,"total_feature_count":66,
        "route_profile_feature_count":12,"target_encoding_columns":23,"inner_target_folds":INNER_FOLDS,
        "cpu_only":True,"gpu_verified":False,"data_source":"competition_train_only",
        "external_labels":False,"leakage_safe_inner_cross_fit":True,
        "outer_validation_used_in_training":False,"submitted":False,
        "runtime_seconds":round(time.monotonic()-started,1),
    })
    print("V2924 Personal Travel OOF AUC",roc_auc_score(y[train_mask],oof[train_mask]),flush=True)


if __name__=="__main__":
    try:main()
    except Exception:
        dump("v2924_error.json",{"traceback":traceback.format_exc()[-8000:]})
        raise
