"""V2.9.26 full RealMLP-TD three-fold experiment with fixed pilot configuration.

This is a genuine three-fold GPU model. Submission is guarded off-kernel.
Inspired by public S6E10 RealMLP investigations and pytabkit docs, not by any
private top-three solution.  Do not train on Kaggle test labels.
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

INPUT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working")
SEED = 20261018
MODEL_SEED = 20261026
FOLDS = 3
PILOT_FOLD = 0
TRAIN_ROWS = 699635
TEST_ROWS = 299844
RUNTIME_BUDGET_SECONDS = 3200
CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
DUAL = ["Age", "Flight Distance", "Inflight wifi service", "Online boarding",
        "Seat comfort", "Inflight entertainment", "Leg room service"]
PAIRS = [("Class", "Type of Travel", "Gender"),
         ("Inflight wifi service", "Online boarding"),
         ("Class", "Customer Type", "Online boarding")]


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


def dump(filename, value):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / filename).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")


def only_competition_file(name):
    files = [p for p in INPUT.rglob(name)
             if "playground-series-s6e10" in str(p.parent)]
    if len(files) != 1:
        raise RuntimeError(f"Exactly one contest {name} required, found {len(files)}")
    return pd.read_csv(files[0])


def prove_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device; refusing CPU fallback or fabricated GPU evidence")
    probe = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=20, check=True)
    devices = [x.strip() for x in probe.stdout.splitlines() if x.strip()]
    if not any("T4" in device for device in devices):
        raise RuntimeError("Kaggle NVIDIA T4 was not confirmed: " + repr(devices))
    evidence = {"ok": True, "device": str(torch.cuda.get_device_name(0)),
                "devices": devices, "torch_version": torch.__version__,
                "cuda_available": True}
    dump("v2926_hardware.json", evidence)
    return evidence


def install_realmlp():
    """Kaggle image may not include pytabkit. Use an explicit pinned release."""
    try:
        import pytabkit
    except ImportError:
        p = subprocess.run([sys.executable, "-m", "pip", "install",
                            "--disable-pip-version-check", "--quiet",
                            "pytabkit==1.7.3"], timeout=240,
                           capture_output=True, text=True)
        if p.returncode:
            raise RuntimeError("Pinned pytabkit install failed; " +
                               (p.stderr or p.stdout)[-1200:])
    from pytabkit import RealMLP_TD_Classifier
    return RealMLP_TD_Classifier


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


def fit_fold(train, test, targets, itr, iva, fold):
    start = time.monotonic()
    # A separate engineering pass uses exactly the same outer-fold train rows,
    # giving test data the same category vocabulary and train-only counts.
    xa, xv, te_train, te_val = engineer(train.iloc[itr], train.iloc[iva])
    _, xt, _, te_test = engineer(train.iloc[itr], test)
    # V2926: add 12 label-free route-profile features fitted on this outer fold only.
    route = route_features(train.iloc[itr], train.iloc[iva], test)
    xa = pd.concat((xa, route[0]), axis=1)
    xv = pd.concat((xv, route[1]), axis=1)
    xt = pd.concat((xt, route[2]), axis=1)
    if xa.columns.duplicated().any() or any(not np.isfinite(z.select_dtypes(include="number").to_numpy(dtype="float32")).all() for z in (xa,xv,xt)):
        raise ValueError("V2926 invalid fold-local route profile matrix")
    if list(xa.columns) != list(xt.columns):
        raise RuntimeError("Train/test feature schemas diverged")
    ytr = targets[itr]
    enc = TargetEncoder(target_type="binary", smooth="auto", cv=5,
                        shuffle=True, random_state=SEED+fold)
    cols = [f"te_{i}" for i in range(te_train.shape[1])]
    xa[cols] = enc.fit_transform(te_train, ytr).astype("float32")
    xv[cols] = enc.transform(te_val).astype("float32")
    xt[cols] = enc.transform(te_test).astype("float32")
    if list(xa.columns) != list(xv.columns) or list(xa.columns) != list(xt.columns):
        raise RuntimeError("Inner cross-fit columns differ")
    Model = install_realmlp()
    model = Model(device="cuda:0", random_state=MODEL_SEED+fold,
                  n_cv=1, n_refit=0, val_fraction=0.0,
                  n_ens=8, n_epochs=5, stop_epoch=5,
                  batch_size=256, n_threads=4, verbosity=1,
                  use_early_stopping=False, lr=0.038, wd=0.015, sq_mom=0.988,
                  lr_sched="flat_anneal", wd_sched="cos_log_15",
                  first_layer_lr_factor=0.25,
                  hidden_sizes=[512,256,128], act="silu",
                  p_drop=0.05, p_drop_sched="invsqrtp1e-3", embedding_size=5,
                  max_one_hot_cat_size=18, plr_hidden_1=16, plr_hidden_2=8,
                  plr_act_name="gelu", plr_lr_factor=0.1151, plr_sigma=2.33,
                  ls_eps=0.01, ls_eps_sched="sqrt_cos",
                  add_front_scale=False, bias_init_mode="neg-uniform-dynamic-2",
                  tfms=["one_hot", "median_center", "robust_scale",
                        "smooth_clip", "embedding", "l2_normalize"])
    # NEVER pass outer-fold validation labels to model.fit().
    model.fit(xa, pd.Series(ytr))
    pv = np.asarray(model.predict_proba(xv)[:,1], dtype="float32")
    pt = np.asarray(model.predict_proba(xt)[:,1], dtype="float32")
    if pv.shape != (len(iva),) or pt.shape != (len(test),):
        raise RuntimeError("Incorrect validation/test prediction lengths")
    if not np.isfinite(pv).all() or not np.isfinite(pt).all():
        raise RuntimeError("Nonfinite RealMLP probability")
    if not (0 <= pv.min() <= pv.max() <= 1 and 0 <= pt.min() <= pt.max() <= 1):
        raise RuntimeError("Predictions outside probability bounds")
    del model
    import torch
    torch.cuda.empty_cache()
    return pv, pt, round(time.monotonic()-start,2), len(xa.columns)


def main():
    start=time.monotonic()
    evidence=prove_gpu()
    train=only_competition_file("train.csv")
    test=only_competition_file("test.csv")
    if len(train)!=TRAIN_ROWS or len(test)!=TEST_ROWS:
        raise RuntimeError("Competition dataset size changed")
    if "satisfaction" not in train or "satisfaction" in test or "id" not in test:
        raise RuntimeError("Invalid target/ID schema")
    y=pd.to_numeric(train["satisfaction"], errors="raise").to_numpy()
    if not np.isin(y,[0,1]).all():
        raise RuntimeError("Only binary 0/1 labels supported")
    fold_splits=list(StratifiedKFold(FOLDS,shuffle=True,random_state=SEED)
                     .split(np.zeros(len(y),dtype=np.uint8),y))
    oof=np.full(len(y),np.nan,dtype="float32")
    test_prediction=np.zeros(len(test),dtype="float64")
    details=[]
    for fold,(itr,iva) in enumerate(fold_splits):
        if time.monotonic()-start>RUNTIME_BUDGET_SECONDS:
            raise TimeoutError("Full threefold RealMLP exceeded budget; no incomplete CV")
        pv,pt,elapsed,num_features=fit_fold(train,test,y,itr,iva,fold)
        oof[iva]=pv
        test_prediction+=pt/FOLDS
        result={"fold":fold,"validation_rows":len(iva),
                "auc":float(roc_auc_score(y[iva],pv)),
                "elapsed_seconds":elapsed,
                "features":num_features}
        details.append(result)
        dump("v2926_progress.json",{"finished":fold+1,"fold_results":details,
                                   "complete":False})
        print("RealMLP full fold",fold,result,flush=True)
    if not np.isfinite(oof).all() or not np.isfinite(test_prediction).all():
        raise RuntimeError("Incomplete full OOF; refusing model validation")
    auc=float(roc_auc_score(y,oof))
    OUTPUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUTPUT/"v2926_realmlp_oof.npz",
                        target=y.astype("int8"),
                        train_id=train["id"].to_numpy(),
                        oof_route_realmlp=oof,
                        test_id=test["id"].to_numpy(),
                        pred_route_realmlp=test_prediction.astype("float32"))
    dump("v2926_metrics.json",{
        "competition":"playground-series-s6e10",
        "model":"RealMLP-TD n_ens=8 5 epochs independent seeded",
        "stage":"three_folds_complete",
        "train_rows":len(train),"test_rows":len(test),
        "validation":"StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)",
        "folds":3,"fold_results":details,"full_oof_auc":auc,
        "gpu_verified":True,"hardware":evidence,
        "leakage_safe_inner_cross_fit":True,
        "outer_validation_used_in_training":False,
        "feature_count":details[0]["features"],
        "model_seed":MODEL_SEED,
        "epochs":5,
        "runtime_seconds":round(time.monotonic()-start,1),
        "ready_for_submission":False,
        "pilot_validated_prior":True,
        "source_inspiration":"V2926 preregistered route-profile representation",
        "feature_family":"fold_local_route_profile_realmlp_v1",
        "route_feature_count":ROUTE_FEATURE_COUNT,
        "model_seed":MODEL_SEED,
        "external_labels":False,
        "submitted":False
    })
    print("V2926 full RealMLP OOF AUC",auc,flush=True)


if __name__=="__main__":
    try:
        main()
    except Exception:
        dump("v2926_error.json",{"traceback":traceback.format_exc()[-8000:]})
        raise
