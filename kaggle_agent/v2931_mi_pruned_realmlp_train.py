"""V2931 preregistered 8 conditional supervised crossfit target encodings + frozen RealMLP-TD.
All new target statistics use inner training rows ONLY. Evaluation-only; never submit.
"""
from __future__ import annotations

import json
import hashlib
import platform
from importlib.metadata import version
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
MODEL_SEED = 20261031
FOLDS = 3
PILOT_FOLD = 0
TRAIN_ROWS = 699635
TEST_ROWS = 299844
RUNTIME_BUDGET_SECONDS = 3200
FEATURE_FAMILY = "v2915_37_features_fold_local_mi_select_29"
PRIOR_WEIGHT = 50.0
INNER_FOLDS = 5
CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
DUAL = ["Age", "Flight Distance", "Inflight wifi service", "Online boarding",
        "Seat comfort", "Inflight entertainment", "Leg room service"]
PAIRS = [("Class", "Type of Travel", "Gender"),
         ("Inflight wifi service", "Online boarding"),
         ("Class", "Customer Type", "Online boarding")]


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
    dump("v2931_hardware.json", evidence)
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



def select_fold_features(xa, xv, xt, ytr):
    """Select 26 of 34 non-TE columns using only outer training labels."""
    from sklearn.feature_selection import mutual_info_classif
    original = list(xa.columns)
    protected = ["te_0", "te_1", "te_2"]
    eligible = [c for c in original if c not in protected]
    if (len(original) != 37 or len(set(original)) != 37 or len(eligible) != 34
            or any(c not in original for c in protected)
            or not (xa.columns.equals(xv.columns) and xa.columns.equals(xt.columns))
            or len(ytr) != len(xa)):
        raise ValueError("V2931 original 37-column schema changed")
    matrix = np.empty((len(xa), 34), dtype=np.float32)
    discrete = np.zeros(34, dtype=bool)
    for j, col in enumerate(eligible):
        series = xa[col]
        if isinstance(series.dtype, pd.CategoricalDtype):
            matrix[:, j] = series.cat.codes.to_numpy(dtype=np.float32)
            discrete[j] = True
        else:
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
            values[~np.isfinite(values)] = np.nan
            median = float(np.nanmedian(values))
            if not np.isfinite(median):
                raise ValueError("All-missing MI feature")
            matrix[:, j] = np.where(np.isfinite(values), values, median).astype(np.float32)
    if not np.isfinite(matrix).all() or not np.isin(ytr, [0, 1]).all():
        raise ValueError("Invalid fold-local MI matrix or labels")
    scores = mutual_info_classif(matrix, np.asarray(ytr), discrete_features=discrete,
                                 n_neighbors=3, random_state=20261031)
    if scores.shape != (34,) or not np.isfinite(scores).all():
        raise ValueError("MI feature ranking failed")
    order = sorted(range(34), key=lambda i: (-float(scores[i]), i))
    keep = set(eligible[i] for i in order[:26]) | set(protected)
    selected = [col for col in original if col in keep]
    if len(selected) != 29:
        raise ValueError("Expected 29 selected columns")
    evidence = {
        "canonical_original_features": original,
        "protected_target_encoded_features": protected,
        "eligible_features": eligible,
        "mi_scores": [float(x) for x in scores],
        "discrete_feature_mask": [bool(x) for x in discrete],
        "selected_features": selected,
        "removed_features": [c for c in original if c not in keep],
        "mi_n_neighbors": 3, "mi_seed": 20261031,
        "mi_tie_break": "canonical_v2915_feature_order",
        "selected_nonprotected_features": 26, "selected_total_features": 29,
    }
    return xa[selected].copy(), xv[selected].copy(), xt[selected].copy(), evidence


def fit_fold(train, test, targets, itr, iva, fold):
    start = time.monotonic()
    # A separate engineering pass uses exactly the same outer-fold train rows,
    # giving test data the same category vocabulary and train-only counts.
    xa, xv, te_train, te_val = engineer(train.iloc[itr], train.iloc[iva])
    _, xt, _, te_test = engineer(train.iloc[itr], test)
    if list(xa.columns) != list(xt.columns):
        raise RuntimeError("Train/test feature schemas diverged")
    ytr = targets[itr]
    enc = TargetEncoder(target_type="binary", smooth="auto", cv=5,
                        shuffle=True, random_state=SEED+fold)
    cols = [f"te_{i}" for i in range(te_train.shape[1])]
    xa[cols] = enc.fit_transform(te_train, ytr).astype("float32")
    xv[cols] = enc.transform(te_val).astype("float32")
    xt[cols] = enc.transform(te_test).astype("float32")
    if len(xa.columns) != 37:
        raise RuntimeError("V2931 requires exactly 37 original V2915 features")
    xa, xv, xt, selection = select_fold_features(xa, xv, xt, ytr)
    if len(xa.columns) != 29 or not (xa.columns.equals(xv.columns) and xa.columns.equals(xt.columns)):
        raise RuntimeError("V2931 fold-local MI feature schema drift")
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
    return pv, pt, round(time.monotonic()-start,2), len(xa.columns), selection


def main():
    start=time.monotonic()
    evidence=prove_gpu()
    train=only_competition_file("train.csv")
    test=only_competition_file("test.csv")
    sample=only_competition_file("sample_submission.csv")
    if list(sample.columns)!=["id","satisfaction"] or not np.array_equal(sample["id"].to_numpy(),test["id"].to_numpy()):
        raise RuntimeError("Official sample submission and test IDs mismatch")
    if len(train)!=TRAIN_ROWS or len(test)!=TEST_ROWS or len(sample)!=TEST_ROWS:
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
        pv,pt,elapsed,num_features,selection=fit_fold(train,test,y,itr,iva,fold)
        oof[iva]=pv
        test_prediction+=pt/FOLDS
        result={"fold":fold,"validation_rows":len(iva),"train_rows":len(itr),"model_seed":MODEL_SEED+fold,
                "auc":float(roc_auc_score(y[iva],pv)),
                "elapsed_seconds":elapsed,
                "features":num_features,"selection":selection}
        details.append(result)
        dump("v2931_progress.json",{"finished":fold+1,"fold_results":details,
                                   "complete":False})
        print("RealMLP full fold",fold,result,flush=True)
    if len(details) != 3 or any(x["features"] != 29 for x in details):
        raise RuntimeError("V2931 full three folds and exactly 29 features required")
    if not np.isfinite(oof).all() or not np.isfinite(test_prediction).all():
        raise RuntimeError("Incomplete full OOF; refusing model validation")
    auc=float(roc_auc_score(y,oof))
    OUTPUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUTPUT/"v2931_realmlp_oof.npz",
                        target=y.astype("int8"),
                        train_id=train["id"].to_numpy(),
                        oof_mi_realmlp=oof,
                        test_id=test["id"].to_numpy(),
                        pred_mi_realmlp=test_prediction.astype("float32"))
    dump("v2931_metrics.json",{
        "competition":"playground-series-s6e10",
        "model":"RealMLP-TD n_ens=8 5 epochs independent seeded",
        "stage":"three_folds_complete",
        "train_rows":len(train),"test_rows":len(test),
        "validation":"StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)",
        "folds":3,"fold_results":details,"full_oof_auc":auc,
        "gpu_verified":True,"hardware":evidence,
        "leakage_safe_inner_cross_fit":True,
        "outer_validation_used_in_training":False,
        "feature_count":29,
        "feature_family":FEATURE_FAMILY,
        "baseline_features":37,
        "protected_target_encoded_features":3,
        "selected_nonprotected_features":26,
        "selected_total_features":29,
        "removed_features":8,
        "feature_selection":"outer_train_only_mutual_info_classif",
        "mi_n_neighbors":3,"mi_seed":20261031,
        "mi_tie_break":"canonical_v2915_feature_order",
        "fold_selection_evidence":[item["selection"] for item in details],
        "original_target_encoding_columns":3,
        "outer_train_only_supervised_statistics":True,
        "external_labels":False,
        "pseudo_labels":False,
        "route_features":False,
        "pca_features":False,
        "ordinal_threshold_features":False,
        "travel_gated_experts":False,
        "data_source":"competition_train_only",
        "submitted":False,
        "source_code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment_versions":{
            "python":platform.python_version(),"numpy":np.__version__,
            "pandas":pd.__version__,"sklearn":__import__("sklearn").__version__,
            "pytabkit":version("pytabkit"),"torch":__import__("torch").__version__,
        },
        "model_seed":MODEL_SEED,
        "epochs":5,
        "runtime_seconds":round(time.monotonic()-start,1),
        "ready_for_submission":False,
        "pilot_validated_prior":True,
        "source_inspiration":"V2931 preregistered fold-local MI pruning of V2915"
    })
    print("V2931 full RealMLP OOF AUC",auc,flush=True)


if __name__=="__main__":
    try:
        main()
    except Exception:
        dump("v2931_error.json",{"traceback":traceback.format_exc()[-8000:]})
        raise
