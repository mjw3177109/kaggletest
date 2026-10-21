"""V2.9.19: independent TabM route-profile 3fold GPU challenger.
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

try:
    import feature_engineering as route
except ImportError:
    import v2918_route_profile_lgbm_train as route

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
    dump("v2919_hardware.json",evidence)
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


def fit_one_fold(train, test, targets, itr, iva, fold):
    from time import monotonic
    started=monotonic()
    xtr, xval, xtest = route.matrices(
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
    train=route.competition_csv("train.csv")
    test=route.competition_csv("test.csv")
    sample=route.competition_csv("sample_submission.csv")
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
        dump("v2919_progress.json",{"folds_finished":fold+1,
             "fold_results":results,"complete":False})
        print("V2919 independent TabM T4 fold",fold,results[-1],flush=True)
    if not (np.isfinite(oof).all() and np.isfinite(test_pred).all()):
        raise RuntimeError("Incomplete TabM OOF or test vector")
    OUTPUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        OUTPUT/"v2919_tabm_oof.npz",
        target=y,train_id=train["id"].to_numpy(),
        oof_tabm_route=oof,
        test_id=test["id"].to_numpy(),
        pred_tabm_route=test_pred.astype(np.float32),
    )
    dump("v2919_metrics.json",{
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
    print("V2919 genuine full 3fold TabM ROC-AUC",roc_auc_score(y,oof),flush=True)


if __name__=="__main__":
    try:
        main()
    except Exception:
        dump("v2919_error.json",{"traceback":traceback.format_exc()[-8000:]})
        raise
