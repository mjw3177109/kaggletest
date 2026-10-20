"""V2.9.5 independent RealMLP-TD *pilot*, using one locked original CV fold.

This is an exploratory GPU neural model, NOT a submission generator.
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
FOLDS = 3
PILOT_FOLD = 0
TRAIN_ROWS = 699635
TEST_ROWS = 299844
RUNTIME_BUDGET_SECONDS = 2400
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
    dump("v295_hardware.json", evidence)
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


def fit_fold(train, targets, itr, iva):
    start = time.monotonic()
    xa, xv, te_train, te_val = engineer(train.iloc[itr], train.iloc[iva])
    ytr = targets[itr]
    enc = TargetEncoder(target_type="binary", smooth="auto", cv=5,
                        shuffle=True, random_state=SEED + PILOT_FOLD)
    names = [f"te_{i}" for i in range(te_train.shape[1])]
    xa[names] = enc.fit_transform(te_train, ytr).astype("float32")
    xv[names] = enc.transform(te_val).astype("float32")

    RealMLP = install_realmlp()
    # Fixed epoch count; don't hand any outer-fold validation labels to fit().
    model = RealMLP(device="cuda:0", random_state=SEED+PILOT_FOLD,
                    n_cv=1, n_refit=0, val_fraction=0.0,
                    n_ens=4, n_epochs=3, stop_epoch=3, batch_size=1024,
                    n_threads=4, verbosity=1,
                    use_early_stopping=False, lr=0.053, wd=0.015,
                    hidden_sizes=[512,256,128], act="silu",
                    p_drop=0.05, embedding_size=5,
                    max_one_hot_cat_size=18)
    model.fit(xa, pd.Series(ytr))
    prediction = np.asarray(model.predict_proba(xv)[:,1], dtype="float32")
    if prediction.shape != (len(iva),) or not np.isfinite(prediction).all():
        raise RuntimeError("RealMLP did not return one finite probability per validation row")
    if np.min(prediction) < 0 or np.max(prediction) > 1:
        raise RuntimeError("RealMLP probabilities are out of bounds")
    if time.monotonic() - start > RUNTIME_BUDGET_SECONDS:
        raise TimeoutError("GPU pilot exceeded time budget; refuse promotion")
    return prediction, time.monotonic()-start, len(xa.columns)


def main():
    start = time.monotonic()
    evidence = prove_gpu()
    train = only_competition_file("train.csv")
    test = only_competition_file("test.csv")
    if len(train) != TRAIN_ROWS or len(test) != TEST_ROWS:
        raise RuntimeError("Unexpected competition sizes")
    if "satisfaction" not in train or "satisfaction" in test:
        raise RuntimeError("Unexpected contest target schema")
    y = pd.to_numeric(train["satisfaction"], errors="raise").to_numpy()
    if not np.isin(y,[0,1]).all():
        raise RuntimeError("Expected numeric zero/one binary label")
    tr,va = list(StratifiedKFold(FOLDS,shuffle=True,random_state=SEED)
                 .split(np.zeros(len(y)), y))[PILOT_FOLD]
    preds,elapsed,n_cols=fit_fold(train,y,tr,va)
    score=float(roc_auc_score(y[va],preds))
    np.savez_compressed(OUTPUT/"v295_pilot_fold0.npz",
                        validation_indices=va.astype("int32"),
                        target=y[va].astype("int8"),
                        probability=preds,
                        train_rows=np.int32(len(train)))
    dump("v295_metrics.json", {
        "competition":"playground-series-s6e10",
        "model":"RealMLP-TD n_ens=4 3 epochs",
        "stage":"pilot_one_fold_complete",
        "validation":"StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)",
        "pilot_fold":PILOT_FOLD,"folds_total":FOLDS,
        "train_rows":len(train),"test_rows":len(test),
        "fold_validation_rows":len(va),"fold_cv_auc":score,
        "leakage_safe_inner_cross_fit":True,
        "outer_validation_used_in_training":False,
        "gpu_verified":True,"hardware":evidence,
        "feature_count":n_cols,
        "elapsed_seconds":round(elapsed,1),
        "runtime_seconds":round(time.monotonic()-start,1),
        "ready_for_submission":False,
        "source_inspiration":"public RealMLP S6E10 independent reproduction"
    })
    print("V295 genuine T4 RealMLP pilot fold AUC",score,flush=True)


if __name__=="__main__":
    try:
        main()
    except Exception:
        dump("v295_error.json", {"traceback":traceback.format_exc()[-8000:]})
        raise
