"""Kaggle Auto Scientist V2.3 GPU branch: 3-fold XGBoost CUDA experiments.

This does not alter V2.1 CPU/TPU kernels. No CPU fallback masquerades as GPU.
"""
from __future__ import annotations

import json
import math
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working")
SEED = 20261018
CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)


def emit(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def verify_nvidia():
    result = {"requested": "NvidiaTeslaT4", "ok": False}
    p = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30
    )
    devices = [x.strip() for x in p.stdout.splitlines() if x.strip()]
    result["devices"] = devices
    result["count"] = len(devices)
    result["returncode"] = p.returncode
    # Explicitly verify NVIDIA T4 rather than some unexpected CPU-only worker.
    result["ok"] = p.returncode == 0 and any("T4" in d for d in devices)
    emit("v23_gpu_hardware.json", result)
    if not result["ok"]:
        raise RuntimeError(f"Kaggle did not allocate NVIDIA Tesla T4 GPU: {result}")
    return result


def find(name):
    items = list(INPUT.rglob(name))
    if len(items) != 1:
        raise FileNotFoundError(f"Expected one {name} under /kaggle/input, got {items[:8]}")
    return items[0]


def y_binary(series):
    if series.isna().any() or series.nunique() != 2:
        raise ValueError("GPU route supports binary targets only.")
    vals = sorted(series.astype(str).unique().tolist())
    return series.astype(str).map({vals[0]: 0, vals[1]: 1}).to_numpy(dtype=np.int8)


def prepare(train, test, sample):
    # The same V2.1 engineered columns and dummy encoding.
    exclude = [c for c in sample if c in test]
    cols = [c for c in test if c not in exclude]
    a, b = train[cols].copy(), test[cols].copy()
    delays = [c for c in cols if "delay" in c.lower() and pd.api.types.is_numeric_dtype(a[c])]
    if len(delays) >= 2:
        for frame in (a, b):
            frame["__auto_total_delay"] = frame[delays].sum(axis=1, skipna=True)
            frame["__auto_max_delay"] = frame[delays].max(axis=1, skipna=True)
    toks = ("service", "comfort", "cleanliness", "booking", "boarding", "location", "entertainment")
    ratings = [c for c in cols if any(t in c.lower() for t in toks) and pd.api.types.is_numeric_dtype(a[c])]
    if len(ratings) >= 3:
        for frame in (a, b):
            frame["__auto_rating_mean"] = frame[ratings].mean(axis=1)
            frame["__auto_rating_min"] = frame[ratings].min(axis=1)
            frame["__auto_rating_max"] = frame[ratings].max(axis=1)
    z = pd.concat([a,b], ignore_index=True)
    cats = list(z.select_dtypes(include=["object", "category", "bool"]).columns)
    for c in z:
        if c not in cats and z[c].isna().any():
            z[c] = z[c].fillna(z[c].median())
    z = pd.get_dummies(z, columns=cats, dummy_na=True, dtype=np.uint8)
    if z.columns.duplicated().any():
        raise ValueError("Duplicate XGBoost feature names after encoding")
    return z.iloc[:len(a)].reset_index(drop=True).astype("float32"), z.iloc[len(a):].reset_index(drop=True).astype("float32")


def gpu_was_used(model):
    conf = json.loads(model.get_booster().save_config())
    learner = conf.get("learner", {})
    device = str(learner.get("generic_param", {}).get("device", ""))
    if not device.startswith("cuda"):
        raise RuntimeError(f"XGBoost did not run in CUDA mode, actual device={device}")
    return device


def main():
    hardware = verify_nvidia()
    import xgboost as xgb
    version = tuple(int(x) for x in xgb.__version__.split(".")[:2])
    if version < (2, 0):
        raise RuntimeError(f"XGBoost >=2 required for device=cuda; found {xgb.__version__}")
    start = time.monotonic()
    tr = pd.read_csv(find("train.csv"))
    te = pd.read_csv(find("test.csv"))
    sample = pd.read_csv(find("sample_submission.csv"))
    targets = [c for c in tr if c not in te]
    if len(targets) != 1:
        raise ValueError(f"Ambiguous target columns: {targets}")
    target = targets[0]
    if target not in sample or len(sample) != len(te):
        raise ValueError("Invalid sample_submission target/row count")
    y = y_binary(tr[target])
    X, Xt = prepare(tr, te, sample)

    variants = {
        "gpu_xgb_depth5": {"max_depth": 5, "learning_rate": .035, "min_child_weight": 3, "reg_lambda": 5.0},
        "gpu_xgb_depth7": {"max_depth": 7, "learning_rate": .025, "min_child_weight": 5, "reg_lambda": 8.0},
    }
    results = {}
    for name, hp in variants.items():
        oof = np.zeros(len(X), dtype=np.float32)
        test_preds = np.zeros(len(Xt), dtype=np.float64)
        fold_rows = []
        for fold, (tr_idx, va_idx) in enumerate(CV.split(X, y), 1):
            model = xgb.XGBClassifier(
                n_estimators=1100,
                learning_rate=hp["learning_rate"],
                max_depth=hp["max_depth"],
                min_child_weight=hp["min_child_weight"],
                reg_lambda=hp["reg_lambda"],
                subsample=.9, colsample_bytree=.9,
                objective="binary:logistic", eval_metric="auc",
                tree_method="hist", device="cuda",
                early_stopping_rounds=80, random_state=SEED + fold,
                n_jobs=4,
            )
            model.fit(X.iloc[tr_idx], y[tr_idx],
                      eval_set=[(X.iloc[va_idx], y[va_idx])], verbose=False)
            actual = gpu_was_used(model)
            vp = model.predict_proba(X.iloc[va_idx])[:, 1]
            tp = model.predict_proba(Xt)[:, 1]
            if not np.isfinite(vp).all() or not np.isfinite(tp).all():
                raise RuntimeError("Non-finite XGBoost predictions")
            oof[va_idx] = vp
            test_preds += tp / 3
            fold_rows.append({"fold": fold, "auc": float(roc_auc_score(y[va_idx], vp)),
                              "best_iteration": int(model.best_iteration), "device": actual})
            print(f"{name}: fold {fold}/3 AUC={fold_rows[-1]['auc']:.8f} device={actual}", flush=True)
        auc = float(roc_auc_score(y, oof))
        results[name] = {"cv_auc": auc, "folds": fold_rows, "hyperparameters": hp}
        sub = sample.copy()
        sub[target] = test_preds
        sub.to_csv(OUT / f"gpu_candidate_{name}.csv", index=False)
        emit("v23_gpu_progress.json", {"completed": list(results), "models": results})

    best = max(results, key=lambda name: results[name]["cv_auc"])
    import shutil
    shutil.copyfile(OUT / f"gpu_candidate_{best}.csv", OUT / "candidate_submission.csv")
    for p in OUT.glob("gpu_candidate_*.csv"):
        p.unlink()
    emit("v23_gpu_metrics.json", {
        "train_rows": len(tr), "test_rows": len(te), "target": target,
        "feature_count": X.shape[1],
        "validation": "StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)",
        "seed": SEED, "folds": 3, "validation_metric": "roc_auc",
        "gpu_verified": True, "hardware": hardware,
        "xgboost_version": xgb.__version__,
        "models": results, "best_model": best, "best_cv_auc": results[best]["cv_auc"],
        "elapsed_seconds": time.monotonic() - start,
    })
    print(f"Best GPU XGBoost CV={results[best]['cv_auc']:.8f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        emit("v23_gpu_error.json", {"error": traceback.format_exc()})
        raise
