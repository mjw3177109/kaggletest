"""V2.9.3: pre-registered low-cost fusion of *real archived* Kaggle OOF models.

No training labels from the Kaggle test set. Only four fixed candidates, a
shared 3-fold validation protocol, and fold-level robustness checks. Never
call this with score-only OOF from a different split or row ordering.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 20261018
VALIDATION = "StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)"
N_TRAIN = 699635
N_TEST = 299844
MIN_CV_GAIN = 0.00003
# Fixed *before* evaluating labels; avoid broad on-OOF hyperparameter search.
PRE_REGISTERED = (
    ("v25_60_v28_40", {"v25": .60, "v28": .40}),
    ("v25_35_v28_65", {"v25": .35, "v28": .65}),
    ("xgb_50_cat_50", {"xgb": .50, "cat": .50}),
    ("v25_25_v28_50_cat_25", {"v25": .25, "v28": .50, "cat": .25}),
)


def rank(arr):
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if not len(arr) or not np.isfinite(arr).all():
        raise ValueError("Empty or nonfinite predictions")
    return rankdata(arr, method="average") / len(arr)


def reconstruct(npz, version):
    """Recreate exactly the preselected rank_blend from original model OOF."""
    y = np.asarray(npz["target"]).reshape(-1)
    if len(y) != N_TRAIN or not np.isin(y, [0, 1]).all():
        raise ValueError("Unexpected training labels or length")
    a = np.asarray(npz["oof_lgb"], dtype=np.float64)
    bkey = "oof_xgb" if version == "v25" else "oof_cat"
    b = np.asarray(npz[bkey], dtype=np.float64)
    for arr in (a, b):
        if len(arr) != len(y) or not np.isfinite(arr).all():
            raise ValueError("Invalid archived OOF")
        if np.min(arr) < 0 or np.max(arr) > 1:
            raise ValueError("Archived OOF probabilities out of range")
    return y, (rank(a) + rank(b)) / 2, b


def verify_sources(npz25, npz28, status25, status28, submission25, submission28):
    if status25.get("stage") != "complete" or not status25.get("submitted"):
        raise ValueError("V2.5 must be a verified submitted champion")
    if status28.get("stage") not in ("complete", "skipped_no_cv_gain"):
        raise ValueError("V2.8 must have completed all folds")
    for ver, source in (("v25", status25), ("v28", status28)):
        m = source.get("metrics") or {}
        h = source.get("hardware") or {}
        if (source.get("competition") != "playground-series-s6e10"
                or m.get("validation") != VALIDATION
                or m.get("best_model") != "blend_rank"
                or int(m.get("train_rows", -1)) != N_TRAIN
                or int(m.get("test_rows", -1)) != N_TEST
                or not m.get("gpu_verified")
                or not h.get("ok")
                or not m.get("leakage_safe_target_encoding")):
            raise ValueError(f"{ver} is not a comparable validated OOF source")
    y25, rank25, xgb = reconstruct(npz25, "v25")
    y28, rank28, cat = reconstruct(npz28, "v28")
    if not np.array_equal(y25, y28):
        raise ValueError("OOF training target order does not match")
    for name, y, pred, state in (("V2.5", y25, rank25, status25), ("V2.8", y28, rank28, status28)):
        measured = roc_auc_score(y, pred)
        recorded = float(state["metrics"]["best_cv_auc"])
        if abs(measured - recorded) > 0.000002:
            raise ValueError(f"{name} archived OOF and recorded CV differ: {measured} vs {recorded}")
    if list(submission25.columns) != ["id", "satisfaction"] or list(submission28.columns) != ["id", "satisfaction"]:
        raise ValueError("Submission schema mismatch")
    if len(submission25) != N_TEST or len(submission28) != N_TEST:
        raise ValueError("Submission row-count mismatch")
    if not np.array_equal(submission25["id"].to_numpy(), submission28["id"].to_numpy()):
        raise ValueError("Kaggle test identifiers are not in the same order")
    # Test predictions are independently scored saved submissions, *not* labels.
    tests = {
        "v25": rank(submission25["satisfaction"].to_numpy()),
        "v28": rank(submission28["satisfaction"].to_numpy()),
        # Individual-model test predictions were not archived; do not silently
        # substitute a blended submission for those individual components.
    }
    return y25, {"v25": rank25, "v28": rank28,
                 "xgb": rank(xgb), "cat": rank(cat)}, tests


def evaluate(y, oofs, tests, benchmark, min_gain=MIN_CV_GAIN):
    """Compare only predeclared candidates reproducibly; reject unavailable test branches."""
    y = np.asarray(y).reshape(-1)
    if len(y) != N_TRAIN or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid labels")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    splits = list(cv.split(np.zeros(len(y), dtype=np.uint8), y))
    candidates = []
    eligible = []
    for name, weights in PRE_REGISTERED:
        if not all(key in tests and key in oofs for key in weights):
            candidates.append({"candidate": name, "status": "unavailable_test_predictions"})
            continue
        fused = sum(weight * rank(oofs[key]) for key, weight in weights.items())
        score = float(roc_auc_score(y, fused))
        folds = [float(roc_auc_score(y[iv], fused[iv])) for _, iv in splits]
        if not np.isfinite(score) or not all(np.isfinite(f) for f in folds):
            raise ValueError("Nonfinite experiment AUC")
        row = {"candidate": name, "cv_auc": score, "fold_auc": folds,
               "delta_vs_best": score - benchmark, "weights": weights}
        candidates.append(row)
        # A small blend win must not cause an outsized single-fold regression.
        if score >= benchmark + min_gain:
            eligible.append((score, name, weights, fused, folds))
    if not eligible:
        return {"stage": "skipped_no_cv_gain", "candidates": candidates, "benchmark_cv": benchmark,
                "min_cv_gain": min_gain}, None
    score, name, weights, _, folds = max(eligible, key=lambda v: (v[0], v[1]))
    prediction = sum(weight * rank(tests[key]) for key, weight in weights.items())
    if len(prediction) != N_TEST or not np.isfinite(prediction).all():
        raise ValueError("Invalid test predictions")
    if np.min(prediction) < 0 or np.max(prediction) > 1:
        raise ValueError("Blended predictions outside [0,1]")
    signature = hashlib.sha256(json.dumps({"experiment": "v293-archived-fusion",
        "validation": VALIDATION, "weights": weights, "sources": ["v25","v28"]},sort_keys=True).encode()).hexdigest()[:12]
    return {"stage": "candidate_ready", "candidate": name, "cv_auc": score,
            "fold_auc": folds, "weights": weights, "benchmark_cv": benchmark,
            "min_cv_gain": min_gain, "delta": score - benchmark,
            "signature": signature, "candidates": candidates}, prediction
