"""V2.9.27 full RealMLP-TD three-fold experiment with fixed pilot configuration.

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
MODEL_SEED = 20261027
FOLDS = 3
EXPERTS = ('Personal Travel', 'Business travel')
TRAIN_ROWS = 699635
TEST_ROWS = 299844
RUNTIME_BUDGET_SECONDS = 5400
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
    dump("v2927_hardware.json", evidence)
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

    # Pre-registered additive feature family. All transformations use only
    # same-row unlabeled input values or outer-training-fold statistics.
    # No target statistics or test-set distribution are fitted here.
    service_cols = ["Inflight wifi service", "Online boarding",
                    "Seat comfort", "Inflight entertainment", "Leg room service"]
    for dst in (tr, va):
        service = dst[service_cols].to_numpy(dtype="float32")
        dst["interaction_service_mean"] = service.mean(axis=1)
        dst["interaction_service_std"] = service.std(axis=1)
        dst["interaction_service_min"] = service.min(axis=1)
        dst["interaction_service_low_count"] = (service <= 2).sum(axis=1).astype("float32")
        dst["interaction_wifi_boarding_gap"] = (dst["Inflight wifi service"] -
                                                  dst["Online boarding"]).astype("float32")
        depart = np.maximum(dst["Departure Delay in Minutes"].to_numpy(dtype="float32"), 0)
        arrive = np.maximum(dst["Arrival Delay in Minutes"].to_numpy(dtype="float32"), 0)
        dst["interaction_delay_log_total"] = np.log1p(depart + arrive).astype("float32")
        dst["interaction_delay_signed_gap"] = (arrive - depart).astype("float32")
        dst["interaction_service_delay"] = (service.mean(axis=1) /
                                           (1.0 + np.log1p(depart + arrive))).astype("float32")
        dst["interaction_flight_distance_log"] = np.log1p(
            np.maximum(dst["Flight Distance"].to_numpy(dtype="float32"),0)).astype("float32")
    if list(tr.columns) != list(va.columns):
        raise RuntimeError("V2927 interaction feature schema differs")
    if not np.isfinite(tr.select_dtypes(include="number").to_numpy()).all():
        raise RuntimeError("V2927 train interactions are nonfinite")
    if not np.isfinite(va.select_dtypes(include="number").to_numpy()).all():
        raise RuntimeError("V2927 validation interactions are nonfinite")

    # Inner cross-fitting, including the training row itself only via out-of-fold
    # mapping. Never use outer validation labels when fitting this encoder.
    return tr, va, te_train, te_val



def travel_mask(frame, group):
    """Gating is observed input Type of Travel, never a learned label."""
    col = frame["Type of Travel"].astype("string")
    if col.isna().any():
        raise ValueError("Missing Type of Travel is not allowed in V2927")
    unique = set(col.unique().tolist())
    if not unique.issubset(set(EXPERTS)):
        raise ValueError(f"Unexpected travel category {unique - set(EXPERTS)}")
    return col.eq(group).to_numpy(dtype=bool)


def make_model(fold, expert_index):
    Model = install_realmlp()
    return Model(
        device="cuda:0", random_state=MODEL_SEED + 2*fold + expert_index,
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


def fit_expert(train, test, targets, itr, iva, fold, expert_index, group):
    """Expert sees only its own outer-training group, never holdout labels."""
    start = time.monotonic()
    fit_idx = itr[travel_mask(train.iloc[itr], group)]
    val_idx = iva[travel_mask(train.iloc[iva], group)]
    test_idx = np.flatnonzero(travel_mask(test, group))
    if min(len(fit_idx), len(val_idx), len(test_idx)) < 100:
        raise RuntimeError(f"Unexpectedly small V2927 group {group}")
    if np.unique(targets[fit_idx]).size != 2 or np.unique(targets[val_idx]).size != 2:
        raise RuntimeError("Missing class in expert fold; cannot report subgroup AUC")
    fit_frame = train.iloc[fit_idx]
    val_frame = train.iloc[val_idx]
    test_frame = test.iloc[test_idx]
    xa, xv, te_train, te_val = engineer(fit_frame, val_frame)
    _, xt, _, te_test = engineer(fit_frame, test_frame)
    if list(xa.columns) != list(xv.columns) or list(xa.columns) != list(xt.columns):
        raise RuntimeError("V2927 expert train/validation/test feature schemas differ")
    ytr = targets[fit_idx]
    enc = TargetEncoder(target_type="binary", smooth="auto", cv=5,
                        shuffle=True, random_state=SEED + 2*fold + expert_index)
    te_cols = [f"te_{i}" for i in range(te_train.shape[1])]
    xa[te_cols] = enc.fit_transform(te_train, ytr).astype("float32")
    xv[te_cols] = enc.transform(te_val).astype("float32")
    xt[te_cols] = enc.transform(te_test).astype("float32")
    if list(xa.columns) != list(xv.columns) or list(xa.columns) != list(xt.columns):
        raise RuntimeError("V2927 cross-fit feature columns differ")
    if len(xa.columns) != 46:  # 37 champion columns + 9 V2916 interactions
        raise RuntimeError(f"Unexpected V2927 engineered feature count: {len(xa.columns)}")
    model = make_model(fold, expert_index)
    model.fit(xa, pd.Series(ytr))
    pv = np.asarray(model.predict_proba(xv)[:, 1], dtype="float32")
    pt = np.asarray(model.predict_proba(xt)[:, 1], dtype="float32")
    if pv.shape != (len(val_idx),) or pt.shape != (len(test_idx),):
        raise RuntimeError("V2927 expert prediction shape mismatch")
    if not np.isfinite(pv).all() or not np.isfinite(pt).all():
        raise RuntimeError("V2927 expert nonfinite probability")
    if not (0 <= pv.min() <= pv.max() <= 1 and 0 <= pt.min() <= pt.max() <= 1):
        raise RuntimeError("V2927 expert probability outside [0,1]")
    del model
    import torch
    torch.cuda.empty_cache()
    return val_idx, test_idx, pv, pt, {
        "expert": group,
        "expert_index": expert_index,
        "model_seed": MODEL_SEED + 2*fold + expert_index,
        "train_rows": int(len(fit_idx)),
        "validation_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "auc": float(roc_auc_score(targets[val_idx], pv)),
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "features": int(len(xa.columns)),
        "inner_target_folds": 5,
    }


def main():
    start = time.monotonic()
    evidence = prove_gpu()
    train = only_competition_file("train.csv")
    test = only_competition_file("test.csv")
    if len(train) != TRAIN_ROWS or len(test) != TEST_ROWS:
        raise RuntimeError("V2927 competition dataset size mismatch")
    if ("id" not in train or "id" not in test or
        "satisfaction" not in train or "satisfaction" in test):
        raise RuntimeError("V2927 ID/target schema mismatch")
    if not train["id"].is_unique or not test["id"].is_unique:
        raise RuntimeError("V2927 duplicate IDs")
    y = pd.to_numeric(train["satisfaction"], errors="raise").to_numpy()
    if not np.isin(y, [0, 1]).all():
        raise RuntimeError("V2927 only binary target allowed")
    if not set(train["Type of Travel"].astype("string").unique()) == set(EXPERTS):
        raise RuntimeError("V2927 training gate category mismatch")
    if not set(test["Type of Travel"].astype("string").unique()) == set(EXPERTS):
        raise RuntimeError("V2927 test gate category mismatch")
    splits = list(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(
        np.zeros(len(y), dtype=np.uint8), y))
    oof = np.full(len(y), np.nan, dtype="float32")
    test_sum = np.zeros(len(test), dtype="float64")
    cover = np.zeros(len(y), dtype="uint8")
    test_cover = np.zeros(len(test), dtype="uint8")
    details = []
    for fold, (itr, iva) in enumerate(splits):
        if time.monotonic() - start > RUNTIME_BUDGET_SECONDS:
            raise TimeoutError("V2927 threefold budget exhausted; no partial CV")
        fold_pred = np.full(len(iva), np.nan, dtype="float32")
        experts = []
        for expert_index, group in enumerate(EXPERTS):
            val_idx, test_idx, pv, pt, result = fit_expert(
                train, test, y, itr, iva, fold, expert_index, group)
            if np.any(cover[val_idx] != 0):
                raise RuntimeError("V2927 overlapping validation expert predictions")
            oof[val_idx] = pv
            cover[val_idx] += 1
            test_sum[test_idx] += pt / FOLDS
            test_cover[test_idx] += 1
            experts.append(result)
        if not np.all(cover[iva] == 1):
            raise RuntimeError("V2927 expert gate did not cover complete outer validation")
        result = {
            "fold": fold, "validation_rows": int(len(iva)),
            "auc": float(roc_auc_score(y[iva], oof[iva])),
            "experts": experts,
        }
        details.append(result)
        dump("v2927_progress.json", {"finished": fold + 1,
                                      "fold_results": details, "complete": False})
        print("V2927 gated full fold", fold, result, flush=True)
    if not np.all(cover == 1) or not np.all(test_cover == FOLDS):
        raise RuntimeError("V2927 incomplete or overlapping expert OOF/test coverage")
    if not np.isfinite(oof).all() or not np.isfinite(test_sum).all():
        raise RuntimeError("V2927 incomplete OOF/test")
    auc = float(roc_auc_score(y, oof))
    group_auc = {}
    for group in EXPERTS:
        mask = travel_mask(train, group)
        group_auc[group] = float(roc_auc_score(y[mask], oof[mask]))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / "v2927_realmlp_oof.npz",
                        target=y.astype("int8"),
                        train_id=train["id"].to_numpy(),
                        oof_gated_realmlp=oof,
                        train_travel_type=train["Type of Travel"].astype(str).to_numpy(),
                        test_id=test["id"].to_numpy(),
                        pred_gated_realmlp=test_sum.astype("float32"),
                        test_travel_type=test["Type of Travel"].astype(str).to_numpy())
    dump("v2927_metrics.json", {
        "competition": "playground-series-s6e10",
        "model": "two travel-type-gated RealMLP-TD n_ens=8 5epoch",
        "stage": "three_folds_complete",
        "train_rows": len(train), "test_rows": len(test),
        "validation": "StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)",
        "folds": 3, "fold_results": details,
        "full_oof_auc": auc, "group_auc": group_auc,
        "gpu_verified": True, "hardware": evidence,
        "leakage_safe_inner_cross_fit": True,
        "outer_validation_used_in_training": False,
        "feature_count": 46, "interaction_feature_count": 9,
        "route_feature_count": 0, "experts": list(EXPERTS),
        "model_seed": MODEL_SEED, "epochs": 5, "n_ens": 8,
        "runtime_seconds": round(time.monotonic() - start, 1),
        "feature_family": "v2916_service_delay_interactions_no_route_profile",
        "external_labels": False, "submitted": False,
        "ready_for_submission": False,
    })
    print("V2927 full gated RealMLP OOF AUC", auc, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        dump("v2927_error.json", {"traceback": traceback.format_exc()[-8000:]})
        raise
