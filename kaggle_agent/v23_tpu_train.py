"""V2.3 independent JAX MLP on a REAL Kaggle TPU. Never silently fall back to CPU/GPU.

The V2.1 boosted-tree training and its committed scores remain unchanged.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

OUT = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
SEED = 20261018
FOLDS = 3
EPOCHS = 14
BATCH = 2048
PRED_BATCH = 16384
PATIENCE = 4


def save(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def probe_real_tpu():
    result = {"requested": "TpuV5E8", "actual": "unknown", "ok": False}
    try:
        import jax
        devices = jax.devices()
        platforms = sorted({d.platform for d in devices})
        result.update({
            "jax_backend": jax.default_backend(),
            "platforms": platforms,
            "device_count": len(devices),
            "jax_version": jax.__version__,
        })
        result["ok"] = jax.default_backend() == "tpu" and "tpu" in platforms
        result["actual"] = jax.default_backend()
        save("v23_hardware.json", result)
        if not result["ok"]:
            raise RuntimeError(
                "Kaggle CLI may have silently provisioned CPU/GPU instead of TPU. "
                "TPU training aborted; no fake TPU benchmark or CPU fallback."
            )
        return jax
    except Exception as exc:
        result["error"] = repr(exc)
        save("v23_hardware.json", result)
        raise


def source(filename):
    matches = list(INPUT.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {filename}; found {matches[:8]}")
    return matches[0]


def features(train, test, sample):
    # Mirror the V2.1 feature recipe to make the same CV comparison meaningful.
    exclude = [c for c in sample if c in test]
    fields = [c for c in test if c not in exclude]
    a = train[fields].copy()
    b = test[fields].copy()
    delay = [c for c in fields if "delay" in c.lower() and pd.api.types.is_numeric_dtype(a[c])]
    if len(delay) >= 2:
        for frame in (a, b):
            frame["__auto_total_delay"] = frame[delay].sum(axis=1, skipna=True)
            frame["__auto_max_delay"] = frame[delay].max(axis=1, skipna=True)
    tokens = ("service", "comfort", "cleanliness", "booking", "boarding", "location", "entertainment")
    ratings = [c for c in fields if any(k in c.lower() for k in tokens) and pd.api.types.is_numeric_dtype(a[c])]
    if len(ratings) >= 3:
        for frame in (a, b):
            frame["__auto_rating_mean"] = frame[ratings].mean(axis=1)
            frame["__auto_rating_min"] = frame[ratings].min(axis=1)
            frame["__auto_rating_max"] = frame[ratings].max(axis=1)
    combined = pd.concat([a, b], ignore_index=True)
    cats = list(combined.select_dtypes(include=["object", "category", "bool"]).columns)
    for col in combined:
        if col not in cats:
            combined[col] = pd.to_numeric(combined[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            if combined[col].isna().any():
                med = combined[col].median()
                combined[col] = combined[col].fillna(0 if pd.isna(med) else med)
    combined = pd.get_dummies(combined, columns=cats, dummy_na=True, dtype=np.uint8)
    x = combined.iloc[:len(a)].to_numpy(dtype=np.float32)
    xt = combined.iloc[len(a):].to_numpy(dtype=np.float32)
    return x, xt, list(combined.columns)


def binary_target(series):
    if series.isna().any() or series.nunique() != 2:
        raise ValueError("V2.3 only supports binary classification with no missing labels")
    values = sorted(series.astype(str).unique().tolist())
    return series.astype(str).map({values[0]: 0, values[1]: 1}).to_numpy(dtype=np.float32)


def main():
    start = time.monotonic()
    jax = probe_real_tpu()
    import jax.numpy as jnp
    from jax import tree_util

    train = pd.read_csv(source("train.csv"))
    test = pd.read_csv(source("test.csv"))
    sample = pd.read_csv(source("sample_submission.csv"))
    target_cols = [c for c in train if c not in test]
    if len(target_cols) != 1:
        raise ValueError(f"Ambiguous classification target: {target_cols}")
    target = target_cols[0]
    if target not in sample or len(sample) != len(test):
        raise ValueError("sample_submission.csv does not match target/test rows")
    y = binary_target(train[target])
    x, xt, names = features(train, test, sample)
    if not np.isfinite(x).all() or not np.isfinite(xt).all():
        raise ValueError("Non-finite matrix values")

    # Single-device JAX is intentional for reliability; TPU is still confirmed at runtime.
    def init_params(key, count):
        sizes = [count, 128, 64, 1]
        keys = jax.random.split(key, 3)
        return tuple(
            (
                jax.random.normal(k, (sizes[i], sizes[i + 1]), dtype=jnp.float32)
                * jnp.sqrt(2.0 / sizes[i]),
                jnp.zeros((sizes[i + 1],), dtype=jnp.float32),
            )
            for i, k in enumerate(keys)
        )

    def forward(p, batch):
        h = batch
        for weight, bias in p[:-1]:
            h = jax.nn.relu(h @ weight + bias)
        w, b = p[-1]
        return (h @ w + b).reshape(-1)

    def loss(p, xb, yb):
        logits = forward(p, xb)
        return jnp.mean(jax.nn.softplus(logits) - yb * logits)

    @jax.jit
    def step(p, moments, velocities, xb, yb, t):
        loss_val, grads = jax.value_and_grad(loss)(p, xb, yb)
        grads = tree_util.tree_map(lambda a: jnp.clip(a, -3.0, 3.0), grads)
        m = tree_util.tree_map(lambda old, grad: 0.9 * old + 0.1 * grad, moments, grads)
        v = tree_util.tree_map(lambda old, grad: 0.999 * old + 0.001 * (grad ** 2), velocities, grads)
        update = tree_util.tree_map(
            lambda a, b: (a / (1.0 - 0.9 ** t)) / (jnp.sqrt(b / (1.0 - 0.999 ** t)) + 1e-8),
            m, v
        )
        p = tree_util.tree_map(lambda par, delta: par - 0.002 * delta, p, update)
        return p, m, v, loss_val

    infer = jax.jit(forward)

    def predict(params, values):
        pieces = []
        for start_idx in range(0, len(values), PRED_BATCH):
            batch = jnp.asarray(values[start_idx:start_idx + PRED_BATCH])
            logits = infer(params, batch)
            pieces.append(np.asarray(jax.nn.sigmoid(logits), dtype=np.float32))
        return np.concatenate(pieces)

    oof = np.zeros(len(x), dtype=np.float32)
    test_pred = np.zeros(len(xt), dtype=np.float64)
    fold_results = []
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(cv.split(x, y), start=1):
        # Fit scaling ONLY on each fold's training rows: no validation-target leakage.
        mean = x[tr].mean(axis=0, dtype=np.float64).astype(np.float32)
        std = x[tr].std(axis=0, dtype=np.float64).astype(np.float32)
        std[std < 1e-6] = 1.0
        xf = np.clip((x[tr] - mean) / std, -10, 10).astype(np.float32)
        vf = np.clip((x[va] - mean) / std, -10, 10).astype(np.float32)
        tf = np.clip((xt - mean) / std, -10, 10).astype(np.float32)
        labels = y[tr]
        p = init_params(jax.random.PRNGKey(SEED + fold), x.shape[1])
        m = tree_util.tree_map(jnp.zeros_like, p)
        v = tree_util.tree_map(jnp.zeros_like, p)
        tick, best_auc, best_epoch, best_p, stalled = 0, -1.0, 0, p, 0
        rng = np.random.default_rng(SEED + fold)
        history = []
        for epoch in range(1, EPOCHS + 1):
            order = rng.permutation(len(tr))
            losses = []
            for begin in range(0, len(order), BATCH):
                ids = order[begin:begin + BATCH]
                if len(ids) != BATCH:
                    # Fixed TPU batch shape avoids repeated recompilation.
                    ids = np.concatenate([ids, order[:BATCH - len(ids)]])
                tick += 1
                p, m, v, l = step(p, m, v, jnp.asarray(xf[ids]), jnp.asarray(labels[ids]), jnp.asarray(tick))
                if not np.isfinite(float(l)):
                    raise RuntimeError("Neural network loss diverged")
                losses.append(float(l))
            pred = predict(p, vf)
            auc = float(roc_auc_score(y[va], pred))
            history.append({"epoch": epoch, "val_auc": auc, "loss": float(np.mean(losses))})
            if auc > best_auc + 0.00002:
                best_auc, best_epoch, best_p, stalled = auc, epoch, p, 0
            else:
                stalled += 1
                if stalled >= PATIENCE:
                    break
            # Hard budget avoids running forever on Kaggle.
            if time.monotonic() - start > 2850:
                raise TimeoutError("TPU training exceeded 2850 second budget; results are incomplete")
        p = best_p
        oof[va] = predict(p, vf)
        test_pred += predict(p, tf) / FOLDS
        fold_results.append({"fold": fold, "best_epoch": best_epoch, "val_auc": float(roc_auc_score(y[va], oof[va])), "history": history})
        save("v23_progress.json", {"finished_folds": fold, "fold_metrics": fold_results})
        print(f"TPU fold {fold}/{FOLDS}: AUC={fold_results[-1]['val_auc']:.8f}", flush=True)
    score = float(roc_auc_score(y, oof))
    submission = sample.copy()
    submission[target] = test_pred
    if submission[target].isna().any() or not np.isfinite(test_pred).all():
        raise ValueError("Invalid neural submission predictions")
    submission.to_csv(OUT / "candidate_submission.csv", index=False)
    save("v23_metrics.json", {
        "method": "JAX two-hidden-layer MLP on TPU v5e-8",
        "target": target, "train_rows": int(len(train)), "test_rows": int(len(test)),
        "feature_count": int(x.shape[1]), "validation": "StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)",
        "folds": FOLDS, "seed": SEED, "cv_auc": score, "fold_metrics": fold_results,
        "backend": jax.default_backend(), "device_count": len(jax.devices()),
        "elapsed_seconds": round(time.monotonic() - start, 1),
        "architecture": "35(or dynamic)-128-64-1 ReLU MLP; foldwise standardization; Adam",
    })
    print(f"TPU 3-fold OOF ROC-AUC = {score:.8f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        save("v23_error.json", {"error": traceback.format_exc()})
        raise
