"""Kaggle V2.2: Optuna LightGBM tuning, full 3-fold OOF and feature attribution.

Runs inside Kaggle Notebook. Reads only competition datasets from /kaggle/input.
Writes reports and one candidate submission to /kaggle/working.
"""
from __future__ import annotations

import json
import os
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 20261018  # Identical to V2.1 so full OOF AUC is comparable
OUT = Path("/kaggle/working")
IN = Path("/kaggle/input")
TRIALS = int(os.environ.get("V22_TRIALS", "8"))
TUNE_ROWS = int(os.environ.get("V22_TUNE_ROWS", "120000"))
TUNE_ROUNDS = int(os.environ.get("V22_TUNE_ROUNDS", "500"))
FULL_ROUNDS = int(os.environ.get("V22_FULL_ROUNDS", "650"))


def output(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def locate(name):
    paths = list(IN.rglob(name))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected exactly one {name} under /kaggle/input, found: {paths[:10]}")
    return paths[0]


def target_to_binary(y):
    if y.isna().any() or y.nunique() != 2:
        raise ValueError("V2.2 supports only clean binary target labels.")
    if pd.api.types.is_bool_dtype(y):
        return y.astype("int8").to_numpy()
    if pd.api.types.is_numeric_dtype(y):
        vals = sorted(y.unique().tolist())
        return y.map({vals[0]: 0, vals[1]: 1}).astype("int8").to_numpy()
    vals = sorted(y.astype(str).unique().tolist())
    return y.astype(str).map({vals[0]: 0, vals[1]: 1}).astype("int8").to_numpy()


def features(train, test, sample, variant):
    # Same 21 raw fields and same engineered columns as V2.1 by default.
    id_cols = [c for c in sample.columns if c in test.columns]
    names = [c for c in test.columns if c not in id_cols]
    a, b = train[names].copy(), test[names].copy()
    delay = [c for c in names if "delay" in c.lower() and pd.api.types.is_numeric_dtype(a[c])]
    if len(delay) >= 2:
        for frame in (a, b):
            frame["__auto_total_delay"] = frame[delay].sum(axis=1, skipna=True)
            frame["__auto_max_delay"] = frame[delay].max(axis=1, skipna=True)
    tokens = ("service", "comfort", "cleanliness", "booking", "boarding", "location", "entertainment")
    rating = [c for c in names if any(t in c.lower() for t in tokens) and pd.api.types.is_numeric_dtype(a[c])]
    if len(rating) >= 3:
        for frame in (a, b):
            frame["__auto_rating_mean"] = frame[rating].mean(axis=1)
            frame["__auto_rating_min"] = frame[rating].min(axis=1)
            frame["__auto_rating_max"] = frame[rating].max(axis=1)
    if variant == "extended":
        # Pure row-wise features: no target encoding or use of unseen labels.
        if len(rating) >= 3:
            for frame in (a, b):
                frame["__rating_std"] = frame[rating].std(axis=1).fillna(0)
                frame["__rating_range"] = frame[rating].max(axis=1) - frame[rating].min(axis=1)
                frame["__rating_low_count"] = (frame[rating] <= 2).sum(axis=1)
                frame["__rating_high_count"] = (frame[rating] >= 4).sum(axis=1)
        if len(delay) >= 2:
            for frame in (a, b):
                frame["__delay_difference"] = frame[delay[0]] - frame[delay[1]]
                frame["__any_delay"] = (frame[delay].fillna(0) > 0).any(axis=1).astype("int8")
    both = pd.concat([a, b], ignore_index=True)
    cats = list(both.select_dtypes(include=["object", "category", "bool"]).columns)
    for col in both:
        if col not in cats:
            both[col] = pd.to_numeric(both[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if both[col].isna().any():
                med = both[col].median()
                both[col] = both[col].fillna(0 if pd.isna(med) else med)
    both = pd.get_dummies(both, columns=cats, dummy_na=True, dtype=np.uint8)
    both.columns = both.columns.astype(str)
    if both.columns.duplicated().any():
        raise ValueError("Duplicate encoded feature names")
    return both.iloc[:len(a)].reset_index(drop=True), both.iloc[len(a):].reset_index(drop=True)


def model_params(params, *, rounds, seed=SEED):
    return {
        "n_estimators": rounds,
        "learning_rate": params["learning_rate"],
        "num_leaves": params["num_leaves"],
        "max_depth": -1,
        "min_child_samples": params["min_child_samples"],
        "reg_lambda": params["reg_lambda"],
        "reg_alpha": params["reg_alpha"],
        "colsample_bytree": params["colsample_bytree"],
        "subsample": 1.0,  # Bagging off unless subsample_freq is enabled
        "random_state": seed,
        "n_jobs": 4,
        "verbosity": -1,
    }


def main():
    import lightgbm as lgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train = pd.read_csv(locate("train.csv"))
    test = pd.read_csv(locate("test.csv"))
    sample = pd.read_csv(locate("sample_submission.csv"))
    target_cols = [c for c in train if c not in test]
    if len(target_cols) != 1:
        raise ValueError(f"Ambiguous target columns: {target_cols}")
    target = target_cols[0]
    if target not in sample or len(sample) != len(test):
        raise ValueError("Sample submission does not match test data or target.")
    y = target_to_binary(train[target])

    X, Xt = features(train, test, sample, "v21")
    Xext, Xtext = features(train, test, sample, "extended")
    if TUNE_ROWS < len(X):
        selected, _ = train_test_split(
            np.arange(len(X)), train_size=min(TUNE_ROWS, len(X) - 2),
            random_state=SEED, stratify=y
        )
    else:
        selected = np.arange(len(X))
    selected = np.sort(selected)
    Xsmall, ysmall = X.iloc[selected], y[selected]
    tuning_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)

    def objective(trial):
        hp = {
            "learning_rate": trial.suggest_float("learning_rate", 0.018, 0.085, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 72),
            "min_child_samples": trial.suggest_int("min_child_samples", 25, 115),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.08, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 3.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.68, 1.0),
        }
        scores = []
        for train_idx, valid_idx in tuning_cv.split(Xsmall, ysmall):
            model = lgb.LGBMClassifier(**model_params(hp, rounds=TUNE_ROUNDS))
            model.fit(
                Xsmall.iloc[train_idx], ysmall[train_idx],
                eval_set=[(Xsmall.iloc[valid_idx], ysmall[valid_idx])],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(55, verbose=False), lgb.log_evaluation(0)],
            )
            pred = model.predict_proba(Xsmall.iloc[valid_idx])[:, 1]
            scores.append(float(roc_auc_score(ysmall[valid_idx], pred)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=TRIALS, timeout=1000, catch=(ValueError,))
    if not study.best_trials:
        raise RuntimeError("No successful Optuna trials")
    hp = study.best_params
    output("v22_trials.json", {
        "search": "Optuna TPESampler",
        "tuning_rows": len(selected),
        "tuning_folds": 2,
        "seed": SEED,
        "best_params": hp,
        "trials": [
            {"number": t.number, "state": t.state.name, "cv_auc": t.value, "params": t.params}
            for t in study.trials
        ],
    })

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    candidates = {}
    gains = {}
    for variant, (mat, mat_test) in {
        "tuned_lightgbm": (X, Xt),
        "tuned_extended": (Xext, Xtext),
    }.items():
        oof = np.zeros(len(y), dtype=np.float32)
        test_pred = np.zeros(len(mat_test), dtype=np.float64)
        fold_scores = []
        importance = np.zeros(mat.shape[1], dtype=np.float64)
        for fi, (tr_idx, va_idx) in enumerate(cv.split(mat, y), 1):
            model = lgb.LGBMClassifier(**model_params(hp, rounds=FULL_ROUNDS, seed=SEED + fi))
            model.fit(
                mat.iloc[tr_idx], y[tr_idx],
                eval_set=[(mat.iloc[va_idx], y[va_idx])],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(65, verbose=False), lgb.log_evaluation(0)]
            )
            pred = model.predict_proba(mat.iloc[va_idx])[:, 1]
            oof[va_idx] = pred
            fold_scores.append(float(roc_auc_score(y[va_idx], pred)))
            test_pred += model.predict_proba(mat_test)[:, 1] / 3.0
            importance += model.booster_.feature_importance(importance_type="gain") / 3.0
        score = float(roc_auc_score(y, oof))
        candidates[variant] = {
            "cv_auc": score,
            "fold_auc": fold_scores,
            "features": int(mat.shape[1]),
            "params": hp,
        }
        gains[variant] = sorted(
            [{"feature": name, "gain": round(float(g), 6)} for name, g in zip(mat.columns, importance)],
            key=lambda a: a["gain"], reverse=True
        )
        sub = sample.copy()
        sub[target] = test_pred
        sub.to_csv(OUT / f"candidate_{variant}.csv", index=False)

    selected_variant = max(candidates, key=lambda k: candidates[k]["cv_auc"])
    winning = OUT / f"candidate_{selected_variant}.csv"
    (OUT / "candidate_submission.csv").write_bytes(winning.read_bytes())
    for intermediate in OUT.glob("candidate_tuned_*.csv"):
        intermediate.unlink(missing_ok=True)
    metrics = {
        "target": target,
        "train_rows": len(train),
        "test_rows": len(test),
        "validation": "StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)",
        "seed": SEED,
        "tuning": {"method": "Optuna TPE", "trials_requested": TRIALS, "trials_completed": sum(t.value is not None for t in study.trials), "sample_rows": len(selected)},
        "candidates": candidates,
        "best_candidate": selected_variant,
        "best_cv_auc": candidates[selected_variant]["cv_auc"],
        "validation_metric": "roc_auc",
        "feature_engineering": "V2.1 compatible and extended row-wise variant",
    }
    output("v22_metrics.json", metrics)
    output("v22_feature_importance.json", {
        "method": "LightGBM gain averaged over 3 folds",
        "candidates": {name: rows[:60] for name, rows in gains.items()},
    })
    print(json.dumps({"best_candidate": selected_variant, "cv_auc": metrics["best_cv_auc"]}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        output("v22_error.json", {"error": traceback.format_exc()})
        raise
