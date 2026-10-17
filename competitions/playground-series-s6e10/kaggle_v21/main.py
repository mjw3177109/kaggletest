from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

INPUT_ROOT = Path("/kaggle/input")
OUT = Path("/kaggle/working")
N_SPLITS = 3
SEED = 20261018


def find(name):
    xs = list(INPUT_ROOT.rglob(name))
    if not xs:
        raise FileNotFoundError(f"Could not find {name} anywhere under {INPUT_ROOT}; top={list(INPUT_ROOT.iterdir())[:20]}")
    return xs[0]


def prepare_features(train, test, target, sample):
    id_cols = [c for c in sample.columns if c in test.columns]
    features = [c for c in test.columns if c not in id_cols]
    X = train[features].copy()
    Xt = test[features].copy()

    # Small automatic feature-engineering layer for common tabular patterns.
    delay_cols = [c for c in features if "delay" in c.lower() and pd.api.types.is_numeric_dtype(X[c])]
    if len(delay_cols) >= 2:
        X["__auto_total_delay"] = X[delay_cols].sum(axis=1, skipna=True)
        Xt["__auto_total_delay"] = Xt[delay_cols].sum(axis=1, skipna=True)
        X["__auto_max_delay"] = X[delay_cols].max(axis=1, skipna=True)
        Xt["__auto_max_delay"] = Xt[delay_cols].max(axis=1, skipna=True)

    rating_tokens = ("service", "comfort", "cleanliness", "booking", "boarding", "location", "entertainment")
    rating_cols = [c for c in features if any(t in c.lower() for t in rating_tokens) and pd.api.types.is_numeric_dtype(X[c])]
    if len(rating_cols) >= 3:
        X["__auto_rating_mean"] = X[rating_cols].mean(axis=1)
        Xt["__auto_rating_mean"] = Xt[rating_cols].mean(axis=1)
        X["__auto_rating_min"] = X[rating_cols].min(axis=1)
        Xt["__auto_rating_min"] = Xt[rating_cols].min(axis=1)
        X["__auto_rating_max"] = X[rating_cols].max(axis=1)
        Xt["__auto_rating_max"] = Xt[rating_cols].max(axis=1)

    combined = pd.concat([X, Xt], axis=0, ignore_index=True)
    cat_cols = list(combined.select_dtypes(include=["object", "category", "bool"]).columns)
    for c in combined.columns:
        if c not in cat_cols and combined[c].isna().any():
            combined[c] = combined[c].fillna(combined[c].median())
    combined = pd.get_dummies(combined, columns=cat_cols, dummy_na=True, dtype=np.uint8)
    X2 = combined.iloc[:len(X)].reset_index(drop=True)
    Xt2 = combined.iloc[len(X):].reset_index(drop=True)
    return X2, Xt2, id_cols


def encode_target(y):
    if y.dtype == bool:
        return y.astype(int).to_numpy(), {"false": 0, "true": 1}
    if pd.api.types.is_numeric_dtype(y) and y.nunique() == 2:
        vals = sorted(y.dropna().unique().tolist())
        mp = {str(vals[0]): 0, str(vals[1]): 1}
        return y.map({vals[0]: 0, vals[1]: 1}).astype(int).to_numpy(), mp
    vals = sorted(y.astype(str).unique().tolist())
    if len(vals) != 2:
        raise RuntimeError(f"V2.1 currently optimizes binary classification; classes={vals[:20]}")
    mp0 = {vals[0]: 0, vals[1]: 1}
    return y.astype(str).map(mp0).astype(int).to_numpy(), mp0


def make_models():
    models = {}
    errors = {}
    try:
        from catboost import CatBoostClassifier
        models["catboost"] = lambda: CatBoostClassifier(
            iterations=420, depth=7, learning_rate=0.055, loss_function="Logloss",
            eval_metric="AUC", random_seed=SEED, verbose=False, thread_count=-1,
            l2_leaf_reg=4.0, random_strength=0.4
        )
    except Exception as e:
        errors["catboost"] = repr(e)
    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = lambda: LGBMClassifier(
            n_estimators=480, learning_rate=0.035, num_leaves=31, max_depth=-1,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=SEED, n_jobs=-1, verbosity=-1
        )
    except Exception as e:
        errors["lightgbm"] = repr(e)
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = lambda: XGBClassifier(
            n_estimators=420, learning_rate=0.04, max_depth=6, min_child_weight=2,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="auc", tree_method="hist",
            random_state=SEED, n_jobs=-1
        )
    except Exception as e:
        errors["xgboost"] = repr(e)
    return models, errors

train = pd.read_csv(find("train.csv"))
test = pd.read_csv(find("test.csv"))
sample = pd.read_csv(find("sample_submission.csv"))
target_candidates = [c for c in train.columns if c not in test.columns]
if len(target_candidates) != 1:
    raise RuntimeError(f"Expected exactly one train-only target; got {target_candidates}")
target = target_candidates[0]
y, target_mapping = encode_target(train[target])
X, Xtest, id_cols = prepare_features(train, test, target, sample)

models, import_errors = make_models()
if len(models) < 2:
    raise RuntimeError(f"Need at least two boosted-tree libraries in Kaggle image; available={list(models)}, errors={import_errors}")

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
metrics = {
    "target": target,
    "rows": int(len(train)),
    "features_after_encoding": int(X.shape[1]),
    "folds": N_SPLITS,
    "target_mapping": target_mapping,
    "import_errors": import_errors,
    "models": {},
}
all_oof = {}
all_test = {}

for name, factory in models.items():
    oof = np.zeros(len(X), dtype=float)
    test_pred = np.zeros(len(Xtest), dtype=float)
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
        model = factory()
        model.fit(X.iloc[tr_idx], y[tr_idx])
        vp = model.predict_proba(X.iloc[va_idx])[:, 1]
        tp = model.predict_proba(Xtest)[:, 1]
        oof[va_idx] = vp
        test_pred += tp / N_SPLITS
        fold_scores.append(float(roc_auc_score(y[va_idx], vp)))
    score = float(roc_auc_score(y, oof))
    metrics["models"][name] = {"cv_auc": score, "fold_auc": fold_scores}
    all_oof[name] = oof
    all_test[name] = test_pred
    sub = sample.copy()
    sub[target] = test_pred
    sub.to_csv(OUT / f"submission_{name}.csv", index=False)

names = list(all_oof)
ensemble_oof = np.mean(np.column_stack([all_oof[n] for n in names]), axis=1)
ensemble_test = np.mean(np.column_stack([all_test[n] for n in names]), axis=1)
ensemble_score = float(roc_auc_score(y, ensemble_oof))
metrics["models"]["ensemble"] = {"cv_auc": ensemble_score, "members": names, "weights": [1 / len(names)] * len(names)}
sub = sample.copy()
sub[target] = ensemble_test
sub.to_csv(OUT / "submission_ensemble.csv", index=False)

best_name = max(metrics["models"], key=lambda n: metrics["models"][n]["cv_auc"])
metrics["best_model"] = best_name
metrics["best_cv_auc"] = metrics["models"][best_name]["cv_auc"]
metrics["ranking"] = sorted(
    [{"model": n, "cv_auc": d["cv_auc"]} for n, d in metrics["models"].items()],
    key=lambda x: x["cv_auc"], reverse=True
)

src = OUT / f"submission_{best_name}.csv"
shutil = __import__("shutil")
shutil.copy2(src, OUT / "best_submission.csv")
(OUT / "v21_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))
