from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

COMPETITION = "playground-series-s6e10"
ROOT = Path("/kaggle/input")
OUT = Path("/kaggle/working")


def find(name):
    xs = list(ROOT.rglob(name))
    if not xs:
        raise FileNotFoundError(f"Could not find {name} under {ROOT}")
    return xs[0]

train = pd.read_csv(find("train.csv"))
test = pd.read_csv(find("test.csv"))
sample = pd.read_csv(find("sample_submission.csv"))

target_candidates = [c for c in train.columns if c not in test.columns]
if len(target_candidates) != 1:
    raise RuntimeError(f"V2 baseline expects one target column, got: {target_candidates}")
target = target_candidates[0]
id_cols = [c for c in sample.columns if c in test.columns]
features = [c for c in test.columns if c not in id_cols]

X = train[features].copy()
X_test = test[features].copy()
y_raw = train[target].copy()

cat_cols = list(X.select_dtypes(include=["object", "category", "bool"]).columns)
num_cols = [c for c in features if c not in cat_cols]

transformers = []
if num_cols:
    transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols))
if cat_cols:
    transformers.append(("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]), cat_cols))
pre = ColumnTransformer(transformers=transformers, remainder="drop")

nunique = y_raw.nunique(dropna=True)
is_classification = (not pd.api.types.is_numeric_dtype(y_raw)) or nunique <= max(20, int(math.sqrt(max(1, len(y_raw)))))
metrics = {"target": target, "features": len(features), "categorical_features": len(cat_cols), "rows": len(train)}

if is_classification:
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))
    stratify = y if np.min(np.bincount(y)) >= 2 else None
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    model = HistGradientBoostingClassifier(max_iter=220, learning_rate=0.08, max_leaf_nodes=31, random_state=42)
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_tr, y_tr)
    if len(le.classes_) == 2:
        va_pred = pipe.predict_proba(X_va)[:, 1]
        metrics["validation_metric"] = "roc_auc"
        metrics["validation_score"] = float(roc_auc_score(y_va, va_pred))
    else:
        va_pred = pipe.predict(X_va)
        metrics["validation_metric"] = "accuracy"
        metrics["validation_score"] = float(accuracy_score(y_va, va_pred))
    pipe.fit(X, y)
    if len(le.classes_) == 2:
        pred = pipe.predict_proba(X_test)[:, 1]
    else:
        pred = le.inverse_transform(pipe.predict(X_test))
else:
    y = pd.to_numeric(y_raw)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)
    model = HistGradientBoostingRegressor(max_iter=220, learning_rate=0.08, max_leaf_nodes=31, random_state=42)
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_tr, y_tr)
    va_pred = pipe.predict(X_va)
    metrics["validation_metric"] = "rmse"
    metrics["validation_score"] = float(mean_squared_error(y_va, va_pred) ** 0.5)
    pipe.fit(X, y)
    pred = pipe.predict(X_test)

submission = sample.copy()
submission[target] = pred
submission.to_csv(OUT / "submission.csv", index=False)
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))
print(submission.head())
