"""V2.9.28 preregistered CPU LightGBM with fold-local service PCA3/contrasts8 and TE23.
Full 3-fold OOF only; zero submission. Outer validation labels NEVER enter preprocessing.
This Notebook is distinct from route-profile and travel-gated experiments.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.decomposition import PCA

DATA = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working")
SEED = 20261018
MODEL_SEED = 20261028
FOLDS = 3
INNER_FOLDS = 5
N_TRAIN = 699635
N_TEST = 299844
MAX_RUNTIME_SECONDS = 3300
CATS = ("Gender", "Customer Type", "Type of Travel", "Class")
SERVICES = (
    "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink",
    "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
)
MODEL = "V2928 fold-local service PCA3 contrasts8 global LightGBM TE23 3fold"
FEATURE_FAMILY = "service_factor_pca3_contrasts8_te23_no_route_profile"
PCA_COMPONENTS = 3
CONTRAST_COUNT = 8
SERVICE_COUNT = 13
TE_COUNT = 23
MODEL_PARAMETERS = {
    "n_estimators":1200, "learning_rate":0.035, "max_depth":8,
    "num_leaves":63, "min_child_samples":60, "subsample":0.9,
    "colsample_bytree":0.85, "reg_lambda":3.0, "reg_alpha":0.0,
    "random_state_base":MODEL_SEED, "n_jobs":4,
}

def dump(name, value):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def competition_csv(name):
    found = [
        path for path in DATA.rglob(name)
        if "playground-series-s6e10" in str(path.parent)
    ]
    if len(found) != 1:
        raise RuntimeError("Expected unique competition " + name + ": " + str(len(found)))
    return pd.read_csv(found[0])


def feature_names(fit):
    return [column for column in fit.columns if column not in ("id", "satisfaction")]


def target_keys(frame, names):
    """Only original input fields; target and row id cannot enter any encoding key."""
    z = pd.DataFrame(index=frame.index)
    for name in names:
        z["raw_" + name] = frame[name].astype("string").fillna("__NA__").astype(str)
    part = [frame[c].astype("string").fillna("__NA__").astype(str) for c in (
        "Class", "Type of Travel", "Customer Type",
    )]
    distance = frame["Flight Distance"].astype("string").fillna("__NA__").astype(str)
    z["cross_traveler"] = part[0] + "|" + part[1] + "|" + part[2]
    z["cross_exact_route_class"] = distance + "|" + part[0]
    if len(z.columns) != TE_COUNT or z.isna().any().any():
        raise RuntimeError("Unexpected 23 leakage-safe TE columns")
    return z


def service_factor_features(fit, holdout, test):
    """Exactly 11 label-free columns, fit medians/scaler/PCA on outer fit ONLY."""
    if len(SERVICES) != SERVICE_COUNT:
        raise ValueError("Service columns preregistration mismatch")
    medians = fit[list(SERVICES)].apply(pd.to_numeric, errors="coerce").median()
    if medians.isna().any():
        raise ValueError("All-missing training-fold service column")
    arrays = []
    for frame in (fit, holdout, test):
        if not set(SERVICES).issubset(frame.columns):
            raise ValueError("Missing canonical service rating column")
        ratings = frame[list(SERVICES)].apply(pd.to_numeric, errors="coerce")
        # The only imputation values come from outer-training rows.
        arrays.append(ratings.fillna(medians).to_numpy(dtype=np.float64))
    scaler = StandardScaler()
    xfit = scaler.fit_transform(arrays[0])
    pca = PCA(n_components=PCA_COMPONENTS, random_state=MODEL_SEED)
    pca.fit(xfit)
    output = []
    digital = [SERVICES.index(name) for name in
               ("Inflight wifi service", "Ease of Online booking", "Online boarding")]
    comfort = [SERVICES.index(name) for name in
               ("Seat comfort", "Leg room service", "Cleanliness")]
    for frame, values in zip((fit, holdout, test), arrays):
        components = pca.transform(scaler.transform(values))
        contrasts = np.column_stack((
            values.mean(axis=1),
            values.std(axis=1),
            values.min(axis=1),
            values.max(axis=1),
            (values == 0).sum(axis=1),
            (values >= 4).sum(axis=1),
            (values <= 2).sum(axis=1),
            values[:, digital].mean(axis=1) - values[:, comfort].mean(axis=1),
        ))
        matrix = np.column_stack((components, contrasts))
        names = [f"service_pca_{i+1}" for i in range(PCA_COMPONENTS)] + [
            "service_mean", "service_std", "service_min", "service_max",
            "service_zero_count", "service_ge4_count", "service_le2_count",
            "service_digital_minus_comfort",
        ]
        if matrix.shape != (len(frame), PCA_COMPONENTS+CONTRAST_COUNT):
            raise ValueError("V2928 fixed service-factor feature shape mismatch")
        if not np.isfinite(matrix).all():
            raise ValueError("V2928 non-finite service-factor values")
        output.append(pd.DataFrame(matrix.astype(np.float32),
                                   index=frame.index, columns=names))
    if not (output[0].columns.equals(output[1].columns)
            and output[1].columns.equals(output[2].columns)):
        raise ValueError("Service-factor schema drift")
    return output

def basic_features(fit, holdout, test, names):
    result = [pd.DataFrame(index=part.index) for part in (fit, holdout, test)]
    for col in names:
        if col in CATS:
            vocabulary = pd.Index(fit[col].astype("string").fillna("__NA__").astype(str).unique())
            for out, frame in zip(result, (fit, holdout, test)):
                vals = frame[col].astype("string").fillna("__NA__").astype(str)
                out[col] = pd.Categorical(vals, categories=vocabulary).codes.astype("float32")
        else:
            fit_vals = pd.to_numeric(fit[col], errors="coerce")
            median = float(fit_vals.median())
            if not np.isfinite(median):
                raise RuntimeError("All-missing numeric field: " + col)
            for out, frame in zip(result, (fit, holdout, test)):
                out[col] = pd.to_numeric(frame[col], errors="coerce").fillna(median).astype("float32")
    return result


def matrices(fit, holdout, test, labels, fold):
    names = feature_names(fit)
    if len(names) != 21 or not set(CATS + SERVICES + ("Flight Distance",)).issubset(names):
        raise ValueError("Official 21-feature schema not verified")
    key_train, key_val, key_test = (target_keys(frame, names) for frame in (fit, holdout, test))
    encoder = TargetEncoder(target_type="binary", smooth="auto", cv=INNER_FOLDS,
                            shuffle=True, random_state=SEED + fold)
    train_te = encoder.fit_transform(key_train, np.asarray(labels, dtype=np.int8))
    val_te = encoder.transform(key_val)
    test_te = encoder.transform(key_test)
    factors = service_factor_features(fit, holdout, test)
    basic = basic_features(fit, holdout, test, names)
    outputs = []
    for i, frame in enumerate((fit, holdout, test)):
        encoded = pd.DataFrame(
            (train_te, val_te, test_te)[i].astype("float32"),
            index=frame.index,
            columns=["te_" + str(j) for j in range(TE_COUNT)],
        )
        values = pd.concat((basic[i], encoded, factors[i]), axis=1)
        if values.columns.duplicated().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError("Invalid or leaked fold-local feature matrix")
        outputs.append(values.astype("float32"))
    if not (outputs[0].columns.equals(outputs[1].columns)
            and outputs[0].columns.equals(outputs[2].columns)):
        raise ValueError("Feature matrix column mismatch")
    return outputs


def main():
    import lightgbm as lgb
    started = time.monotonic()
    train, test = competition_csv("train.csv"), competition_csv("test.csv")
    sample = competition_csv("sample_submission.csv")
    if len(train) != N_TRAIN or len(test) != N_TEST or len(sample) != N_TEST:
        raise RuntimeError("Unexpected official competition row counts")
    if "satisfaction" not in train or "satisfaction" in test:
        raise RuntimeError("Target schema mismatch")
    if list(sample.columns) != ["id", "satisfaction"] or not np.array_equal(
        sample["id"].to_numpy(), test["id"].to_numpy()
    ):
        raise RuntimeError("Official Kaggle sample ID order mismatch")
    y = pd.to_numeric(train["satisfaction"], errors="raise").to_numpy(dtype=np.int8)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Unexpected nonbinary target")
    oof = np.full(len(train), np.nan, dtype=np.float32)
    pred_test = np.zeros(len(test), dtype=np.float64)
    fold_results = []
    for fold, (itr, iva) in enumerate(StratifiedKFold(
        n_splits=FOLDS, shuffle=True, random_state=SEED
    ).split(np.zeros(len(y), dtype=np.uint8), y)):
        if time.monotonic() - started > MAX_RUNTIME_SECONDS:
            raise TimeoutError("Full threefold OOF time budget exceeded; no partial promotion")
        xtr, xval, xtest = matrices(train.iloc[itr], train.iloc[iva], test, y[itr], fold)
        model = lgb.LGBMClassifier(
            objective="binary", metric="auc", n_estimators=1200,
            learning_rate=0.035, max_depth=8, num_leaves=63,
            min_child_samples=60, subsample=0.9, colsample_bytree=0.85,
            reg_lambda=3.0, reg_alpha=0.0, random_state=MODEL_SEED+fold,
            n_jobs=4, verbosity=-1,
        )
        # No validation labels, sample weighting, or early stopping in fitting.
        model.fit(xtr, y[itr])
        pv = np.asarray(model.predict_proba(xval)[:, 1], dtype=np.float32)
        pt = np.asarray(model.predict_proba(xtest)[:, 1], dtype=np.float32)
        if not (np.isfinite(pv).all() and np.isfinite(pt).all()
                and (pv >= 0).all() and (pv <= 1).all()
                and (pt >= 0).all() and (pt <= 1).all()):
            raise RuntimeError("Invalid LightGBM OOF/test probabilities")
        oof[iva] = pv
        pred_test += pt / FOLDS
        fold_results.append({
            "fold":fold,"validation_rows":len(iva),
            "auc":float(roc_auc_score(y[iva], pv)),
            "features":len(xtr.columns),
            "train_rows":len(itr),"model_seed":MODEL_SEED+fold,
        })
        dump("v2928_progress.json", {
            "folds_finished":fold+1, "fold_results":fold_results,
            "complete":False,
        })
        print("V2928 independent service-factor LightGBM fold", fold, fold_results[-1], flush=True)
        del model, xtr, xval, xtest
    if len(fold_results)!=3 or any(item["features"]!=55 for item in fold_results):
        raise ValueError("V2928 all three folds/55 features not verified")
    if not np.isfinite(oof).all() or not np.isfinite(pred_test).all():
        raise RuntimeError("Incomplete threefold OOF/test")
    if (pred_test < 0).any() or (pred_test > 1).any():
        raise RuntimeError("Out-of-range test prediction")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT/"v2928_service_factor_lgbm_oof.npz",
        target=y,
        train_id=train["id"].to_numpy(),
        oof_service_factor_lgbm=oof,
        test_id=test["id"].to_numpy(),
        pred_service_factor_lgbm=pred_test.astype(np.float32),
    )
    dump("v2928_metrics.json", {
        "competition":"playground-series-s6e10",
        "model":MODEL,"stage":"three_folds_complete",
        "train_rows":len(train),"test_rows":len(test),
        "validation":"StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)",
        "folds":FOLDS,"fold_results":fold_results,
        "full_oof_auc":float(roc_auc_score(y, oof)),
        "model_seed":MODEL_SEED,
        "feature_family":FEATURE_FAMILY,
        "model_parameters":MODEL_PARAMETERS,
        "pca_components":PCA_COMPONENTS, "service_contrast_features":CONTRAST_COUNT,
        "service_rating_columns":SERVICE_COUNT, "route_feature_count":0,
        "feature_count":55,
        "pseudo_labels":False,
        "target_encoding_columns":TE_COUNT,"inner_target_folds":INNER_FOLDS,
        "cpu_only":True,"gpu_verified":False,"data_source":"competition_train_only",
        "hardware":{"device":"cpu","cpu_verified":True},
        "external_labels":False,"leakage_safe_inner_cross_fit":True,
        "outer_validation_used_in_training":False,
        "runtime_seconds":round(time.monotonic()-started, 1),
        "submitted":False,
    })
    print("V2928 genuine independent 3-fold OOF AUC", roc_auc_score(y, oof), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        dump("v2928_error.json", {"traceback":traceback.format_exc()[-8000:]})
        raise
