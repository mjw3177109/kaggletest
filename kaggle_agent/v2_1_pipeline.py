from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "agent.yaml"


def run(cmd: list[str], *, check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if not env.get("KAGGLE_API_TOKEN"):
        raise RuntimeError("Missing KAGGLE_API_TOKEN")
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    return proc


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def list_json(text: str) -> list[dict[str, Any]]:
    value = json.loads(text.strip() or "[]")
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for k in ("results", "items", "submissions", "data"):
            if isinstance(value.get(k), list):
                return value[k]
        return [value]
    return []


def poll_kernel(handle: str, timeout_seconds: int, poll_seconds: int) -> tuple[str, str]:
    started = time.time()
    last = ""
    while time.time() - started < timeout_seconds:
        p = run(["kaggle", "kernels", "status", handle], check=False)
        last = ((p.stdout or "") + "\n" + (p.stderr or "")).strip()
        low = last.lower()
        if any(x in low for x in ("complete", "success")):
            return "complete", last
        if any(x in low for x in ("error", "failed", "cancel")):
            return "failed", last
        time.sleep(poll_seconds)
    return "timeout", last


def latest_submission(slug: str) -> dict[str, Any] | None:
    p = run(["kaggle", "competitions", "submissions", slug, "--page-size", "20", "--format", "json"], check=False)
    if p.returncode != 0:
        return None
    try:
        rows = list_json(p.stdout)
        return rows[0] if rows else None
    except Exception:
        return None


TRAIN_SCRIPT = r'''from pathlib import Path
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
'''


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    v2 = load_json(ROOT / "state" / "v2_status.json", {}) or {}
    if v2.get("stage") != "complete" or not v2.get("competition"):
        raise RuntimeError("V2 must complete before V2.1 can optimize the selected competition")

    slug = str(v2["competition"])
    project = ROOT / "competitions" / slug
    project.mkdir(parents=True, exist_ok=True)
    old_submission = v2.get("latest_submission") or {}
    old_lb = old_submission.get("publicScore")
    old_cv = (v2.get("metrics") or {}).get("validation_score")
    now = datetime.now(timezone.utc).isoformat()

    username = os.environ.get("KAGGLE_USERNAME") or "mjw3177109"
    kernel_slug = ("kas-v21-" + re.sub(r"[^a-z0-9-]+", "-", slug.lower()))[:48].rstrip("-")
    handle = f"{username}/{kernel_slug}"
    kernel_dir = project / "kaggle_v21"
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "main.py").write_text(TRAIN_SCRIPT, encoding="utf-8")
    metadata = {
        "id": handle,
        "title": f"KAS V2.1 {slug}"[:80],
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": [slug],
        "kernel_sources": [],
        "model_sources": [],
    }
    write_json(kernel_dir / "kernel-metadata.json", metadata)

    status: dict[str, Any] = {
        "generated_at": now,
        "version": "V2.1",
        "competition": slug,
        "kernel": handle,
        "stage": "kernel_push",
        "baseline_cv": old_cv,
        "baseline_public_score": old_lb,
    }
    write_json(ROOT / "state" / "v2_1_status.json", status)

    push = run(["kaggle", "kernels", "push", "-p", str(kernel_dir), "-t", "3600"], check=False, timeout=300)
    (kernel_dir / "push_output.txt").write_text((push.stdout or "") + "\n" + (push.stderr or ""), encoding="utf-8")
    if push.returncode != 0:
        status["stage"] = "kernel_push_failed"
        status["error"] = (push.stderr or push.stdout)[:12000]
        write_json(ROOT / "state" / "v2_1_status.json", status)
        return

    state, state_text = poll_kernel(handle, 3500, 20)
    status["kernel_state"] = state
    (kernel_dir / "status_output.txt").write_text(state_text, encoding="utf-8")
    if state != "complete":
        logs = run(["kaggle", "kernels", "logs", handle], check=False)
        (kernel_dir / "logs.txt").write_text((logs.stdout or "") + "\n" + (logs.stderr or ""), encoding="utf-8")
        status["stage"] = f"kernel_{state}"
        write_json(ROOT / "state" / "v2_1_status.json", status)
        return

    raw = project / "results_v21" / "raw"
    if raw.exists():
        shutil.rmtree(raw)
    raw.mkdir(parents=True, exist_ok=True)
    out = run(["kaggle", "kernels", "output", handle, "-p", str(raw), "-o", "-q"], check=False, timeout=600)
    if out.returncode != 0:
        status["stage"] = "kernel_output_failed"
        status["error"] = (out.stderr or out.stdout)[:12000]
        write_json(ROOT / "state" / "v2_1_status.json", status)
        return

    metric_paths = list(raw.rglob("v21_metrics.json"))
    best_paths = list(raw.rglob("best_submission.csv"))
    if not metric_paths or not best_paths:
        status["stage"] = "kernel_artifacts_missing"
        status["error"] = "Kernel completed but v21_metrics.json or best_submission.csv is missing"
        write_json(ROOT / "state" / "v2_1_status.json", status)
        return

    metrics = json.loads(metric_paths[0].read_text(encoding="utf-8"))
    results = project / "results_v21"
    write_json(results / "metrics.json", metrics)
    status["metrics"] = metrics
    status["stage"] = "submitting_best_cv"
    write_json(ROOT / "state" / "v2_1_status.json", status)

    best_name = metrics["best_model"]
    message = f"KAS V2.1 {best_name} CV={metrics['best_cv_auc']:.6f}"
    submit = run([
        "kaggle", "competitions", "submit", slug,
        "-f", str(best_paths[0]), "-m", message,
        "--wait", "600", "--poll-interval", "30",
    ], check=False, timeout=900)
    (results / "submit_output.txt").write_text((submit.stdout or "") + "\n" + (submit.stderr or ""), encoding="utf-8")
    status["submit_returncode"] = submit.returncode

    latest = latest_submission(slug)
    status["latest_submission"] = latest
    new_lb = latest.get("publicScore") if latest else None
    try:
        status["public_score_delta"] = float(new_lb) - float(old_lb) if new_lb not in (None, "") and old_lb not in (None, "") else None
    except Exception:
        status["public_score_delta"] = None
    status["stage"] = "complete"
    write_json(ROOT / "state" / "v2_1_status.json", status)
    if latest:
        write_json(results / "latest_submission.json", latest)

    exp_dir = project / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    order = [("002", "catboost"), ("003", "lightgbm"), ("004", "xgboost"), ("005", "ensemble")]
    for run_id, name in order:
        if name not in metrics["models"]:
            continue
        d = metrics["models"][name]
        submitted = name == best_name
        lb = new_lb if submitted else None
        body = [
            f"# Run {run_id} — {name}", "",
            f"- CV ROC-AUC: `{d['cv_auc']}`",
            f"- Kaggle Public LB: `{lb if lb is not None else 'not submitted'}`",
            f"- Submitted: `{submitted}`",
            f"- Previous best LB: `{old_lb}`",
            "",
            "## Change", "",
            "V2.1 boosted-tree comparison with 3-fold OOF validation and automatic tabular feature engineering.", "",
        ]
        if "fold_auc" in d:
            body += [f"- Fold AUC: `{d['fold_auc']}`", ""]
        (exp_dir / f"run_{run_id}_{name}.md").write_text("\n".join(body), encoding="utf-8")

    ranking = metrics.get("ranking", [])
    report = [
        "# Kaggle Auto Scientist V2.1 — Leaderboard Optimization", "",
        f"Generated: `{now}`", "",
        f"Competition: **{slug}**", "",
        f"Baseline CV: `{old_cv}`", "",
        f"Baseline Public LB: `{old_lb}`", "",
        "## Model comparison", "",
        "| Rank | Model | 3-Fold OOF ROC-AUC |", "|---:|---|---:|",
    ]
    for i, row in enumerate(ranking, 1):
        report.append(f"| {i} | {row['model']} | {row['cv_auc']:.8f} |")
    report += [
        "", f"Selected for submission: **{best_name}**", "",
        f"New Public LB: `{new_lb}`", "",
        f"LB delta vs V2 baseline: `{status.get('public_score_delta')}`", "",
        "## Decision", "",
    ]
    delta = status.get("public_score_delta")
    if isinstance(delta, (int, float)) and delta > 0:
        report += ["V2.1 improved the leaderboard score. Promote this experiment as the new online baseline.", ""]
    else:
        report += ["V2.1 did not improve the public leaderboard. Keep the V2 baseline as the online champion and use V2.1 results to guide the next feature/model search.", ""]
    (ROOT / "reports" / "v2_1_latest.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
