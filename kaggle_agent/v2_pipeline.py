from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "agent.yaml"


def run(cmd: list[str], *, check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if not env.get("KAGGLE_API_TOKEN"):
        raise RuntimeError("Missing KAGGLE_API_TOKEN")
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def load_json_output(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if not text:
        return []
    value = json.loads(text)
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("results", "competitions", "items", "data"):
            if isinstance(value.get(key), list):
                return value[key]
        return [value]
    return []


def slug_from_ref(ref: str) -> str:
    ref = str(ref).rstrip("/")
    return ref.split("/")[-1]


def parse_deadline(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    for candidate in (text, text.replace(" ", "T", 1)):
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass
    return None


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def competition_score(row: dict[str, Any], now: datetime) -> float:
    slug = slug_from_ref(row.get("ref", "")).lower()
    category = str(row.get("category", "")).lower()
    teams = int(row.get("teamCount") or row.get("team_count") or 0)
    deadline = parse_deadline(row.get("deadline"))

    score = 0.0
    if category == "playground":
        score += 120
    elif category.lower() == "gettingstarted":
        score += 70
    if "playground-series" in slug:
        score += 120
    if any(x in slug for x in ("tabular", "classification", "regression", "predict")):
        score += 35
    if teams and teams <= 5000:
        score += 15
    if deadline:
        days = max(0.0, (deadline - now).total_seconds() / 86400)
        score += max(0.0, 30.0 - days / 2.0)

    hard = ("arc-prize", "llm", "image", "audio", "video", "segmentation", "simulation", "protein", "medical")
    if any(x in slug for x in hard):
        score -= 80
    return round(score, 2)


def list_candidates(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
    for category in cfg["v2"]["categories"]:
        proc = run([
            "kaggle", "competitions", "list",
            "--group", "general",
            "--category", str(category),
            "--sort-by", "latestDeadline",
            "--page-size", str(cfg["v2"].get("page_size", 100)),
            "--format", "json",
        ])
        rows.extend(load_json_output(proc.stdout))

    dedup: dict[str, dict[str, Any]] = {}
    min_days = float(cfg["v2"].get("min_days_remaining", 2))
    for row in rows:
        ref = row.get("ref") or row.get("url") or ""
        slug = slug_from_ref(ref)
        if not slug:
            continue
        deadline = parse_deadline(row.get("deadline"))
        if not deadline or (deadline - now).total_seconds() < min_days * 86400:
            continue
        normalized = dict(row)
        normalized["ref"] = ref
        normalized["slug"] = slug
        normalized["deadline"] = deadline.isoformat()
        normalized["score"] = competition_score(normalized, now)
        dedup[slug] = normalized

    return sorted(dedup.values(), key=lambda x: (x.get("score", 0), x.get("deadline", "")), reverse=True)


def safe_page(slug: str, page_name: str) -> str:
    proc = run([
        "kaggle", "competitions", "pages", slug,
        "--page-name", page_name,
        "--content",
    ], check=False)
    text = (proc.stdout or proc.stderr or "").strip()
    return text[:50000]


def safe_files(slug: str) -> tuple[list[dict[str, Any]], str]:
    proc = run([
        "kaggle", "competitions", "files", slug,
        "--page-size", "200",
        "--format", "json",
    ], check=False)
    if proc.returncode != 0:
        return [], (proc.stderr or proc.stdout).strip()[:8000]
    try:
        return load_json_output(proc.stdout), ""
    except Exception as exc:
        return [], f"Could not parse files output: {exc}\n{proc.stdout[:4000]}"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def excerpt(text: str, limit: int = 3500) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    return text[:limit]


def write_problem_docs(project: Path, selected: dict[str, Any], pages: dict[str, str], files: list[dict[str, Any]]) -> None:
    docs = project / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    for name, content in pages.items():
        (docs / f"{name}.txt").write_text(content, encoding="utf-8")
    write_json(docs / "files.json", files)

    problem = [
        f"# {selected['slug']} — Problem Analysis",
        "",
        f"- Kaggle: {selected.get('ref')}",
        f"- Category: {selected.get('category')}",
        f"- Deadline (UTC): {selected.get('deadline')}",
        f"- Teams: {selected.get('teamCount', selected.get('team_count', 0))}",
        f"- Auto-selection score: {selected.get('score')}",
        "",
        "## Competition description excerpt",
        "",
        excerpt(pages.get("description", "No description page returned.")),
        "",
        "## Evaluation excerpt",
        "",
        excerpt(pages.get("evaluation", "No evaluation page returned.")),
        "",
        "## V2 execution decision",
        "",
        "V2 prioritizes currently-open Playground / Getting Started competitions that are suitable for a reproducible automated baseline. It never accepts competition rules on the user's behalf.",
        "",
    ]
    (project / "problem_analysis.md").write_text("\n".join(problem), encoding="utf-8")

    plan = f"""# Solution Plan — {selected['slug']}

## Run 001: reproducible tabular baseline

1. Identify `train.csv`, `test.csv`, and `sample_submission.csv` from the Kaggle competition source.
2. Infer the target as the column present in train but absent from test.
3. Exclude submission ID columns from model features.
4. Impute numeric and categorical missing values independently.
5. Ordinal-encode categorical columns with unknown-category handling.
6. Use HistGradientBoosting for a dependency-light baseline that is available in the Kaggle sklearn image.
7. Hold out validation data and record ROC-AUC for binary classification, accuracy for multiclass, or RMSE for regression.
8. Refit on full training data and write `submission.csv`.
9. V2 pushes the script to Kaggle, waits for completion, downloads output, and submits automatically when rules have already been accepted.

## Next iterations after V2 baseline

- CatBoost with native categorical handling.
- LightGBM / XGBoost comparison.
- Cross-validation and out-of-fold predictions.
- Feature engineering guided by leakage-safe validation.
- Seed/model ensembling only when CV and leaderboard movement agree.
"""
    (project / "solution_plan.md").write_text(plan, encoding="utf-8")


def locate(data_dir: Path, filename: str) -> Path | None:
    matches = list(data_dir.rglob(filename))
    return matches[0] if matches else None


def profile_csv(path: Path | None, max_rows: int) -> dict[str, Any] | None:
    if not path:
        return None
    df = pd.read_csv(path, nrows=max_rows)
    return {
        "path": str(path.relative_to(ROOT)),
        "sampled_rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(map(str, df.columns)),
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "missing_top": {str(k): int(v) for k, v in df.isna().sum().sort_values(ascending=False).head(20).items()},
        "file_bytes": path.stat().st_size,
    }


def build_data_analysis(project: Path, train_path: Path, test_path: Path, sample_path: Path, max_rows: int) -> dict[str, Any]:
    train = pd.read_csv(train_path, nrows=max_rows)
    test = pd.read_csv(test_path, nrows=max_rows)
    sample = pd.read_csv(sample_path, nrows=min(max_rows, 1000))
    target_candidates = [c for c in train.columns if c not in test.columns]
    id_candidates = [c for c in sample.columns if c in test.columns]
    target = target_candidates[0] if len(target_candidates) == 1 else None

    profile = {
        "train": profile_csv(train_path, max_rows),
        "test": profile_csv(test_path, max_rows),
        "sample_submission": profile_csv(sample_path, min(max_rows, 1000)),
        "target_candidates": target_candidates,
        "target": target,
        "id_candidates": id_candidates,
    }
    if target:
        series = train[target]
        profile["target_dtype"] = str(series.dtype)
        profile["target_nunique_sample"] = int(series.nunique(dropna=False))
        profile["target_value_counts_sample"] = {str(k): int(v) for k, v in series.value_counts(dropna=False).head(20).items()}

    write_json(project / "data_profile.json", profile)
    lines = [
        f"# Data Analysis — {project.name}", "",
        f"Profile sample cap: **{max_rows:,} rows per CSV**", "",
        f"- Train sample shape: `{train.shape}`",
        f"- Test sample shape: `{test.shape}`",
        f"- Target candidates: `{target_candidates}`",
        f"- Selected target: `{target}`",
        f"- Submission/ID candidates: `{id_candidates}`",
        "",
        "## Missing values (train sample)", "",
        "| Column | Missing |", "|---|---:|",
    ]
    for col, count in train.isna().sum().sort_values(ascending=False).head(20).items():
        lines.append(f"| {md_escape(str(col))} | {int(count)} |")
    lines += ["", "## Dtypes", "", "| Column | Dtype |", "|---|---|"]
    for col, dtype in train.dtypes.items():
        lines.append(f"| {md_escape(str(col))} | {dtype} |")
    (project / "data_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return profile


BASELINE = r'''from pathlib import Path
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

COMPETITION = "__COMPETITION__"
ROOT = Path("/kaggle/input") / COMPETITION
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
'''


def build_kernel(project: Path, slug: str) -> tuple[Path, str]:
    username = os.environ.get("KAGGLE_USERNAME") or os.environ.get("GITHUB_REPOSITORY_OWNER") or "mjw3177109"
    base = re.sub(r"[^a-z0-9-]+", "-", slug.lower()).strip("-")
    kernel_slug = ("kas-v2-" + base)[:48].rstrip("-")
    handle = f"{username}/{kernel_slug}"
    kernel_dir = project / "kaggle"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "main.py").write_text(BASELINE.replace("__COMPETITION__", slug), encoding="utf-8")
    metadata = {
        "id": handle,
        "title": f"KAS V2 {slug}"[:80],
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
    return kernel_dir, handle


def poll_kernel(handle: str, timeout_seconds: int, poll_seconds: int) -> tuple[str, str]:
    started = time.time()
    last = ""
    while time.time() - started < timeout_seconds:
        proc = run(["kaggle", "kernels", "status", handle], check=False)
        last = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        low = last.lower()
        if any(x in low for x in ("complete", "success")):
            return "complete", last
        if any(x in low for x in ("error", "failed", "cancel")):
            return "failed", last
        time.sleep(poll_seconds)
    return "timeout", last


def latest_submission(slug: str) -> dict[str, Any] | None:
    proc = run(["kaggle", "competitions", "submissions", slug, "--page-size", "10", "--format", "json"], check=False)
    if proc.returncode != 0:
        return None
    try:
        rows = load_json_output(proc.stdout)
        return rows[0] if rows else None
    except Exception:
        return None


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    candidates = list_candidates(cfg)
    if not candidates:
        raise RuntimeError("No currently-open Playground/Getting Started competition matched V2 filters")

    selected = candidates[0]
    slug = selected["slug"]
    project = ROOT / "competitions" / slug
    project.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    pages = {
        "description": safe_page(slug, "description"),
        "evaluation": safe_page(slug, "evaluation"),
        "data-description": safe_page(slug, "data-description"),
    }
    files, files_error = safe_files(slug)
    write_problem_docs(project, selected, pages, files)
    write_json(ROOT / "state" / "v2_candidates.json", candidates[:20])
    write_json(ROOT / "state" / "v2_selection.json", selected)

    status: dict[str, Any] = {
        "generated_at": now,
        "version": "V2",
        "competition": slug,
        "competition_url": selected.get("ref"),
        "entered": as_bool(selected.get("userHasEntered", selected.get("user_has_entered", False))),
        "stage": "selected",
        "files_error": files_error,
    }

    report = [
        "# Kaggle Auto Scientist V2 — Latest Run", "",
        f"Generated: `{now}`", "",
        f"Selected competition: **{slug}**", "",
        f"Kaggle: {selected.get('ref')}", "",
        f"Deadline: `{selected.get('deadline')}`", "",
    ]

    if not status["entered"]:
        status["stage"] = "rules_required"
        status["next_action"] = "Open the competition URL, click Join Competition, and accept the rules; then rerun Kaggle Auto Scientist V2."
        report += [
            "## Status: RULES REQUIRED", "",
            "The competition was selected and its public pages were captured, but your Kaggle account is not marked as entered.", "",
            "V2 deliberately does **not** accept legal/competition rules for you. Open the competition link, click **Join Competition**, accept its rules, then rerun the workflow. After that, V2 continues with data download → Kaggle training → submission → score write-back automatically.", "",
        ]
        write_json(ROOT / cfg["output"]["v2_status"], status)
        (ROOT / cfg["output"]["v2_report"]).write_text("\n".join(report), encoding="utf-8")
        print(json.dumps(status, indent=2))
        return

    data_dir = project / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dl = run(["kaggle", "competitions", "download", slug, "-p", str(data_dir), "--unzip", "-q"], check=False, timeout=1200)
    if dl.returncode != 0:
        status["stage"] = "download_failed"
        status["error"] = (dl.stderr or dl.stdout).strip()[:10000]
        write_json(ROOT / cfg["output"]["v2_status"], status)
        report += ["## Status: DOWNLOAD FAILED", "", "```text", status["error"], "```", ""]
        (ROOT / cfg["output"]["v2_report"]).write_text("\n".join(report), encoding="utf-8")
        return

    train_path = locate(data_dir, "train.csv")
    test_path = locate(data_dir, "test.csv")
    sample_path = locate(data_dir, "sample_submission.csv")
    if not all((train_path, test_path, sample_path)):
        status["stage"] = "unsupported_dataset_layout"
        status["error"] = "V2 baseline currently requires train.csv, test.csv, and sample_submission.csv."
        write_json(ROOT / cfg["output"]["v2_status"], status)
        report += ["## Status: DATA LAYOUT NEEDS V3 ADAPTER", "", status["error"], ""]
        (ROOT / cfg["output"]["v2_report"]).write_text("\n".join(report), encoding="utf-8")
        return

    profile = build_data_analysis(project, train_path, test_path, sample_path, int(cfg["v2"].get("profile_rows", 50000)))
    if not profile.get("target"):
        status["stage"] = "target_inference_failed"
        status["error"] = f"Expected exactly one train-only target column; got {profile.get('target_candidates')}"
        write_json(ROOT / cfg["output"]["v2_status"], status)
        report += ["## Status: TARGET INFERENCE FAILED", "", status["error"], ""]
        (ROOT / cfg["output"]["v2_report"]).write_text("\n".join(report), encoding="utf-8")
        return

    kernel_dir, handle = build_kernel(project, slug)
    status["kernel"] = handle
    status["stage"] = "kernel_push"
    push = run(["kaggle", "kernels", "push", "-p", str(kernel_dir), "-t", "3600"], check=False, timeout=300)
    (project / "kaggle" / "push_output.txt").write_text((push.stdout or "") + "\n" + (push.stderr or ""), encoding="utf-8")
    if push.returncode != 0:
        status["stage"] = "kernel_push_failed"
        status["error"] = (push.stderr or push.stdout).strip()[:10000]
        write_json(ROOT / cfg["output"]["v2_status"], status)
        report += ["## Status: KERNEL PUSH FAILED", "", "```text", status["error"], "```", ""]
        (ROOT / cfg["output"]["v2_report"]).write_text("\n".join(report), encoding="utf-8")
        return

    kernel_state, kernel_status_text = poll_kernel(
        handle,
        int(cfg["v2"].get("kernel_timeout_seconds", 3300)),
        int(cfg["v2"].get("kernel_poll_seconds", 20)),
    )
    (project / "kaggle" / "status_output.txt").write_text(kernel_status_text, encoding="utf-8")
    status["kernel_state"] = kernel_state
    if kernel_state != "complete":
        status["stage"] = f"kernel_{kernel_state}"
        logs = run(["kaggle", "kernels", "logs", handle], check=False)
        (project / "kaggle" / "logs.txt").write_text((logs.stdout or "") + "\n" + (logs.stderr or ""), encoding="utf-8")
        write_json(ROOT / cfg["output"]["v2_status"], status)
        report += [f"## Status: KERNEL {kernel_state.upper()}", "", "See `competitions/.../kaggle/logs.txt`.", ""]
        (ROOT / cfg["output"]["v2_report"]).write_text("\n".join(report), encoding="utf-8")
        return

    raw_out = project / "results" / "raw"
    raw_out.mkdir(parents=True, exist_ok=True)
    out = run(["kaggle", "kernels", "output", handle, "-p", str(raw_out), "-o", "-q"], check=False, timeout=600)
    if out.returncode != 0:
        status["stage"] = "kernel_output_failed"
        status["error"] = (out.stderr or out.stdout).strip()[:10000]
        write_json(ROOT / cfg["output"]["v2_status"], status)
        report += ["## Status: OUTPUT DOWNLOAD FAILED", "", status["error"], ""]
        (ROOT / cfg["output"]["v2_report"]).write_text("\n".join(report), encoding="utf-8")
        return

    submission = locate(raw_out, "submission.csv")
    metrics_path = locate(raw_out, "metrics.json")
    results_dir = project / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {}
    if metrics_path:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        write_json(results_dir / "metrics.json", metrics)
    if not submission:
        status["stage"] = "submission_missing"
        status["error"] = "Kernel completed but did not produce submission.csv"
        write_json(ROOT / cfg["output"]["v2_status"], status)
        report += ["## Status: SUBMISSION FILE MISSING", "", status["error"], ""]
        (ROOT / cfg["output"]["v2_report"]).write_text("\n".join(report), encoding="utf-8")
        return

    if bool(cfg["v2"].get("auto_submit", True)):
        submit = run([
            "kaggle", "competitions", "submit", slug,
            "-f", str(submission),
            "-m", "Kaggle Auto Scientist V2 baseline",
            "--wait", "600",
            "--poll-interval", "30",
        ], check=False, timeout=900)
        (results_dir / "submit_output.txt").write_text((submit.stdout or "") + "\n" + (submit.stderr or ""), encoding="utf-8")
        status["submit_returncode"] = submit.returncode

    latest = latest_submission(slug)
    if latest:
        write_json(results_dir / "latest_submission.json", latest)
    status["stage"] = "complete"
    status["metrics"] = metrics
    status["latest_submission"] = latest
    write_json(ROOT / cfg["output"]["v2_status"], status)

    report += ["## Status: COMPLETE", ""]
    if metrics:
        report += [
            f"- Validation metric: `{metrics.get('validation_metric')}`",
            f"- Validation score: `{metrics.get('validation_score')}`",
            f"- Target: `{metrics.get('target')}`",
            f"- Features: `{metrics.get('features')}`",
            "",
        ]
    if latest:
        report += ["## Latest Kaggle submission", "", "```json", json.dumps(latest, ensure_ascii=False, indent=2), "```", ""]
    (ROOT / cfg["output"]["v2_report"]).write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
