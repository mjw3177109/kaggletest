"""V2.2 GitHub controller: Kaggle kernel push -> poll -> CV gate -> optional submit -> archive.

Idempotent after success; never submits a worse/equal CV candidate.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from v2_1_pipeline import ROOT, run, load_json, write_json, latest_submission

BASE = ROOT / "state" / "v2_1_status.json"
STATE = ROOT / "state" / "v2_2_status.json"
REPORT = ROOT / "reports" / "v2_2_latest.md"
CONF = ROOT / "config" / "agent.yaml"
V22_SOURCE = ROOT / "kaggle_agent" / "v22_train.py"


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def model_gate(candidate_cv: float, incumbent_cv: float, min_delta: float) -> dict[str, Any]:
    delta = float(candidate_cv) - float(incumbent_cv)
    return {"candidate_cv": float(candidate_cv), "champion_cv": float(incumbent_cv),
            "delta": delta, "min_delta": min_delta, "passed": delta > min_delta}


def kernel_handle(stdout: str, expected_owner: str, expected_slug: str) -> str:
    match = re.search(r"https?://(?:www\.)?kaggle\.com/code/([A-Za-z0-9_-]+)/([a-z0-9-]+)", stdout)
    if not match:
        raise ValueError("Kernel push gave no Kaggle code URL; cannot assume a slug.")
    owner, slug = match.groups()
    if owner.lower() != expected_owner.lower() or slug != expected_slug:
        raise ValueError(f"Expected {expected_owner}/{expected_slug} but Kaggle returned {owner}/{slug}")
    return f"{owner}/{slug}"


def poll(handle: str, limit: int, every: int) -> tuple[str, str]:
    stop = time.monotonic() + limit
    result = ""
    while time.monotonic() < stop:
        p = run(["kaggle", "kernels", "status", handle], check=False, timeout=90)
        result = (p.stdout or "") + "\n" + (p.stderr or "")
        low = result.lower()
        if p.returncode or "permission" in low or "denied" in low:
            return "kernel_access_error", result
        if any(term in low for term in ("failed", "error", "cancelled", "canceled")):
            return "kernel_failed", result
        if any(term in low for term in ("complete", "success")):
            return "kernel_complete", result
        time.sleep(every)
    return "kernel_running", result


def validate_submission(path: Path, target: str, expected_rows: int):
    import pandas as pd
    import numpy as np
    df = pd.read_csv(path)
    if len(df) != expected_rows or target not in df:
        raise ValueError("Candidate row count or target column mismatches Kaggle sample.")
    pred = pd.to_numeric(df[target], errors="coerce").to_numpy()
    if not np.isfinite(pred).all() or (pred < 0).any() or (pred > 1).any():
        raise ValueError("Candidate probabilities must be finite numbers between 0 and 1.")


def write_report(status: dict[str, Any], project: Path):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    m = status.get("metrics", {})
    imp = load_json(project / "results_v22" / "feature_importance.json", {}) or {}
    lines = [
        "# Kaggle Auto Scientist V2.2 — 自动参数搜索与实验复盘",
        "",
        f"- 比赛：{status.get('competition')}",
        f"- 当前阶段：{status.get('stage')}",
        f"- Kaggle Notebook：{status.get('kernel_url') or '暂无'}",
        f"- 上一轮 V2.1 Public LB：{status.get('incumbent_lb')}",
        f"- 上一轮 V2.1 3-Fold OOF：{status.get('incumbent_cv')}",
        "",
        "## 本轮做了什么",
        "",
        "Optuna TPE 搜索 LightGBM 超参数；然后用与 V2.1 相同的 3-Fold",
        "StratifiedKFold 和随机种子，分别验证旧特征方案与扩展特征方案。",
        "两套方案都完成后才决定是否提交。重要性使用 LightGBM Gain。",
        "",
    ]
    if m:
        lines.extend([
            f"- Optuna trials：{m.get('tuning',{}).get('trials_completed')} / {m.get('tuning',{}).get('trials_requested')}",
            f"- 本轮最好的候选：{m.get('best_candidate')}",
            "",
            "| 方案 | Full 3-Fold OOF ROC-AUC |", "|---|---:|",
        ])
        for name, row in m.get("candidates", {}).items():
            lines.append(f"| {name} | {float(row['cv_auc']):.8f} |")
        gate = status.get("cv_gate", {})
        lines += [
            "",
            f"- 与 V2.1 3-Fold 的差值：{gate.get('delta')}",
            f"- 提交阈值：必须提高超过 {gate.get('min_delta')}",
            f"- 是否通过：{'是' if gate.get('passed') else '否（保留原冠军）'}",
        ]
    top = imp.get("candidates", {}).get(m.get("best_candidate"), [])[:15]
    if top:
        lines += ["", "## Top 15 特征重要性（Gain）", "",
                  "| 特征 | Gain |", "|---|---:|"]
        for x in top:
            lines.append(f"| {x['feature']} | {x['gain']} |")
    latest = status.get("latest_submission") or {}
    lines += [
        "", "## Kaggle 结果", "",
        f"- V2.2 提交：{status.get('submitted', False)}",
        f"- V2.2 Public LB：{latest.get('publicScore') if status.get('submitted') else '未提交'}",
        f"- 新旧线上分数差：{status.get('lb_delta')}",
        f"- 最终冠军：{status.get('champion', 'V2.1 ensemble')}",
        "",
        "## 下一轮建议", "",
        "若本轮未通过 3-Fold 门槛，可重点分析服务评分、出行类型、延误时长相关交互特征；",
        "避免仅根据一次 Public LB 调参，以免过拟合公共榜。",
    ]
    if status.get("error"):
        lines += ["", "## 排障", "", "错误信息详见 state/v2_2_status.json。"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline():
    config = yaml.safe_load(CONF.read_text(encoding="utf-8")) or {}
    v22 = config.get("v22", {})
    previous = load_json(BASE, {}) or {}
    if previous.get("stage") != "complete" or not previous.get("metrics"):
        raise RuntimeError("V2.1 completed metrics required before V2.2.")
    slug = previous["competition"]
    incumbent_cv = float(previous["metrics"]["best_cv_auc"])
    incumbent_lb = str((previous.get("latest_submission") or {}).get("publicScore") or "")
    if not incumbent_lb:
        raise RuntimeError("V2.1 has no verified Public LB to compare.")
    project = ROOT / "competitions" / slug
    project.mkdir(parents=True, exist_ok=True)

    old_status = load_json(STATE, {}) or {}
    if old_status.get("stage") in ("complete", "skipped_no_cv_gain", "submission_pending") and old_status.get("competition") == slug:
        print("V2.2 experiment already finalized. Not training/submitting twice.")
        return

    status = dict(old_status) if old_status.get("competition") == slug else {}
    status.update({
        "version": "V2.2", "competition": slug, "updated_at": timestamp(),
        "incumbent_cv": incumbent_cv, "incumbent_lb": incumbent_lb,
    })
    target_folder = project / "kaggle_v22"
    target_folder.mkdir(parents=True, exist_ok=True)
    results = project / "results_v22"
    results.mkdir(parents=True, exist_ok=True)
    username = (os.environ.get("KAGGLE_USERNAME") or previous["kernel"].split("/")[0]).strip()
    slugged = "kas-v22-" + re.sub(r"[^a-z0-9-]", "-", slug.lower()).strip("-")
    slugged = slugged[:49].rstrip("-")
    predicted = f"{username}/{slugged}"

    # Make the title precisely equal to Kaggle's expected URL slug.
    metadata = {
        "id": predicted, "title": slugged, "code_file": "main.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": False, "enable_internet": False, "dataset_sources": [],
        "competition_sources": [slug], "kernel_sources": [], "model_sources": [],
    }
    write_json(target_folder / "kernel-metadata.json", metadata)
    shutil.copy2(V22_SOURCE, target_folder / "main.py")
    status["kernel"] = predicted
    status["kernel_url"] = f"https://www.kaggle.com/code/{predicted}"
    try:
        if status.get("stage") not in ("kernel_running", "kernel_complete", "kernel_output_failed", "kernel_artifacts_missing"):
            status["stage"] = "pushing_kernel"
            write_json(STATE, status)
            p = run(["kaggle", "kernels", "push", "-p", str(target_folder), "-t",
                     str(v22.get("kernel_timeout_seconds", 3600))],
                    check=False, timeout=300)
            push_txt = (p.stdout or "") + "\n" + (p.stderr or "")
            (target_folder / "push_output.txt").write_text(push_txt, encoding="utf-8")
            if p.returncode:
                status.update({"stage": "kernel_push_failed", "error": push_txt[-8000:]})
                return
            status["kernel"] = kernel_handle(push_txt, username, slugged)
            status["kernel_url"] = f"https://www.kaggle.com/code/{status['kernel']}"
            status["stage"] = "kernel_running"
            write_json(STATE, status)

        result, message = poll(status["kernel"], int(v22.get("poll_timeout_seconds", 3000)), 20)
        (target_folder / "status_output.txt").write_text(message, encoding="utf-8")
        status["kernel_state"] = result
        status["stage"] = result
        if result != "kernel_complete":
            if result in ("kernel_failed", "kernel_access_error"):
                p = run(["kaggle", "kernels", "logs", status["kernel"]], check=False, timeout=120)
                logs = (p.stdout or "") + "\n" + (p.stderr or "")
                (target_folder / "logs.txt").write_text(logs[-25000:], encoding="utf-8")
                status["error"] = logs[-5000:] if result == "kernel_failed" else message[-5000:]
            return

        raw = results / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        p = run(["kaggle", "kernels", "output", status["kernel"],
                 "-p", str(raw), "-o", "-q"], check=False, timeout=600)
        if p.returncode:
            status.update({"stage": "kernel_output_failed", "error": (p.stderr or p.stdout)[-8000:]})
            return
        mp = list(raw.rglob("v22_metrics.json"))
        ip = list(raw.rglob("v22_feature_importance.json"))
        tp = list(raw.rglob("v22_trials.json"))
        sp = list(raw.rglob("candidate_submission.csv"))
        if not (mp and ip and tp and sp):
            err = list(raw.rglob("v22_error.json"))
            message = err[0].read_text(encoding="utf-8") if err else "Missing metrics/importance/trials/submission in kernel outputs"
            status.update({"stage": "kernel_artifacts_missing", "error": message[-8000:]})
            return
        metrics = json.loads(mp[0].read_text(encoding="utf-8"))
        feature_importance = json.loads(ip[0].read_text(encoding="utf-8"))
        trials = json.loads(tp[0].read_text(encoding="utf-8"))
        if (metrics.get("validation") != "StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)"
                or metrics.get("validation_metric") != "roc_auc"):
            raise ValueError("Refuse CV comparison with incompatible validation settings")
        if int(metrics.get("train_rows", 0)) != int(previous["metrics"]["rows"]):
            raise ValueError("V2.2/V2.1 have different row counts: cannot compare CV scores")
        validate_submission(sp[0], metrics["target"], metrics["test_rows"])
        status["metrics"] = metrics
        write_json(results / "metrics.json", metrics)
        write_json(results / "feature_importance.json", feature_importance)
        write_json(results / "trials.json", trials)
        min_gain = float(v22.get("min_cv_improvement", 0.00008))
        status["cv_gate"] = model_gate(metrics["best_cv_auc"], incumbent_cv, min_gain)

        if not status["cv_gate"]["passed"]:
            status.update({
                "stage": "skipped_no_cv_gain", "submitted": False,
                "champion": "V2.1 ensemble", "decision": "No submission: CV threshold not met",
            })
            return
        if not bool(v22.get("auto_submit", True)):
            status.update({"stage": "awaiting_submit_approval", "submitted": False})
            return

        best = metrics["best_candidate"]
        marker = f"KAS V2.2 {best} CV={float(metrics['best_cv_auc']):.6f}"
        latest = latest_submission(slug)
        if latest and marker in str(latest.get("description", "")):
            status["submission_reused"] = True
        else:
            status["stage"] = "submitting"
            write_json(STATE, status)
            p = run(["kaggle", "competitions", "submit", slug,
                     "-f", str(sp[0]), "-m", marker,
                     "--wait", "600", "--poll-interval", "30"], check=False, timeout=900)
            (results / "submit_output.txt").write_text((p.stdout or "") + "\n" + (p.stderr or ""), encoding="utf-8")
            if p.returncode:
                status.update({"stage": "submission_failed", "error": (p.stderr or p.stdout)[-8000:]})
                return
            latest = latest_submission(slug)
        if not latest or marker not in str(latest.get("description", "")):
            status.update({"stage": "submission_unverified", "error": "Latest Kaggle submission cannot be verified by its message"})
            return
        status["submitted"] = True
        status["latest_submission"] = latest
        write_json(results / "latest_submission.json", latest)
        new_lb = latest.get("publicScore")
        if new_lb in ("", None):
            status.update({"stage": "submission_pending", "champion": "V2.1 ensemble (pending new LB)"})
            return
        status["lb_delta"] = round(float(new_lb) - float(incumbent_lb), 8)
        status["champion"] = "V2.2 " + best if status["lb_delta"] > 0 else "V2.1 ensemble"
        status["stage"] = "complete"
    except Exception as e:
        status["stage"] = "controller_failed"
        status["error"] = repr(e)
    finally:
        status["updated_at"] = timestamp()
        write_json(STATE, status)
        write_report(status, project)
        metrics = status.get("metrics") or {}
        if metrics:
            exp = project / "experiments"
            exp.mkdir(parents=True, exist_ok=True)
            for idx, (name, v) in enumerate(metrics.get("candidates", {}).items(), 6):
                text = [
                    f"# Run {idx:03d} — {name}", "",
                    f"- Full 3-fold OOF ROC-AUC: {v['cv_auc']:.8f}",
                    f"- Fold AUCs: {v.get('fold_auc')}",
                    f"- Previous ensemble: {incumbent_cv:.8f}",
                    f"- Submitted: {bool(status.get('submitted') and name == metrics.get('best_candidate'))}",
                    f"- Public LB (if submitted): {(status.get('latest_submission') or {}).get('publicScore') if name == metrics.get('best_candidate') and status.get('submitted') else 'not submitted'}",
                    "", "Tuned with Optuna on a stratified sample; full validation uses the original V2.1 folds.",
                ]
                (exp / f"run_{idx:03d}_{name}.md").write_text("\n".join(text) + "\n", encoding="utf-8")
        print(json.dumps(status, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    run_pipeline()
