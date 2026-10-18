"""Recover an already-pushed Kaggle V2.1 kernel. Never start new training."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from v2_1_pipeline import ROOT, run, load_json, write_json, latest_submission


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_actual_kernel(project: Path) -> str:
    push = (project / "kaggle_v21" / "push_output.txt").read_text(encoding="utf-8")
    m = re.search(r"https?://(?:www\\.)?kaggle\\.com/code/([A-Za-z0-9_-]+)/([a-z0-9-]+)", push)
    if not m:
        raise RuntimeError("No real Kaggle notebook URL found in push_output.txt; cannot guess kernel ID.")
    return f"{m.group(1)}/{m.group(2)}"


def status_from_cli(handle: str) -> tuple[str, str]:
    p = run(["kaggle", "kernels", "status", handle], check=False, timeout=90)
    message = ((p.stdout or "") + "\n" + (p.stderr or "")).strip()
    low = message.lower()
    if p.returncode != 0 or "permission" in low or "denied" in low:
        return "kernel_access_error", message
    if any(x in low for x in ("error", "failed", "cancel")):
        return "kernel_failed", message
    if any(x in low for x in ("complete", "success")):
        return "kernel_complete", message
    return "kernel_running", message


def make_report(project: Path, status: dict[str, Any]) -> None:
    slug = status["competition"]
    metrics = status.get("metrics") or {}
    ranking = metrics.get("ranking") or []
    lines = [
        "# Kaggle Auto Scientist V2.1 - Recovery and Model Comparison",
        "",
        f"- Competition: \`{slug}\`",
        f"- Actual Kaggle kernel: \`{status.get('kernel')}\`",
        f"- Recovery stage: \`{status.get('stage')}\`",
        f"- Previous public LB: \`{status.get('baseline_public_score')}\`",
        "",
    ]
    if ranking:
        lines += ["| Rank | Model | 3-Fold OOF ROC-AUC |", "|---:|---|---:|"]
        for i, row in enumerate(ranking, 1):
            lines.append(f"| {i} | {row['model']} | {float(row['cv_auc']):.8f} |")
        lines += ["", f"Selected candidate: **{metrics.get('best_model')}**", ""]
    latest = status.get("latest_submission") or {}
    lines += [
        f"New public LB: \`{latest.get('publicScore') or 'not available'}\`",
        f"Online delta from V2: \`{status.get('public_score_delta')}\`",
        "",
        "No new Kaggle training was started by the recovery job.",
    ]
    if status.get("error"):
        lines += ["", "## Error or current condition", "", "```text", str(status["error"])[:6000], "```"]
    dest = ROOT / "reports" / "v2_1_latest.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_run_files(project: Path, metrics: dict[str, Any], online_model: str, new_lb: Any, old_lb: Any) -> None:
    exp = project / "experiments"
    exp.mkdir(parents=True, exist_ok=True)
    for ident, name in (("002", "catboost"), ("003", "lightgbm"), ("004", "xgboost"), ("005", "ensemble")):
        d = metrics.get("models", {}).get(name)
        if not d:
            continue
        lines = [
            f"# Run {ident} - {name}",
            "",
            f"- OOF CV ROC-AUC: \`{d.get('cv_auc')}\`",
            f"- Individual fold scores: \`{d.get('fold_auc', 'ensemble')}\`",
            f"- Submitted: \`{name == online_model}\`",
            f"- Public LB: \`{new_lb if name == online_model else 'not submitted'}\`",
            f"- Prior V2 LB: \`{old_lb}\`",
            "",
            "3-fold comparison using the same data split; do not compare the V2 holdout score as if it were the same CV method.",
        ]
        (exp / f"run_{ident}_{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    original = load_json(ROOT / "state" / "v2_1_status.json", {}) or {}
    v2 = load_json(ROOT / "state" / "v2_status.json", {}) or {}
    slug = original.get("competition") or v2.get("competition")
    if not slug:
        raise RuntimeError("No active competition found.")
    project = ROOT / "competitions" / slug
    status: dict[str, Any] = dict(original)
    status.update({"competition": slug, "recovered_at": now()})
    stage_file = ROOT / "state" / "v2_1_status.json"
    try:
        handle = extract_actual_kernel(project)
        status["kernel"] = handle
        stage, response = status_from_cli(handle)
        (project / "kaggle_v21" / "status_output.txt").write_text(response + "\n", encoding="utf-8")
        status["stage"] = stage
        if stage == "kernel_access_error":
            status["error"] = response
            return
        if stage == "kernel_failed":
            p = run(["kaggle", "kernels", "logs", handle], check=False, timeout=90)
            status["error"] = ((p.stdout or "") + "\n" + (p.stderr or ""))[-8000:]
            (project / "kaggle_v21" / "logs.txt").write_text(status["error"], encoding="utf-8")
            return
        if stage == "kernel_running":
            status["error"] = "Kaggle reports the existing kernel is still running; no new kernel was pushed."
            return

        # Completed notebook: download its existing outputs without re-training.
        raw = project / "results_v21" / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        p = run(["kaggle", "kernels", "output", handle, "-p", str(raw), "-o", "-q"], check=False, timeout=600)
        if p.returncode:
            status["stage"] = "kernel_output_failed"
            status["error"] = ((p.stdout or "") + "\n" + (p.stderr or ""))[-8000:]
            return
        mp = sorted(raw.rglob("v21_metrics.json"))
        sp = sorted(raw.rglob("best_submission.csv"))
        if not mp or not sp:
            status["stage"] = "kernel_artifacts_missing"
            status["error"] = "Kernel completed, but v21_metrics.json or best_submission.csv was not produced."
            return
        metrics = json.loads(mp[0].read_text(encoding="utf-8"))
        best = str(metrics.get("best_model") or "")
        if not best or not metrics.get("ranking"):
            raise RuntimeError("Candidate ranking/best_model missing in Kaggle metrics.")
        import pandas as pd
        import numpy as np
        submission = pd.read_csv(sp[0])
        target = metrics.get("target")
        if not target or target not in submission.columns or len(submission) == 0:
            raise RuntimeError("submission.csv has no rows or missing target column.")
        if submission[target].isna().any() or not np.isfinite(submission[target].to_numpy(dtype=float)).all():
            raise RuntimeError("submission.csv contains missing or non-finite predictions.")

        results = project / "results_v21"
        write_json(results / "metrics.json", metrics)
        status["metrics"] = metrics
        old_lb = status.get("baseline_public_score") or (v2.get("latest_submission") or {}).get("publicScore")
        status["baseline_public_score"] = old_lb
        marker = f"KAS V2.1 {best} CV={float(metrics['best_cv_auc']):.6f}"
        before = latest_submission(slug)
        if before and marker in str(before.get("description", "")):
            status["submit_returncode"] = 0
            status["submission_reused"] = True
            latest = before
        else:
            status["stage"] = "submitting_best_cv"
            write_json(stage_file, status)
            p = run([
                "kaggle", "competitions", "submit", slug, "-f", str(sp[0]),
                "-m", marker, "--wait", "600", "--poll-interval", "30",
            ], check=False, timeout=900)
            (results / "submit_output.txt").write_text((p.stdout or "") + "\n" + (p.stderr or ""), encoding="utf-8")
            status["submit_returncode"] = p.returncode
            latest = latest_submission(slug)
            if p.returncode:
                status["stage"] = "submission_failed"
                status["error"] = (p.stderr or p.stdout)[-6000:]
                return
        if not latest or marker not in str(latest.get("description", "")):
            status["stage"] = "submission_unverified"
            status["error"] = "Could not verify that the latest competition submission belongs to this recovery."
            return
        write_json(results / "latest_submission.json", latest)
        status["latest_submission"] = latest
        new_lb = latest.get("publicScore")
        try:
            status["public_score_delta"] = float(new_lb) - float(old_lb) if new_lb not in ("", None) and old_lb not in ("", None) else None
        except (TypeError, ValueError):
            status["public_score_delta"] = None
        save_run_files(project, metrics, best, new_lb, old_lb)
        status["stage"] = "complete" if new_lb not in ("", None) else "submission_pending"
        status.pop("error", None)
    except Exception as exc:
        status["stage"] = "recovery_failed"
        status["error"] = str(exc)[:8000]
    finally:
        write_json(stage_file, status)
        make_report(project, status)
        print(json.dumps(status, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
