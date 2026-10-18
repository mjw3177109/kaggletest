"""V2.3 GPU orchestration: launch independent Kaggle T4, verify CUDA, CV-gate results.

Never modify V2.1 tree artifacts or TPU status. Recovery only reads existing kernel.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime, timezone

from v2_1_pipeline import ROOT, run, load_json, write_json, latest_submission
from v22_pipeline import kernel_handle, validate_submission
from v23_tpu_pipeline import poll_kernel, check_cv

STATE = ROOT / "state" / "v2_3_gpu_status.json"
REPORT = ROOT / "reports" / "v2_3_gpu_latest.md"
COMP = "playground-series-s6e10"
ACCEL = "NvidiaTeslaT4"


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def report(status):
    m = status.get("metrics", {})
    gate = status.get("cv_gate", {})
    lines = [
        "# V2.3 GPU 独立训练线：Kaggle T4 XGBoost", "",
        f"- 比赛：{status.get('competition')}",
        f"- 状态：{status.get('stage')}",
        f"- Notebook：{status.get('kernel_url')}",
        f"- GPU 实际配置：{status.get('hardware', {}).get('devices', '未核验')}",
        f"- 旧冠军 V2.1 3-Fold OOF：{status.get('incumbent_cv')}",
        f"- 旧冠军 Public LB：{status.get('incumbent_lb')}",
        f"- 本轮 GPU 3-Fold OOF：{m.get('best_cv_auc', '暂无')}",
        f"- CV 与旧冠军差值：{gate.get('delta', '暂无')}",
        f"- CV 门槛通过：{gate.get('passed', False)}",
        f"- Kaggle 已提交：{status.get('submitted', False)}",
        f"- 新 Public LB：{(status.get('latest_submission') or {}).get('publicScore', '未提交或未评分')}",
        f"- 线上冠军：{status.get('champion', 'V2.1 ensemble')}",
        "", "## GPU 训练模型", "",
        "原有 CPU CatBoost/LightGBM/XGBoost Ensemble 保留不变；GPU 训练仅是新的候选模型。",
        "使用 NVIDIA T4 ×2 实例，但当前单进程 XGBoost 仅使用一个 CUDA 设备，",
        "不能声称同时使用了两张 GPU。", "",
    ]
    if m.get("models"):
        lines.extend(["| 模型 | 3-Fold OOF ROC-AUC |", "|---|---:|"])
        for name, data in m["models"].items():
            lines.append(f"| {name} | {data['cv_auc']:.8f} |")
    if status.get("stage") == "kernel_queued":
        lines.extend(["", "## 排队说明", "", "Kaggle 报告 QUEUED，仅表示已排队，不能视为 GPU 正在训练。"])
    if status.get("error"):
        lines.extend(["", "## 诊断", "", str(status["error"])[-6500:]])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(recover_only=False):
    previous = load_json(ROOT / "state" / "v2_1_status.json", {}) or {}
    if previous.get("stage") != "complete":
        raise RuntimeError("GPU experiment requires completed V2.1 champion")
    slug = previous["competition"]
    old_cv = float(previous["metrics"]["best_cv_auc"])
    old_lb = (previous.get("latest_submission") or {}).get("publicScore")
    if old_lb in (None, ""):
        raise RuntimeError("Need known champion Public LB")
    project = ROOT / "competitions" / slug
    folder = project / "kaggle_gpu_v23"
    results = project / "results_v23_gpu"
    folder.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    existing = load_json(STATE, {}) or {}
    if existing.get("competition") == slug and existing.get("stage") in (
        "complete", "skipped_no_cv_gain", "submission_pending",
        "tpu_unavailable", "gpu_unavailable"
    ):
        print("GPU experiment already finalized; refusing duplicate training/submission")
        return
    status = dict(existing) if existing.get("competition") == slug else {}
    status.update({"version": "V2.3 GPU", "competition": slug, "updated_at": utcnow(),
                   "incumbent_cv": old_cv, "incumbent_lb": old_lb,
                   "champion": "V2.1 ensemble", "submitted": bool(status.get("submitted", False))})
    username = os.environ.get("KAGGLE_USERNAME") or previous.get("kernel", "").split("/")[0]
    kslug = ("kas-v23-gpu-" + re.sub(r"[^a-z0-9-]+", "-", slug.lower()))[:48].rstrip("-")
    expected = f"{username}/{kslug}"
    status.setdefault("kernel", expected)
    status.setdefault("kernel_url", f"https://www.kaggle.com/code/{expected}")

    try:
        if not recover_only and status.get("stage") not in (
            "kernel_queued", "kernel_running", "kernel_complete", "kernel_access_error"
        ):
            shutil.copy2(ROOT / "kaggle_agent" / "v23_gpu_train.py", folder / "main.py")
            write_json(folder / "kernel-metadata.json", {
                "id": expected, "title": kslug, "code_file": "main.py",
                "language": "python", "kernel_type": "script", "is_private": True,
                "enable_gpu": True, "machine_shape": ACCEL, "enable_internet": False,
                "dataset_sources": [], "competition_sources": [slug],
                "kernel_sources": [], "model_sources": []
            })
            status["stage"] = "pushing_gpu_kernel"
            write_json(STATE, status)
            pushed = run(["kaggle", "kernels", "push", "-p", str(folder), "-t", "3600",
                          "--accelerator", ACCEL], check=False, timeout=300)
            txt = (pushed.stdout or "") + "\n" + (pushed.stderr or "")
            (folder / "push_output.txt").write_text(txt, encoding="utf-8")
            if pushed.returncode:
                status.update({"stage": "gpu_push_failed", "error": txt[-8000:]})
                return
            status["kernel"] = kernel_handle(txt, username, kslug)
        elif recover_only and not existing.get("kernel"):
            raise RuntimeError("Cannot recover: no previously created GPU kernel")
        status["kernel_url"] = f"https://www.kaggle.com/code/{status['kernel']}"
        stage, desc = poll_kernel(status["kernel"], 90 if recover_only else 600, 20)
        (folder / "status_output.txt").write_text(desc, encoding="utf-8")
        status["stage"] = stage
        status["kernel_state"] = stage
        if stage in ("kernel_queued", "kernel_running", "kernel_access_error"):
            status["error"] = desc[-6500:]
            return
        # A previously queued status is not an error after Kaggle has completed.
        status.pop("error", None)

        raw = results / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        out = run(["kaggle", "kernels", "output", status["kernel"],
                   "-p", str(raw), "-o", "-q"], check=False, timeout=600)
        hwfiles = list(raw.rglob("v23_gpu_hardware.json"))
        if hwfiles:
            hardware = json.loads(hwfiles[0].read_text(encoding="utf-8"))
            status["hardware"] = hardware
            write_json(results / "hardware.json", hardware)
            if not hardware.get("ok"):
                status.update({"stage": "gpu_unavailable",
                               "error": "Kaggle did not allocate NVIDIA T4; GPU training refused."})
                return
        if stage != "kernel_complete":
            p = run(["kaggle", "kernels", "logs", status["kernel"]], check=False, timeout=120)
            logs = (p.stdout or "") + "\n" + (p.stderr or "")
            (folder / "logs.txt").write_text(logs[-25000:], encoding="utf-8")
            status.update({"stage": "kernel_failed", "error": logs[-7000:]})
            return
        if out.returncode or not hwfiles:
            status.update({"stage": "output_failed", "error": "Kaggle missing hardware probe or output files."})
            return
        mp = list(raw.rglob("v23_gpu_metrics.json"))
        sp = list(raw.rglob("candidate_submission.csv"))
        if not mp or not sp:
            status.update({"stage": "artifacts_missing", "error": "Kernel missing CV metrics or submission."})
            return
        m = json.loads(mp[0].read_text(encoding="utf-8"))
        if (not m.get("gpu_verified") or m.get("validation") !=
                "StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)"
                or int(m.get("train_rows", -1)) != int(previous["metrics"]["rows"])
                or int(m.get("folds", 0)) != 3):
            status.update({"stage": "incomparable_results", "error": "GPU backend or CV settings invalid."})
            return
        for item in m.get("models", {}).values():
            if not all(str(f.get("device", "")).startswith("cuda") for f in item.get("folds", [])):
                status.update({"stage": "cuda_unverified", "error": "Not all folds verified CUDA."})
                return
        validate_submission(sp[0], m["target"], m["test_rows"])
        status["metrics"] = m
        write_json(results / "metrics.json", m)
        status["cv_gate"] = check_cv(m["best_cv_auc"], old_cv, .00010)
        if not status["cv_gate"]["passed"]:
            status.update({"stage": "skipped_no_cv_gain", "submitted": False})
            return
        # Maximum ONE automatically submitted candidate; reuse if already submitted.
        note = f"KAS V2.3 GPU {m['best_model']} CV={m['best_cv_auc']:.6f}"
        latest = latest_submission(slug)
        if not latest or note not in str(latest.get("description", "")):
            status["stage"] = "submitting"
            write_json(STATE, status)
            p = run(["kaggle", "competitions", "submit", slug, "-f", str(sp[0]),
                     "-m", note, "--wait", "600", "--poll-interval", "30"],
                    check=False, timeout=900)
            (results / "submit_output.txt").write_text(
                (p.stdout or "") + "\n" + (p.stderr or ""), encoding="utf-8"
            )
            if p.returncode:
                status.update({"stage": "submission_failed", "error": (p.stderr or p.stdout)[-6500:]})
                return
            latest = latest_submission(slug)
        if not latest or note not in str(latest.get("description", "")):
            status.update({"stage": "submission_unverified", "error": "Kaggle submission cannot be verified."})
            return
        status["submitted"] = True
        status["latest_submission"] = latest
        write_json(results / "latest_submission.json", latest)
        score = latest.get("publicScore")
        if score in (None, ""):
            status["stage"] = "submission_pending"
            return
        delta = round(float(score) - float(old_lb), 8)
        status["public_lb_delta"] = delta
        status["champion"] = f"V2.3 GPU {m['best_model']}" if delta > 0 else "V2.1 ensemble"
        status["stage"] = "complete"
    except Exception as exc:
        status.update({"stage": "controller_failed", "error": repr(exc)})
    finally:
        status["updated_at"] = utcnow()
        write_json(STATE, status)
        report(status)
        print(json.dumps(status, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--recover-only", action="store_true")
    main(args.parse_args().recover_only)
