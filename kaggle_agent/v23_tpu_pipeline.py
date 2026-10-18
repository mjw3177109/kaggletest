"""Independent V2.3 TPU controller with verified backend and conservative submissions."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone

import yaml

from v2_1_pipeline import ROOT, load_json, write_json, run, latest_submission
from v22_pipeline import kernel_handle, validate_submission

STATE = ROOT / "state" / "v2_3_status.json"
REPORT = ROOT / "reports" / "v2_3_latest.md"


def now():
    return datetime.now(timezone.utc).isoformat()


def check_cv(candidate, benchmark, minimum):
    delta = float(candidate) - float(benchmark)
    return {"candidate": float(candidate), "benchmark": float(benchmark),
            "delta": delta, "threshold": float(minimum), "passed": delta > float(minimum)}


def poll_kernel(handle, wait_seconds=3000, interval=25):
    stop = time.monotonic() + wait_seconds
    last = ""
    while time.monotonic() < stop:
        p = run(["kaggle", "kernels", "status", handle], check=False, timeout=90)
        last = ((p.stdout or "") + "\n" + (p.stderr or "")).strip()
        low = last.lower()
        if p.returncode or "permission" in low or "denied" in low:
            return "kernel_access_error", last
        if "kernelworkerstatus.queued" in low or 'has status "queued"' in low:
            return "kernel_queued", last
        if any(x in low for x in ("failed", "error", "cancelled", "canceled")):
            return "kernel_failed", last
        if any(x in low for x in ("complete", "success")):
            return "kernel_complete", last
        time.sleep(interval)
    return "kernel_running", last


def write_report(status):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    metrics = status.get("metrics") or {}
    hw = status.get("hardware") or {}
    gate = status.get("cv_gate") or {}
    lines = [
        "# Kaggle Auto Scientist V2.3 - 独立 TPU 神经网络实验",
        "", "## 训练线状态", "",
        "- 树模型 V2.1 Ensemble 保留不变。",
        "- 独立的 TPU 线用 JAX MLP；统一 3 折验证、再判断是否提交。",
        f"- 比赛：{status.get('competition')}",
        f"- 当前阶段：{status.get('stage')}",
        f"- Kaggle Notebook：{status.get('kernel_url', '尚未创建')}",
        "", "## 硬件核验", "",
        f"- 请求硬件：TpuV5E8",
        f"- 真实 JAX backend：{hw.get('actual', '未知')}",
        f"- 真正使用 TPU：{hw.get('ok', False)}",
        f"- JAX device count：{hw.get('device_count', '未知')}",
        "", "## 3-Fold OOF 对比", "",
        f"- V2.1 树模型 CV：{status.get('incumbent_cv')}",
        f"- V2.1 Public LB：{status.get('incumbent_lb')}",
        f"- V2.3 JAX TPU CV：{metrics.get('cv_auc', '尚未完成')}",
        f"- CV 提升差值：{gate.get('delta', '未知')}",
        f"- 提交门槛：{gate.get('threshold', '未知')}",
        f"- 达到提交条件：{gate.get('passed', False)}",
        "", "## 结果与保底", "",
        f"- 已提交：{status.get('submitted', False)}",
        f"- 新 Public LB：{(status.get('latest_submission') or {}).get('publicScore', '未提交或未评分')}",
        f"- 当前线上冠军：{status.get('champion', 'V2.1 ensemble')}",
        "",
        "V2.1 未存储逐行 OOF 预测，因此不能声称已比较了 TPU/树模型的联合融合分数。",
        "TPU 线单独验证，不覆盖 V2.1 树模型。",
    ]
    if status.get("stage") == "kernel_queued":
        lines.extend([
            "", "## Kaggle 仍在排队", "",
            "KernelWorkerStatus.QUEUED 表示 Kaggle 尚未安排执行，",
            "因此现在没有 TPU JAX 检测或 CV 结果；绝不能将它写成训练中或训练失败。",
            "可在 Kaggle Notebook 页面查看排队状态，若排队持续很久再检查 TPU 额度。",
        ])
    if status.get("stage") == "tpu_unavailable":
        lines.extend([
            "", "## Kaggle 没有真正分配 TPU 时", "",
            "Kaggle CLI 可能返回上传成功，但服务端仍运行 CPU 镜像。",
            "本系统会检测并拒绝使用 CPU 结果。",
            "请打开上述 Kaggle Notebook，在网页选择 Settings → Accelerator → TPU v5e-8。",
            "点击 Save & Run All，确认 TPU 运行完成，然后在 GitHub Actions",
            "选择 V2.3 的 recover_existing 模式，只读取已有结果，不重新提交训练。",
        ])
    if status.get("error"):
        lines.extend(["", "## 诊断信息", "", str(status["error"])[-6000:]])
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(recover_only=False):
    base = load_json(ROOT / "state" / "v2_1_status.json", {}) or {}
    if base.get("stage") != "complete":
        raise RuntimeError("V2.1 complete status required before TPU experiment")
    config = (yaml.safe_load((ROOT / "config" / "agent.yaml").read_text(encoding="utf-8")) or {}).get("v23", {})
    slug = str(base["competition"])
    old_cv = float(base["metrics"]["best_cv_auc"])
    old_lb = (base.get("latest_submission") or {}).get("publicScore")
    if old_lb in (None, ""):
        raise RuntimeError("No V2.1 scored submission exists")
    project = ROOT / "competitions" / slug
    project.mkdir(parents=True, exist_ok=True)
    previous = load_json(STATE, {}) or {}
    if previous.get("competition") == slug and previous.get("stage") in (
        "complete", "skipped_no_cv_gain", "submission_pending"
    ):
        print("V2.3 already finalized; refusing duplicate training and submissions")
        return
    if previous.get("stage") == "tpu_unavailable" and not recover_only:
        print("TPU unavailable via CLI; choose recover_existing after running on Kaggle UI")
        return
    status = dict(previous) if previous.get("competition") == slug else {}
    status.update({
        "version": "V2.3", "competition": slug, "incumbent_cv": old_cv,
        "incumbent_lb": old_lb, "updated_at": now(),
        "champion": "V2.1 ensemble", "submitted": bool(status.get("submitted", False)),
    })
    folder = project / "kaggle_tpu_v23"
    results = project / "results_v23"
    folder.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    username = os.environ.get("KAGGLE_USERNAME") or base.get("kernel", "").split("/")[0]
    if not username:
        raise RuntimeError("Kaggle username unavailable")
    notebook_slug = ("kas-v23-tpu-" + re.sub(r"[^a-z0-9-]", "-", slug.lower()))[:48].rstrip("-")
    handle = f"{username}/{notebook_slug}"
    status["kernel"] = handle
    status["kernel_url"] = f"https://www.kaggle.com/code/{handle}"
    try:
        if not recover_only and status.get("stage") not in ("kernel_running", "kernel_complete"):
            shutil.copy2(ROOT / "kaggle_agent" / "v23_tpu_train.py", folder / "main.py")
            accelerator = str(config.get("accelerator", "TpuV5E8"))
            write_json(folder / "kernel-metadata.json", {
                "id": handle, "title": notebook_slug,
                "code_file": "main.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": False, "enable_tpu": True,
                "machine_shape": accelerator, "enable_internet": False,
                "dataset_sources": [], "competition_sources": [slug],
                "kernel_sources": [], "model_sources": [],
            })
            status["stage"] = "pushing_tpu_kernel"
            write_json(STATE, status)
            p = run([
                "kaggle", "kernels", "push", "-p", str(folder),
                "-t", str(config.get("kernel_timeout_seconds", 3600)),
                "--accelerator", accelerator,
            ], check=False, timeout=300)
            output = (p.stdout or "") + "\n" + (p.stderr or "")
            (folder / "push_output.txt").write_text(output, encoding="utf-8")
            if p.returncode:
                status.update({"stage": "tpu_push_failed", "error": output[-7000:]})
                return
            status["kernel"] = kernel_handle(output, username, notebook_slug)
            status["kernel_url"] = f"https://www.kaggle.com/code/{status['kernel']}"
        elif recover_only and not previous.get("kernel"):
            raise RuntimeError("No previously pushed kernel to recover")
        status["stage"] = "kernel_running"
        write_json(STATE, status)
        stage, message = poll_kernel(
            status["kernel"], 90 if recover_only else int(config.get("poll_timeout_seconds", 2950))
        )
        (folder / "status_output.txt").write_text(message, encoding="utf-8")
        status["kernel_state"] = stage
        if stage in ("kernel_access_error", "kernel_running", "kernel_queued"):
            status["stage"] = stage
            status["error"] = message[-5000:] if stage != "kernel_running" else "Kernel is still running"
            return
        # Even after Kaggle reports a failed run, retrieve the hardware probe if possible.
        raw = results / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        out = run([
            "kaggle", "kernels", "output", status["kernel"], "-p", str(raw), "-o", "-q"
        ], check=False, timeout=600)
        hp = list(raw.rglob("v23_hardware.json"))
        if hp:
            hardware = json.loads(hp[0].read_text(encoding="utf-8"))
            status["hardware"] = hardware
            write_json(results / "hardware.json", hardware)
            if not hardware.get("ok") or hardware.get("actual") != "tpu":
                status.update({
                    "stage": "tpu_unavailable",
                    "error": "Actual TPU backend missing. Refusing fake TPU results or CPU fallback.",
                })
                return
        if stage != "kernel_complete":
            p = run(["kaggle", "kernels", "logs", status["kernel"]], check=False, timeout=120)
            logs = (p.stdout or "") + "\n" + (p.stderr or "")
            (folder / "logs.txt").write_text(logs[-25000:], encoding="utf-8")
            status.update({"stage": "kernel_failed", "error": logs[-7000:]})
            return
        if out.returncode:
            status.update({"stage": "output_failed", "error": (out.stderr or out.stdout)[-7000:]})
            return
        if not hp:
            status.update({"stage": "tpu_unverified", "error": "Missing TPU hardware evidence."})
            return
        mp = list(raw.rglob("v23_metrics.json"))
        sp = list(raw.rglob("candidate_submission.csv"))
        if not mp or not sp:
            status.update({"stage": "artifacts_missing", "error": "Missing metrics or submission file."})
            return
        metrics = json.loads(mp[0].read_text(encoding="utf-8"))
        required_cv = "StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)"
        if (metrics.get("backend") != "tpu" or metrics.get("validation") != required_cv
                or int(metrics.get("train_rows", -1)) != int(base["metrics"]["rows"])
                or int(metrics.get("folds", 0)) != 3):
            status.update({"stage": "incomparable_results", "error": "TPU hardware or CV settings do not match."})
            return
        validate_submission(sp[0], metrics["target"], int(metrics["test_rows"]))
        status["metrics"] = metrics
        write_json(results / "metrics.json", metrics)
        gate = check_cv(
            metrics["cv_auc"], old_cv, float(config.get("min_cv_improvement", 0.00010))
        )
        status["cv_gate"] = gate
        if not gate["passed"]:
            status["stage"] = "skipped_no_cv_gain"
            status["submitted"] = False
            return
        if not config.get("auto_submit", True):
            status["stage"] = "awaiting_submission_approval"
            return
        note = f"KAS V2.3 JAX-TPU CV={metrics['cv_auc']:.6f}"
        latest = latest_submission(slug)
        if not latest or note not in str(latest.get("description", "")):
            status["stage"] = "submitting"
            write_json(STATE, status)
            response = run([
                "kaggle", "competitions", "submit", slug, "-f", str(sp[0]),
                "-m", note, "--wait", "600", "--poll-interval", "30",
            ], check=False, timeout=900)
            (results / "submit_output.txt").write_text(
                (response.stdout or "") + "\n" + (response.stderr or ""), encoding="utf-8"
            )
            if response.returncode:
                status.update({"stage": "submission_failed", "error": (response.stderr or response.stdout)[-7000:]})
                return
            latest = latest_submission(slug)
        if not latest or note not in str(latest.get("description", "")):
            status.update({"stage": "submission_unverified", "error": "Unverified Kaggle submission."})
            return
        status["submitted"] = True
        status["latest_submission"] = latest
        write_json(results / "latest_submission.json", latest)
        new_lb = latest.get("publicScore")
        if new_lb in (None, ""):
            status["stage"] = "submission_pending"
            return
        delta = round(float(new_lb) - float(old_lb), 8)
        status["public_lb_delta"] = delta
        if delta > 0:
            status["champion"] = "V2.3 JAX TPU MLP"
        status["stage"] = "complete"
    except Exception as exc:
        status.update({"stage": "controller_failed", "error": repr(exc)})
    finally:
        status["updated_at"] = now()
        write_json(STATE, status)
        write_report(status)
        print(json.dumps(status, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recover-only", action="store_true")
    args = parser.parse_args()
    main(recover_only=args.recover_only)
