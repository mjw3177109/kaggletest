"""V2.9.24 guarded pipeline: Personal Travel specialist residual correction.

One owner-verified CPU Notebook, 3-fold OOF, fixed alpha, no submission.
Existing Notebook handles are recovered only; terminal states are immutable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import v2923_pipeline as v23

ROOT = Path(__file__).resolve().parents[1]
SLUG = "playground-series-s6e10"
KERNEL = "mjw3177109/kas-v2924-personal-travel-specialist-cpu-s6e10"
STATE = ROOT / "state/v2_9_24_status.json"
V2923_STATE = ROOT / "state/v2_9_23_status.json"
REPORT = ROOT / "reports/v2_9_24_personal_travel_specialist.md"
WORKDIR = ROOT / "competitions" / SLUG / "kaggle_cpu_v2924_personal_travel"
OUTDIR = ROOT / "competitions" / SLUG / "results_v2924_personal_travel"
LEDGER = ROOT / "state/v2_9_experiments.json"

BASELINE = 0.9612850399332336
CV_FLOOR = 0.96125635576833
ALPHA = 0.10
MIN_GAIN = 0.00005
SEED = 20261018
MODEL_SEED = 20261032
VALIDATION = "StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)"
TERMINAL = {
    "candidate_ready_unsubmitted", "skipped_no_cv_gain", "kernel_failed",
    "output_failed", "validation_mismatch", "push_failed", "kernel_access_error",
}
SAFE_TO_ADVANCE_V2923 = {
    "skipped_no_cv_gain", "kernel_failed", "output_failed",
    "validation_mismatch", "push_failed", "kernel_access_error",
}


def read(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def cmd(parts, timeout=180, check=True):
    return v23.cmd(parts, timeout=timeout, check=check)


def residual_correct(base, specialist, mask, alpha=ALPHA):
    """Apply a fixed within-subgroup rank residual to baseline scores only on mask."""
    base = np.asarray(base, dtype=np.float64)
    specialist = np.asarray(specialist, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if base.ndim != 1 or mask.shape != base.shape:
        raise ValueError("Baseline and subgroup mask must be aligned one-dimensional arrays")
    if specialist.shape != (int(mask.sum()),):
        raise ValueError("Specialist scores must contain exactly the masked subgroup rows")
    if not np.isfinite(base).all() or not np.isfinite(specialist).all():
        raise ValueError("Residual correction inputs must be finite")
    if not (0.0 < float(alpha) <= 0.25):
        raise ValueError("Residual alpha is outside the pre-registered safe range")
    result = base.copy()
    result[mask] = base[mask] + float(alpha) * (
        v23.audit.rank(specialist) - v23.audit.rank(base[mask])
    )
    if not np.isfinite(result).all():
        raise ValueError("Residual correction produced non-finite values")
    return result


def preregistered_v2923_is_terminal(state):
    stage = state.get("stage")
    message = str(state.get("kernel_message") or "")
    if stage == "skipped_no_cv_gain":
        return (
            state.get("kernel_state") == "kernel_complete"
            and "KernelWorkerStatus.COMPLETE" in message
            and "KernelWorkerStatus.RUNNING" not in message
        )
    if stage == "kernel_failed":
        return (
            state.get("kernel_state") == "kernel_failed"
            and any(token in message for token in ("KernelWorkerStatus.ERROR", "KernelWorkerStatus.FAILED", "KernelWorkerStatus.CANCELLED"))
            and "KernelWorkerStatus.RUNNING" not in message
        )
    if stage in {"output_failed", "validation_mismatch"}:
        return (
            state.get("kernel_state") == "kernel_complete"
            and "KernelWorkerStatus.COMPLETE" in message
            and "KernelWorkerStatus.RUNNING" not in message
        )
    # Ambiguous push/access failures are not proof that a Notebook is terminal.
    return False

def preflight():
    champion = v23.preflight()
    official = read(ROOT / "state/v2_9_15_submission.json")
    ranking = read(ROOT / "state/score_rank_latest.json")
    ladder = read(ROOT / "state/dual_ladder.json")
    v2923 = read(V2923_STATE)
    if not preregistered_v2923_is_terminal(v2923):
        raise RuntimeError("V2924 is blocked until the prior V2923 Notebook is confirmed terminal")
    if v2923.get("stage") == "candidate_ready_unsubmitted":
        raise RuntimeError("V2923 candidate is awaiting evaluation; do not advance")
    if v2923.get("submitted") is not False:
        raise RuntimeError("V2923 must remain non-submitting")
    comparison = v2923.get("comparison") or {}
    if v2923.get("stage") == "skipped_no_cv_gain" and float(comparison.get("gain", 1)) >= 0:
        raise RuntimeError("V2923 negative-gain terminal evidence changed")
    try:
        deadline = datetime.fromisoformat(
            str(ladder.get("active_deadline", "")).replace("Z", "+00:00")
        )
        active = deadline.tzinfo is not None and datetime.now(timezone.utc) < deadline
    except (TypeError, ValueError):
        active = False
    sub = official.get("submission") or {}
    score = ranking.get("score") or {}
    if not (
        active
        and ladder.get("active_competition") == SLUG
        and official.get("stage") == "complete"
        and official.get("submitted") is True
        and official.get("submission_attempted") is True
        and str(sub.get("ref")) == "57446286"
        and sub.get("status") == "SubmissionStatus.COMPLETE"
        and abs(float(official.get("official_lb", 0)) - 0.96099) < 1e-9
        and official.get("prediction_hash") == champion.get("candidate_test_hash")
        and str(score.get("submission_id")) == "57446286"
        and abs(float(score.get("public_score", 0)) - 0.96099) < 1e-9
    ):
        raise RuntimeError("V2924 champion/official-score/deadline gate failed")
    return champion


def persist(state):
    state["updated_at"] = now()
    v23.write(STATE, state)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    m = state.get("metrics") or {}
    c = state.get("comparison") or {}
    REPORT.write_text(
        "# V2.9.24 Personal Travel specialist residual-correction audit\n\n"
        f"- UTC: {state['updated_at']}\n"
        f"- Stage: {state.get('stage')}\n"
        f"- Notebook: https://www.kaggle.com/code/{state.get('kernel') or KERNEL}\n"
        f"- Frozen champion: V2.9.15, CV {BASELINE:.14f}, official Public LB 0.96099 (submission 57446286)\n"
        f"- Specialist: Personal Travel only; LightGBM; 66 features; 3 outer folds; 5-fold inner TargetEncoder\n"
        f"- Personal subgroup OOF AUC: {m.get('personal_oof_auc', 'pending')}\n"
        f"- Fixed residual alpha: {ALPHA}\n"
        f"- Candidate full OOF AUC: {c.get('candidate_auc', 'pending')}\n"
        f"- Candidate gain vs champion: {c.get('gain', 'pending')}\n"
        f"- Fold deltas: {c.get('fold_gain', 'pending')}\n"
        f"- Promotion: CV >= {CV_FLOOR}, gain >= {MIN_GAIN}, at least 2 positive folds, worst fold > -0.00005\n"
        "- External labels: none. Submission: never performed by this evaluator.\n"
        + (f"- Error: {state['error']}\n" if state.get("error") else ""),
        encoding="utf-8",
    )


def existing_or_absent():
    status = cmd(["kaggle", "kernels", "status", KERNEL], timeout=90, check=False)
    if status.returncode == 0:
        return True
    mine = cmd(
        ["kaggle", "kernels", "list", "--mine", "--search",
         KERNEL.split("/", 1)[1], "--page-size", "50", "-v"],
        timeout=120, check=False,
    )
    if mine.returncode:
        raise RuntimeError("Authenticated owner list unavailable: refusing Notebook push")
    if KERNEL.lower() in mine.stdout.lower():
        return True
    if mine.stdout.strip().lower() != "not found":
        raise RuntimeError("Ambiguous owner listing: refusing Notebook push")
    witness = cmd(
        ["kaggle", "kernels", "status", v23.v15.KERNEL],
        timeout=90, check=False,
    )
    if witness.returncode or "KernelWorkerStatus." not in witness.stdout:
        raise RuntimeError("Known owner Notebook control failed: refusing Notebook push")
    return False


def poll_kernel_strict(handle, wait_seconds=300, interval=20):
    """Parse the explicit current status token; never infer completion from a substring."""
    stop = time.monotonic() + wait_seconds
    last = ""
    while time.monotonic() < stop:
        p = cmd(["kaggle", "kernels", "status", handle], timeout=90, check=False)
        last = ((p.stdout or "") + "\n" + (p.stderr or "")).strip()
        low = last.lower()
        if p.returncode or "permission" in low or "denied" in low:
            return "kernel_access_error", last
        tokens = re.findall(r"KernelWorkerStatus\.([A-Z_]+)", last)
        token = tokens[-1] if tokens else None
        if token is None:
            match = re.search(
                r'has status\s+["\'](?:KernelWorkerStatus\.)?([A-Z_]+)',
                last, re.I,
            )
            token = match.group(1).upper() if match else None
        if token == "QUEUED":
            return "kernel_queued", last
        if token in {"RUNNING", "INITIALIZED", "RESTARTING", "CANCEL_REQUESTED"}:
            time.sleep(interval)
            continue
        if token in {"COMPLETE", "SUCCEEDED", "SUCCESS"}:
            return "kernel_complete", last
        if token in {"ERROR", "FAILED", "CANCELLED", "CANCELED"}:
            return "kernel_failed", last
        return "kernel_access_error", last or "Unrecognized Kaggle status response"
    return "kernel_running", last


def baseline_arrays(tempdir, champion):
    older = {
        "v25": read(ROOT / "state/v2_5_status.json"),
        "v28": read(ROOT / "state/v2_8_status.json"),
        "v296": read(ROOT / "state/v2_9_6_status.json"),
        "v297": read(ROOT / "state/v2_9_7_status.json"),
    }
    v23.audit.check_provenance(older)
    train, test, data, files = v23.audit.download_inputs(tempdir, older)
    train_id, test_id, y = v23.audit.validate_official(train, test)
    replay = v23.audit.evaluate(train, test, data, older, audit_legacy_order=True)
    if abs(float(replay["candidate_cv"]) - 0.9610183558267038) > 0.000002:
        raise ValueError("V298 archived benchmark changed")
    teacher_oof, teacher_test, teacher_meta = v23.teacher.get_teacher(
        tempdir, read(ROOT / "state/v2_9_4_status.json"), train_id, test_id, y
    )
    files["v294"] = teacher_meta
    outputs = {}
    for version, kernel in (
        ("v2913", v23.v13.KERNEL),
        ("v2914", v23.v14.KERNEL),
        ("v2915", v23.v15.KERNEL),
    ):
        folder = Path(tempdir) / version
        folder.mkdir(parents=True, exist_ok=True)
        p = v23.cmd(
            ["kaggle", "kernels", "output", kernel, "-p", str(folder), "-o", "-q"],
            timeout=600, check=False,
        )
        if p.returncode:
            raise RuntimeError(f"Could not retrieve frozen {version} output: {(p.stderr or p.stdout)[-1000:]}")
        if version == "v2915":
            file, arrays = v23.v15.read_new(folder)
            train_key, test_key = "oof_realmlp_5epoch", "pred_realmlp_5epoch"
        else:
            file, arrays, train_key, test_key = v23.v14.read_oof(folder, version)
        files[version] = {"notebook": kernel, "sha256": v23.audit.sha256(file)}
        outputs[version] = v23.v14.align_new(
            arrays, train_key, test_key, train_id, test_id, y, version
        )
    prior_files = champion.get("source_artifacts") or {}
    if set(files) != set(prior_files):
        raise ValueError("V2915 frozen source artifact inventory changed")
    for key, fields in prior_files.items():
        for field, value in fields.items():
            if files[key].get(field) != value:
                raise ValueError("Historical Kaggle source SHA changed: " + key + "/" + field)

    old_oof, old_test = v23.verify_sources(
        data["v25"], data["v28"], older["v25"], older["v28"],
        data["v25_submission"], data["v28_submission"],
    )[1:]
    prior_history = 0.35 * v23.audit.rank(old_oof["v25"]) + 0.65 * v23.audit.rank(old_oof["v28"])
    prior_history_test = 0.35 * old_test["v25"] + 0.65 * old_test["v28"]
    real_oof = v23.audit.align(data["v296"]["oof_realmlp"], data["v296"]["train_id"], train_id, "V296")
    real_test = v23.audit.align(data["v296"]["pred_realmlp"], data["v296"]["test_id"], test_id, "V296 test")
    conditional_oof = v23.audit.align(data["v297"]["oof_conditional"], data["v297"]["train_id"], train_id, "V297")
    conditional_test = v23.audit.align(data["v297"]["test_conditional"], data["v297"]["test_id"], test_id, "V297 test")
    historical = v23.teacher.fuse_fixed(
        y, prior_history, real_oof, conditional_oof, teacher_oof, v23.audit.EXPECTED_V296
    )
    if abs(float(historical["candidate_cv"]) - 0.9610561936366048) > 0.000002:
        raise ValueError("V299 historical OOF changed")
    v299_oof = (
        0.60 * v23.audit.rank(prior_history)
        + 0.20 * v23.audit.rank(real_oof)
        + 0.15 * v23.audit.rank(conditional_oof)
        + 0.05 * v23.audit.rank(teacher_oof)
    )
    v2913_oof = 0.85 * v23.audit.rank(v299_oof) + 0.15 * v23.audit.rank(outputs["v2913"][0])
    v15_result = v23.v15.evaluate_blend(y, v2913_oof, outputs["v2914"][0], outputs["v2915"][0])
    if abs(float(v15_result["candidate_auc"]) - BASELINE) > 0.000002:
        raise ValueError("V2915 frozen champion OOF baseline did not reproduce")
    baseline_oof = (
        0.80 * v23.audit.rank(v2913_oof)
        + 0.10 * v23.audit.rank(outputs["v2914"][0])
        + 0.10 * v23.audit.rank(outputs["v2915"][0])
    )
    if abs(float(roc_auc_score(y, baseline_oof)) - BASELINE) > 0.000002:
        raise ValueError("V2915 reconstructed OOF score mismatch")
    v2913_test = 0.85 * v23.audit.rank(
        0.60 * v23.audit.rank(prior_history_test)
        + 0.20 * v23.audit.rank(real_test)
        + 0.15 * v23.audit.rank(conditional_test)
        + 0.05 * v23.audit.rank(teacher_test)
    ) + 0.15 * v23.audit.rank(outputs["v2913"][1])
    baseline_test = (
        0.80 * v23.audit.rank(v2913_test)
        + 0.10 * v23.audit.rank(outputs["v2914"][1])
        + 0.10 * v23.audit.rank(outputs["v2915"][1])
    )
    baseline_test_hash = hashlib.sha256(baseline_test.astype("<f8").tobytes()).hexdigest()
    if baseline_test_hash != champion.get("candidate_test_hash"):
        raise ValueError("Reconstructed V2915 frozen test prediction hash changed")
    official = read(ROOT / "state/v2_9_15_submission.json")
    if baseline_test_hash != official.get("prediction_hash"):
        raise ValueError("Reconstructed V2915 test hash differs from official submission reservation")
    return train, test, train_id, test_id, y, baseline_oof, baseline_test, files


def load_specialist(raw, train_id, test_id, y, train, test):
    metrics_file = v23.unique(raw, "v2924_metrics.json")
    pred_file = v23.unique(raw, "v2924_personal_travel_oof.npz")
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    expected = {
        "competition": SLUG,
        "stage": "three_folds_complete",
        "model": "Personal-travel specialist LightGBM TE23 route-profile interactions 3fold seeded",
        "validation": VALIDATION,
        "folds": 3,
        "train_rows": 699635,
        "test_rows": 299844,
        "model_seed": MODEL_SEED,
        "feature_family": "personal_travel_te23_route_interactions_v1",
        "specialist_group": "Personal Travel",
        "group_only_training": True,
        "interaction_feature_count": 10,
        "total_feature_count": 66,
        "route_profile_feature_count": 12,
        "target_encoding_columns": 23,
        "inner_target_folds": 5,
        "cpu_only": True,
        "gpu_verified": False,
        "data_source": "competition_train_only",
        "external_labels": False,
        "submitted": False,
        "leakage_safe_inner_cross_fit": True,
        "outer_validation_used_in_training": False,
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise ValueError(f"V2924 provenance mismatch for {key}")
    if metrics.get("personal_train_rows", 0) < 150000 or metrics.get("personal_test_rows", 0) < 50000:
        raise ValueError("V2924 Personal Travel subgroup unexpectedly small")
    with np.load(pred_file, allow_pickle=False) as f:
        required = {"train_id", "target", "oof_personal", "train_mask", "test_id", "pred_personal", "test_mask"}
        if not required.issubset(f.files):
            raise ValueError("V2924 canonical OOF/test arrays missing")
        arrays = {key: np.asarray(f[key]).copy() for key in required}
    if not np.array_equal(arrays["train_id"], train_id) or not np.array_equal(arrays["test_id"], test_id):
        raise ValueError("V2924 official train/test IDs or ordering changed")
    if not np.array_equal(arrays["target"], y):
        raise ValueError("V2924 output target differs from official training target")
    expected_train_mask = train["Type of Travel"].astype("string").fillna("__NA__").eq("Personal Travel").to_numpy()
    expected_test_mask = test["Type of Travel"].astype("string").fillna("__NA__").eq("Personal Travel").to_numpy()
    train_mask = arrays["train_mask"].astype(bool)
    test_mask = arrays["test_mask"].astype(bool)
    if not np.array_equal(train_mask, expected_train_mask) or not np.array_equal(test_mask, expected_test_mask):
        raise ValueError("V2924 specialist subgroup mask mismatch")
    oof = arrays["oof_personal"].astype(np.float64)
    pred_test = arrays["pred_personal"].astype(np.float64)
    if not np.array_equal(np.isfinite(oof), train_mask) or not np.array_equal(np.isfinite(pred_test), test_mask):
        raise ValueError("V2924 OOF/test finite masks do not exactly match Personal Travel")
    if not ((oof[train_mask] >= 0).all() and (oof[train_mask] <= 1).all()
            and (pred_test[test_mask] >= 0).all() and (pred_test[test_mask] <= 1).all()):
        raise ValueError("V2924 probabilities fall outside [0,1]")
    if abs(float(roc_auc_score(y[train_mask], oof[train_mask])) - float(metrics.get("personal_oof_auc", 0))) > 0.000002:
        raise ValueError("V2924 personal subgroup OOF AUC did not reproduce")
    folds = metrics.get("fold_results") or []
    if len(folds) != 3:
        raise ValueError("V2924 requires exactly three fold results")
    for fold, (_, valid_idx) in enumerate(StratifiedKFold(
        n_splits=3, shuffle=True, random_state=SEED
    ).split(np.zeros(len(y), dtype=np.uint8), y)):
        idx = valid_idx[train_mask[valid_idx]]
        result = folds[fold]
        if result.get("fold") != fold or result.get("validation_rows") != len(idx):
            raise ValueError("V2924 fold membership or row count changed")
        if abs(float(result.get("auc", 0)) - float(roc_auc_score(y[idx], oof[idx]))) > 0.000002:
            raise ValueError("V2924 per-fold specialist AUC did not reproduce")
    return metrics, pred_file, oof[train_mask], pred_test[test_mask], train_mask, test_mask


def validate(raw, state):
    champion = preflight()
    with tempfile.TemporaryDirectory(prefix="kas_v2924_verify_") as tempdir:
        train, test, train_id, test_id, y, baseline_oof, baseline_test, source_files = baseline_arrays(tempdir, champion)
        metrics, pred_file, specialist_oof, specialist_test, train_mask, test_mask = load_specialist(
            raw, train_id, test_id, y, train, test
        )
        candidate_oof = residual_correct(baseline_oof, specialist_oof, train_mask)
        candidate_auc = float(roc_auc_score(y, candidate_oof))
        fold_gain = []
        for _, idx in StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(
            np.zeros(len(y), dtype=np.uint8), y
        ):
            fold_gain.append(float(
                roc_auc_score(y[idx], candidate_oof[idx]) - roc_auc_score(y[idx], baseline_oof[idx])
            ))
        comparison = {
            "baseline_auc": float(roc_auc_score(y, baseline_oof)),
            "standalone_personal_auc": float(metrics["personal_oof_auc"]),
            "candidate_auc": candidate_auc,
            "gain": candidate_auc - BASELINE,
            "fold_gain": fold_gain,
            "positive_folds": sum(delta > 0 for delta in fold_gain),
            "fixed_residual_alpha": ALPHA,
            "promotion_floor": CV_FLOOR,
            "min_gain": MIN_GAIN,
        }
        promoted = (
            candidate_auc >= CV_FLOOR
            and comparison["gain"] >= MIN_GAIN
            and comparison["positive_folds"] >= 2
            and min(fold_gain) > -0.00005
        )
        state.update({
            "metrics": metrics,
            "comparison": comparison,
            "source_artifacts": {
                **source_files,
                "v2924": {"notebook": state["kernel"], "sha256": v23.audit.sha256(pred_file)},
            },
            "stage": "candidate_ready_unsubmitted" if promoted else "skipped_no_cv_gain",
            "submitted": False,
            "submission_attempted": False,
        })
        if promoted:
            candidate_test = residual_correct(baseline_test, specialist_test, test_mask)
            if len(candidate_test) != len(test_id) or not np.isfinite(candidate_test).all():
                raise ValueError("V2924 candidate test vector invalid")
            digest = hashlib.sha256(candidate_test.astype("<f8").tobytes()).hexdigest()
            state["candidate_test_hash"] = digest
            OUTDIR.mkdir(parents=True, exist_ok=True)
            candidate_csv = OUTDIR / "candidate_submission.csv"
            pd.DataFrame({"id": test_id, "satisfaction": candidate_test}).to_csv(
                candidate_csv, index=False
            )
            state["candidate_file_sha256"] = v23.audit.sha256(candidate_csv)
            state["candidate_file"] = str(candidate_csv.relative_to(ROOT))
        else:
            candidate_csv = OUTDIR / "candidate_submission.csv"
            if candidate_csv.exists():
                candidate_csv.unlink()
        v23.record_experiment(LEDGER, {
            "competition": SLUG,
            "candidate": "v2924_personal_travel_residual_correction",
            "validation": VALIDATION,
            "seed": SEED,
            "parameters": {
                "model_seed": MODEL_SEED,
                "model": "LightGBM_personal_travel_specialist",
                "feature_family": "personal_travel_te23_route_interactions_v1",
                "n_estimators": 1000,
                "residual_alpha": ALPHA,
                "min_gain": MIN_GAIN,
                "cv_floor": CV_FLOOR,
            },
            "status": "completed" if promoted else "skipped_no_cv_gain",
            "completed_at": now(),
            "cv": candidate_auc,
            "cpu_seconds": metrics.get("runtime_seconds"),
            "gpu_seconds": 0,
            "kernel_sources": [KERNEL],
        })
    return state


def run(recover_only=False):
    previous = read(STATE)
    if previous.get("stage") in TERMINAL:
        print("V2924 terminal; never create another Notebook or submit")
        return previous
    try:
        preflight()
    except Exception as exc:
        state = previous or {"competition": SLUG, "stage": "preflight_blocked", "kernel": None, "submitted": False}
        state.update({"stage": "preflight_blocked", "error": str(exc)[-1800:]})
        persist(state)
        return state
    state = previous or {"competition": SLUG, "stage": "preflight", "kernel": None, "submitted": False}
    if not state.get("kernel"):
        if recover_only:
            print("Recovery-only forbids creating an unrecorded Notebook")
            return state
        try:
            exists = existing_or_absent()
        except Exception as exc:
            state.update({"stage": "kernel_access_error", "error": str(exc)[-1200:]})
            persist(state)
            return state
        if exists:
            state.update({"stage": "kernel_queued", "kernel": KERNEL, "preexisting_kernel_recovered": True})
            persist(state)
    if not state.get("kernel"):
        WORKDIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "kaggle_agent/v2924_personal_travel_specialist_train.py", WORKDIR / "main.py")
        dump(WORKDIR / "kernel-metadata.json", {
            "id": KERNEL,
            "title": KERNEL.split("/", 1)[1],
            "code_file": "main.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": False,
            "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": [SLUG],
            "kernel_sources": [],
            "model_sources": [],
        })
        state.update({"stage": "pushing_kernel", "kernel": KERNEL, "submitted": False})
        persist(state)
        push = cmd(["kaggle", "kernels", "push", "-p", str(WORKDIR), "-t", "3600"], timeout=300, check=False)
        if push.returncode:
            state.update({"stage": "push_failed", "error": (push.stderr or push.stdout)[-1500:]})
            persist(state)
            return state
        match = re.search(
            r"https?://(?:www\.)?kaggle\.com/code/([A-Za-z0-9_-]+)/([a-z0-9-]+)",
            (push.stdout or "") + "\n" + (push.stderr or ""),
        )
        if not match or match.group(1).lower() != "mjw3177109":
            state.update({"stage": "kernel_access_error", "error": "Kaggle push returned no canonical owned Notebook URL"})
            persist(state)
            return state
        actual = "/".join(match.groups())
        if actual != KERNEL:
            state.update({"stage": "kernel_access_error", "kernel": actual,
                          "error": "Kaggle canonical Notebook slug differs; never repush"})
            persist(state)
            return state
        persist(state)
    handle = state.get("kernel")
    if not isinstance(handle, str) or handle != KERNEL or not handle.startswith("mjw3177109/"):
        state.update({"stage": "kernel_access_error", "error": "Untrusted or unexpected Notebook ownership/slug"})
        persist(state)
        return state
    status, message = poll_kernel_strict(handle, 90 if recover_only else 300, 20)
    state["kernel_state"] = status
    state["kernel_message"] = str(message)[-1200:]
    if status in {"kernel_queued", "kernel_running", "kernel_access_error"}:
        state["stage"] = status
        persist(state)
        return state
    raw = OUTDIR / "raw"
    if raw.exists():
        shutil.rmtree(raw)
    raw.mkdir(parents=True, exist_ok=True)
    output = cmd(["kaggle", "kernels", "output", handle, "-p", str(raw), "-o", "-q"], timeout=600, check=False)
    if status != "kernel_complete" or output.returncode:
        errors = list(raw.rglob("v2924_error.json"))
        state.update({
            "stage": "kernel_failed" if status != "kernel_complete" else "output_failed",
            "error": errors[0].read_text(encoding="utf-8")[-2000:] if errors else str(output.stderr or message)[-2000:],
        })
        persist(state)
        return state
    try:
        state = validate(raw, state)
    except Exception as exc:
        state.update({"stage": "validation_mismatch", "error": str(exc)[-2000:]})
        persist(state)
        raise
    persist(state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recover-only", action="store_true")
    args = parser.parse_args()
    result = run(recover_only=args.recover_only)
    print(json.dumps({
        "stage": result.get("stage"),
        "kernel": result.get("kernel"),
        "comparison": result.get("comparison"),
        "error": result.get("error"),
    }, ensure_ascii=False, indent=2))
