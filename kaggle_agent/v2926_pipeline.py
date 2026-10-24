"""V2.9.26 guarded one-Notebook evaluator for route-profile RealMLP-TD.
Run mode may create only the unique preregistered owner Notebook.
Recovery mode NEVER creates or pushes a Notebook. No submission API calls here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import v2924_pipeline as v24

ROOT = Path(__file__).resolve().parents[1]
SLUG = "playground-series-s6e10"
KERNEL = "mjw3177109/kas-v2926-route-profile-realmlp-t4-s6e10"
STATUS = ROOT / "state/v2_9_26_status.json"
REPORT = ROOT / "reports/v2_9_26_result.md"
WORKDIR = ROOT / "competitions" / SLUG / "kaggle_gpu_v2926_route_profile"
OUTDIR = ROOT / "competitions" / SLUG / "results_v2926_route_profile"
LEDGER = ROOT / "state/v2_9_experiments.json"
BASELINE = 0.9612850399332336
FLOOR = 0.96125635576833
MIN_GAIN = 0.00005
SEED = 20261018
MODEL_SEED = 20261026
NEW_WEIGHT = 0.20
TERMINAL = {"candidate_ready_unsubmitted", "skipped_no_cv_gain", "kernel_failed",
            "output_failed", "validation_mismatch", "push_failed", "kernel_access_error"}


def read(path):
    return v24.read(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def verify_prereg(s):
    if (s.get("competition") != SLUG
        or s.get("version") != "V2.9.26"
        or s.get("intended_kernel") != KERNEL
        or s.get("method") != "fold_local_route_profile_realmlp_5epoch_t4"
        or s.get("seed") != SEED or s.get("model_seed") != MODEL_SEED
        or s.get("outer_folds") != 3 or s.get("epochs") != 5
        or s.get("n_ens") != 8 or s.get("route_feature_count") != 12
        or s.get("route_statistics_use_outer_train_only") is not True
        or s.get("external_labels") is not False
        or s.get("blend") != {"v2915_rank": 0.8, "v2926_rank": NEW_WEIGHT}
        or s.get("baseline_cv") != BASELINE
        or s.get("promotion") != {
            "min_cv": FLOOR, "min_gain_vs_v2915": MIN_GAIN,
            "min_positive_folds": 2, "worst_fold_strict_gt": -0.00005
        }
        or s.get("submission_policy") != "evaluation_only_no_submission"
        or s.get("submitted") is not False
        or s.get("submission_attempted") is not False
        or s.get("stage") == "preregistered_not_started"
        and (s.get("kernel") or s.get("notebook_created") is not False)):
        raise RuntimeError("V2926 frozen preregistration/owner/anti-submit gate failed")
    if s.get("stage") in TERMINAL:
        return False
    return True


def preflight():
    s = read(STATUS)
    if not verify_prereg(s):
        return None
    champion = v24.preflight()  # Also verifies official submitted ID, SHA, old CV and deadline.
    previous = read(ROOT / "state/v2_9_25_status.json")
    if (previous.get("stage") != "skipped_no_cv_gain"
        or previous.get("submitted") is not False
        or previous.get("training_started") is not False
        or abs(float(previous.get("full_cv", 0)) - 0.961332372152315) > 1e-12
        or previous.get("promotable") is not False):
        raise RuntimeError("The last frozen OOF audit V2925 is not a verified terminal experiment")
    if champion.get("candidate_test_hash") != "906185547dcd5f0cac3ac173de644fd9997f20ac863c6745bcb1fd3e9a89e5f7":
        raise RuntimeError("V2915 frozen official prediction SHA changed")
    return s


def persist(s):
    s["updated_at"] = now()
    v24.dump(STATUS, s)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    c = s.get("comparison") or {}
    m = s.get("metrics") or {}
    REPORT.write_text(
        "# V2.9.26 fixed route-profile RealMLP threefold OOF\n\n"
        + f"- Updated UTC: {s['updated_at']}\n"
        + f"- Stage: {s.get('stage')}\n"
        + f"- Kaggle Notebook: https://www.kaggle.com/code/{KERNEL}\n"
        + f"- Frozen champion CV: {BASELINE}\n"
        + f"- New independent model full OOF: {m.get('full_oof_auc', 'pending')}\n"
        + f"- Frozen 80/20 fusion full OOF: {c.get('candidate_auc', 'pending')}\n"
        + f"- Candidate gain: {c.get('gain', 'pending')}\n"
        + f"- Paired fold gains: {c.get('fold_gain', 'pending')}\n"
        + f"- CV gain gate: >= {MIN_GAIN}; official Public LB stays 0.96099 until independently scored.\n"
        + "- No Kaggle competition submission is performed by this evaluator.\n"
        + (f"- Error: {s['error']}\n" if s.get("error") else ""),
        encoding="utf-8",
    )


def checked_owner_exists():
    # First authenticate exact canonical owner Notebook.
    q = v24.cmd(["kaggle", "kernels", "status", KERNEL], timeout=90, check=False)
    if q.returncode == 0 and "KernelWorkerStatus." in (q.stdout or ""):
        return True
    mine = v24.cmd(
        ["kaggle", "kernels", "list", "--mine", "--search",
         KERNEL.split("/", 1)[1], "--page-size", "50", "-v"],
        timeout=120, check=False,
    )
    if mine.returncode:
        raise RuntimeError("Kaggle owner list failed; fail closed, no Notebook push")
    if KERNEL.lower() in (mine.stdout or "").lower():
        return True
    if (mine.stdout or "").strip().lower() != "not found":
        raise RuntimeError("Ambiguous Kaggle Notebook listing; fail closed, no push")
    reference = v24.cmd(
        ["kaggle", "kernels", "status", v24.v23.v15.KERNEL],
        timeout=90, check=False,
    )
    if reference.returncode or "KernelWorkerStatus.COMPLETE" not in (reference.stdout or ""):
        raise RuntimeError("Known owner identity witness unavailable: no Notebook push")
    return False


def verify_metrics(metrics, y, oof):
    hardware = metrics.get("hardware") or {}
    if (metrics.get("competition") != SLUG
        or metrics.get("stage") != "three_folds_complete"
        or metrics.get("validation") != "StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)"
        or metrics.get("train_rows") != 699635 or metrics.get("test_rows") != 299844
        or metrics.get("folds") != 3 or len(metrics.get("fold_results") or []) != 3
        or metrics.get("model_seed") != MODEL_SEED
        or metrics.get("epochs") != 5 or metrics.get("route_feature_count") != 12
        or metrics.get("feature_family") != "fold_local_route_profile_realmlp_v1"
        or metrics.get("external_labels") is not False
        or metrics.get("submitted") is not False
        or metrics.get("gpu_verified") is not True
        or hardware.get("ok") is not True
        or not any("T4" in str(device) for device in hardware.get("devices", []))
        or metrics.get("leakage_safe_inner_cross_fit") is not True
        or metrics.get("outer_validation_used_in_training") is not False):
        raise ValueError("V2926 immutable GPU/3fold/feature provenance mismatch")
    for fold, (_, indices) in enumerate(StratifiedKFold(
        n_splits=3, shuffle=True, random_state=SEED
    ).split(np.zeros(len(y), dtype=np.uint8), y)):
        item = metrics["fold_results"][fold]
        if item.get("fold") != fold or item.get("validation_rows") != len(indices):
            raise ValueError("V2926 validation fold membership mismatch")
        measured = float(roc_auc_score(y[indices], oof[indices]))
        if abs(measured - float(item.get("auc", -1))) > 2e-6:
            raise ValueError("V2926 saved fold AUC failed independent replay")
    if abs(float(roc_auc_score(y, oof)) - float(metrics.get("full_oof_auc", -1))) > 2e-6:
        raise ValueError("V2926 saved full OOF failed independent replay")


def validate(raw, s):
    newfile = v24.v23.unique(raw, "v2926_realmlp_oof.npz")
    metrics = read(v24.v23.unique(raw, "v2926_metrics.json"))
    with np.load(newfile, allow_pickle=False) as f:
        keys = {"train_id", "target", "oof_route_realmlp", "test_id", "pred_route_realmlp"}
        if not keys.issubset(f.files):
            raise ValueError("Missing canonical V2926 train/test OOF keys")
        arrays = {k: np.asarray(f[k]).copy() for k in keys}
    champion = v24.preflight()
    with tempfile.TemporaryDirectory(prefix="kas_v2926_immutable_") as temporary:
        train, test, train_id, test_id, y, champion_oof, champion_test, sources = \
            v24.baseline_arrays(temporary, champion)
        # Do not assume row order. Proven ID + label alignment is compulsory.
        new_oof, new_test = v24.v23.v14.align_new(
            arrays, "oof_route_realmlp", "pred_route_realmlp",
            train_id, test_id, y, "V2926",
        )
        verify_metrics(metrics, y, new_oof)
        source_digest = v24.v23.audit.sha256(newfile)
        if len(source_digest) != 64:
            raise ValueError("Invalid new prediction archive hash")
        rank = v24.v23.audit.rank
        candidate_oof = 0.80 * rank(champion_oof) + NEW_WEIGHT * rank(new_oof)
        candidate_test = 0.80 * rank(champion_test) + NEW_WEIGHT * rank(new_test)
        if not np.isfinite(candidate_oof).all() or not np.isfinite(candidate_test).all():
            raise ValueError("Non-finite new OOF/test predictions")
        measured_baseline = float(roc_auc_score(y, champion_oof))
        if abs(measured_baseline - BASELINE) > 1e-12:
            raise ValueError("Frozen champion OOF score was not reproduced")
        score = float(roc_auc_score(y, candidate_oof))
        fold_gain = []
        fold_auc = []
        for _, indices in StratifiedKFold(3, shuffle=True, random_state=SEED).split(
            np.zeros(len(y), dtype=np.uint8), y
        ):
            a = float(roc_auc_score(y[indices], candidate_oof[indices]))
            fold_auc.append(a)
            fold_gain.append(a - float(roc_auc_score(y[indices], champion_oof[indices])))
        positive = sum(g > 0 for g in fold_gain)
        promoted = bool(score >= FLOOR and score - BASELINE >= MIN_GAIN
                        and positive >= 2 and min(fold_gain) > -0.00005)
        result = {
            "baseline_auc": measured_baseline,
            "standalone_auc": float(roc_auc_score(y, new_oof)),
            "candidate_auc": score,
            "gain": score - BASELINE,
            "fold_auc": fold_auc,
            "fold_gain": fold_gain,
            "positive_folds": positive,
            "fixed_weights": {"v2915_rank": 0.8, "v2926_rank": NEW_WEIGHT},
            "promotable": promoted,
        }
        s.update({
            "stage": "candidate_ready_unsubmitted" if promoted else "skipped_no_cv_gain",
            "metrics": metrics, "comparison": result,
            "source_artifacts": {**sources, "v2926": {"notebook": KERNEL, "sha256": source_digest}},
            "candidate_oof_hash": digest(candidate_oof),
            "candidate_test_hash": digest(candidate_test),
            "submitted": False, "submission_attempted": False,
        })
        v24.v23.record_experiment(LEDGER, {
            "competition": SLUG,
            "candidate": "v2926_route_profile_realmlp_5epoch",
            "validation": metrics["validation"], "seed": SEED,
            "parameters": {
                "model_seed": MODEL_SEED, "epochs": 5, "n_ens": 8,
                "route_feature_count": 12,
                "fusion_weights": dict(result["fixed_weights"]),
                "min_gain": MIN_GAIN,
            },
            "status": "completed" if promoted else "skipped_no_cv_gain",
            "cv": score, "completed_at": now(),
            "gpu_seconds": metrics.get("runtime_seconds"),
            "kernel_sources": [KERNEL],
        })
    return s


def run(recover_only=False):
    previous = read(STATUS)
    if previous.get("stage") in TERMINAL:
        print("V2926 terminal; no Notebook creation, recovery or submission")
        return previous
    s = preflight()
    if s is None:
        return previous
    if not s.get("kernel"):
        if recover_only:
            print("RECOVERY ONLY: refusing to create unrecorded Kaggle Notebook")
            return s
        try:
            exists = checked_owner_exists()
        except Exception as e:
            s.update({"stage": "kernel_access_error", "error": str(e)[-1400:]})
            persist(s)
            return s
        if exists:
            s.update({"kernel": KERNEL, "stage": "kernel_queued",
                      "preexisting_kernel_recovered": True})
            persist(s)
    if not s.get("kernel"):
        WORKDIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "kaggle_agent/v2926_route_realmlp_train.py", WORKDIR / "main.py")
        v24.dump(WORKDIR / "kernel-metadata.json", {
            "id": KERNEL, "title": KERNEL.split("/", 1)[1],
            "code_file": "main.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [], "competition_sources": [SLUG],
            "kernel_sources": [], "model_sources": [],
        })
        s.update({"stage": "pushing_kernel", "kernel": KERNEL, "notebook_created": False})
        persist(s)  # durable stopgap: any subsequent run MUST enter recovery-only.
        pushed = v24.cmd(["kaggle", "kernels", "push", "-p", str(WORKDIR), "-t", "3600"],
                         timeout=300, check=False)
        if pushed.returncode:
            s.update({"stage": "push_failed", "error": (pushed.stderr or pushed.stdout)[-1800:]})
            persist(s)
            return s
        match = re.search(
            r"https?://(?:www\.)?kaggle\.com/code/([a-z0-9_-]+)/([a-z0-9-]+)",
            (pushed.stdout or "") + "\n" + (pushed.stderr or ""), re.I,
        )
        if not match or "/".join(match.groups()).lower() != KERNEL:
            s.update({"stage": "kernel_access_error",
                      "error": "Canonical owner Notebook URL missing or changed; refuse second push"})
            persist(s)
            return s
        s.update({"notebook_created": True, "training_started": True})
        persist(s)
    if s.get("kernel") != KERNEL:
        raise RuntimeError("Saved Notebook ownership/slug mismatch")
    stage, message = v24.poll_kernel_strict(KERNEL, 90 if recover_only else 300, 20)
    s["kernel_state"] = stage
    s["kernel_message"] = message[-1600:]
    if stage in {"kernel_running", "kernel_queued", "kernel_access_error"}:
        s["stage"] = stage
        persist(s)
        return s
    raw = OUTDIR / "raw"
    if raw.exists():
        shutil.rmtree(raw)
    raw.mkdir(parents=True, exist_ok=True)
    got = v24.cmd(["kaggle", "kernels", "output", KERNEL, "-p", str(raw), "-o", "-q"],
                  timeout=600, check=False)
    if stage != "kernel_complete" or got.returncode:
        errors = list(raw.rglob("v2926_error.json"))
        s.update({"stage": "kernel_failed" if stage != "kernel_complete" else "output_failed",
                  "error": errors[0].read_text(encoding="utf-8")[-2000:] if errors else (got.stderr or message)[-1600:]})
        persist(s)
        return s
    try:
        validate(raw, s)
    except Exception as e:
        s.update({"stage": "validation_mismatch", "error": str(e)[-1800:]})
        persist(s)
        raise
    persist(s)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recover-only", action="store_true")
    args = parser.parse_args()
    result = run(recover_only=args.recover_only)
    print(json.dumps({"stage": result.get("stage"), "kernel": result.get("kernel"),
                      "comparison": result.get("comparison"), "error": result.get("error")},
                     ensure_ascii=False, indent=2))
