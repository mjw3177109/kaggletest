"""V2.9.27 guarded one-Notebook evaluator for travel-gated dual RealMLP-TD.
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
KERNEL = "mjw3177109/kas-v2927-travel-gated-dual-realmlp-t4-s6e10"
STATUS = ROOT / "state/v2_9_27_status.json"
REPORT = ROOT / "reports/v2_9_27_result.md"
WORKDIR = ROOT / "competitions" / SLUG / "kaggle_gpu_v2927_travel_gated"
OUTDIR = ROOT / "competitions" / SLUG / "results_v2927_travel_gated"
LEDGER = ROOT / "state/v2_9_experiments.json"
BASELINE = 0.9612850399332336
FLOOR = 0.96125635576833
MIN_GAIN = 0.00005
SEED = 20261018
MODEL_SEED = 20261027
NEW_WEIGHT = 0.15
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
        or s.get("version") != "V2.9.27"
        or s.get("intended_kernel") != KERNEL
        or s.get("hypothesis") != "travel_type_gated_dual_realmlp_interaction_experts"
        or s.get("model_family") != "two_fold_local_RealMLP_TD_travel_experts"
        or s.get("feature_family") != "v2916_service_delay_interactions_no_travel_gated"
        or s.get("seed") != SEED or s.get("model_seed") != MODEL_SEED
        or s.get("outer_folds") != 3 or s.get("epochs") != 5
        or s.get("n_ens") != 8 or s.get("interaction_feature_count") != 9
        or s.get("route_feature_count") != 0
        or s.get("experts") != ["Personal Travel", "Business travel"]
        or s.get("external_labels") is not False
        or s.get("blend") != {"v2915_rank": 0.85, "v2927_rank": NEW_WEIGHT}
        or s.get("baseline_cv") != BASELINE
        or s.get("promotion") != {
            "min_cv": FLOOR, "min_gain_vs_v2915": MIN_GAIN,
            "min_positive_folds": 2, "worst_fold_strict_gt": -0.00005
        }
        or s.get("submission_policy") != "evaluation_only_no_submission"
        or s.get("submitted") is not False
        or s.get("submission_attempted") is not False
        or (s.get("kernel") not in ("", None, KERNEL))
        or (s.get("stage") == "preregistered_not_started"
            and (s.get("kernel") or s.get("notebook_created") is not False))):
        raise RuntimeError("V2927 frozen preregistration/owner/anti-submit gate failed")
    return s.get("stage") not in TERMINAL


def preflight():
    s = read(STATUS)
    if not verify_prereg(s):
        return None
    champion = v24.preflight()  # Replays official submission, SHA and deadline gates.
    previous = read(ROOT / "state/v2_9_26_status.json")
    if (previous.get("stage") != "skipped_no_cv_gain"
        or previous.get("submitted") is not False
        or previous.get("kernel_state") != "kernel_complete"
        or abs(float((previous.get("comparison") or {}).get("candidate_auc", 0))
               - 0.9613072330727896) > 1e-12
        or (previous.get("source_artifacts") or {}).get("v2926", {}).get("sha256")
           != "1026cf4aa6990f7eb3fc9e9e5c21209cbe4ae8b6fd42e8398fb9f9a24853e2e2"):
        raise RuntimeError("V2926 terminal no-gain provenance changed; do not launch")
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
        "# V2.9.27 fixed travel-gated dual RealMLP threefold OOF\n\n"
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


def verify_metrics(metrics, y, oof, train, test):
    hardware = metrics.get("hardware") or {}
    experts = ["Personal Travel", "Business travel"]
    if (metrics.get("competition") != SLUG
        or metrics.get("stage") != "three_folds_complete"
        or metrics.get("validation") != "StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)"
        or metrics.get("train_rows") != 699635 or metrics.get("test_rows") != 299844
        or metrics.get("folds") != 3 or len(metrics.get("fold_results") or []) != 3
        or metrics.get("model_seed") != MODEL_SEED
        or metrics.get("epochs") != 5 or metrics.get("n_ens") != 8
        or metrics.get("interaction_feature_count") != 9
        or metrics.get("route_feature_count") != 0
        or metrics.get("feature_count") != 46
        or metrics.get("experts") != experts
        or metrics.get("feature_family") != "v2916_service_delay_interactions_no_travel_gated"
        or metrics.get("external_labels") is not False
        or metrics.get("submitted") is not False
        or metrics.get("gpu_verified") is not True
        or hardware.get("ok") is not True
        or not any("T4" in str(device) for device in hardware.get("devices", []))
        or metrics.get("leakage_safe_inner_cross_fit") is not True
        or metrics.get("outer_validation_used_in_training") is not False):
        raise ValueError("V2927 immutable GPU/3fold/feature provenance mismatch")
    if set(train["Type of Travel"].astype(str).unique()) != set(experts):
        raise ValueError("V2927 unexpected training gate categories")
    if set(test["Type of Travel"].astype(str).unique()) != set(experts):
        raise ValueError("V2927 unexpected test gate categories")
    for fold, (itr, indices) in enumerate(StratifiedKFold(
        n_splits=3, shuffle=True, random_state=SEED
    ).split(np.zeros(len(y), dtype=np.uint8), y)):
        item = metrics["fold_results"][fold]
        if item.get("fold") != fold or item.get("validation_rows") != len(indices):
            raise ValueError("V2927 validation fold membership mismatch")
        if len(item.get("experts") or []) != 2:
            raise ValueError("V2927 expected two independently trained experts per fold")
        for expert_index, group in enumerate(experts):
            m = item["experts"][expert_index]
            train_mask = train["Type of Travel"].astype(str).eq(group).to_numpy()
            test_mask = test["Type of Travel"].astype(str).eq(group).to_numpy()
            valid_idx = indices[train_mask[indices]]
            train_idx = itr[train_mask[itr]]
            if (m.get("expert") != group or m.get("expert_index") != expert_index
                or m.get("model_seed") != MODEL_SEED + 2*fold + expert_index
                or m.get("train_rows") != len(train_idx)
                or m.get("validation_rows") != len(valid_idx)
                or m.get("test_rows") != int(test_mask.sum())
                or m.get("features") != 46 or m.get("inner_target_folds") != 5):
                raise ValueError("V2927 expert membership/seed/features mismatch")
            measured = float(roc_auc_score(y[valid_idx], oof[valid_idx]))
            if abs(measured - float(m.get("auc", -1))) > 2e-6:
                raise ValueError("V2927 subgroup fold AUC failed replay")
        measured = float(roc_auc_score(y[indices], oof[indices]))
        if abs(measured - float(item.get("auc", -1))) > 2e-6:
            raise ValueError("V2927 saved fold AUC failed independent replay")
    if abs(float(roc_auc_score(y, oof)) - float(metrics.get("full_oof_auc", -1))) > 2e-6:
        raise ValueError("V2927 full OOF AUC failed independent replay")
    for group in experts:
        mask = train["Type of Travel"].astype(str).eq(group).to_numpy()
        measured = float(roc_auc_score(y[mask], oof[mask]))
        if abs(measured - float((metrics.get("group_auc") or {}).get(group, -1))) > 2e-6:
            raise ValueError("V2927 group AUC failed independent replay")


def validate(raw, s):
    newfile = v24.v23.unique(raw, "v2927_realmlp_oof.npz")
    metrics = read(v24.v23.unique(raw, "v2927_metrics.json"))
    with np.load(newfile, allow_pickle=False) as f:
        keys = {"train_id", "target", "oof_gated_realmlp", "test_id", "pred_gated_realmlp", "train_travel_type", "test_travel_type"}
        if not keys.issubset(f.files):
            raise ValueError("Missing canonical V2927 train/test OOF keys")
        arrays = {k: np.asarray(f[k]).copy() for k in keys}
    champion = v24.preflight()
    with tempfile.TemporaryDirectory(prefix="kas_v2927_immutable_") as temporary:
        train, test, train_id, test_id, y, champion_oof, champion_test, sources = \
            v24.baseline_arrays(temporary, champion)
        # Do not assume row order. Proven ID + label alignment is compulsory.
        new_oof, new_test = v24.v23.v14.align_new(
            arrays, "oof_gated_realmlp", "pred_gated_realmlp",
            train_id, test_id, y, "V2927",
        )
        if not np.array_equal(arrays["train_travel_type"].astype(str),
                              train["Type of Travel"].astype(str).to_numpy()):
            raise ValueError("V2927 saved training travel gate differs from canonical competition")
        if not np.array_equal(arrays["test_travel_type"].astype(str),
                              test["Type of Travel"].astype(str).to_numpy()):
            raise ValueError("V2927 saved test travel gate differs from canonical competition")
        verify_metrics(metrics, y, new_oof, train, test)
        source_digest = v24.v23.audit.sha256(newfile)
        if len(source_digest) != 64:
            raise ValueError("Invalid new prediction archive hash")
        rank = v24.v23.audit.rank
        candidate_oof = 0.85 * rank(champion_oof) + NEW_WEIGHT * rank(new_oof)
        candidate_test = 0.85 * rank(champion_test) + NEW_WEIGHT * rank(new_test)
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
            "fixed_weights": {"v2915_rank": 0.85, "v2927_rank": NEW_WEIGHT},
            "promotable": promoted,
        }
        s.update({
            "stage": "candidate_ready_unsubmitted" if promoted else "skipped_no_cv_gain",
            "metrics": metrics, "comparison": result,
            "source_artifacts": {**sources, "v2927": {"notebook": KERNEL, "sha256": source_digest}},
            "candidate_oof_hash": digest(candidate_oof),
            "candidate_test_hash": digest(candidate_test),
            "submitted": False, "submission_attempted": False,
        })
        v24.v23.record_experiment(LEDGER, {
            "competition": SLUG,
            "candidate": "v2927_travel_gated_realmlp_5epoch",
            "validation": metrics["validation"], "seed": SEED,
            "parameters": {
                "model_seed": MODEL_SEED, "epochs": 5, "n_ens": 8,
                "interaction_feature_count": 9, "route_feature_count": 0,
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
        print("V2927 terminal; no Notebook creation, recovery or submission")
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
        shutil.copy2(ROOT / "kaggle_agent/v2927_travel_gated_realmlp_train.py", WORKDIR / "main.py")
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
        errors = list(raw.rglob("v2927_error.json"))
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
