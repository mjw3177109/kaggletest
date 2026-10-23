"""V2.9.25 preregistered read-only dual-axis fusion evaluator.

No Notebook creation, training, hyperparameter search or Kaggle submission.
Reads only existing, immutable Kaggle Notebook outputs. Fails closed on SHA,
fold, source, deadline, official champion or preregistration mismatches.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import v2916_pipeline as v16
import v2924_pipeline as v24

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "state/v2_9_25_status.json"
AUDIT = ROOT / "state/v2_9_24_readonly_audit.json"
REPORT = ROOT / "reports/v2_9_25_readonly_result.md"
EVIDENCE = ROOT / "v2925_readonly_evidence.json"
BASELINE = 0.9612850399332336
FLOOR = 0.96125635576833
MIN_GAIN = 0.00005
WEIGHT_INTERACTION = 0.15
ALPHA = 0.15
KERNEL = "mjw3177109/kas-v2925-dual-axis-residual-cpu-s6e10"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def preflight():
    prereg = read(STATUS)
    audit = read(AUDIT)
    champion = read(ROOT / "state/v2_9_15_status.json")
    official = read(ROOT / "state/v2_9_15_submission.json")
    rank = read(ROOT / "state/score_rank_latest.json")
    ladder = read(ROOT / "state/dual_ladder.json")
    prior16 = read(ROOT / "state/v2_9_16_status.json")
    prior24 = read(ROOT / "state/v2_9_24_status.json")
    p = prereg.get("pre_registered_parameters") or {}
    expected = {
        "base_champion_cv": BASELINE,
        "source_v2916_sha256": "05facb3098732f1348928c8de62cefae3ecb64b5b1fabdb3a7a84342b7bfe0d7",
        "source_v2924_sha256": "47b5aa6931ab925e6cb2ec89dfd262c508da1445af8766f69eb8525bc93d9568",
        "source_champion_test_hash": "906185547dcd5f0cac3ac173de644fd9997f20ac863c6745bcb1fd3e9a89e5f7",
        "residual_alpha": ALPHA,
        "outer_folds": 3,
        "no_new_model_fit": True,
        "cpu_only": True,
        "threshold_full_cv": FLOOR,
        "threshold_gain_vs_champion": MIN_GAIN,
        "threshold_positive_folds": 2,
        "threshold_worst_fold_strictly_greater_than": -0.00005,
    }
    for key, value in expected.items():
        if p.get(key) != value:
            raise ValueError("Preregistered parameter mismatch: " + key)
    if p.get("rank_fusion_weights") != {"v2915": 0.85, "v2916": 0.15}:
        raise ValueError("Frozen blend weights changed")
    if (prereg.get("stage") != "preregistered_not_started"
        or prereg.get("unique_kernel_slug") != KERNEL
        or prereg.get("notebook_created") is not False
        or prereg.get("training_started") is not False
        or prereg.get("submitted") is not False
        or prereg.get("evaluation_only") is not True
        or audit.get("stage") != "skipped_no_cv_gain_verified_readonly"
        or audit.get("kernel_state") != "COMPLETE_authenticated_cli"
        or audit.get("reproducibility", {}).get("frozen_champion_test_sha_verified") is not True
        or prior24.get("stage") != "validation_mismatch"
        or prior24.get("kernel_state") != "kernel_complete"
        or prior24.get("submitted") is not False
        or prior16.get("stage") != "skipped_no_cv_gain"
        or prior16.get("submitted") is not False
        or official.get("stage") != "complete"
        or official.get("submitted") is not True
        or str((official.get("submission") or {}).get("ref")) != "57446286"
        or (official.get("submission") or {}).get("status") != "SubmissionStatus.COMPLETE"
        or official.get("official_lb") != 0.96099
        or official.get("prediction_hash") != champion.get("candidate_test_hash")
        or champion.get("candidate_test_hash") != p["source_champion_test_hash"]
        or str((rank.get("score") or {}).get("submission_id")) != "57446286"
        or (rank.get("score") or {}).get("public_score") != 0.96099
        or ladder.get("active_competition") != "playground-series-s6e10"):
        raise ValueError("Read-only preflight state/official evidence mismatch")
    deadline = datetime.fromisoformat(ladder["active_deadline"].replace("Z", "+00:00"))
    if deadline.tzinfo is None or datetime.now(timezone.utc) >= deadline:
        raise ValueError("Competition deadline passed; no further evaluation")
    if abs(float((prior16.get("comparison") or {}).get("candidate_auc", 0)) - 0.9613243650285448) > 1e-12:
        raise ValueError("V2916 verified full CV changed")
    if abs(float((audit.get("full_candidate") or {}).get("cv_auc", 0)) - 0.9612947642045307) > 1e-12:
        raise ValueError("V2924 independently audited full CV changed")
    return prereg, p, champion, prior16


def status_complete(kernel):
    result = subprocess.run(["kaggle", "kernels", "status", kernel],
                            capture_output=True, text=True, timeout=90, check=False)
    if result.returncode != 0 or "KernelWorkerStatus.COMPLETE" not in result.stdout:
        raise RuntimeError("Authenticated Kaggle COMPLETE status unavailable for frozen source")
    return "COMPLETE"


def fused(base, interaction, specialist, mask):
    base = np.asarray(base, dtype=np.float64)
    interaction = np.asarray(interaction, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if base.ndim != 1 or interaction.shape != base.shape or mask.shape != base.shape:
        raise ValueError("Full-array dimensions/ordering mismatch")
    if not np.isfinite(base).all() or not np.isfinite(interaction).all():
        raise ValueError("Nonfinite full-array scores")
    blend = 0.85 * v24.v23.audit.rank(base) + WEIGHT_INTERACTION * v24.v23.audit.rank(interaction)
    candidate = v24.residual_correct(blend, specialist, mask, alpha=ALPHA)
    if not np.array_equal(candidate[~mask], blend[~mask]):
        raise ValueError("Non-Personal Travel predictions changed")
    return candidate


def evaluate():
    prereg, p, champion, prior16 = preflight()
    status_complete(v16.KERNEL)
    status_complete(v24.KERNEL)
    with tempfile.TemporaryDirectory(prefix="kas_v2925_readonly_") as temp:
        train, test, train_id, test_id, y, base_oof, base_test, sources = v24.baseline_arrays(temp, champion)
        folder16 = Path(temp) / "v2916"
        folder24 = Path(temp) / "v2924"
        folder16.mkdir()
        folder24.mkdir()
        for kernel, folder in ((v16.KERNEL, folder16), (v24.KERNEL, folder24)):
            cmd = subprocess.run(["kaggle", "kernels", "output", kernel, "-p", str(folder), "-o", "-q"],
                                 capture_output=True, text=True, timeout=600, check=False)
            if cmd.returncode:
                raise RuntimeError("Read-only source Notebook output download failed")
        file16, arrays16 = v16.read_new(folder16)
        sha16 = v16.audit.sha256(file16)
        if sha16 != p["source_v2916_sha256"] or sha16 != prior16["source_artifacts"]["v2916"]["sha256"]:
            raise ValueError("V2916 immutable artifact SHA changed")
        metrics16 = read(v16.unique(folder16, "v2916_metrics.json"))
        v16.validate_metrics(metrics16)
        int_oof, int_test = v16.v14.align_new(arrays16, "oof_realmlp_interactions",
                                               "pred_realmlp_interactions", train_id, test_id, y, "V2916")
        if abs(float(roc_auc_score(y, int_oof)) - float(metrics16["full_oof_auc"])) > 1e-12:
            raise ValueError("V2916 standalone full OOF AUC changed")
        v16.v13.evaluate_fold_safety(y, int_oof, metrics16)
        metrics24, file24, spec_oof, spec_test, train_mask, test_mask = v24.load_specialist(
            folder24, train_id, test_id, y, train, test)
        sha24 = v24.v23.audit.sha256(file24)
        if sha24 != p["source_v2924_sha256"]:
            raise ValueError("V2924 immutable artifact SHA changed")
        candidate = fused(base_oof, int_oof, spec_oof, train_mask)
        candidate_test = fused(base_test, int_test, spec_test, test_mask)
        score = float(roc_auc_score(y, candidate))
        baseline_score = float(roc_auc_score(y, base_oof))
        if abs(baseline_score - BASELINE) > 1e-12:
            raise ValueError("Frozen champion full OOF AUC changed")
        fold_deltas = []
        fold_aucs = []
        for _, valid in StratifiedKFold(3, shuffle=True, random_state=v24.SEED).split(
                np.zeros(len(y), dtype=np.uint8), y):
            fold_auc = float(roc_auc_score(y[valid], candidate[valid]))
            fold_aucs.append(fold_auc)
            fold_deltas.append(fold_auc - float(roc_auc_score(y[valid], base_oof[valid])))
        positive = sum(x > 0 for x in fold_deltas)
        gain = score - BASELINE
        promotable = bool(score >= FLOOR and gain >= MIN_GAIN and positive >= 2
                          and min(fold_deltas) > -0.00005)
        result = {
            "competition": "playground-series-s6e10",
            "version": "V2.9.25",
            "stage": "candidate_ready_unsubmitted" if promotable else "skipped_no_cv_gain",
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "evaluation_only": True, "no_new_model_fit": True,
            "notebook_created": False, "training_started": False,
            "submitted": False, "submission_attempted": False,
            "official_submission_id": 57446286, "official_public_lb": 0.96099,
            "private_lb": None,
            "baseline_cv": baseline_score, "full_cv": score,
            "gain_vs_champion": gain, "fold_auc": fold_aucs,
            "fold_gain": fold_deltas, "positive_folds": positive,
            "promotable": promotable,
            "candidate_oof_hash": digest(candidate),
            "candidate_test_hash": digest(candidate_test),
            "source_v2916_sha256": sha16,
            "source_v2924_sha256": sha24,
            "source_champion_test_hash": digest(base_test),
            "source_artifacts": sources,
            "v2924_personal_oof_auc": float(metrics24["personal_oof_auc"]),
            "promotion_gates": {"cv_floor": FLOOR, "min_gain": MIN_GAIN,
                                "min_positive_folds": 2, "worst_fold_gt": -0.00005},
        }
        EVIDENCE.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        prereg.update({k: v for k, v in result.items()
                       if k not in {"source_artifacts", "source_v2916_sha256",
                                    "source_v2924_sha256", "source_champion_test_hash"}})
        prereg["source_artifacts_verified"] = {"v2916": sha16, "v2924": sha24,
                                                 "v2915_test": digest(base_test)}
        prereg["updated_at"] = result["checked_at"]
        prereg["next_action"] = ("Verify a separate persistent one-shot submission workflow; do not submit from evaluator."
                                  if promotable else "Candidate below preregistered promotion gate; no submission.")
        STATUS.write_text(json.dumps(prereg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        REPORT.write_text(
            "# V2.9.25 independent read-only dual-axis OOF audit\n\n"
            + f"- Checked UTC: {result['checked_at']}\n"
            + f"- Frozen V2.9.15 CV: {baseline_score:.15f}\n"
            + f"- New threefold full OOF CV: {score:.15f}\n"
            + f"- Gain: {gain:+.15f}; fold gains: {fold_deltas}\n"
            + f"- Positive folds: {positive}/3; promotable: {promotable}\n"
            + f"- V2916 SHA256: {sha16}; V2924 SHA256: {sha24}\n"
            + f"- Candidate OOF SHA256: {result['candidate_oof_hash']}\n"
            + f"- Candidate test prediction SHA256: {result['candidate_test_hash']}\n"
            + "- No new Notebook, training, or Kaggle submission. Public LB remains independently authenticated 0.96099.\n",
            encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "source_artifacts"}, indent=2))
        return result


if __name__ == "__main__":
    evaluate()
