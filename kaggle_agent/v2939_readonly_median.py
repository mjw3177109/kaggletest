"""V2939 immutable conservative rank-median consensus; CPU/read-only, no fitting or submission."""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import v2916_pipeline as v16
import v2924_pipeline as v24
import v2930_pipeline as v30
import v2935_readonly_consensus as old
import v2939_actions_gate as gate

ROOT = gate.ROOT
STATUS = ROOT / "state/v2_9_39_status.json"
REPORT = ROOT / "reports/v2_9_39_readonly_result.md"
EVIDENCE = ROOT / "v2939_readonly_evidence.json"
BASELINE = gate.BASELINE
EXPECTED16 = 0.9605256772889431
EXPECTED30 = 0.9606450272778854
N_TRAIN, N_TEST = 699635, 299844

def median_consensus(champion, source16, source30):
    """Frozen label-free order statistic, not V2935's arithmetic agreement correction."""
    n = len(champion)
    if n == 0:
        raise ValueError("Empty prediction vector")
    a = old.checked_vector(champion, n, "champion")
    b = old.checked_vector(source16, n, "v2916")
    c = old.checked_vector(source30, n, "v2930")
    r0 = rankdata(a, method="average") / n
    r16 = rankdata(b, method="average") / n
    r30 = rankdata(c, method="average") / n
    median = np.median(np.stack((r0, r16, r30), axis=0), axis=0)
    candidate = r0 + 0.15 * (median - r0)
    disagree = (r16 - r0) * (r30 - r0) <= 0
    if not np.array_equal(candidate[disagree], r0[disagree]):
        raise ValueError("Disagreement or tie changed champion rank")
    if not np.isfinite(candidate).all():
        raise ValueError("Nonfinite median candidate")
    return candidate, disagree

def evaluate():
    s = gate.preregistration(allow_reserved=True)
    if s.get("stage") != "run_reserved":
        raise RuntimeError("V2939 not exclusively reserved; no evaluation")
    gate.source_gate.common.assert_no_other_actions()
    gate.source_gate.assert_remote_head()
    with tempfile.TemporaryDirectory(prefix="kas_v2939_median_readonly_") as td:
        temp = Path(td)
        champion = v24.preflight()
        train, test, train_id, test_id, y, oof0, test0, sources = v24.baseline_arrays(
            temp, champion)
        if sources.get("v2915", {}).get("sha256") != gate.SOURCES["v2915_npz"]:
            raise ValueError("Champion NPZ SHA mismatch")
        if old.digest(test0) != gate.CHAMPION_SHA:
            raise ValueError("Official champion test SHA mismatch")
        if abs(float(roc_auc_score(y, oof0)) - BASELINE) > 1e-12:
            raise ValueError("Champion threefold OOF CV mismatch")
        folder16 = temp / "v2916"
        folder30 = temp / "v2930"
        old.download(v16.KERNEL, folder16)
        old.download(v30.KERNEL, folder30)
        file16, arrays16 = v16.read_new(folder16)
        file30, arrays30 = old.read_v2930(folder30)
        if old.sha_file(file16) != gate.SOURCES["v2916_npz"]:
            raise ValueError("V2916 source NPZ SHA mismatch")
        if old.sha_file(file30) != gate.SOURCES["v2930_npz"]:
            raise ValueError("V2930 source NPZ SHA mismatch")
        metrics16 = old.read(v24.v23.unique(folder16, "v2916_metrics.json"))
        metrics30 = old.read(v24.v23.unique(folder30, "v2930_metrics.json"))
        v16.validate_metrics(metrics16)
        oof16, test16 = v16.v14.align_new(
            arrays16, "oof_realmlp_interactions", "pred_realmlp_interactions",
            train_id, test_id, y, "V2916")
        oof30, test30 = v16.v14.align_new(
            arrays30, "oof_conditional_realmlp", "pred_conditional_realmlp",
            train_id, test_id, y, "V2930")
        for name, values, expected in (
            ("V2916", oof16, EXPECTED16), ("V2930", oof30, EXPECTED30)
        ):
            old.checked_vector(values, N_TRAIN, name + " OOF")
            if abs(float(roc_auc_score(y, values)) - expected) > 2e-6:
                raise ValueError(name + " standalone OOF AUC mismatch")
        for name, values in (("V2916", test16), ("V2930", test30)):
            old.checked_vector(values, N_TEST, name + " test")
        v16.v13.evaluate_fold_safety(y, oof16, metrics16)
        v30.verify_metrics(metrics30, y, oof30)
        candidate_oof, disagree_oof = median_consensus(oof0, oof16, oof30)
        candidate_test, disagree_test = median_consensus(test0, test16, test30)
        auc = float(roc_auc_score(y, candidate_oof))
        fold_auc, fold_gain, fold_hashes = [], [], []
        for fold, (_, indices) in enumerate(StratifiedKFold(
                n_splits=3, shuffle=True, random_state=20261018
            ).split(np.zeros(len(y), dtype=np.uint8), y)):
            fold_auc.append(float(roc_auc_score(y[indices], candidate_oof[indices])))
            fold_gain.append(fold_auc[-1] - float(roc_auc_score(y[indices], oof0[indices])))
            fold_hashes.append(hashlib.sha256(
                np.asarray(indices, dtype="<i8").tobytes()).hexdigest())
        gain = auc - BASELINE
        positive = sum(x > 0 for x in fold_gain)
        promotable = bool(auc >= 0.96125635576833 and gain >= 0.00005
                          and positive >= 2 and min(fold_gain) > -0.00005)
        result = {
            "version": "V2.9.39",
            "competition": "playground-series-s6e10",
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "stage": "candidate_ready_unsubmitted" if promotable else "skipped_no_cv_gain",
            "evaluation_only": True, "cpu_only": True, "no_new_model_fit": True,
            "notebook_created": False, "training_started": False,
            "submitted": False, "submission_attempted": False,
            "official_submission_id": 57446286, "official_public_lb": 0.96099,
            "baseline_cv": BASELINE, "full_cv": auc, "gain_vs_champion": gain,
            "fold_auc": fold_auc, "fold_gain": fold_gain,
            "fold_validation_index_sha256": fold_hashes,
            "positive_folds": positive, "promotable": promotable,
            "median_disagreement_fraction_oof": float(disagree_oof.mean()),
            "median_disagreement_fraction_test": float(disagree_test.mean()),
            "candidate_oof_sha256": old.digest(candidate_oof),
            "candidate_test_sha256": old.digest(candidate_test),
            "source_sha256": dict(gate.SOURCES),
            "verified_source_notebooks": list(gate.source_gate.KERNELS[:3]),
            "train_rows": N_TRAIN, "test_rows": N_TEST,
            "source_artifacts": sources,
            "source_metrics": {
                "v2916_full_oof_auc": metrics16["full_oof_auc"],
                "v2930_full_oof_auc": metrics30["full_oof_auc"],
            },
        }
        EVIDENCE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        s.update({key: value for key, value in result.items() if key != "source_artifacts"})
        s["updated_at"] = result["checked_at"]
        s["next_action"] = ("Separate durable one-shot submission gate required; evaluator never submits."
                            if promotable else "Terminal: below frozen promotion gate, never submit.")
        STATUS.write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        REPORT.write_text(
            "# V2939 immutable read-only conservative rank-median audit\n\n"
            + f"- UTC: {result['checked_at']}\n"
            + f"- Threefold full CV: **{auc:.16f}**; champion {BASELINE:.16f}\n"
            + f"- Paired gain: **{gain:+.16f}**; fold gains {fold_gain}\n"
            + f"- Positive folds: {positive}/3; promotable: {promotable}\n"
            + f"- Disagreement fractions OOF/test: {result['median_disagreement_fraction_oof']:.8f} / {result['median_disagreement_fraction_test']:.8f}\n"
            + f"- OOF/test SHA256: {result['candidate_oof_sha256']} / {result['candidate_test_sha256']}\n"
            + "- No Notebook, training or official submission. CV is not Kaggle Public LB.\n",
            encoding="utf-8")
        if promotable:
            pd.DataFrame({"id": test_id, "satisfaction": candidate_test}).to_csv(
                temp / "v2939_candidate_unsubmitted.csv", index=False)
            # The candidate vector is hashed, but a later separately authorized
            # submission must reconstruct and verify it; do not submit here.
        print(json.dumps({key: value for key, value in result.items()
                          if key not in ("source_artifacts", "source_metrics")}, indent=2))
        return result

if __name__ == "__main__":
    import os
    if os.environ.get("V2939_GATE_PASSED") != "true":
        raise SystemExit("V2939 authenticated gate required; no ad-hoc evaluation")
    # Guarantee >=3s after the separate preflight process's final Kaggle call.
    time.sleep(3.1)
    subprocess.run = old.paced_run
    try:
        evaluate()
    finally:
        subprocess.run = old.ORIGINAL_RUN
