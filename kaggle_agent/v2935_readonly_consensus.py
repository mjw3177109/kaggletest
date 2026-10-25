"""V2935 immutable three-source agreement gate. CPU/read-only, never trains or submits."""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import v2916_pipeline as v16
import v2924_pipeline as v24
import v2930_pipeline as v30
import v2935_actions_gate as gate

ROOT = gate.ROOT
STATUS = ROOT / "state/v2_9_35_status.json"
REPORT = ROOT / "reports/v2_9_35_readonly_result.md"
EVIDENCE = ROOT / "v2935_readonly_evidence.json"
BASELINE = gate.BASELINE
EXPECTED16 = 0.9605256772889431
EXPECTED30 = 0.9606450272778854
N_TRAIN, N_TEST = 699635, 299844
ORIGINAL_RUN = subprocess.run
LAST_CALL = None


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def paced_run(args, *pos, **kw):
    """Pace all nested legacy Kaggle CLI calls, including original source replay."""
    global LAST_CALL
    if isinstance(args, (list, tuple)) and args and args[0] == "kaggle":
        if LAST_CALL is not None:
            wait = 3.0 - (time.monotonic() - LAST_CALL)
            if wait > 0:
                time.sleep(wait)
        LAST_CALL = time.monotonic()
        response = ORIGINAL_RUN(args, *pos, **kw)
        if response.returncode:
            err = str(response.stderr or "") + str(response.stdout or "")
            category = "rate_limited" if "429" in err else "source_unavailable"
            raise RuntimeError("Read-only Kaggle CLI failed closed: " + category)
        return response
    return ORIGINAL_RUN(args, *pos, **kw)


def checked_vector(values, length, name):
    v = np.asarray(values)
    if v.shape != (length,) or v.dtype.kind not in "iuf" or not np.isfinite(v).all():
        raise ValueError(name + " invalid numeric length, dtype or finite range")
    return v.astype(np.float64)


def agreement_gate(champion, source16, source30):
    n = len(champion)
    if n == 0:
        raise ValueError("Empty prediction array")
    a = checked_vector(champion, n, "champion")
    b = checked_vector(source16, n, "v2916")
    c = checked_vector(source30, n, "v2930")
    r0 = rankdata(a, method="average") / n
    r16 = rankdata(b, method="average") / n
    r30 = rankdata(c, method="average") / n
    mask = (r16 - r0) * (r30 - r0) > 0
    candidate = r0 + mask * (0.075 * (r16 - r0) + 0.075 * (r30 - r0))
    if not np.array_equal(candidate[~mask], r0[~mask]):
        raise ValueError("Disagreement or ties changed champion ranking")
    if not np.isfinite(candidate).all():
        raise ValueError("Nonfinite candidate predictions")
    return candidate, mask


def read_v2930(folder):
    path = v24.v23.unique(folder, "v2930_realmlp_oof.npz")
    with np.load(path, allow_pickle=False) as npz:
        keys = ("train_id", "target", "oof_conditional_realmlp",
                "test_id", "pred_conditional_realmlp")
        if not set(keys).issubset(npz.files):
            raise ValueError("V2930 incomplete canonical OOF/test source")
        arrays = {key: np.asarray(npz[key]).copy() for key in keys}
    for key, size in (("train_id", N_TRAIN), ("target", N_TRAIN),
                      ("oof_conditional_realmlp", N_TRAIN),
                      ("test_id", N_TEST), ("pred_conditional_realmlp", N_TEST)):
        checked_vector(arrays[key], size, key)
    return path, arrays


def download(kernel, destination):
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "kernels", "output", kernel, "-p",
                    str(destination), "-o", "-q"], check=True,
                   capture_output=True, text=True, timeout=600)


def evaluate():
    if read(STATUS).get("stage") != "preregistered_not_started":
        raise RuntimeError("V2935 is no longer pristine; never repeat a terminal evaluation")
    s = gate.preregistration()
    gate.assert_remote_head()
    with tempfile.TemporaryDirectory(prefix="kas_v2935_frozen_readonly_") as tmp:
        temp = Path(tmp)
        # Rebuild champion from all original immutable sources, including official
        # train/test ID and target, 3fold OOF and official prediction SHA.
        champion = v24.preflight()
        train, test, train_id, test_id, y, oof0, test0, sources = v24.baseline_arrays(
            temp, champion)
        if sources.get("v2915", {}).get("sha256") != gate.SOURCES["v2915_npz"]:
            raise ValueError("Champion NPZ source SHA mismatch")
        if digest(test0) != gate.CHAMPION_SHA:
            raise ValueError("Official V2915 test prediction SHA mismatch")
        if abs(float(roc_auc_score(y, oof0)) - BASELINE) > 1e-12:
            raise ValueError("Champion full OOF AUC mismatch")
        path16 = temp / "v2916"
        path30 = temp / "v2930"
        download(v16.KERNEL, path16)
        download(v30.KERNEL, path30)
        file16, arrays16 = v16.read_new(path16)
        file30, arrays30 = read_v2930(path30)
        if sha_file(file16) != gate.SOURCES["v2916_npz"]:
            raise ValueError("V2916 raw NPZ SHA mismatch")
        if sha_file(file30) != gate.SOURCES["v2930_npz"]:
            raise ValueError("V2930 raw NPZ SHA mismatch")
        m16 = read(v24.v23.unique(path16, "v2916_metrics.json"))
        m30 = read(v24.v23.unique(path30, "v2930_metrics.json"))
        v16.validate_metrics(m16)
        oof16, test16 = v16.v14.align_new(
            arrays16, "oof_realmlp_interactions", "pred_realmlp_interactions",
            train_id, test_id, y, "V2916")
        oof30, test30 = v16.v14.align_new(
            arrays30, "oof_conditional_realmlp", "pred_conditional_realmlp",
            train_id, test_id, y, "V2930")
        for name, values, expected in (("V2916", oof16, EXPECTED16),
                                       ("V2930", oof30, EXPECTED30)):
            checked_vector(values, N_TRAIN, name + " OOF")
            if abs(float(roc_auc_score(y, values)) - expected) > 2e-6:
                raise ValueError(name + " standalone CV mismatch")
        for name, values in (("V2916", test16), ("V2930", test30)):
            checked_vector(values, N_TEST, name + " test")
        v16.v13.evaluate_fold_safety(y, oof16, m16)
        v30.verify_metrics(m30, y, oof30)
        candidate_oof, mask_oof = agreement_gate(oof0, oof16, oof30)
        candidate_test, mask_test = agreement_gate(test0, test16, test30)
        auc = float(roc_auc_score(y, candidate_oof))
        fold_auc, fold_gain, fold_hashes = [], [], []
        for fold, (_, idx) in enumerate(StratifiedKFold(
                n_splits=3, shuffle=True, random_state=20261018
                ).split(np.zeros(len(y), dtype=np.uint8), y)):
            fold_auc.append(float(roc_auc_score(y[idx], candidate_oof[idx])))
            fold_gain.append(fold_auc[-1] - float(roc_auc_score(y[idx], oof0[idx])))
            fold_hashes.append(hashlib.sha256(np.asarray(idx, dtype="<i8").tobytes()).hexdigest())
        gain = auc - BASELINE
        positive = sum(x > 0 for x in fold_gain)
        promotable = bool(auc >= 0.96125635576833 and gain >= 0.00005
                          and positive >= 2 and min(fold_gain) > -0.00005)
        result = {
            "version": "V2.9.35", "competition": "playground-series-s6e10",
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "stage": "candidate_ready_unsubmitted" if promotable else "skipped_no_cv_gain",
            "evaluation_only": True, "cpu_only": True, "no_new_model_fit": True,
            "notebook_created": False, "training_started": False,
            "submitted": False, "submission_attempted": False,
            "official_submission_id": 57446286, "official_public_lb": 0.96099,
            "baseline_cv": BASELINE, "full_cv": auc, "gain_vs_champion": gain,
            "fold_auc": fold_auc, "fold_gain": fold_gain,
            "fold_validation_index_sha256": fold_hashes, "positive_folds": positive,
            "promotable": promotable, "agreement_fraction_oof": float(mask_oof.mean()),
            "agreement_fraction_test": float(mask_test.mean()),
            "candidate_oof_sha256": digest(candidate_oof),
            "candidate_test_sha256": digest(candidate_test),
            "source_sha256": dict(gate.SOURCES),
            "verified_source_notebooks": list(gate.KERNELS[:3]),
            "train_rows": N_TRAIN, "test_rows": N_TEST,
            "source_artifacts": sources,
            "source_metrics": {
                "v2916_full_oof_auc": m16["full_oof_auc"],
                "v2930_full_oof_auc": m30["full_oof_auc"],
            },
        }
        EVIDENCE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        s.update({key: val for key, val in result.items() if key != "source_artifacts"})
        s["updated_at"] = result["checked_at"]
        s["next_action"] = ("Independent durable one-shot submission gate required; evaluator never submits."
                            if promotable else "Terminal: insufficient preregistered gain, never submit.")
        STATUS.write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        REPORT.write_text(
            "# V2.9.35 immutable read-only agreement-gated OOF audit\n\n"
            + f"- UTC: {result['checked_at']}\n"
            + f"- Threefold full CV: **{auc:.16f}**; champion {BASELINE:.16f}\n"
            + f"- Paired gain: **{gain:+.16f}**; fold gains {fold_gain}\n"
            + f"- Positive folds: {positive}/3; promotable: {promotable}\n"
            + f"- Agreement fractions OOF/test: {result['agreement_fraction_oof']:.8f} / {result['agreement_fraction_test']:.8f}\n"
            + f"- OOF/test SHA256: {result['candidate_oof_sha256']} / {result['candidate_test_sha256']}\n"
            + "- No Notebook, training or official submission. Official Public LB remains independently certified 0.96099 until refreshed.\n",
            encoding="utf-8")
        print(json.dumps({key: val for key, val in result.items()
                          if key not in ("source_artifacts", "source_metrics")}, indent=2))
        return result


if __name__ == "__main__":
    # The gate in the Actions workflow is mandatory. Refuse ad-hoc execution.
    import os
    if os.environ.get("V2935_GATE_PASSED") != "true":
        raise SystemExit("V2935 authenticated gate required; no ad-hoc evaluation")
    subprocess.run = paced_run
    try:
        evaluate()
    finally:
        subprocess.run = ORIGINAL_RUN
