"""V2931 fail-closed preflight for one Notebook: read-only Kaggle + all GitHub Actions."""
from __future__ import annotations
import csv
import io
import json
import os
import subprocess
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = "mjw3177109/kas-v2931-conditional-crossfit-te-realmlp-t4-s6e10"
COMPETITION = "playground-series-s6e10"
ACTIVE = ("in_progress", "queued", "waiting", "requested", "pending")


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def github_runs(status):
    token = os.environ.get("GH_API_TOKEN", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not token or repo != "mjw3177109/kaggletest":
        raise RuntimeError("GitHub Actions authentication unavailable")
    q = urllib.parse.urlencode({"status": status, "per_page": 100})
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/actions/runs?{q}",
        headers={"Authorization": "Bearer " + token,
                 "Accept": "application/vnd.github+json",
                 "User-Agent": "kaggle-autoscientist-v2931"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        result = json.load(response)
    runs = result.get("workflow_runs") or []
    if int(result.get("total_count", -1)) > len(runs):
        raise RuntimeError("GitHub active Actions result paginated; fail closed")
    return runs


def assert_no_other_actions():
    own = int(os.environ.get("GITHUB_RUN_ID", "0"))
    if not own:
        raise RuntimeError("No current GitHub run ID")
    for status in ACTIVE:
        others = [r for r in github_runs(status) if int(r.get("id", 0)) != own]
        if others:
            raise RuntimeError("Other GitHub Actions active; refuse new or recovery Notebook work")


def kaggle(args, timeout=120):
    p = subprocess.run(["kaggle", *args], capture_output=True, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError("Authenticated read-only Kaggle CLI verification unavailable")
    return p.stdout


def assert_kaggle_idle_and_official():
    if os.environ.get("KAGGLE_USERNAME") != "mjw3177109" or not os.environ.get("KAGGLE_API_TOKEN"):
        raise RuntimeError("Expected Kaggle account identity not verified")
    submissions = kaggle(["competitions", "submissions", "-c", COMPETITION])
    if "57446286" not in submissions or "0.96099" not in submissions:
        raise RuntimeError("Official V2915 submission/score absent from authenticated history")
    listing = kaggle(["kernels", "list", "--mine", "--page-size", "100", "-v"])
    rows = list(csv.DictReader(io.StringIO(listing)))
    if not rows or len(rows) >= 100 or "ref" not in rows[0]:
        raise RuntimeError("Owned Kaggle Notebook listing ambiguous or incomplete")
    refs = [str(row.get("ref", "")).strip() for row in rows]
    if any(not ref.startswith("mjw3177109/") for ref in refs):
        raise RuntimeError("Owned Kaggle Notebook identity mismatch")
    known = {
        "mjw3177109/kas-v2927-travel-gated-dual-realmlp-t4-s6e10",
        "mjw3177109/kas-v2928-service-factor-pca-lgbm-cpu-s6e10",
        "mjw3177109/kas-v2929-fullservice-ordinal-realmlp-t4-s6e10",
        "mjw3177109/kas-v2930-conditional-crossfit-te-realmlp-t4-s6e10",
    }
    relevant = {ref for ref in refs if "s6e10" in ref or "playground-series-s6e10" in ref}
    for ref in sorted((relevant | known) - {KERNEL}):
        status = kaggle(["kernels", "status", ref], timeout=90)
        if any("KernelWorkerStatus." + x in status for x in ("QUEUED", "RUNNING", "PENDING")):
            raise RuntimeError("Another Kaggle S6E10 Notebook still active; refuse concurrent training")
        if "KernelWorkerStatus." not in status:
            raise RuntimeError("Unknown Kaggle S6E10 Notebook status; fail closed")


def assert_preregistration():
    s = read("state/v2_9_30_status.json")
    p = read("state/v2_9_30_status.json")
    c = read("state/v2_9_15_status.json")
    o = read("state/v2_9_15_submission.json")
    rank = read("state/score_rank_latest.json")
    ladder = read("state/dual_ladder.json")
    deadline = datetime.fromisoformat(ladder["active_deadline"].replace("Z", "+00:00"))
    frozen = {
        "competition": COMPETITION, "version": "V2.9.31",
        "intended_kernel": KERNEL,
        "hypothesis": "fold_local_supervised_mutual_information_pruning_of_champion_features",
        "model_family": "global_RealMLP_TD_T4",
        "feature_family": "v2915_37_features_fold_local_mi_select_29",
        "baseline_features": 37, "protected_target_encoded_features": 3,
        "selected_nonprotected_features": 26, "selected_total_features": 29,
        "removed_features": 8, "feature_selection": "outer_train_only_mutual_info_classif",
        "mi_n_neighbors": 3, "mi_seed": 20261031,
        "mi_tie_break": "canonical_v2915_feature_order",
        "outer_folds": 3, "seed": 20261018, "model_seed": 20261031,
        "epochs": 5, "n_ens": 8, "baseline_cv": 0.9612850399332336,
        "official_submission_id": 57446286, "official_public_lb": 0.96099,
        "blend": {"v2915_rank": .85, "v2931_rank": .15},
        "promotion": {"min_cv": 0.96125635576833,
                      "min_gain_vs_v2915": .00005,
                      "min_positive_folds": 2,
                      "worst_fold_strict_gt": -.00005},
        "submission_policy": "evaluation_only_no_submission",
    }
    if any(s.get(k) != v for k, v in frozen.items()):
        raise RuntimeError("V2931 frozen preregistration mismatch")
    if any(s.get(k) is not False for k in (
        "submitted", "submission_attempted", "external_labels", "pseudo_labels",
        "route_features", "pca_features", "ordinal_threshold_features",
        "travel_gated_experts", "new_conditional_te_features")):
        raise RuntimeError("V2931 prohibited option enabled")
    stored = s.get("kernel")
    if stored not in ("", None, KERNEL):
        raise RuntimeError("V2931 stored Notebook owner mismatch")
    if s.get("stage") not in {"preregistered_not_started", "kernel_queued",
                              "kernel_running", "pushing_kernel"}:
        raise RuntimeError("V2931 already terminal or unknown stage")
    if (s.get("stage") == "preregistered_not_started"
            and (stored or s.get("notebook_created") is not False
                 or s.get("training_started") is not False)):
        raise RuntimeError("V2931 not-started status is inconsistent")
    if (s.get("stage") != "preregistered_not_started" and stored != KERNEL):
        raise RuntimeError("V2931 active status lacks durable Notebook owner")
    if (deadline.tzinfo is None or datetime.now(timezone.utc) >= deadline
            or ladder.get("active_competition") != COMPETITION):
        raise RuntimeError("Competition deadline gate blocked")
    if (p.get("stage") != "skipped_no_cv_gain" or p.get("submitted") is not False
            or p.get("kernel_state") != "kernel_complete"
            or (p.get("source_artifacts") or {}).get("v2930", {}).get("sha256")
               != "45e96e595676ec91a1e6c8eea98f975e63c9c0762412b898fe549c0636107362"
            or abs(float((p.get("comparison") or {}).get("candidate_auc", 0))
                   - 0.9613286615476094) > 1e-12):
        raise RuntimeError("V2930 terminal evidence mismatch")
    if (c.get("candidate_test_hash") != o.get("prediction_hash")
            or o.get("stage") != "complete" or o.get("submitted") is not True
            or str((o.get("submission") or {}).get("ref")) != "57446286"
            or abs(float(o.get("official_lb", 0)) - .96099) > 1e-9
            or str((rank.get("score") or {}).get("submission_id")) != "57446286"
            or abs(float((rank.get("score") or {}).get("public_score", 0)) - .96099) > 1e-9):
        raise RuntimeError("V2915 official submission/score/hash provenance mismatch")
    return "recover_only" if stored else (os.environ.get("REQUESTED_MODE") or "run")


def main():
    assert_no_other_actions()
    mode = assert_preregistration()
    if mode not in ("run", "recover_only"):
        raise RuntimeError("Unsupported V2931 mode")
    assert_kaggle_idle_and_official()
    assert_no_other_actions()
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
        f.write("allowed=true\nmode=" + mode + "\n")
    print("V2931 strict preregistration, Kaggle owner, official submission and global Actions gates passed")


if __name__ == "__main__":
    main()
