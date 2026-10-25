"""V2935 read-only CPU gate: fail closed; never train, push kernels, or submit."""
from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone

import v2934_actions_gate as common

ROOT = common.ROOT
BRANCH = "feature/v29-autonomous-evolution"
COMPETITION = "playground-series-s6e10"
BASELINE = 0.9612850399332336
CHAMPION_SHA = "906185547dcd5f0cac3ac173de644fd9997f20ac863c6745bcb1fd3e9a89e5f7"
SOURCES = {
    "v2915_npz": "46ad63f08537cd8d61c4c8e8698efc20bb3c8fbad4eb0d94ecd7c993b2d894b9",
    "v2915_test": CHAMPION_SHA,
    "v2916_npz": "05facb3098732f1348928c8de62cefae3ecb64b5b1fabdb3a7a84342b7bfe0d7",
    "v2930_npz": "45e96e595676ec91a1e6c8eea98f975e63c9c0762412b898fe549c0636107362",
}
KERNELS = (
    "mjw3177109/kas-v2915-realmlp-5epoch-t4-s6e10",
    "mjw3177109/kas-v2916-interaction-realmlp-ens8-t4-s6e10",
    "mjw3177109/kas-v2930-conditional-crossfit-te-realmlp-t4-s6e10",
    "mjw3177109/kas-v2934-explicit-secondorder-deepfm-t4-s6e10",
)


def preregistration():
    s = common.read("state/v2_9_35_status.json")
    official = common.read("state/v2_9_15_submission.json")
    champion = common.read("state/v2_9_15_status.json")
    old16 = common.read("state/v2_9_16_status.json")
    old30 = common.read("state/v2_9_30_status.json")
    old34 = common.read("state/v2_9_34_status.json")
    ranking = common.read("state/score_rank_latest.json")
    ladder = common.read("state/dual_ladder.json")
    frozen = {
        "competition": COMPETITION,
        "version": "V2.9.35",
        "stage": "preregistered_not_started",
        "hypothesis": "label_free_agreement_gated_frozen_v2916_v2930_rank_corrections",
        "model_family": "read_only_three_source_consensus_gate_no_fit",
        "notebook_created": False,
        "training_started": False,
        "submitted": False,
        "submission_attempted": False,
        "evaluation_only": True,
        "no_new_model_fit": True,
        "cpu_only": True,
        "outer_folds": 3,
        "seed": 20261018,
        "train_rows": 699635,
        "test_rows": 299844,
        "agreement_rule": "(rank(v2916)-rank(v2915))*(rank(v2930)-rank(v2915))>0",
        "tie_rule": "zero_or_disagreement_preserve_champion",
        "rank_method": "scipy.stats.rankdata(method=average)/n",
        "fixed_weights": {
            "champion_when_agree": 0.85,
            "v2916_when_agree": 0.075,
            "v2930_when_agree": 0.075,
            "champion_when_disagree": 1,
        },
        "baseline_cv": BASELINE,
        "official_public_lb": 0.96099,
        "official_submission_id": 57446286,
        "source_sha256": SOURCES,
        "promotion": {
            "min_cv": 0.96125635576833,
            "min_gain_vs_v2915": 0.00005,
            "min_positive_folds": 2,
            "worst_fold_strict_gt": -0.00005,
        },
        "submission_policy": "evaluation_only_no_submission",
        "implementation_status": "awaiting_readonly_evaluator_tests_and_guarded_workflow",
    }
    if any(s.get(k) != value for k, value in frozen.items()):
        raise RuntimeError("V2935 immutable preregistration mismatch")
    if s.get("kernel") is not None or s.get("previous_v2934", {}).get("stage") != "skipped_no_cv_gain":
        raise RuntimeError("V2935 owner/previous experiment mismatch")
    if (champion.get("candidate_test_hash") != CHAMPION_SHA
        or champion.get("stage") != "candidate_ready_unsubmitted"
        or (champion.get("source_artifacts") or {}).get("v2915", {}).get("sha256") != SOURCES["v2915_npz"]
        or official.get("stage") != "complete" or official.get("submitted") is not True
        or official.get("prediction_hash") != CHAMPION_SHA
        or str((official.get("submission") or {}).get("ref")) != "57446286"
        or (official.get("submission") or {}).get("status") != "SubmissionStatus.COMPLETE"
        or official.get("official_lb") != 0.96099
        or str((ranking.get("score") or {}).get("submission_id")) != "57446286"
        or (ranking.get("score") or {}).get("public_score") != 0.96099):
        raise RuntimeError("Frozen champion official score/hash mismatch")
    for prior, key in ((old16, "v2916_npz"), (old30, "v2930_npz")):
        name = key[:5]
        if (prior.get("stage") != "skipped_no_cv_gain"
            or prior.get("submitted") is not False
            or prior.get("kernel_state") != "kernel_complete"
            or (prior.get("source_artifacts") or {}).get(name, {}).get("sha256") != SOURCES[key]):
            raise RuntimeError("Frozen source terminal artifact mismatch: " + key)
    if (old34.get("stage") != "skipped_no_cv_gain"
        or old34.get("submitted") is not False
        or old34.get("kernel_state") != "kernel_complete"
        or (old34.get("source_artifacts") or {}).get("v2934", {}).get("sha256")
           != "e0a3b343c4aad51f9ca80130e0f7f41dc69298b625778686db37a7cf9ef07d79"):
        raise RuntimeError("V2934 terminal evidence mismatch")
    deadline = datetime.fromisoformat(str(ladder.get("active_deadline", "")).replace("Z", "+00:00"))
    if (ladder.get("active_competition") != COMPETITION
        or deadline.tzinfo is None or datetime.now(timezone.utc) >= deadline):
        raise RuntimeError("Kaggle competition deadline reached or ambiguous")
    return s


def assert_remote_head():
    local = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, check=True, timeout=30).stdout.strip()
    remote = subprocess.run(["git", "ls-remote", "origin", "refs/heads/" + BRANCH],
                            cwd=ROOT, capture_output=True, text=True,
                            check=True, timeout=45).stdout.strip()
    if remote != local + "\trefs/heads/" + BRANCH:
        raise RuntimeError("Remote branch HEAD changed; refuse evaluation")


def authenticated_idle():
    if os.environ.get("KAGGLE_USERNAME") != "mjw3177109" or not os.environ.get("KAGGLE_API_TOKEN"):
        raise RuntimeError("Authenticated Kaggle owner unavailable")
    common.assert_kaggle_idle_and_official()
    for kernel in KERNELS:
        status = common.kaggle(["kernels", "status", kernel], timeout=90)
        if "KernelWorkerStatus.COMPLETE" not in status:
            raise RuntimeError("Frozen source Notebook not COMPLETE: " + kernel)


def main():
    if os.environ.get("GITHUB_REPOSITORY") != "mjw3177109/kaggletest":
        raise RuntimeError("Wrong GitHub repository")
    common.assert_no_other_actions()
    assert_remote_head()
    preregistration()
    authenticated_idle()
    common.assert_no_other_actions()
    assert_remote_head()
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
        f.write("allowed=true\n")
    print("V2935 authenticated read-only gate passed; no Notebook creation/training/submission")


if __name__ == "__main__":
    main()
