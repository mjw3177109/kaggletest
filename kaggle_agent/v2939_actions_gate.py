"""V2939 immutable read-only rank-median safety gate; never trains or submits."""
from __future__ import annotations

import os
from datetime import datetime, timezone

import v2935_actions_gate as source_gate

ROOT = source_gate.ROOT
BASELINE = source_gate.BASELINE
CHAMPION_SHA = source_gate.CHAMPION_SHA
SOURCES = dict(source_gate.SOURCES)
COMPETITION = "playground-series-s6e10"
BRANCH = "feature/v29-autonomous-evolution"
FROZEN = {
    "competition": COMPETITION,
    "version": "V2.9.39",
    "stage": "preregistered_not_started",
    "hypothesis": "conservative_order_statistic_median_of_champion_v2916_v2930_ranks",
    "model_family": "read_only_robust_three_source_median_no_fit",
    "execution": "guarded_one_shot_readonly_cpu",
    "notebook_created": False,
    "training_started": False,
    "submitted": False,
    "submission_attempted": False,
    "evaluation_only": True,
    "no_new_model_fit": True,
    "cpu_only": True,
    "run_reservation_id": None,
    "outer_folds": 3,
    "seed": 20261018,
    "train_rows": 699635,
    "test_rows": 299844,
    "rank_method": "scipy.stats.rankdata(method=average)/n",
    "median_rule": "np.median(np.stack([r0,r16,r30],axis=0),axis=0)",
    "candidate_formula": "r0+0.15*(median(r0,r16,r30)-r0)",
    "fixed_shrinkage": 0.15,
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
    "preregistration_report": "reports/v2_9_39_preregistration.md",
    "failure_diagnostic": "reports/v2_9_38_failure_diagnostic.md",
}

def preregistration():
    read = source_gate.common.read
    state = read("state/v2_9_39_status.json")
    if any(state.get(key) != value for key, value in FROZEN.items()):
        raise RuntimeError("V2939 frozen preregistration or source SHA mismatch")
    if state.get("full_cv") is not None or state.get("promotable") is not None:
        raise RuntimeError("V2939 already evaluated; never evaluate twice")
    previous = state.get("previous_v2938") or {}
    s38 = read("state/v2_9_38_status.json")
    if (s38.get("stage") != "skipped_no_cv_gain"
        or str(s38.get("run_reservation_id")) != "40164585333"
        or s38.get("submitted") is not False
        or abs(float(s38.get("full_cv", 0)) - 0.9611212529755768) > 1e-12
        or previous.get("stage") != "skipped_no_cv_gain"
        or previous.get("run_id") != 40164585333
        or abs(float(previous.get("gain", 0)) + 0.0001637869576567308) > 1e-12):
        raise RuntimeError("V2938 terminal one-shot evidence mismatch")
    for name, required in (
        ("state/v2_9_35_status.json", "skipped_no_cv_gain"),
        ("state/v2_9_36_status.json", "kernel_failed"),
        ("state/v2_9_37_status.json", "skipped_no_cv_gain"),
    ):
        prior = read(name)
        if prior.get("stage") != required or prior.get("submitted") is not False:
            raise RuntimeError("Prior terminal experiment changed: " + name)
    official = read("state/v2_9_15_submission.json")
    champion = read("state/v2_9_15_status.json")
    ranking = read("state/score_rank_latest.json")
    if (official.get("stage") != "complete"
        or official.get("submitted") is not True
        or official.get("prediction_hash") != CHAMPION_SHA
        or champion.get("candidate_test_hash") != CHAMPION_SHA
        or str((official.get("submission") or {}).get("ref")) != "57446286"
        or (official.get("submission") or {}).get("status") != "SubmissionStatus.COMPLETE"
        or float(official.get("official_lb", 0)) != 0.96099
        or str((ranking.get("score") or {}).get("submission_id")) != "57446286"
        or float((ranking.get("score") or {}).get("public_score", 0)) != 0.96099):
        raise RuntimeError("Official V2915 submission/score/hash evidence mismatch")
    ladder = read("state/dual_ladder.json")
    deadline = datetime.fromisoformat(str(ladder.get("active_deadline", "")).replace("Z", "+00:00"))
    if (ladder.get("active_competition") != COMPETITION
        or deadline.tzinfo is None or datetime.now(timezone.utc) >= deadline):
        raise RuntimeError("Competition deadline reached or ambiguous")
    return state

def main():
    if os.environ.get("GITHUB_REPOSITORY") != "mjw3177109/kaggletest":
        raise RuntimeError("Wrong GitHub repository")
    source_gate.common.assert_no_other_actions()
    source_gate.assert_remote_head()
    preregistration()
    source_gate.authenticated_idle()
    source_gate.common.assert_no_other_actions()
    source_gate.assert_remote_head()
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
        stream.write("allowed=true\\n".replace("\\n", "\n"))
    print("V2939 authenticated read-only gate passed; no Notebook, training or submission")

if __name__ == "__main__":
    main()
