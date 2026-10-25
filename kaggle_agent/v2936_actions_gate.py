"""V2936 fail-closed gate: global GitHub Actions, Kaggle owner, official champion, deadline."""
from __future__ import annotations
import csv
import io
import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
KERNEL="mjw3177109/kas-v2936-original37-xgboost-dart-t4-s6e10"
COMPETITION="playground-series-s6e10"
ACTIVE=("in_progress","queued","waiting","requested","pending")
# Rate-limit prevention: every authenticated Kaggle request remains mandatory.
# Never treat a 429 as idle or silently skip any owner kernel.
_MIN_KAGGLE_CALL_GAP_SECONDS=3.0
_last_kaggle_call_at=None


def read(path):
    return json.loads((ROOT/path).read_text(encoding="utf-8"))


def github_runs(status):
    token=os.environ.get("GH_API_TOKEN","")
    repo=os.environ.get("GITHUB_REPOSITORY","")
    if not token or repo!="mjw3177109/kaggletest":
        raise RuntimeError("GitHub Actions authentication unavailable")
    q=urllib.parse.urlencode({"status":status,"per_page":100})
    req=urllib.request.Request(
        f"https://api.github.com/repos/{repo}/actions/runs?{q}",
        headers={"Authorization":"Bearer "+token,
                 "Accept":"application/vnd.github+json",
                 "User-Agent":"kaggle-autoscientist-v2936"})
    with urllib.request.urlopen(req,timeout=30) as response:
        result=json.load(response)
    runs=result.get("workflow_runs") or []
    if int(result.get("total_count",-1))>len(runs):
        raise RuntimeError("GitHub active Actions result incomplete; fail closed")
    return runs


def assert_no_other_actions():
    own=int(os.environ.get("GITHUB_RUN_ID","0"))
    if not own:
        raise RuntimeError("No authenticated current GitHub run ID")
    for status in ACTIVE:
        others=[r for r in github_runs(status) if int(r.get("id",0))!=own]
        if others:
            raise RuntimeError("Other global GitHub Actions active; refuse V2936 work")


def kaggle(args,timeout=120):
    global _last_kaggle_call_at
    # Bound the request rate across list/submission/status reads in this process.
    # The gate still checks every owned kernel and fails closed on any 429.
    if _last_kaggle_call_at is not None:
        remaining=_MIN_KAGGLE_CALL_GAP_SECONDS-(time.monotonic()-_last_kaggle_call_at)
        if remaining>0:
            time.sleep(remaining)
    _last_kaggle_call_at=time.monotonic()
    p=subprocess.run(["kaggle",*args],capture_output=True,text=True,timeout=timeout)
    if p.returncode:
        diagnostic=(p.stderr or p.stdout or "").lower()
        category=("not_found" if any(x in diagnostic for x in ("404","not found","notfound"))
                  else "forbidden" if any(x in diagnostic for x in ("403","forbidden"))
                  else "rate_limited" if "429" in diagnostic else "unknown")
        if category=="not_found" and args[:2]==["kernels","status"]:
            raise LookupError("Kaggle kernel status authenticated NOT_FOUND")
        raise RuntimeError("Authenticated read-only Kaggle CLI unavailable; category="+category+
                           "; exit="+str(p.returncode))
    return p.stdout


def assert_kaggle_idle_and_official():
    if os.environ.get("KAGGLE_USERNAME")!="mjw3177109" or not os.environ.get("KAGGLE_API_TOKEN"):
        raise RuntimeError("Kaggle account identity unavailable")
    submissions=kaggle(["competitions","submissions","-c",COMPETITION])
    if "57446286" not in submissions or "0.96099" not in submissions:
        raise RuntimeError("Official V2915 submission/score not authenticated")
    listing=kaggle(["kernels","list","--mine","--page-size","100","-v"])
    rows=list(csv.DictReader(io.StringIO(listing)))
    if not rows or len(rows)>=100 or "ref" not in rows[0]:
        raise RuntimeError("Owned Kaggle Notebook list incomplete or ambiguous")
    refs=[str(row.get("ref","")).strip() for row in rows]
    if any(not ref.startswith("mjw3177109/") for ref in refs):
        raise RuntimeError("Kaggle Notebook owner mismatch")
    known={
        "mjw3177109/kas-v2927-travel-gated-dual-realmlp-t4-s6e10",
        "mjw3177109/kas-v2928-service-factor-pca-lgbm-cpu-s6e10",
        "mjw3177109/kas-v2929-fullservice-ordinal-realmlp-t4-s6e10",
        "mjw3177109/kas-v2930-conditional-crossfit-te-realmlp-t4-s6e10",
        "mjw3177109/kas-v2931-foldlocal-mi-pruned-realmlp-t4-s6e10",
        "mjw3177109/kas-v2932-attention-baseline37-t4-s6e10",
        "mjw3177109/kas-v2933-pairwise-ranknet-mlp-t4-s6e10",
        "mjw3177109/kas-v2934-explicit-secondorder-deepfm-t4-s6e10",
    }
    for ref in sorted((set(refs)|known)-{KERNEL}):
        try:
            status=kaggle(["kernels","status",ref],timeout=90)
        except LookupError as exc:
            if ref in known or "s6e10" in ref or "playground-series-s6e10" in ref:
                raise RuntimeError("Required S6E10 Kaggle Notebook status NOT_FOUND: "+ref) from exc
            print("Authenticated NOT_FOUND for historical non-S6E10 Notebook, not active: "+ref)
            continue
        except RuntimeError as exc:
            raise RuntimeError("Kaggle owner Notebook status inaccessible for "+ref) from exc
        if any("KernelWorkerStatus."+x in status for x in
               ("QUEUED","RUNNING","PENDING","CANCEL_REQUESTED","NEW_SCRIPT")):
            raise RuntimeError("Another Kaggle Notebook may be active: refuse V2936: "+ref)
        if "KernelWorkerStatus.CANCEL_ACKNOWLEDGED" in status:
            # Kaggle reports an acknowledged cancellation, not a successful run.
            # Require two authenticated identical reads and an unrelated old slug.
            if ref in known or "s6e10" in ref or "playground-series-s6e10" in ref:
                raise RuntimeError("S6E10 canceled Notebook requires manual review: "+ref)
            confirmation=kaggle(["kernels","status",ref],timeout=90)
            if "KernelWorkerStatus.CANCEL_ACKNOWLEDGED" not in confirmation:
                raise RuntimeError("Kaggle cancellation changed during verification: "+ref)
            print("Historical non-S6E10 Notebook cancellation acknowledged twice: "+ref)
            continue
        if not any("KernelWorkerStatus."+x in status for x in ("COMPLETE","ERROR","CANCELLED")):
            raise RuntimeError("Unknown Kaggle Notebook status for "+ref+": "+repr(status.strip()[:160]))


def assert_preregistration():
    s=read("state/v2_9_36_status.json")
    p=read("state/v2_9_35_status.json")
    c=read("state/v2_9_15_status.json")
    o=read("state/v2_9_15_submission.json")
    rank=read("state/score_rank_latest.json")
    ladder=read("state/dual_ladder.json")
    deadline=datetime.fromisoformat(ladder["active_deadline"].replace("Z","+00:00"))
    frozen={
        "competition":COMPETITION,"version":"V2.9.36",
        "intended_kernel":KERNEL,
        "hypothesis":"original37_dropout_regularized_dart_boosted_trees_distinct_from_v2920_routeprofile_hist",
        "model_family":"XGBoost_DART_T4_original37",
        "feature_family":"v2915_original37_3_inner5fold_TE_no_extra_features",
        "baseline_features":37,"protected_target_encoded_features":3,
        "outer_folds":3,"inner_target_folds":5,"seed":20261018,
        "model_seed":20261104,"early_stopping":False,
        "weight_search":False,"feature_search":False,
        "model_parameters":{
            "booster":"dart","objective":"binary:logistic","eval_metric":"auc",
            "tree_method":"hist","device":"cuda","n_estimators":900,
            "learning_rate":0.035,"max_depth":4,"min_child_weight":10,
            "subsample":0.85,"colsample_bytree":0.85,"max_bin":256,
            "reg_lambda":5.0,"reg_alpha":0.0,"gamma":0.0,
            "rate_drop":0.05,"skip_drop":0.0,"sample_type":"uniform",
            "normalize_type":"tree","n_jobs":4},
        "baseline_cv":0.9612850399332336,
        "official_submission_id":57446286,"official_public_lb":0.96099,
        "blend":{"v2915_rank":0.85,"v2936_rank":0.15},
        "promotion":{"min_cv":0.96125635576833,
                     "min_gain_vs_v2915":0.00005,
                     "min_positive_folds":2,
                     "worst_fold_strict_gt":-0.00005},
        "submission_policy":"evaluation_only_no_submission",
    }
    if any(s.get(k)!=v for k,v in frozen.items()):
        raise RuntimeError("V2936 frozen preregistration mismatch")
    if any(s.get(k) is not False for k in (
            "submitted","submission_attempted","external_labels","pseudo_labels",
            "route_features","pca_features","ordinal_threshold_features",
            "new_target_encoding_features","feature_pruning")):
        raise RuntimeError("V2936 prohibited option enabled")
    stored=s.get("kernel")
    if stored not in ("",None,KERNEL):
        raise RuntimeError("V2936 Notebook slug changed")
    if s.get("stage") not in {"preregistered_not_started","kernel_queued",
                              "kernel_running","pushing_kernel"}:
        raise RuntimeError("V2936 terminal or unknown stage")
    if (s.get("stage")=="preregistered_not_started" and
        (stored or s.get("notebook_created") is not False
         or s.get("training_started") is not False)):
        raise RuntimeError("V2936 pristine preregistration mismatch")
    if s.get("stage")!="preregistered_not_started" and stored!=KERNEL:
        raise RuntimeError("V2936 active state lacks durable owner")
    if (deadline.tzinfo is None or datetime.now(timezone.utc)>=deadline
        or ladder.get("active_competition")!=COMPETITION):
        raise RuntimeError("Kaggle competition deadline gate blocked")
    if (p.get("stage")!="skipped_no_cv_gain" or p.get("submitted") is not False
        or abs(float(p.get("full_cv",0))-0.9613302676316021)>1e-12
        or p.get("candidate_oof_sha256")!="e3b7b0cc288e995393c8ab688bfa7e2683f3eb1643ad05fa8dfd0e18d34d9b32"):
        raise RuntimeError("V2935 terminal evidence mismatch")
    if (c.get("candidate_test_hash")!=o.get("prediction_hash")
        or o.get("stage")!="complete" or o.get("submitted") is not True
        or str((o.get("submission") or {}).get("ref"))!="57446286"
        or abs(float(o.get("official_lb",0))-0.96099)>1e-9
        or str((rank.get("score") or {}).get("submission_id"))!="57446286"
        or abs(float((rank.get("score") or {}).get("public_score",0))-0.96099)>1e-9):
        raise RuntimeError("Official V2915 submission/score/hash evidence mismatch")
    return "recover_only" if stored else (os.environ.get("REQUESTED_MODE") or "run")


def main():
    assert_no_other_actions()
    mode=assert_preregistration()
    if mode not in ("run","recover_only"):
        raise RuntimeError("Unsupported V2936 launch mode")
    assert_kaggle_idle_and_official()
    assert_no_other_actions()
    with open(os.environ["GITHUB_OUTPUT"],"a",encoding="utf-8") as f:
        f.write("allowed=true\nmode="+mode+"\n")
    print("V2936 owner, official champion, deadline and global Actions gates passed")


if __name__=="__main__":
    main()
