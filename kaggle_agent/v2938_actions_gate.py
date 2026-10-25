"""V2938 fail-closed preflight and one-shot GitHub run reservation.
No Kaggle Notebook creation and no competition submission. All owner kernels checked.
"""
from __future__ import annotations
import argparse, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
import v2936_actions_gate as inherited

ROOT=Path(__file__).resolve().parents[1]
STATUS=ROOT/"state/v2_9_38_status.json"
BRANCH="feature/v29-autonomous-evolution"
COMPETITION="playground-series-s6e10"
ACTIVE=("in_progress","queued","waiting","requested","pending")
CHAMPION_SHA="906185547dcd5f0cac3ac173de644fd9997f20ac863c6745bcb1fd3e9a89e5f7"
TERMINAL36="mjw3177109/kas-v2936-original37-xgboost-dart-t4-s6e10"
FROZEN={"loss_function":"Logloss","eval_metric":"AUC","boosting_type":"Ordered",
 "task_type":"CPU","iterations":600,"depth":6,"learning_rate":0.035,
 "l2_leaf_reg":5.0,"random_strength":1.0,"bootstrap_type":"Bayesian",
 "bagging_temperature":0.5,"rsm":0.85,"border_count":128,
 "thread_count":4,"allow_writing_files":False,"verbose":False}
ENV={"python":"3.11","catboost":"1.2.8","numpy":"2.2.6","pandas":"2.2.3",
     "scipy":"1.15.3","sklearn":"1.6.1"}

def read(name):
    return json.loads((ROOT/name).read_text(encoding="utf-8"))

def fingerprint(state):
    trainer=ROOT/"kaggle_agent/v2938_ordered_train.py"
    if not trainer.is_file(): raise RuntimeError("V2938 trainer missing")
    frozen={k:state[k] for k in ("competition","version","model_family",
        "model_parameters","seed","model_seed","outer_folds","inner_target_folds",
        "baseline_features","protected_target_encoded_features","blend","promotion")}
    frozen["trainer_sha256"]=hashlib.sha256(trainer.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(frozen,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def prove_cpu_synthetic_preflight():
    """Version + frozen Ordered fitting capability, using synthetic data only."""
    import platform, catboost, numpy, pandas, scipy, sklearn
    actual={"python":".".join(platform.python_version_tuple()[:2]),
        "catboost":catboost.__version__,"numpy":numpy.__version__,
        "pandas":pandas.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__}
    if actual!=ENV:
        raise RuntimeError("V2938 pinned CPU environment mismatch: "+repr(actual))
    from catboost import CatBoostClassifier
    rng=numpy.random.default_rng(20261106)
    x=rng.normal(size=(128,37)).astype("float32")
    y=(x[:,0]+x[:,1]>0).astype("int8")
    model=CatBoostClassifier(**dict(FROZEN,iterations=5),random_seed=20261106)
    model.fit(x,y)
    pred=model.predict_proba(x[:8])[:,1]
    if model.tree_count_!=5 or not numpy.isfinite(pred).all():
        raise RuntimeError("V2938 Ordered CPU synthetic fit failed")
    print("V2938 pinned CPU Ordered synthetic preflight PASS")

def assert_preregistration():
    s=read("state/v2_9_38_status.json")
    if s.get("stage")!="preregistered_not_started":
        raise RuntimeError("V2938 run already reserved or terminal; never train again")
    if (s.get("model_parameters")!=FROZEN or s.get("environment")!=dict(ENV,device="cpu")
        or s.get("version")!="V2.9.38" or s.get("competition")!=COMPETITION
        or s.get("model_seed")!=20261106 or s.get("seed")!=20261018
        or s.get("outer_folds")!=3 or s.get("inner_target_folds")!=5
        or s.get("baseline_features")!=37 or s.get("protected_target_encoded_features")!=3
        or s.get("blend")!={"v2915_rank":0.85,"v2938_rank":0.15}
        or s.get("promotion")!={"min_cv":0.96125635576833,"min_gain_vs_v2915":0.00005,
                                "min_positive_folds":2,"worst_fold_strict_gt":-0.00005}
        or s.get("submission_policy")!="evaluation_only_no_submission"):
        raise RuntimeError("V2938 immutable preregistration mismatch")
    if any(s.get(k) is not False for k in ("notebook_created","training_started",
                                           "submitted","submission_attempted")):
        raise RuntimeError("V2938 prior action already attempted")
    if s.get("run_reservation_id") is not None or s.get("run_reservation_sha256") is not None:
        raise RuntimeError("V2938 already reserved; never retrain")
    p=read("state/v2_9_36_status.json")
    if (p.get("stage")!="kernel_failed" or p.get("submitted") is not False
        or "xgboost==3.0.5 required; got 3.4.1" not in str(p.get("error"))):
        raise RuntimeError("V2936 terminal version-failure evidence changed")
    prev=read("state/v2_9_37_status.json")
    if (prev.get("stage")!="skipped_no_cv_gain" or prev.get("submitted") is not False
        or str(prev.get("run_reservation_id"))!="40157906219"
        or abs(float(prev.get("gain_vs_champion",0))-(-0.00004434252994589549))>1e-12):
        raise RuntimeError("V2937 terminal OOF evidence changed")
    champion=read("state/v2_9_15_status.json")
    official=read("state/v2_9_15_submission.json")
    rank=read("state/score_rank_latest.json")
    if (champion.get("candidate_test_hash")!=CHAMPION_SHA
        or official.get("prediction_hash")!=CHAMPION_SHA
        or official.get("submitted") is not True or official.get("stage")!="complete"
        or str((official.get("submission") or {}).get("ref"))!="57446286"
        or (official.get("submission") or {}).get("status")!="SubmissionStatus.COMPLETE"
        or abs(float(official.get("official_lb",0))-0.96099)>1e-9
        or str((rank.get("score") or {}).get("submission_id"))!="57446286"):
        raise RuntimeError("Official V2915 champion submission/hash changed")
    ladder=read("state/dual_ladder.json")
    deadline=datetime.fromisoformat(ladder["active_deadline"].replace("Z","+00:00"))
    if (ladder.get("active_competition")!=COMPETITION or deadline.tzinfo is None
        or datetime.now(timezone.utc)>=deadline):
        raise RuntimeError("Competition deadline reached or unverified")
    if os.getenv("GITHUB_REF_NAME")!=BRANCH or not os.getenv("GITHUB_RUN_ID"):
        raise RuntimeError("Wrong branch or absent GitHub run identity")
    return s

def assert_owner_idle():
    # The previous gate skipped its own V2936 slug. V2938 has NO Notebook:
    # force all owned refs, including V2936, through the owner-wide status scan.
    inherited.KERNEL="__V2938_NO_KERNEL_TO_EXCLUDE__"
    inherited.assert_kaggle_idle_and_official()
    # V2936 is mandatory even if Kaggle's list omitted it.
    terminal=inherited.kaggle(["kernels","status",TERMINAL36],timeout=90)
    if "KernelWorkerStatus.ERROR" not in terminal:
        raise RuntimeError("V2936 owner terminal witness changed")

def assert_global_idle():
    inherited.assert_no_other_actions()

def reserve():
    s=assert_preregistration()
    assert_global_idle()
    assert_owner_idle()
    assert_global_idle()
    s["stage"]="run_reserved"
    s["run_reservation_id"]=str(os.environ["GITHUB_RUN_ID"])
    s["run_reservation_sha256"]=fingerprint(s)
    s["training_started"]=False
    s["updated_at"]=datetime.now(timezone.utc).isoformat()
    STATUS.write_text(json.dumps(s,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("V2938 one-shot reservation prepared for run",s["run_reservation_id"])
    return s

def verify_reserved():
    s=read("state/v2_9_38_status.json")
    if (s.get("stage")!="run_reserved"
        or str(s.get("run_reservation_id"))!=os.getenv("GITHUB_RUN_ID")
        or s.get("run_reservation_sha256")!=fingerprint(s)
        or s.get("submitted") is not False or s.get("training_started") is not False):
        raise RuntimeError("V2938 durable one-shot reservation mismatch")
    assert_global_idle()
    assert_owner_idle()
    assert_global_idle()
    return s

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["preflight","reserve","verify"])
    args=parser.parse_args()
    if args.mode=="preflight":
        prove_cpu_synthetic_preflight()
        assert_preregistration();assert_global_idle();assert_owner_idle();assert_global_idle()
    elif args.mode=="reserve":reserve()
    else:verify_reserved()
