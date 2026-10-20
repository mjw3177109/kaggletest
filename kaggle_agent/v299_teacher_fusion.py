"""V2.9.9: pre-registered archived V294 teacher complement; never train/submit."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import v298_fusion as p
from v291_ledger import read, write, record_experiment
from v293_blend_math import verify_sources, VALIDATION, N_TRAIN, N_TEST

STATUS=p.ROOT/"state/v2_9_9_status.json"
REPORT=p.ROOT/"reports/v2_9_9_teacher_audit.md"
WEIGHTS={"v293":.60,"realmlp":.20,"conditional":.15,"teacher_v294":.05}
MIN_GAIN=.00010
FOLD_FLOOR=-.00005
V294_SCORE=.9607525307822926
TERMINAL={"candidate_ready_unsubmitted","skipped_no_cv_gain","validation_blocked"}

def fuse_fixed(y,baseline,real,conditional,teacher,benchmark):
    """Exactly one fixed challenger; label-independent weights."""
    y=np.asarray(y,dtype=np.int8)
    base=p.rank(baseline)
    old=.8*base+.2*p.rank(real)
    incumbent=float(roc_auc_score(y,old))
    if abs(incumbent-benchmark)>0.000002:
        raise ValueError("Official V296 OOF reconstruction mismatch")
    proposed=(WEIGHTS["v293"]*base + WEIGHTS["realmlp"]*p.rank(real)
              + WEIGHTS["conditional"]*p.rank(conditional)
              + WEIGHTS["teacher_v294"]*p.rank(teacher))
    auc=float(roc_auc_score(y,proposed))
    folds=[];gains=[]
    for _,iv in StratifiedKFold(3,shuffle=True,random_state=20261018).split(
            np.zeros(len(y),dtype=np.uint8),y):
        folds.append(float(roc_auc_score(y[iv],proposed[iv])))
        gains.append(float(folds[-1]-roc_auc_score(y[iv],old[iv])))
    passed=(auc-incumbent>=MIN_GAIN and sum(x>0 for x in gains)>=2
            and min(gains)>FOLD_FLOOR)
    return {"stage":"candidate_ready_unsubmitted" if passed else "skipped_no_cv_gain",
            "candidate_cv":auc,"incumbent_v296_cv":incumbent,"gain":auc-incumbent,
            "fold_auc":folds,"fold_gain":gains,"positive_folds":sum(x>0 for x in gains),
            "weights":WEIGHTS,"min_gain":MIN_GAIN,"submitted":False}

def get_teacher(work, status, train_id, test_id, y):
    """V294 predates ID archives, so require canonical target and test order."""
    m=status.get("metrics") or {}
    if (status.get("competition")!=p.SLUG or status.get("stage")!="skipped_no_cv_gain"
        or m.get("validation")!=VALIDATION or m.get("train_rows")!=N_TRAIN
        or m.get("test_rows")!=N_TEST or m.get("best_model")!="blend_rank"
        or m.get("outer_valid_used_for_early_stopping") is not False
        or status.get("kernel")!="mjw3177109/kas-v294-original-route-playground-series-s6e10"):
        raise ValueError("V294 teacher source provenance mismatch")
    folder=Path(work)/"v294";folder.mkdir(parents=True,exist_ok=True)
    p.checked_command(["kaggle","kernels","output",status["kernel"],
                       "-p",str(folder),"-o","-q"])
    path=p.unique_file(folder,"v294_oof.npz")
    arr=p.load_npz(path,("target","oof_lgb","oof_xgb"))
    if not np.array_equal(arr["target"],y):
        raise ValueError("V294 historical OOF target order does not match official train")
    teacher=.5*p.rank(p.vector(arr["oof_lgb"],N_TRAIN,"v294 LGB OOF"))+ \
            .5*p.rank(p.vector(arr["oof_xgb"],N_TRAIN,"v294 XGB OOF"))
    if abs(roc_auc_score(y,teacher)-V294_SCORE)>0.000002:
        raise ValueError("Cannot reproduce V294 historical threefold CV")
    subfile=p.unique_file(folder,"candidate_submission.csv")
    sub=pd.read_csv(subfile)
    if list(sub.columns)!=["id","satisfaction"] or not np.array_equal(
            sub["id"].to_numpy(),test_id):
        raise ValueError("V294 test row ordering or submission schema mismatch")
    prediction=p.vector(sub["satisfaction"],N_TEST,"v294 test")
    return teacher,prediction,{"notebook":status["kernel"],"sha256":p.sha256(path),
                               "submission_sha256":p.sha256(subfile)}

def run():
    previous=read(STATUS)
    if previous.get("stage") in TERMINAL:
        print("V299 already evaluated; no duplicate");return
    if read(p.STATUS).get("stage") not in p.TERMINAL:
        print("V298 audit not terminal; avoid concurrent artifact audit");return
    p.check_competition()
    path={"v25":"state/v2_5_status.json","v28":"state/v2_8_status.json",
          "v296":"state/v2_9_6_status.json","v297":"state/v2_9_7_status.json"}
    statuses={k:read(p.ROOT/name) for k,name in path.items()}
    p.check_provenance(statuses)
    teacher_status=read(p.ROOT/"state/v2_9_4_status.json")
    with tempfile.TemporaryDirectory(prefix="kas_v299_") as scratch:
        train,test,archive,files=p.download_inputs(scratch,statuses)
        train_id,test_id,y=p.validate_official(train,test)
        # Source-order audit is mandatory for 3 legacy OOF archives.
        p.evaluate(train,test,archive,statuses,audit_legacy_order=True)
        teacher,test_teacher,tinfo=get_teacher(scratch,teacher_status,train_id,test_id,y)
        files["v294"]=tinfo
        _,oofs,tests=verify_sources(archive["v25"],archive["v28"],
                                   statuses["v25"],statuses["v28"],
                                   archive["v25_submission"],archive["v28_submission"])
        baseline=.35*p.rank(oofs["v25"])+.65*p.rank(oofs["v28"])
        real=p.align(archive["v296"]["oof_realmlp"],archive["v296"]["train_id"],
                     train_id,"V296 train")
        conditional=p.align(archive["v297"]["oof_conditional"],archive["v297"]["train_id"],
                            train_id,"V297 train")
        result=fuse_fixed(y,baseline,real,conditional,teacher,p.EXPECTED_V296)
        if result["stage"]=="candidate_ready_unsubmitted":
            # This hash is evidence, not a Kaggle submit file.
            base_test=.35*tests["v25"]+.65*tests["v28"]
            ntest=N_TEST
            v296_test=p.align(archive["v296"]["pred_realmlp"],
                              archive["v296"]["test_id"],test_id,"V296 test")
            v297_test=p.align(archive["v297"]["test_conditional"],
                              archive["v297"]["test_id"],test_id,"V297 test")
            blended=(.60*p.rank(base_test)+.20*p.rank(v296_test)
                     +.15*p.rank(v297_test)+.05*p.rank(test_teacher))
            import hashlib
            result["candidate_test_hash"]=hashlib.sha256(
                blended.astype("<f8").tobytes()).hexdigest()
    result.update({"competition":p.SLUG,"validation":VALIDATION,
                   "verified_at_utc":p.timestamp(),"source_artifacts":files})
    write(STATUS,result)
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text("# V2.9.9 archived teacher complementary fusion\n\n"
                      +f"- UTC: {result['verified_at_utc']}\n"
                      +f"- Candidate status: {result['stage']} (NOT submitted)\n"
                      +f"- Baseline V296: {result['incumbent_v296_cv']}\n"
                      +f"- Fixed weights: {WEIGHTS}\n"
                      +f"- Full 3fold CV: {result['candidate_cv']}\n"
                      +f"- Gain: {result['gain']}\n"
                      +f"- Fold gains: {result['fold_gain']}\n"
                      +"- No Kaggle kernels launched or scores submitted.\n",encoding="utf-8")
    record_experiment(p.LEDGER,{"competition":p.SLUG,
       "candidate":"v299_archived_teacher_fixed_four_model",
       "validation":VALIDATION,"seed":20261018,
       "parameters":{"weights":WEIGHTS,"min_gain":MIN_GAIN},
       "status":"completed" if result["stage"]=="candidate_ready_unsubmitted"
                else "skipped_no_cv_gain",
       "completed_at":result["verified_at_utc"],"cv":result["candidate_cv"],
       "gpu_seconds":0,
       "kernel_sources":[statuses[k]["kernel"] for k in ("v25","v28","v296","v297")]
                        +[teacher_status["kernel"]]})
    print(json.dumps({k:result[k] for k in ("stage","candidate_cv","gain","fold_gain")},
                     ensure_ascii=False,indent=2))

if __name__=="__main__":
    run()
