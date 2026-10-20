"""V2.9.10: paired full-size OOF robustness audit (no new CV folds, no submit)."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import v298_fusion as p
import v299_teacher_fusion as teacher
from v291_ledger import read,write

ROOT=p.ROOT
STATUS=ROOT/"state/v2_9_10_robustness_status.json"
REPORT=ROOT/"reports/v2_9_10_robustness.md"
SEED=20261020
REPLICATES=120
SCORE_EXPECTED={
    "v296":0.9609541991872741,
    "v298":0.9610183558267038,
    "v299":0.9610561936366048,
}
TERMINAL={"robustness_complete","audit_failed"}

def matched_bootstrap(y,incumbent,candidates,replicates=REPLICATES,seed=SEED):
    y=np.asarray(y,dtype=np.int8)
    n=len(y)
    if (np.unique(y).tolist()!=[0,1] or n<60 or replicates<20):
        raise ValueError("Paired stratified bootstrap requires both labels and >=20 samples")
    vectors={"incumbent":incumbent,**candidates}
    checked={k:p.vector(v,n,k) for k,v in vectors.items()}
    index=np.zeros(n,dtype=np.uint8)
    splitter=StratifiedKFold(3,shuffle=True,random_state=20261018)
    for fold,(_,valid) in enumerate(splitter.split(np.zeros(n,dtype=np.uint8),y)):
        index[valid]=fold
    strata=[np.flatnonzero((index==fold)&(y==label))
            for fold in range(3) for label in (0,1)]
    if any(len(ix)<2 for ix in strata):
        raise ValueError("Insufficient fold/target rows for bootstrap")
    rng=np.random.default_rng(seed)
    draws={key:[] for key in candidates}
    for _ in range(replicates):
        take=np.concatenate([rng.choice(ix,size=len(ix),replace=True) for ix in strata])
        yy=y[take]
        previous=float(roc_auc_score(yy,checked["incumbent"][take]))
        for key in candidates:
            observed=float(roc_auc_score(yy,checked[key][take]))
            draws[key].append(observed-previous)
    return {key:{"lower_025":float(np.quantile(v,.025)),
                 "median":float(np.quantile(v,.5)),
                 "upper_975":float(np.quantile(v,.975)),
                 "positive_fraction":float(np.mean(np.asarray(v)>0)),
                 "draws":replicates,"seed":seed}
            for key,v in draws.items()}

def subgroup_audit(train,y,incumbent,candidate,min_group=10000):
    """Only aggregate slices; no row IDs or personal records retained."""
    columns=("Type of Travel","Class","Customer Type","Gender")
    rows=[]
    for column in columns:
        if column not in train:raise ValueError(f"Missing competition subgroup column: {column}")
        cats=train[column].astype("string").fillna("__NA__")
        for value,count in cats.value_counts().items():
            if count<min_group:continue
            sub=np.flatnonzero((cats==value).to_numpy())
            if len(np.unique(y[sub]))!=2:continue
            old=float(roc_auc_score(y[sub],incumbent[sub]))
            new=float(roc_auc_score(y[sub],candidate[sub]))
            rows.append({"feature":column,"group":str(value),"rows":len(sub),
                         "baseline_auc":old,"candidate_auc":new,
                         "gain":new-old})
    return sorted(rows,key=lambda r:r["gain"])

def reconstruct(train,test,archive,states,tstatus,work):
    train_id,test_id,y=p.validate_official(train,test)
    # Existing full V298 evaluator performs baseline provenance and ID validation.
    r298=p.evaluate(train,test,archive,states,audit_legacy_order=True)
    if abs(r298["candidate_cv"]-SCORE_EXPECTED["v298"])>0.000002:
        raise ValueError("V298 OOF score changed from immutable prior audit")
    old25,old28=archive["v25"],archive["v28"]
    _,oofs,tests=p.verify_sources(old25,old28,states["v25"],states["v28"],
                                  archive["v25_submission"],archive["v28_submission"])
    base=.35*p.rank(oofs["v25"])+.65*p.rank(oofs["v28"])
    real=p.align(archive["v296"]["oof_realmlp"],
                 archive["v296"]["train_id"],train_id,"v296")
    conditional=p.align(archive["v297"]["oof_conditional"],
                        archive["v297"]["train_id"],train_id,"v297")
    t,_,teacher_record=teacher.get_teacher(work,tstatus,train_id,test_id,y)
    v296=.8*p.rank(base)+.2*p.rank(real)
    v298=.65*p.rank(base)+.2*p.rank(real)+.15*p.rank(conditional)
    v299=.60*p.rank(base)+.20*p.rank(real)+.15*p.rank(conditional)+.05*p.rank(t)
    out={"v296":v296,"v298":v298,"v299":v299}
    measured={k:float(roc_auc_score(y,v)) for k,v in out.items()}
    for k,auc in measured.items():
        if abs(auc-SCORE_EXPECTED[k])>0.000002:
            raise ValueError(f"{k} reproduced CV mismatch {auc} != {SCORE_EXPECTED[k]}")
    return y,out,measured,teacher_record

def run():
    if read(STATUS).get("stage") in TERMINAL:
        print("Already evaluated paired bootstrap; no repeat");return
    if read(teacher.STATUS).get("stage") not in teacher.TERMINAL:
        print("V299 is not terminal; no overlap");return
    p.check_competition()
    paths={"v25":"state/v2_5_status.json","v28":"state/v2_8_status.json",
           "v296":"state/v2_9_6_status.json","v297":"state/v2_9_7_status.json"}
    states={k:read(ROOT/v) for k,v in paths.items()}
    p.check_provenance(states)
    ts=read(ROOT/"state/v2_9_4_status.json")
    with tempfile.TemporaryDirectory(prefix="kas_v210_") as folder:
        train,test,archive,source_files=p.download_inputs(folder,states)
        y,vectors,measured,teacher_file=reconstruct(train,test,archive,states,ts,folder)
    source_files["v294"]=teacher_file
    confidence=matched_bootstrap(y,vectors["v296"],
            {"v298":vectors["v298"],"v299":vectors["v299"]})
    segments=subgroup_audit(train,y,vectors["v296"],vectors["v299"])
    result={"stage":"robustness_complete","competition":p.SLUG,
            "verified_at_utc":p.timestamp(),"validation":p.VALIDATION,
            "source_artifacts":source_files,"auc":measured,
            "paired_bootstrap":confidence,"worst_subgroups":segments[:8],
            "subgroups_audited":len(segments),
            "submission":False,"new_model_training":False,
            "warning":"Bootstrap reuses original OOF. It is not a new CV seed or an official LB."}
    write(STATUS,result)
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text("# V2.9.10 paired OOF robustness audit\n\n"
       +f"- UTC: {result['verified_at_utc']}\n"
       +f"- Threefold reproduced AUC: {measured}\n"
       +f"- Bootstrap (paired folds × targets, B={REPLICATES}): {confidence}\n"
       +f"- Worst subgroup deltas (aggregate only): {segments[:8]}\n"
       +"- These are resamples of existing OOF, not independent training or Public LB.\n"
       +"- No Kaggle Notebook launched or submission made.\n",encoding="utf-8")
    print(json.dumps({"status":result["stage"],"auc":measured,
                      "bootstrap":confidence,"worst_groups":segments[:5]},
                     ensure_ascii=False,indent=2))

if __name__=="__main__":
    run()
