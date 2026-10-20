"""V2.9.11 CPU LightGBM with distinct traveler gap / context TE.

One LightGBM, locked 3-fold split. Different from the original-teacher
V2.9.4 branch; NO external data and NO neural model blending in training.
"""
from __future__ import annotations
import json
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score

DATA=Path("/kaggle/input")
OUTPUT=Path("/kaggle/working")
SEED=20261018
FOLDS=3
CATS=("Gender","Customer Type","Type of Travel","Class")
SERVICES=("Inflight wifi service","Departure/Arrival time convenient",
"Ease of Online booking","Gate location","Food and drink","Online boarding",
"Seat comfort","Inflight entertainment","On-board service","Leg room service",
"Baggage handling","Checkin service","Cleanliness")
PAIRS=(
("Inflight wifi service","Online boarding"),
("Inflight wifi service","Ease of Online booking"),
("Online boarding","Seat comfort"),
("Seat comfort","Inflight entertainment"),
("Inflight entertainment","Cleanliness"),
)
FREQUENCY_COLUMNS=(
"pair_0","pair_1","pair_2","segment_Inflight wifi service",
"segment_Online boarding","segment_Inflight entertainment",
"gap_0","gap_1","gap_class_0","delay_distance_context")
MAX_RUNTIME_SECONDS=2900


def dump(name,thing):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/name).write_text(json.dumps(thing,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")


def data_file(name):
    found=[p for p in DATA.rglob(name) if "playground-series-s6e10" in str(p.parent)]
    if len(found)!=1:raise RuntimeError(f"Expected unique competition {name}: {len(found)}")
    return pd.read_csv(found[0])


def keys(frame):
    if not set(CATS)|set(SERVICES) <= set(frame):
        raise ValueError("Missing categorical/service columns")
    z=pd.DataFrame(index=frame.index)
    part=frame["Type of Travel"].astype("string").fillna("__NA__")
    client=frame["Customer Type"].astype("string").fillna("__NA__")
    cabin=frame["Class"].astype("string").fillna("__NA__")
    segment=part+"|"+client+"|"+cabin
    traveler=part+"|"+client
    for service in SERVICES:
        z["segment_"+service]=frame[service].astype("string").fillna("__NA__")+"|"+segment
    for i,(a,b) in enumerate(PAIRS):
        z[f"pair_{i}"]=(
            frame[a].astype("string").fillna("__NA__")+"|"+
            frame[b].astype("string").fillna("__NA__")+"|"+traveler)
    for numeric in ("Age","Flight Distance"):
        for cls in ("Type of Travel","Customer Type","Class"):
            z[f"numeric_{numeric}_{cls}"]=(
                frame[numeric].astype("string").fillna("__NA__")+"|"+
                frame[cls].astype("string").fillna("__NA__"))
    # Prespecified unsupervised bins and group interactions, never fitted on targets.
    age=pd.to_numeric(frame["Age"],errors="coerce").fillna(-1)
    distance=pd.to_numeric(frame["Flight Distance"],errors="coerce").fillna(-1)
    delay=pd.to_numeric(frame["Departure Delay in Minutes"],errors="coerce").fillna(-1)
    age_band=pd.cut(age,[-2,18,30,45,60,200],labels=False).astype("Int64").astype("string").fillna("__NA__")
    distance_band=pd.cut(distance,[-2,500,1500,3000,6500,20000],labels=False).astype("Int64").astype("string").fillna("__NA__")
    delay_band=pd.cut(delay,[-2,0,5,15,30,90,20000],labels=False).astype("Int64").astype("string").fillna("__NA__")
    for i,(a,b) in enumerate((
            ("Online boarding","Inflight wifi service"),
            ("Inflight entertainment","Seat comfort"),
            ("On-board service","Baggage handling"),
            ("Checkin service","Ease of Online booking"),
            )):
        gap=(pd.to_numeric(frame[a],errors="coerce")-
             pd.to_numeric(frame[b],errors="coerce")).fillna(0).clip(-6,6)
        as_key=gap.astype("int16").astype("string")
        z[f"gap_{i}"]=(as_key+"|"+traveler+"|"+age_band)
        z[f"gap_class_{i}"]=(as_key+"|"+cabin+"|"+distance_band)
    z["delay_distance_context"]=delay_band+"|"+distance_band+"|"+part+"|"+cabin
    z["age_distance_context"]=age_band+"|"+distance_band+"|"+client
    if z.isna().any().any() or z.columns.duplicated().any():
        raise ValueError("Unsafe or duplicate conditional keys")
    return z.astype(str)


def split_frequencies(tr,va,te,y,fold):
    """Frequency of a training row never includes that same row."""
    n=len(tr)
    holdouts=list(StratifiedKFold(5,shuffle=True,random_state=SEED+100+fold).split(
        np.zeros(n,dtype=np.uint8),y))
    result=[pd.DataFrame(index=f.index) for f in (tr,va,te)]
    for column in FREQUENCY_COLUMNS:
        for train_idx,hold_idx in holdouts:
            counts=tr.iloc[train_idx][column].value_counts(normalize=True)
            result[0].loc[tr.iloc[hold_idx].index,"freq_"+column]=np.log1p(
                tr.iloc[hold_idx][column].map(counts).fillna(0).to_numpy()*1e6)
        counts=tr[column].value_counts(normalize=True)
        for destination,frame in zip(result[1:],(va,te)):
            destination["freq_"+column]=np.log1p(frame[column].map(counts).fillna(0).to_numpy()*1e6)
    for f in result:
        if f.isna().any().any():raise ValueError("Incomplete OOF frequency feature")
    return result


def matrices(training,validation,test,labels,fold):
    kt,kv,ke=(keys(frame) for frame in (training,validation,test))
    enc=TargetEncoder(target_type="binary",smooth="auto",cv=5,
                      shuffle=True,random_state=SEED+fold)
    et=enc.fit_transform(kt,labels).astype("float32")
    ev=enc.transform(kv).astype("float32")
    ee=enc.transform(ke).astype("float32")
    names=[f"conditional_te_{i}" for i in range(et.shape[1])]
    frequencies=split_frequencies(kt,kv,ke,labels,fold)
    raw=(training,validation,test)
    out=[]
    for idx,frame in enumerate(raw):
        basic=pd.DataFrame(index=frame.index)
        for col in training:
            if col in ("id","satisfaction"):continue
            if col in CATS:
                vocab=pd.Index(training[col].astype("string").fillna("__NA__").unique())
                basic[col]=pd.Categorical(frame[col].astype("string").fillna("__NA__"),
                                           categories=vocab).codes.astype("float32")
            else:
                source=pd.to_numeric(training[col],errors="coerce")
                median=float(source.median())
                basic[col]=pd.to_numeric(frame[col],errors="coerce").fillna(median).astype("float32")
        encoded=pd.DataFrame((et,ev,ee)[idx],index=frame.index,columns=names)
        prepared=pd.concat((basic,encoded,frequencies[idx]),axis=1)
        if prepared.columns.duplicated().any() or not np.isfinite(prepared.to_numpy()).all():
            raise ValueError("Invalid fold-only frequency / conditional-TE features")
        out.append(prepared.astype("float32"))
    if not(out[0].columns.equals(out[1].columns) and out[0].columns.equals(out[2].columns)):
        raise ValueError("Fold feature schema mismatch")
    return out


def main():
    import lightgbm as lgb
    started=time.monotonic()
    train=data_file("train.csv")
    test=data_file("test.csv")
    sample=data_file("sample_submission.csv")
    if len(train)!=699635 or len(test)!=299844 or len(sample)!=len(test):
        raise ValueError("Unknown Kaggle dataset version")
    if list(sample.columns)!=["id","satisfaction"] or "satisfaction" not in train:
        raise ValueError("Unexpected Kaggle schema")
    y=pd.to_numeric(train["satisfaction"],errors="raise").to_numpy(dtype="int8")
    if not np.isin(y,[0,1]).all():raise ValueError("Nonbinary target")
    if not np.array_equal(sample["id"].to_numpy(),test["id"].to_numpy()):
        raise ValueError("Test IDs differ from sample submission")
    skf=StratifiedKFold(n_splits=FOLDS,shuffle=True,random_state=SEED)
    oof=np.full(len(y),np.nan,dtype="float32")
    pred=np.zeros(len(test),dtype="float64")
    folds=[]
    for fold,(fit,valid) in enumerate(skf.split(train,y)):
        if time.monotonic()-started>MAX_RUNTIME_SECONDS:
            raise TimeoutError("Exceeded budget before all three folds")
        xtr,xva,xt=matrices(train.iloc[fit],train.iloc[valid],test,y[fit],fold)
        m=lgb.LGBMClassifier(objective="binary",metric="auc",n_estimators=1100,
            learning_rate=0.035,num_leaves=39,min_child_samples=80,
            colsample_bytree=.80,subsample=.90,subsample_freq=1,
            reg_alpha=.20,reg_lambda=7.0,n_jobs=4,
            verbosity=-1,random_state=SEED+fold)
        m.fit(xtr,y[fit])  # Never use outer validation for early stopping.
        pv=m.predict_proba(xva)[:,1]
        pt=m.predict_proba(xt)[:,1]
        if not np.isfinite(pv).all() or not np.isfinite(pt).all():
            raise RuntimeError("Nonfinite predictions")
        oof[valid]=pv
        pred+=pt/FOLDS
        folds.append({"fold":fold,"validation_rows":len(valid),
                      "auc":float(roc_auc_score(y[valid],pv)),
                      "features":len(xtr.columns),
                      "top_importances":sorted(zip(xtr.columns,m.feature_importances_.astype(float)),
                                                key=lambda x:-x[1])[:15]})
        dump("v2911_progress.json",{"folds_finished":fold+1,"fold_results":folds})
        print("V2911 conditional GBDT",fold,folds[-1]["auc"],flush=True)
        del xtr,xva,xt,m
    if not np.isfinite(oof).all():
        raise RuntimeError("Incomplete OOF: will not publish any CV")
    result=sample.copy()
    result["satisfaction"]=pred
    if not np.isfinite(pred).all() or (pred<0).any() or (pred>1).any():
        raise ValueError("Unbounded Kaggle test predictions")
    OUTPUT.mkdir(parents=True,exist_ok=True)
    result.to_csv(OUTPUT/"v2911_candidate_unsubmitted.csv",index=False)
    np.savez_compressed(OUTPUT/"v2911_oof.npz",target=y,train_id=train["id"].to_numpy(),
                        oof_segment=oof,test_id=test["id"].to_numpy(),
                        test_segment=pred.astype("float32"))
    dump("v2911_metrics.json",{"competition":"playground-series-s6e10",
        "model":"LightGBM traveler-contrast-context independent",
        "train_rows":len(y),"test_rows":len(test),
        "validation":"StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)",
        "folds":3,"fold_results":folds,"full_cv_auc":float(roc_auc_score(y,oof)),
        "inner_target_folds":5,"inner_frequency_folds":5,
        "conditional_keys":len(keys(train.iloc[:3]).columns),
        "data_source":"competition_train_only","external_teacher":False,
        "outer_validation_used_for_early_stopping":False,
        "stage":"full_threefold_complete","elapsed_seconds":time.monotonic()-started,
        "submitted":False})
    print("V2911 genuine full OOF",float(roc_auc_score(y,oof)),flush=True)


if __name__=="__main__":
    try:
        main()
    except Exception:
        dump("v2911_error.json",{"traceback":traceback.format_exc()[-6000:]})
        raise
