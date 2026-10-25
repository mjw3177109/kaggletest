"""V2938 immutable CPU-only LightGBM Ordered original37 threefold OOF.
No Notebook, no submission, no tuning. Outer-fold-only encoders and TE. Exactly 37 numeric-encoded features.
"""
from __future__ import annotations
import hashlib, json, os, platform, time
from pathlib import Path
import catboost
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import scipy, sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

SEED=20261018
MODEL_SEED=20261106
TRAIN_ROWS=699635
TEST_ROWS=299844
CATS=["Gender","Customer Type","Type of Travel","Class"]
DUAL=["Age","Flight Distance","Inflight wifi service","Online boarding","Seat comfort","Inflight entertainment","Leg room service"]
PAIRS=[("Class","Type of Travel","Gender"),("Inflight wifi service","Online boarding"),("Class","Customer Type","Online boarding")]
CANONICAL=["Age","Flight Distance","Inflight wifi service","Departure/Arrival time convenient","Ease of Online booking","Gate location","Food and drink","Online boarding","Seat comfort","Inflight entertainment","On-board service","Leg room service","Baggage handling","Checkin service","Cleanliness","Departure Delay in Minutes","Arrival Delay in Minutes","Gender","Customer Type","Type of Travel","Class","dual_Age","dual_Flight Distance","dual_Inflight wifi service","dual_Online boarding","dual_Seat comfort","dual_Inflight entertainment","dual_Leg room service","combo_0","count_combo_0","combo_1","count_combo_1","combo_2","count_combo_2","te_0","te_1","te_2"]
CATEGORICAL=set(CATS+["dual_"+c for c in DUAL]+["combo_0","combo_1","combo_2"])
PARAMS={"loss_function":"Logloss","eval_metric":"AUC","boosting_type":"Ordered",
"task_type":"CPU","iterations":600,"depth":6,"learning_rate":0.035,
"l2_leaf_reg":5.0,"random_strength":1.0,"bootstrap_type":"Bayesian",
"bagging_temperature":0.5,"rsm":0.85,"border_count":128,
"thread_count":4,"allow_writing_files":False,"verbose":False}
VERSIONS={"catboost":"1.2.8","numpy":"2.2.6","pandas":"2.2.3","scipy":"1.15.3","sklearn":"1.6.1"}

def prove_environment():
    actual={"catboost":catboost.__version__,"numpy":np.__version__,"pandas":pd.__version__,
            "scipy":scipy.__version__,"sklearn":sklearn.__version__}
    if actual!=VERSIONS or platform.python_version_tuple()[:2]!=("3","11"):
        raise RuntimeError("Pinned V2938 environment mismatch: "+repr(actual)+" python="+platform.python_version())
    if not os.getenv("GITHUB_RUN_ID") or os.getenv("V2938_RESERVED_RUN_ID")!=os.getenv("GITHUB_RUN_ID"):
        raise RuntimeError("V2938 durable GitHub run reservation missing")
    return dict(actual,python=platform.python_version(),device="cpu")

def _categorical(train, validation, name, left, right):
    fit = left.astype("string").fillna("__NA__")
    val = right.astype("string").fillna("__NA__")
    categories = pd.Index(fit.unique()).astype(str).tolist()
    train[name] = pd.Categorical(fit.astype(str), categories=categories)
    validation[name] = pd.Categorical(val.astype(str), categories=categories)


def engineer(train, validation):
    """All encoders/maps are fitted only on the *outer training partition*."""
    features = [c for c in train.columns if c not in ("id", "satisfaction")]
    tr = pd.DataFrame(index=train.index)
    va = pd.DataFrame(index=validation.index)
    for col in features:
        if col in CATS:
            _categorical(tr, va, col, train[col], validation[col])
        else:
            a = pd.to_numeric(train[col], errors="coerce")
            b = pd.to_numeric(validation[col], errors="coerce")
            median = float(a.median())
            tr[col] = a.fillna(median).astype("float32")
            va[col] = b.fillna(median).astype("float32")
    for col in DUAL:
        _categorical(tr, va, "dual_" + col, train[col], validation[col])

    te_train = pd.DataFrame(index=train.index)
    te_val = pd.DataFrame(index=validation.index)
    for i, cols in enumerate(PAIRS):
        name = "combo_" + str(i)
        tr_key = train[list(cols)].astype("string").fillna("__NA__").agg("|".join, axis=1)
        va_key = validation[list(cols)].astype("string").fillna("__NA__").agg("|".join, axis=1)
        _categorical(tr, va, name, tr_key, va_key)
        count = tr_key.value_counts()
        tr["count_" + name] = np.log1p(tr_key.map(count).astype("float32")).to_numpy()
        va["count_" + name] = np.log1p(va_key.map(count).fillna(0).astype("float32")).to_numpy()
        te_train[name] = tr_key.astype(str)
        te_val[name] = va_key.astype(str)

    # Inner cross-fitting, including the training row itself only via out-of-fold
    # mapping. Never use outer validation labels when fitting this encoder.
    return tr, va, te_train, te_val




def fold_matrix(xa,xv,xt):
    """Frozen order, fold-local categories, unseen category sentinel -1, finite numeric."""
    columns=list(xa.columns)
    if columns!=CANONICAL or list(xv.columns)!=CANONICAL or list(xt.columns)!=CANONICAL:
        raise ValueError("V2938 frozen 37-feature order mismatch")
    if len(columns)!=37 or columns[-3:]!=["te_0","te_1","te_2"]:
        raise ValueError("V2938 protected TE columns mismatch")
    vocab={}
    frames=[]
    for col in columns:
        if col in CATEGORICAL:
            if not isinstance(xa[col].dtype,pd.CategoricalDtype):
                raise ValueError("Outer-train categorical vocabulary missing: "+col)
            vocab[col]=list(map(str,xa[col].cat.categories))
    for frame in (xa,xv,xt):
        out=pd.DataFrame(index=frame.index)
        for col in columns:
            if col in CATEGORICAL:
                if not isinstance(frame[col].dtype,pd.CategoricalDtype):
                    raise ValueError("Fold-local category type mismatch: "+col)
                if list(map(str,frame[col].cat.categories))!=vocab[col]:
                    raise ValueError("Validation/test category vocabulary differs: "+col)
                out[col]=frame[col].cat.codes.to_numpy(dtype="float32")
            else:
                arr=pd.to_numeric(frame[col],errors="raise").to_numpy(dtype="float32")
                if not np.isfinite(arr).all():
                    raise ValueError("Nonfinite numeric field: "+col)
                out[col]=arr
        frames.append(out)
    vocab_sha=hashlib.sha256(json.dumps(vocab,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    return frames,vocab_sha

def fit_fold(train,test,y,itr,iva,fold):
    start=time.monotonic()
    xa,xv,te_train,te_val=engineer(train.iloc[itr],train.iloc[iva])
    _,xt,_,te_test=engineer(train.iloc[itr],test)
    if list(te_train.columns)!=["combo_0","combo_1","combo_2"]:
        raise RuntimeError("Original three inner TE keys changed")
    enc=TargetEncoder(target_type="binary",smooth="auto",cv=5,shuffle=True,random_state=SEED+fold)
    te_cols=["te_0","te_1","te_2"]
    xa[te_cols]=enc.fit_transform(te_train,y[itr]).astype("float32")
    xv[te_cols]=enc.transform(te_val).astype("float32")
    xt[te_cols]=enc.transform(te_test).astype("float32")
    frames,vocab_sha=fold_matrix(xa,xv,xt)
    model=CatBoostClassifier(**PARAMS,random_seed=MODEL_SEED+fold)
    cat_cols=[c for c in CANONICAL if c in CATEGORICAL]
    # All 37 columns are numeric-encoded; no native CatBoost CTR/category fitting.
    model.fit(frames[0],y[itr])
    if model.tree_count_!=PARAMS["iterations"]:
        raise RuntimeError("Incomplete fixed Ordered rounds")
    pv=np.asarray(model.predict_proba(frames[1])[:,1],dtype="float32")
    pt=np.asarray(model.predict_proba(frames[2])[:,1],dtype="float32")
    if pv.shape!=(len(iva),) or pt.shape!=(len(test),):
        raise RuntimeError("Incomplete fold prediction")
    if any(not np.isfinite(a).all() or np.any((a<0)|(a>1)) for a in (pv,pt)):
        raise RuntimeError("Invalid probability")
    info={"fold":fold,"train_rows":len(itr),"validation_rows":len(iva),
      "model_seed":MODEL_SEED+fold,"features":37,"feature_order":CANONICAL,
      "categorical_columns":cat_cols,"category_vocabulary_sha256":vocab_sha,
      "validation_index_sha256":hashlib.sha256(np.asarray(iva,dtype="<i8").tobytes()).hexdigest(),
      "auc":float(roc_auc_score(y[iva],pv)),"rounds":PARAMS["iterations"],
      "elapsed_seconds":round(time.monotonic()-start,2)}
    return pv,pt,info

def load_official(directory):
    root=Path(directory)
    files={}
    for name in ("train.csv","test.csv","sample_submission.csv"):
        hits=list(root.rglob(name))
        if len(hits)!=1: raise RuntimeError("Exactly one official file required: "+name)
        files[name]=pd.read_csv(hits[0])
    train,test,sample=(files[k] for k in ("train.csv","test.csv","sample_submission.csv"))
    if len(train)!=TRAIN_ROWS or len(test)!=TEST_ROWS or len(sample)!=TEST_ROWS:
        raise RuntimeError("Official competition row counts changed")
    if list(sample.columns)!=["id","satisfaction"] or not np.array_equal(sample["id"],test["id"]):
        raise RuntimeError("Test IDs/sample schema mismatch")
    if "id" not in train or "satisfaction" not in train or "satisfaction" in test:
        raise RuntimeError("Target/id schema mismatch")
    if train["id"].duplicated().any() or test["id"].duplicated().any():
        raise RuntimeError("Duplicate official IDs")
    y=pd.to_numeric(train["satisfaction"],errors="raise").to_numpy()
    if not np.isin(y,[0,1]).all(): raise RuntimeError("Nonbinary target")
    if list(train.columns.drop(["id","satisfaction"]))!=CANONICAL[:21]:
        raise RuntimeError("Original 21 raw train columns changed")
    if list(test.columns.drop(["id"]))!=CANONICAL[:21]:
        raise RuntimeError("Original 21 raw test columns changed")
    return train,test,y

def run(official_directory,output_directory):
    env=prove_environment()
    train,test,y=load_official(official_directory)
    out=Path(output_directory);out.mkdir(parents=True,exist_ok=True)
    oof=np.full(len(y),np.nan,dtype="float32")
    test_pred=np.zeros(len(test),dtype="float64")
    results=[]
    folds=StratifiedKFold(3,shuffle=True,random_state=SEED)
    for fold,(itr,iva) in enumerate(folds.split(np.zeros(len(y),dtype="uint8"),y)):
        pv,pt,info=fit_fold(train,test,y,itr,iva,fold)
        oof[iva]=pv;test_pred+=pt/3;results.append(info)
        (out/"v2938_progress.json").write_text(json.dumps({"finished":fold+1,"complete":False,
             "fold_results":results},indent=2),encoding="utf-8")
    if len(results)!=3 or not np.isfinite(oof).all() or not np.isfinite(test_pred).all():
        raise RuntimeError("Incomplete threefold OOF")
    artifact=out/"v2938_ordered_oof.npz"
    np.savez_compressed(artifact,target=y.astype("int8"),train_id=train["id"].to_numpy(),
       oof_ordered=oof,test_id=test["id"].to_numpy(),pred_ordered=test_pred.astype("float32"))
    metrics={"competition":"playground-series-s6e10","stage":"three_folds_complete",
       "model":"V2938 CatBoost 1.2.8 CPU Ordered original37","train_rows":len(train),
       "test_rows":len(test),"folds":3,"fold_results":results,
       "full_oof_auc":float(roc_auc_score(y,oof)),"feature_count":37,
       "inner_target_folds":5,"leakage_safe_inner_cross_fit":True,
       "outer_validation_used_in_training":False,"external_labels":False,
       "submitted":False,"cpu_only":True,"model_seed":MODEL_SEED,
       "model_parameters":PARAMS,"environment_versions":env,
       "source_code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/"v2938_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    return artifact,metrics

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--official-directory",required=True)
    parser.add_argument("--output-directory",required=True)
    args=parser.parse_args()
    run(args.official_directory,args.output_directory)
