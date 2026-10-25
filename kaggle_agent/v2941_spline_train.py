"""V2941 fold-safe CPU spline-additive logistic learner. Never submits or creates Kaggle kernels."""
from __future__ import annotations
import hashlib,json,os,platform,time,warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy,sklearn
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,SplineTransformer,StandardScaler,TargetEncoder
SEED=20261018
TRAIN_ROWS=699635
TEST_ROWS=299844
CATS=["Gender","Customer Type","Type of Travel","Class"]
DUAL=["Age","Flight Distance","Inflight wifi service","Online boarding","Seat comfort","Inflight entertainment","Leg room service"]
PAIRS=[("Class","Type of Travel","Gender"),("Inflight wifi service","Online boarding"),("Class","Customer Type","Online boarding")]
CANONICAL=["Age","Flight Distance","Inflight wifi service","Departure/Arrival time convenient","Ease of Online booking","Gate location","Food and drink","Online boarding","Seat comfort","Inflight entertainment","On-board service","Leg room service","Baggage handling","Checkin service","Cleanliness","Departure Delay in Minutes","Arrival Delay in Minutes","Gender","Customer Type","Type of Travel","Class","dual_Age","dual_Flight Distance","dual_Inflight wifi service","dual_Online boarding","dual_Seat comfort","dual_Inflight entertainment","dual_Leg room service","combo_0","count_combo_0","combo_1","count_combo_1","combo_2","count_combo_2","te_0","te_1","te_2"]
CAT_ORDER=[c for c in CANONICAL if c in set(CATS+["dual_"+d for d in DUAL]+["combo_0","combo_1","combo_2"])]
SERVICE=["Inflight wifi service","Online boarding","Seat comfort","Inflight entertainment"]
NUM_ORDER=[c for c in CANONICAL if c not in set(CAT_ORDER+SERVICE)]
PARAMS={"penalty":"l2","C":0.5,"solver":"lbfgs","max_iter":400,"tol":0.0001,"fit_intercept":True,"class_weight":None}
VERSIONS={"numpy":"2.2.6","pandas":"2.2.3","scipy":"1.15.3","sklearn":"1.6.1"}
def frozen_schema(columns):
    if list(columns)!=CANONICAL or len(columns)!=37 or len(set(columns))!=37:
        raise RuntimeError("V2941 original37 feature schema mismatch")
    if (len(CAT_ORDER),len(SERVICE),len(NUM_ORDER))!=(14,4,19):
        raise RuntimeError("V2941 14+4+19 groups mismatch")
    return hashlib.sha256(json.dumps(CANONICAL,separators=(",",":")).encode()).hexdigest()
def prove_environment():
    actual={"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__}
    if actual!=VERSIONS or platform.python_version_tuple()[:2]!=("3","11"):
        raise RuntimeError("V2941 pinned environment mismatch: "+repr(actual))
    if os.getenv("OMP_NUM_THREADS")!="4" or not os.getenv("GITHUB_RUN_ID") or os.getenv("V2941_RESERVED_RUN_ID")!=os.getenv("GITHUB_RUN_ID"):
        raise RuntimeError("V2941 CPU environment or durable reservation missing")
    return dict(actual,python=platform.python_version(),device="cpu",omp_num_threads=4)
def preprocessor():
    return ColumnTransformer([
      ("categorical",OneHotEncoder(handle_unknown="infrequent_if_exist",max_categories=32,sparse_output=True),CAT_ORDER),
      ("spline",Pipeline([("median",SimpleImputer(strategy="median")),("spline",SplineTransformer(n_knots=5,degree=3,knots="quantile",extrapolation="constant",include_bias=False))]),SERVICE),
      ("numeric",Pipeline([("median",SimpleImputer(strategy="median")),("scale",StandardScaler())]),NUM_ORDER)],
      sparse_threshold=0.3,remainder="drop")
def prepare(frame):
    frozen_schema(frame.columns)
    out=frame.copy()
    for col in CAT_ORDER:out[col]=out[col].astype("string").fillna("__UNKNOWN__").astype(str)
    for col in SERVICE+NUM_ORDER:out[col]=pd.to_numeric(out[col],errors="raise")
    return out
def strict_fit(model,x,y):
    with warnings.catch_warnings(record=True) as events:
        warnings.simplefilter("always",ConvergenceWarning)
        model.fit(x,y)
    if any(issubclass(w.category,ConvergenceWarning) for w in events):
        raise RuntimeError("V2941 frozen LBFGS failed convergence; fail closed")
    return model
def synthetic_preflight():
    actual={"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__}
    if actual!=VERSIONS or platform.python_version_tuple()[:2]!=("3","11") or os.getenv("OMP_NUM_THREADS")!="4":
        raise RuntimeError("V2941 synthetic pinned environment mismatch")
    rng=np.random.default_rng(20261108)
    d={c:(rng.choice(["a","b","c"],256) if c in CAT_ORDER else rng.normal(size=256)) for c in CANONICAL}
    x=prepare(pd.DataFrame(d)[CANONICAL])
    y=(rng.random(256)<1/(1+np.exp(-x["Age"].to_numpy()))).astype("int8")
    a=preprocessor().fit_transform(x)
    strict_fit(LogisticRegression(**PARAMS),a,y)
    if a.shape[0]!=256 or not np.isfinite(a.data if hasattr(a,"toarray") else a).all():
        raise RuntimeError("V2941 synthetic pipeline invalid")
    print("V2941 pinned synthetic spline logistic PASS")
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




def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,default=str,separators=(",",":")).encode()).hexdigest()
def fit_fold(train,test,y,itr,iva,fold):
    start=time.monotonic()
    xa,xv,te_train,te_val=engineer(train.iloc[itr],train.iloc[iva])
    _,xt,_,te_test=engineer(train.iloc[itr],test)
    if list(te_train.columns)!=["combo_0","combo_1","combo_2"]:
        raise RuntimeError("Protected TE keys changed")
    enc=TargetEncoder(target_type="binary",smooth="auto",cv=5,shuffle=True,random_state=SEED+fold)
    cols=["te_0","te_1","te_2"]
    xa[cols]=enc.fit_transform(te_train,y[itr]).astype("float32")
    xv[cols]=enc.transform(te_val).astype("float32")
    xt[cols]=enc.transform(te_test).astype("float32")
    a,b,c=[prepare(f[CANONICAL]) for f in (xa,xv,xt)]
    prep=preprocessor()
    aa=prep.fit_transform(a)
    vv=prep.transform(b)
    tt=prep.transform(c)
    if aa.shape[1]!=vv.shape[1] or aa.shape[1]!=tt.shape[1]:
        raise RuntimeError("V2941 expanded feature dimensions changed")
    ohe=prep.named_transformers_["categorical"]
    spline=prep.named_transformers_["spline"].named_steps["spline"]
    model=strict_fit(LogisticRegression(**PARAMS),aa,y[itr])
    pv=np.asarray(model.predict_proba(vv)[:,1],dtype="float32")
    pt=np.asarray(model.predict_proba(tt)[:,1],dtype="float32")
    if pv.shape!=(len(iva),) or pt.shape!=(len(test),) or any(not np.isfinite(x).all() or np.any((x<0)|(x>1)) for x in (pv,pt)):
        raise RuntimeError("V2941 invalid fold probabilities")
    info={"fold":fold,"train_rows":len(itr),"validation_rows":len(iva),
          "features":37,"feature_order":CANONICAL,"category_columns":CAT_ORDER,
          "spline_columns":SERVICE,"numeric_columns":NUM_ORDER,
          "expanded_features":int(aa.shape[1]),
          "category_vocabulary_sha256":digest([v.tolist() for v in ohe.categories_]),
          "spline_knots_sha256":digest([bs.t.tolist() for bs in spline.bsplines_]),
          "feature_schema_sha256":frozen_schema(CANONICAL),
          "validation_index_sha256":hashlib.sha256(np.asarray(iva,dtype="<i8").tobytes()).hexdigest(),
          "auc":float(roc_auc_score(y[iva],pv)),
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
    for fold,(itr,iva) in enumerate(StratifiedKFold(3,shuffle=True,random_state=SEED).split(np.zeros(len(y),dtype="uint8"),y)):
        pv,pt,info=fit_fold(train,test,y,itr,iva,fold)
        oof[iva]=pv;test_pred+=pt/3;results.append(info)
        (out/"v2941_progress.json").write_text(json.dumps({"finished":fold+1,"complete":False,"fold_results":results},indent=2),encoding="utf-8")
    if len(results)!=3 or not np.isfinite(oof).all() or not np.isfinite(test_pred).all():
        raise RuntimeError("V2941 incomplete OOF")
    artifact=out/"v2941_spline_oof.npz"
    np.savez_compressed(artifact,target=y.astype("int8"),train_id=train["id"].to_numpy(),oof_spline=oof,test_id=test["id"].to_numpy(),pred_spline=test_pred.astype("float32"))
    metrics={"competition":"playground-series-s6e10","stage":"three_folds_complete",
       "model":"V2941 sklearn CPU spline-additive logistic original37","train_rows":len(train),
       "test_rows":len(test),"folds":3,"fold_results":results,"full_oof_auc":float(roc_auc_score(y,oof)),
       "feature_count":37,"inner_target_folds":5,"leakage_safe_inner_cross_fit":True,
       "outer_validation_used_in_training":False,"external_labels":False,
       "submitted":False,"cpu_only":True,"model_parameters":PARAMS,
       "feature_schema_sha256":frozen_schema(CANONICAL),
       "category_columns":CAT_ORDER,"spline_columns":SERVICE,"numeric_columns":NUM_ORDER,
       "environment_versions":env,"source_code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/"v2941_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    return artifact,metrics
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--official-directory",required=True)
    p.add_argument("--output-directory",required=True)
    a=p.parse_args();run(a.official_directory,a.output_directory)
