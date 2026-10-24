"""V2.9.33 pre-registered Pairwise RankNet MLP, original V2915 37 columns, T4 only.
One independent, leakage-safe threefold OOF. NO Kaggle competition submission.
This file is copied to the sole Kaggle Notebook as main.py; no repository imports.
"""
from __future__ import annotations
import hashlib
import json
import os
import platform
import subprocess
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
from torch import nn
import math
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

INPUT=Path("/kaggle/input")
OUTPUT=Path("/kaggle/working")
SEED=20261018
MODEL_SEED=20261102
FOLDS=3
TRAIN_ROWS=699635
TEST_ROWS=299844
RUNTIME_BUDGET_SECONDS=30000
CATS=["Gender","Customer Type","Type of Travel","Class"]
DUAL=["Age","Flight Distance","Inflight wifi service","Online boarding",
      "Seat comfort","Inflight entertainment","Leg room service"]
PAIRS=[("Class","Type of Travel","Gender"),
       ("Inflight wifi service","Online boarding"),
       ("Class","Customer Type","Online boarding")]
FEATURE_FAMILY="v2915_original37_3_inner5fold_TE_no_extra_features"
FEATURE_COUNT=37
EMBEDDING_DIM=16
HIDDEN_LAYERS=(256,128)
DROPOUT=0.10
EPOCHS=5
BATCH_SIZE=512
POSITIVE_PER_BATCH=256
NEGATIVE_PER_BATCH=256
LOSS_NAME="0.5*BCEWithLogitsLoss+0.5*softplus(-(positive_logit-negative_logit)).mean"
LR=0.001
WEIGHT_DECAY=0.0001

def dump(filename, value):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / filename).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")


def only_competition_file(name):
    files = [p for p in INPUT.rglob(name)
             if "playground-series-s6e10" in str(p.parent)]
    if len(files) != 1:
        raise RuntimeError(f"Exactly one contest {name} required, found {len(files)}")
    return pd.read_csv(files[0])


def prove_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device; refusing CPU fallback or fabricated GPU evidence")
    probe = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=20, check=True)
    devices = [x.strip() for x in probe.stdout.splitlines() if x.strip()]
    if not any("T4" in device for device in devices):
        raise RuntimeError("Kaggle NVIDIA T4 was not confirmed: " + repr(devices))
    evidence = {"ok": True, "device": str(torch.cuda.get_device_name(0)),
                "devices": devices, "torch_version": torch.__version__,
                "cuda_available": True}
    dump("v2933_hardware.json", evidence)
    return evidence


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



def feature_tokens(xa, xv, xt):
    """Only the outer train fold fits means/stds and categorical vocabulary."""
    columns=list(xa.columns)
    if (len(columns)!=FEATURE_COUNT or len(set(columns))!=FEATURE_COUNT
        or columns[-3:]!=["te_0","te_1","te_2"]
        or columns!=list(xv.columns) or columns!=list(xt.columns)):
        raise ValueError("V2933 frozen canonical 37-feature order mismatch")
    cat_cols=[c for c in columns if isinstance(xa[c].dtype,pd.CategoricalDtype)]
    num_cols=[c for c in columns if c not in cat_cols]
    token_types=["categorical" if c in cat_cols else "numeric" for c in columns]
    numeric=[]
    for frame in (xa,xv,xt):
        matrix=[]
        for col in num_cols:
            values=pd.to_numeric(frame[col],errors="raise").to_numpy(dtype="float64")
            ref=pd.to_numeric(xa[col],errors="raise").to_numpy(dtype="float64")
            if not np.isfinite(ref).all() or not np.isfinite(values).all():
                raise ValueError("Nonfinite numeric feature: "+col)
            mean=float(ref.mean())
            std=float(ref.std())
            if not np.isfinite(mean) or not np.isfinite(std):
                raise ValueError("Nonfinite fold-local scaler")
            matrix.append(((values-mean)/max(std,1e-6)).astype("float32"))
        numeric.append(np.ascontiguousarray(np.stack(matrix,axis=1)))
    categorical=[]
    cards=[]
    for col in cat_cols:
        if not isinstance(xa[col].dtype,pd.CategoricalDtype):
            raise ValueError("Category vocabulary not fitted on outer train")
        cards.append(len(xa[col].cat.categories)+1)
    for frame in (xa,xv,xt):
        matrix=[]
        for col in cat_cols:
            if not isinstance(frame[col].dtype,pd.CategoricalDtype):
                raise ValueError("Fold-local category schema mismatch")
            values=frame[col].cat.codes.to_numpy(dtype="int64")+1
            matrix.append(values)
        categorical.append(np.ascontiguousarray(np.stack(matrix,axis=1)))
    for values in categorical:
        if any(np.any(values[:,i]<0) or np.any(values[:,i]>=card)
               for i,card in enumerate(cards)):
            raise ValueError("Categorical ID outside train-fold vocabulary")
    return numeric,categorical,columns,token_types,cat_cols,num_cols,cards


class TabularRankMLP(nn.Module):
    """Frozen original-37-feature tabular MLP, no attention or extra features."""
    def __init__(self,columns,cat_cols,num_cols,cards):
        super().__init__()
        if len(columns)!=FEATURE_COUNT or len(cards)!=len(cat_cols) or len(num_cols)!=23 or len(cat_cols)!=14:
            raise ValueError("V2933 frozen 23-numeric/14-categorical schema mismatch")
        self.embeddings=nn.ModuleList([nn.Embedding(card,EMBEDDING_DIM) for card in cards])
        self.net=nn.Sequential(
            nn.Linear(len(num_cols)+len(cards)*EMBEDDING_DIM,HIDDEN_LAYERS[0]),
            nn.GELU(),nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_LAYERS[0],HIDDEN_LAYERS[1]),
            nn.GELU(),nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_LAYERS[1],1))
    def forward(self,num,cat):
        if num.ndim!=2 or cat.ndim!=2:
            raise ValueError("V2933 tensor rank mismatch")
        return self.net(torch.cat([num]+[emb(cat[:,i]) for i,emb in enumerate(self.embeddings)],dim=1)).squeeze(-1)


class BalancedPairSampler:
    """Outer-train-only deterministic shuffled/reshuffled positive-negative pairs."""
    def __init__(self,labels,seed):
        labels=np.asarray(labels)
        if not np.isin(labels,[0,1]).all():
            raise ValueError("Binary 0/1 labels required")
        self.rng=np.random.default_rng(seed)
        self.groups={1:np.flatnonzero(labels==1),0:np.flatnonzero(labels==0)}
        if min(len(self.groups[1]),len(self.groups[0]))<1:
            raise ValueError("Both classes required for pairwise RankNet")
        self.order={key:np.empty(0,dtype="int64") for key in (0,1)}
        self.cursor={0:0,1:0}
    def _take(self,key,n):
        chunks=[]
        while n:
            if self.cursor[key]==len(self.order[key]):
                self.order[key]=self.rng.permutation(self.groups[key])
                self.cursor[key]=0
            available=min(n,len(self.order[key])-self.cursor[key])
            chunks.append(self.order[key][self.cursor[key]:self.cursor[key]+available])
            self.cursor[key]+=available
            n-=available
        return np.concatenate(chunks)
    def batch(self):
        return np.concatenate((self._take(1,POSITIVE_PER_BATCH),
                               self._take(0,NEGATIVE_PER_BATCH)))


def ranknet_loss(logits):
    if logits.ndim!=1 or logits.shape[0]!=BATCH_SIZE:
        raise ValueError("Frozen RankNet batch must contain 256 positive then 256 negative")
    labels=torch.cat((torch.ones(POSITIVE_PER_BATCH,device=logits.device),
                      torch.zeros(NEGATIVE_PER_BATCH,device=logits.device)))
    bce=F.binary_cross_entropy_with_logits(logits,labels)
    pair=F.softplus(-(logits[:POSITIVE_PER_BATCH]-logits[POSITIVE_PER_BATCH:])).mean()
    return 0.5*bce+0.5*pair


@torch.no_grad()
def predict(model,num,cat):
    model.eval()
    output=[]
    for start in range(0,len(num),BATCH_SIZE):
        n=torch.as_tensor(num[start:start+BATCH_SIZE],device="cuda:0")
        c=torch.as_tensor(cat[start:start+BATCH_SIZE],device="cuda:0")
        output.append(torch.sigmoid(model(n,c)).cpu().numpy())
    return np.concatenate(output).astype("float32")


def fit_fold(train,test,y,itr,iva,fold):
    started=time.monotonic()
    xa,xv,te_train,te_val=engineer(train.iloc[itr],train.iloc[iva])
    _,xt,_,te_test=engineer(train.iloc[itr],test)
    if list(xa.columns)!=list(xt.columns):
        raise RuntimeError("Train/test engineered feature order mismatch")
    enc=TargetEncoder(target_type="binary",smooth="auto",cv=5,
                      shuffle=True,random_state=SEED+fold)
    te_cols=["te_0","te_1","te_2"]
    if list(te_train.columns)!=["combo_0","combo_1","combo_2"]:
        raise RuntimeError("Original three inner-fivefold TE sources changed")
    xa[te_cols]=enc.fit_transform(te_train,y[itr]).astype("float32")
    xv[te_cols]=enc.transform(te_val).astype("float32")
    xt[te_cols]=enc.transform(te_test).astype("float32")
    numeric,categorical,columns,types,cat_cols,num_cols,cards=feature_tokens(xa,xv,xt)
    torch.manual_seed(MODEL_SEED+fold)
    torch.cuda.manual_seed_all(MODEL_SEED+fold)
    np.random.seed(MODEL_SEED+fold)
    model=TabularRankMLP(columns,cat_cols,num_cols,cards).to("cuda:0")
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    steps=math.ceil(len(itr)/BATCH_SIZE)
    sampler=BalancedPairSampler(y[itr],MODEL_SEED+fold)
    num_cpu=torch.from_numpy(numeric[0])
    cat_cpu=torch.from_numpy(categorical[0])
    for epoch in range(EPOCHS):
        model.train()
        for step in range(steps):
            ids=torch.from_numpy(sampler.batch())
            n=num_cpu[ids].to("cuda:0",non_blocking=True)
            c=cat_cpu[ids].to("cuda:0",non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss=ranknet_loss(model(n,c))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite V2933 RankNet loss")
            loss.backward()
            optimizer.step()
        print("V2933 fold",fold,"epoch",epoch+1,"complete",flush=True)
    pv=predict(model,numeric[1],categorical[1])
    pt=predict(model,numeric[2],categorical[2])
    if pv.shape!=(len(iva),) or pt.shape!=(len(test),):
        raise RuntimeError("Incomplete validation/test predictions")
    if not (np.isfinite(pv).all() and np.isfinite(pt).all()
            and (pv>=0).all() and (pv<=1).all()
            and (pt>=0).all() and (pt<=1).all()):
        raise RuntimeError("Nonfinite/out-of-range Pairwise RankNet MLP probabilities")
    info={"fold":fold,"train_rows":len(itr),"validation_rows":len(iva),
          "model_seed":MODEL_SEED+fold,"features":FEATURE_COUNT,
          "feature_order":columns,"token_types":types,
          "categorical_columns":cat_cols,"numeric_columns":num_cols,
          "categorical_cardinalities":cards,
          "validation_index_sha256":hashlib.sha256(np.asarray(iva,dtype="<i8").tobytes()).hexdigest(),
          "auc":float(roc_auc_score(y[iva],pv)),
          "elapsed_seconds":round(time.monotonic()-started,2),
          "epochs":EPOCHS,"steps_per_epoch":steps}
    del model,num_cpu,cat_cpu,sampler
    torch.cuda.empty_cache()
    return pv,pt,info


def main():
    started=time.monotonic()
    hardware=prove_gpu()
    if "T4" not in str(torch.cuda.get_device_name(0)):
        raise RuntimeError("Primary CUDA GPU is not NVIDIA T4")
    train=only_competition_file("train.csv")
    test=only_competition_file("test.csv")
    sample=only_competition_file("sample_submission.csv")
    if len(train)!=TRAIN_ROWS or len(test)!=TEST_ROWS or len(sample)!=TEST_ROWS:
        raise RuntimeError("Competition train/test/sample row counts changed")
    if list(sample.columns)!=["id","satisfaction"] or not np.array_equal(sample["id"].to_numpy(),test["id"].to_numpy()):
        raise RuntimeError("Official sample and test IDs mismatch")
    if "satisfaction" not in train or "satisfaction" in test or "id" not in train:
        raise RuntimeError("Competition target or IDs missing")
    if train["id"].duplicated().any() or test["id"].duplicated().any():
        raise RuntimeError("Duplicate train/test IDs")
    y=pd.to_numeric(train["satisfaction"],errors="raise").to_numpy()
    if not np.isin(y,[0,1]).all():
        raise RuntimeError("Binary 0/1 labels required")
    folds=list(StratifiedKFold(FOLDS,shuffle=True,random_state=SEED).split(np.zeros(len(y),dtype="uint8"),y))
    oof=np.full(len(y),np.nan,dtype="float32")
    test_pred=np.zeros(len(test),dtype="float64")
    results=[]
    for fold,(itr,iva) in enumerate(folds):
        if time.monotonic()-started>RUNTIME_BUDGET_SECONDS:
            raise TimeoutError("V2933 exceeded budget; refuse incomplete OOF")
        pv,pt,info=fit_fold(train,test,y,itr,iva,fold)
        oof[iva]=pv
        test_pred+=pt/FOLDS
        results.append(info)
        dump("v2933_progress.json",{"finished":fold+1,"complete":False,"fold_results":results})
    if len(results)!=3 or not np.isfinite(oof).all() or not np.isfinite(test_pred).all():
        raise RuntimeError("Incomplete V2933 threefold OOF")
    np.savez_compressed(OUTPUT/"v2933_ranknet_oof.npz",
                        target=y.astype("int8"),
                        train_id=train["id"].to_numpy(),
                        oof_ranknet=oof,
                        test_id=test["id"].to_numpy(),
                        pred_ranknet=test_pred.astype("float32"))
    dump("v2933_metrics.json",{
        "competition":"playground-series-s6e10",
        "stage":"three_folds_complete",
        "model":"Pairwise RankNet MLP PyTorch original V2915 37-feature independent",
        "model_family":"PyTorch_Tabular_MLP_Pairwise_RankNet_T4",
        "feature_family":FEATURE_FAMILY,
        "train_rows":len(train),"test_rows":len(test),
        "validation":"StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)",
        "folds":FOLDS,"fold_results":results,
        "full_oof_auc":float(roc_auc_score(y,oof)),
        "gpu_verified":True,"hardware":hardware,
        "feature_count":FEATURE_COUNT,"inner_target_folds":5,
        "leakage_safe_inner_cross_fit":True,
        "outer_validation_used_in_training":False,
        "external_labels":False,"pseudo_labels":False,
        "submitted":False,"model_seed":MODEL_SEED,"epochs":EPOCHS,
        "batch_size":BATCH_SIZE,"optimizer":"AdamW",
        "learning_rate":LR,"weight_decay":WEIGHT_DECAY,
        "loss":LOSS_NAME,"early_stopping":False,
        "ensemble":False,
        "embedding_dim":EMBEDDING_DIM,"hidden_layers":list(HIDDEN_LAYERS),
        "activation":"GELU","dropout":DROPOUT,
        "positive_per_batch":POSITIVE_PER_BATCH,
        "negative_per_batch":NEGATIVE_PER_BATCH,
        "numeric_standardization":"outer_train_only_mean_std",
        "categorical_vocabulary":"outer_train_only_unknown_missing_id",
        "source_code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment_versions":{
            "python":platform.python_version(),"torch":torch.__version__,
            "cuda":str(torch.version.cuda),"numpy":np.__version__,
            "pandas":pd.__version__,"sklearn":__import__("sklearn").__version__,
        },
        "runtime_seconds":round(time.monotonic()-started,1),
        "ready_for_submission":False
    })
    print("V2933 independent full OOF AUC",roc_auc_score(y,oof),flush=True)


if __name__=="__main__":
    try:
        main()
    except Exception:
        dump("v2933_error.json",{"traceback":traceback.format_exc()[-8000:]})
        raise
