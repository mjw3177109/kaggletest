"""V2.9.22 read-only subgroup and diversity audit for completed S6E10 OOF artifacts.
No model training, no label-derived features, no leaderboard submissions, no weight search.
"""
from __future__ import annotations
import hashlib,json,subprocess,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
SLUG="playground-series-s6e10"
STATE_DIR=ROOT/"state"
REPORT_DIR=ROOT/"reports"
OUTPUT_DIR=ROOT/"competitions"/SLUG/"results_v2922_subgroups"
MODELS=[
 {"name":"V2918 LightGBM","state":"v2_9_18_status.json","kernel":"mjw3177109/kas-v2918-routeprofile-lightgbm-cpu-s6e10","file":"v2918_route_lgbm_oof.npz","key":"oof_route_lgbm"},
 {"name":"V2919R2 TabM","state":"v2_9_19r2_status.json","kernel":"mjw3177109/kas-v2919r2-tabm-fixed-te23-t4-s6e10","file":"v2919r2_tabm_oof.npz","key":"oof_tabm_route"},
 {"name":"V2920 XGBoost","state":"v2_9_20_status.json","kernel":"mjw3177109/kas-v2920-routeprofile-xgboost-hist-cpu-s6e10","file":"v2920_route_xgb_oof.npz","key":"oof_route_xgb"},
 {"name":"V2921 CatBoost","state":"v2_9_21_status.json","kernel":"mjw3177109/kas-v2921-catboost-native-route-cpu-s6e10","file":"v2921_catboost_native_route_oof.npz","key":"oof_catboost_route"},
]
TERMINAL={"candidate_ready_unsubmitted","skipped_no_cv_gain"}


def command(argv,timeout=300):
    result=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError("Read-only audit command failed: "+(result.stderr or result.stdout)[-1600:])
    return result.stdout


def align_predictions(train,ids,target,pred):
    ids=np.asarray(ids); target=np.asarray(target); pred=np.asarray(pred,dtype=np.float64)
    if not (len(ids)==len(target)==len(pred)==len(train)):
        raise ValueError("OOF vector length mismatch")
    if len(np.unique(ids))!=len(ids) or train["id"].duplicated().any():
        raise ValueError("Duplicate OOF or official train IDs")
    positions=pd.Index(train["id"].to_numpy()).get_indexer(ids)
    if (positions<0).any() or len(np.unique(positions))!=len(train):
        raise ValueError("OOF IDs are missing or outside official train")
    y=np.empty(len(train),dtype=np.int8)
    oof=np.empty(len(train),dtype=np.float64)
    y[positions]=target.astype(np.int8)
    oof[positions]=pred
    official=pd.to_numeric(train["satisfaction"],errors="raise").to_numpy(dtype=np.int8)
    if not np.array_equal(y,official):
        raise ValueError("OOF target vector does not match official labels by ID")
    if not np.isfinite(oof).all() or (oof<0).any() or (oof>1).any():
        raise ValueError("OOF probability contains nonfinite or out-of-range values")
    return official,oof


def _subgroup_columns(train):
    groups={}
    for col in ("Class","Type of Travel","Customer Type","Gender"):
        groups[col]=train[col].astype("string").fillna("__NA__").astype(str)
    distance=pd.to_numeric(train["Flight Distance"],errors="coerce")
    groups["Flight Distance quintile"]=pd.qcut(distance,q=5,duplicates="drop").astype("string").fillna("__NA__").astype(str)
    for col in ("Departure Delay in Minutes","Arrival Delay in Minutes"):
        delay=pd.to_numeric(train[col],errors="coerce").fillna(0)
        groups[col+" band"]=pd.Series(np.select(
            [delay<=0,delay<=15,delay<=60],
            ["0","1-15","16-60"],default=">60"),index=train.index)
    age=pd.to_numeric(train["Age"],errors="coerce")
    groups["Age band"]=pd.cut(age,[-np.inf,24,34,44,54,np.inf],
        labels=["<=24","25-34","35-44","45-54","55+"],include_lowest=True).astype("string").fillna("__NA__").astype(str)
    return groups


def subgroup_metrics(train,y,pred,min_rows=1000):
    result={}
    for name,series in _subgroup_columns(train).items():
        entries={}
        for value in sorted(series.unique()):
            mask=series.to_numpy()==value
            n=int(mask.sum())
            if n<min_rows or len(np.unique(y[mask]))<2:
                continue
            entries[value]={"rows":n,"positive_rate":float(np.mean(y[mask])),
                            "auc":float(roc_auc_score(y[mask],pred[mask]))}
        result[name]=entries
    return result


def load_official_train(directory):
    directory.mkdir(parents=True,exist_ok=True)
    zip_path=directory/(SLUG+".zip")
    if not zip_path.exists():
        command(["kaggle","competitions","download","-c",SLUG,"-p",str(directory)],timeout=300)
    if not zip_path.exists():
        raise RuntimeError("Kaggle competition download did not create expected ZIP")
    with zipfile.ZipFile(zip_path) as archive:
        names=[n for n in archive.namelist() if n.rsplit("/",1)[-1]=="train.csv"]
        if len(names)!=1:
            raise RuntimeError("Expected one official train.csv in competition ZIP")
        with archive.open(names[0]) as stream:
            train=pd.read_csv(stream)
    required={"id","satisfaction","Class","Type of Travel","Customer Type","Gender","Flight Distance","Age","Departure Delay in Minutes","Arrival Delay in Minutes"}
    if len(train)!=699635 or not required.issubset(train.columns):
        raise RuntimeError("Official train.csv row count or required subgroup fields mismatch")
    return train


def load_model_oof(model,train,folder):
    state_path=STATE_DIR/model["state"]
    if not state_path.exists():
        raise RuntimeError("Missing completed model state: "+model["state"])
    state=json.loads(state_path.read_text(encoding="utf-8"))
    metrics=state.get("metrics") or {}
    if (state.get("stage") not in TERMINAL
        or state.get("submitted") is not False
        or state.get("kernel")!=model["kernel"]
        or metrics.get("stage")!="three_folds_complete"
        or len(metrics.get("fold_results") or [])!=3):
        raise RuntimeError("Model is not terminal with complete 3fold OOF: "+model["name"])
    folder.mkdir(parents=True,exist_ok=True)
    command(["kaggle","kernels","output",model["kernel"],"-p",str(folder),"-o","-q"],timeout=300)
    matches=list(folder.rglob(model["file"]))
    if len(matches)!=1:
        raise RuntimeError("Expected one "+model["file"]+"; found "+str(len(matches)))
    artifact=matches[0]
    with np.load(artifact,allow_pickle=False) as archive:
        required={"train_id","target",model["key"]}
        if not required.issubset(archive.files):
            raise RuntimeError("OOF artifact missing required arrays: "+model["name"])
        y,pred=align_predictions(train,archive["train_id"],archive["target"],archive[model["key"]])
    return pred,{"kernel":model["kernel"],"state_stage":state["stage"],
                 "standalone_auc":float(roc_auc_score(y,pred)),
                 "oof_sha256":hashlib.sha256(artifact.read_bytes()).hexdigest(),
                 "subgroups":subgroup_metrics(train,y,pred)}


def main():
    work=OUTPUT_DIR/"inputs"
    train=load_official_train(work)
    y=pd.to_numeric(train["satisfaction"],errors="raise").to_numpy(dtype=np.int8)
    predictions={}; summaries={}
    for model in MODELS:
        pred,summary=load_model_oof(model,train,OUTPUT_DIR/model["name"].replace(" ","_"))
        predictions[model["name"]]=pred
        summaries[model["name"]]=summary
        print(model["name"],"AUC",summary["standalone_auc"],flush=True)
    correlations={}
    names=list(predictions)
    for i,left in enumerate(names):
        for right in names[i+1:]:
            correlations[left+" <> "+right]=float(spearmanr(predictions[left],predictions[right]).statistic)
    result={
        "competition":SLUG,"analysis":"V2922 subgroup stability and model rank diversity",
        "train_rows":len(train),"validation":"OOF vectors from fixed common StratifiedKFold seed 20261018",
        "label_source":"official competition train.csv only","training_performed":False,
        "submission_performed":False,"blend_weight_search":False,
        "models":summaries,"pairwise_spearman_rank_correlation":correlations,
        "notes":["Subgroup AUC is descriptive and does not prove a blend gain.",
                 "Pairwise rank correlation is a diversity diagnostic, not a promotion criterion.",
                 "All OOF artifacts are ID/target aligned to the official train file before analysis."]
    }
    REPORT_DIR.mkdir(parents=True,exist_ok=True)
    json_path=REPORT_DIR/"v2_9_22_subgroup_stability.json"
    json_path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    md=["# V2.9.22 OOF subgroup stability and model diversity","",
        "- Rows: 699,635","- Training performed: no","- Kaggle submission: no",
        "- Model weights searched: no","","## Overall OOF AUC","",
        "| Model | AUC | OOF SHA-256 |","|---|---:|---|"]
    for name,summary in summaries.items():
        md.append('| '+name+' | '+format(summary["standalone_auc"],".8f")+' | '+summary["oof_sha256"]+' |')
    md.extend(["","## Pairwise Spearman rank correlation","","| Model pair | Correlation |","|---|---:|"])
    for pair,value in correlations.items():md.append("| "+pair+" | "+format(value,".6f")+" |")
    for name,summary in summaries.items():
        md.extend(["","## "+name+" — subgroup AUC","","| Group field | Value | Rows | Positive rate | AUC |","|---|---|---:|---:|---:|"])
        for field,values in summary["subgroups"].items():
            for value,metrics in values.items():
                md.append("| "+field+" | "+value+" | "+str(metrics["rows"])+" | "+format(metrics["positive_rate"],".4f")+" | "+format(metrics["auc"],".6f")+" |")
    md.extend(["","## Interpretation","",
        "Subgroup AUCs show where each model ranks passengers well or poorly. Correlation reveals overlap among model rankings. Neither metric alone justifies promotion; any blend must be re-evaluated on the same frozen OOF and fold splits."])
    (REPORT_DIR/"v2_9_22_subgroup_stability.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    print("V2922 diagnostic completed; no training, no submission.",flush=True)


if __name__=="__main__":
    main()
