"""V2.9.6: submit a uniquely identified Kaggle GPU RealMLP full 3-fold job.

Never auto-submit to leaderboard here. Recover only the prior notebook, and
mark a candidate ready only after checked OOF labels, real T4, all three folds,
and identical historical CV protocol.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from v291_ledger import read,write,record_experiment
from v293_blend_math import verify_sources,VALIDATION,N_TRAIN,N_TEST
from v23_tpu_pipeline import poll_kernel

ROOT=Path(__file__).resolve().parents[1]
SLUG="playground-series-s6e10"
KERNEL="mjw3177109/kas-v296-realmlp-full-s6e10"
STATUS=ROOT/"state/v2_9_6_status.json"
LEDGER=ROOT/"state/v2_9_experiments.json"
REPORT=ROOT/"reports/v2_9_6_realmlp_full.md"
OUT=ROOT/"competitions"/SLUG/"results_v296"
FOLDER=ROOT/"competitions"/SLUG/"kaggle_gpu_v296"
TRIAL_GAIN=0.00006
TERMINAL={"skipped_no_cv_gain","candidate_ready_unsubmitted","model_failed",
          "kernel_failed","invalid_results","validation_mismatch",
          "gpu_unverified","output_failed","push_failed","access_error"}


def now():
    return datetime.now(timezone.utc)


def cli(cmd,timeout=240,check=True):
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout,cwd=str(ROOT))
    if check and p.returncode:
        raise RuntimeError(f"Command failed ({cmd[:3]}): "+(p.stderr or p.stdout)[-1200:])
    return p


def eligible():
    ladder=read(ROOT/"state/dual_ladder.json")
    if ladder.get("active_competition")!=SLUG:
        raise RuntimeError("Contest changed; refuse S6E10 kernel")
    deadline=ladder.get("active_deadline")
    if not deadline:
        raise RuntimeError("Unknown deadline")
    end=datetime.fromisoformat(deadline.replace("Z","+00:00"))
    if end.tzinfo is None or now()>=end:
        raise RuntimeError("Deadline passed or timezone missing")
    pilot=read(ROOT/"state/v2_9_5_realmlp_status.json")
    cmp=pilot.get("comparison") or {}
    if (pilot.get("stage")!="pilot_evaluated"
            or not (pilot.get("hardware") or {}).get("ok")
            or float(cmp.get("paired_delta",0))<=0.00005
            or cmp.get("single_fold_only") is not True):
        raise RuntimeError("No validated positive, single-fold RealMLP pilot")
    other=read(ROOT/"state/v2_9_4_status.json")
    if other.get("stage") not in ("skipped_no_cv_gain","candidate_ready_unsubmitted","complete","kernel_failed"):
        raise RuntimeError("V2.9.4 is not terminal; do not overlap GPU")


def save(s):
    s["updated_at"]=now().isoformat()
    write(STATUS,s)
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    cmp=s.get("comparison") or {}
    m=s.get("metrics") or {}
    rows=["# V2.9.6 RealMLP 完整三折 GPU 实验","",
          f"- 核验 UTC：{s['updated_at']}",
          f"- Kaggle kernel：https://www.kaggle.com/code/{s.get('kernel') or KERNEL}",
          f"- 当前阶段：{s.get('stage')}",
          f"- 验证协议：{VALIDATION}",
          "- 方法：固定 4 子模型、3 epochs 的 RealMLP-TD；训练集内 5fold 交叉目标编码。",
          "- 新预测不修改既有 V2.5、V2.8、V2.9.3 的实验输出。",
          f"- RealMLP 独立三折 OOF：{m.get('full_oof_auc','尚未训练')}",
          f"- 历史最佳三折 OOF：{cmp.get('baseline_auc','0.9607535043719558')}",
          f"- 新固定 80/20 融合 OOF：{cmp.get('candidate_auc','尚未完成')}",
          f"- 融合相对提升：{cmp.get('delta','尚未完成')}",
          f"- 单折改变量：{cmp.get('fold_gain','尚未完成')}",
          f"- 真正分配 Tesla T4：{(s.get('hardware') or {}).get('ok',False)}",
          f"- 是否已经 Kaggle 提交：{s.get('submitted',False)}",
          "- 正式 Public LB：尚无本版本新成绩；不能将 CV 当成 Public LB。",
          "- 本版本只负责训练与 CV gate；提交由独立预留后的一次性控制器执行。",
          ""]
    if s.get("error"):rows.extend(["## 错误与拦截",str(s["error"])[:2000]])
    REPORT.write_text("\n".join(rows)+"\n",encoding="utf-8")


def only(folder,name):
    found=list(folder.rglob(name))
    if len(found)!=1:
        raise RuntimeError(f"Exactly one {name} required, found {len(found)}")
    return found[0]


def evaluate(raw,s):
    mp=json.loads(only(raw,"v296_metrics.json").read_text(encoding="utf-8"))
    if (mp.get("competition")!=SLUG or mp.get("stage")!="three_folds_complete"
        or mp.get("model")!="RealMLP-TD n_ens=4 3 epochs"
        or mp.get("validation")!=VALIDATION or mp.get("folds")!=3
        or len(mp.get("fold_results",[]))!=3
        or mp.get("train_rows")!=N_TRAIN or mp.get("test_rows")!=N_TEST
        or mp.get("gpu_verified") is not True
        or mp.get("leakage_safe_inner_cross_fit") is not True
        or mp.get("outer_validation_used_in_training") is not False):
        raise ValueError("Incomplete or unverifiable full threefold RealMLP")
    hardware=json.loads(only(raw,"v295_hardware.json").read_text(encoding="utf-8"))
    if not hardware.get("ok") or not any("T4" in str(x) for x in hardware.get("devices",[])):
        raise ValueError("True Kaggle Tesla T4 evidence missing")
    with np.load(only(raw,"v296_realmlp_oof.npz"),allow_pickle=False) as z:
        labels=np.asarray(z["target"])
        ids=np.asarray(z["test_id"])
        train_ids=np.asarray(z["train_id"])
        oof=np.asarray(z["oof_realmlp"],dtype=np.float64)
        test_prediction=np.asarray(z["pred_realmlp"],dtype=np.float64)
    if (len(oof)!=N_TRAIN or len(labels)!=N_TRAIN
        or len(train_ids)!=N_TRAIN or len(test_prediction)!=N_TEST
        or len(ids)!=N_TEST or not np.isfinite(oof).all()
        or not np.isfinite(test_prediction).all()
        or np.unique(train_ids).size!=N_TRAIN
        or np.unique(ids).size!=N_TEST):
        raise ValueError("Truncated or duplicate output rows")
    if not (0<=oof.min()<=oof.max()<=1 and 0<=test_prediction.min()<=test_prediction.max()<=1):
        raise ValueError("RealMLP probabilities outside [0,1]")
    baseline=raw.parent/"baseline"
    baseline.mkdir(exist_ok=True)
    prior={}
    source_files={}
    for tag,version in (("v25","v2_5"),("v28","v2_8")):
        old=read(ROOT/f"state/{version}_status.json")
        if old.get("competition")!=SLUG or (old.get("metrics") or {}).get("validation")!=VALIDATION:
            raise ValueError("Archived source not comparable")
        handle=old.get("kernel")
        if not isinstance(handle,str) or not handle.startswith("mjw3177109/"):
            raise ValueError("Expected own previous Kaggle source")
        folder=baseline/tag
        folder.mkdir(exist_ok=True)
        cli(["kaggle","kernels","output",handle,"-p",str(folder),"-o","-q"],timeout=600)
        prior[tag]=old
        source_files[tag]=(only(folder,f"{tag}_oof.npz"),
                           pd.read_csv(only(folder,"candidate_submission.csv")))
    with np.load(source_files["v25"][0],allow_pickle=False) as a, \
         np.load(source_files["v28"][0],allow_pickle=False) as b:
        truth, sources, tests=verify_sources(
            a,b,prior["v25"],prior["v28"],
            source_files["v25"][1],source_files["v28"][1])
    if not np.array_equal(truth,labels):
        raise ValueError("RealMLP target order differs from own historical OOF")
    if not np.array_equal(source_files["v25"][1]["id"].to_numpy(),ids):
        raise ValueError("RealMLP test row IDs differ from archived Kaggle submission")
    base=.35*(rankdata(sources["v25"])/N_TRAIN)+.65*(rankdata(sources["v28"])/N_TRAIN)
    historical=float(roc_auc_score(truth,base))
    if abs(historical-0.9607535043719558)>0.000002:
        raise ValueError("Historical V2.9.3 baseline cannot be reproduced")
    if abs(roc_auc_score(truth,oof)-float(mp["full_oof_auc"]))>0.000002:
        raise ValueError("Full RealMLP self-reported score cannot be reproduced")
    # Precommitted fixed 80% old + 20% RealMLP; DO NOT grid-search OOF weights.
    mixed=.8*rankdata(base)/N_TRAIN+.2*rankdata(oof)/N_TRAIN
    full=float(roc_auc_score(truth,mixed))
    splits=list(StratifiedKFold(3,shuffle=True,random_state=20261018).split(
        np.zeros(len(truth),dtype=np.uint8),truth))
    fold_old=[];fold_new=[];fold_gain=[]
    for fold,(_,iv) in enumerate(splits):
        old_auc=float(roc_auc_score(truth[iv],base[iv]))
        new_auc=float(roc_auc_score(truth[iv],mixed[iv]))
        if abs(float(mp["fold_results"][fold]["auc"])-
               float(roc_auc_score(truth[iv],oof[iv])))>0.000002:
            raise ValueError("Neural fold CV does not match logged result")
        fold_old.append(old_auc);fold_new.append(new_auc);fold_gain.append(new_auc-old_auc)
    result={"baseline_auc":historical,
            "realmlp_auc":float(roc_auc_score(truth,oof)),
            "candidate_auc":full,"delta":full-historical,
            "fold_baseline":fold_old,"fold_candidate":fold_new,
            "fold_gain":fold_gain,"pre_registered_weights":{"v293":.8,"realmlp":.2},
            "positive_folds":sum(x>0 for x in fold_gain)}
    s["metrics"]=mp;s["hardware"]=hardware;s["comparison"]=result
    # All three folds are required, with at least 2 positive and no material
    # single-fold harm. This is a minimum for exploration, not a guarantee.
    qualified=(result["delta"]>=TRIAL_GAIN and result["positive_folds"]>=2
               and min(fold_gain)>-0.00005)
    if qualified:
        test_values=.35*tests["v25"]+.65*tests["v28"]
        candidate=.8*rankdata(test_values)/N_TEST+.2*rankdata(test_prediction)/N_TEST
        output=raw.parent/"v296_submission.csv"
        frame=source_files["v25"][1].copy()
        frame["satisfaction"]=candidate
        if len(frame)!=N_TEST or frame.isna().any().any():
            raise ValueError("New submission file invalid")
        frame.to_csv(output,index=False)
        sha=hashlib.sha256(output.read_bytes()).hexdigest()
        s["candidate_file_sha256"]=sha
        s["submission_note"]="KAS V296 REALMLP "+sha[:12]
        s["stage"]="candidate_ready_unsubmitted"
        s["candidate_file"]=str(output)
    else:
        s["stage"]="skipped_no_cv_gain"
    record_experiment(LEDGER,{
       "competition":SLUG,"candidate":"v296_realmlp_full_threefold",
       "validation":VALIDATION,"seed":20261018,
       "parameters":{"model":"realmlp_td","n_ens":4,"epochs":3,
                     "weight_old":.8,"weight_nn":.2},
       "status":"completed" if qualified else "skipped_no_cv_gain",
       "completed_at":now().isoformat(),
       "cv":full,"gpu_seconds":mp.get("runtime_seconds"),
       "kernel_sources":[s["kernel"]],"submission_id":None,
       "notes":"Kaggle GPU time is wall clock runtime, not metered utilization"})
    return s


def run(recover_only=False):
    prior=read(STATUS)
    if prior.get("stage") in TERMINAL:
        print("V2.9.6 already evaluated, never duplicate GPU kernel or submission")
        return prior
    eligible()
    s=prior if prior else {"competition":SLUG,"stage":"preflight",
                           "kernel":None,"submitted":False,"pilot_used":True}
    if not s.get("kernel"):
        if recover_only:
            print("No existing Notebook to recover");return s
        FOLDER.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/"kaggle_agent/v296_realmlp_full_train.py",FOLDER/"main.py")
        write(FOLDER/"kernel-metadata.json",{
            "id":KERNEL,"title":KERNEL.split("/",1)[1],
            "code_file":"main.py","language":"python","kernel_type":"script",
            "is_private":True,"enable_gpu":True,"machine_shape":"NvidiaTeslaT4",
            "enable_internet":True,"dataset_sources":[],
            "competition_sources":[SLUG],"kernel_sources":[],"model_sources":[]})
        s["kernel"]=KERNEL
        s["stage"]="pushing_kernel"
        save(s)
        response=cli(["kaggle","kernels","push","-p",str(FOLDER),"-t","3600",
                      "--accelerator","NvidiaTeslaT4"],timeout=300,check=False)
        if response.returncode:
            s.update({"stage":"push_failed","error":(response.stderr or response.stdout)[-1500:]})
            save(s);return s
        combined=(response.stdout or "")+"\n"+(response.stderr or "")
        match=re.search(r"https?://(?:www\.)?kaggle\.com/code/([A-Za-z0-9_-]+)/([a-z0-9-]+)",combined)
        if not match or match.group(1).lower()!="mjw3177109":
            s.update({"stage":"access_error","error":"No authenticated owner Notebook URL"})
            save(s);return s
        actual="/".join(match.groups())
        if actual!=KERNEL:
            s["kernel"]=actual
            s["kernel_alias"]=KERNEL
            s["canonical_note"]="Kaggle returned a title-derived canonical slug. The same Notebook is recovered, never repushed."
        else:
            s["kernel"]=KERNEL
        save(s)
    handle=s.get("kernel")
    if not isinstance(handle,str) or not handle.startswith("mjw3177109/"):
        raise RuntimeError("Untrusted Notebook owner")
    stage,message=poll_kernel(handle,90 if recover_only else 300,20)
    s["kernel_state"]=stage;s["stage"]=stage
    if stage in ("kernel_queued","kernel_running","kernel_access_error"):
        s["kernel_message"]=message[-700:];save(s);return s
    raw=OUT/"raw"
    raw.mkdir(parents=True,exist_ok=True)
    p=cli(["kaggle","kernels","output",handle,"-p",str(raw),"-o","-q"],
          timeout=600,check=False)
    if stage!="kernel_complete" or p.returncode:
        s["stage"]="kernel_failed" if stage!="kernel_complete" else "output_failed"
        errors=list(raw.rglob("v296_error.json"))
        s["error"]=(errors[0].read_text(encoding="utf-8")[-3000:] if errors
                     else (p.stderr or message or p.stdout)[-2000:])
        save(s);return s
    try:
        result=evaluate(raw,s)
    except Exception as exc:
        s["stage"]="validation_mismatch"
        s["error"]=str(exc)[-2000:]
        save(s);raise
    save(result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--recover-only",action="store_true")
    args=p.parse_args()
    s=run(recover_only=args.recover_only)
    print(json.dumps({"stage":s.get("stage"),"kernel":s.get("kernel"),
                      "comparison":s.get("comparison"),
                      "error":s.get("error")},ensure_ascii=False,indent=2,default=str))


if __name__=="__main__":
    main()
