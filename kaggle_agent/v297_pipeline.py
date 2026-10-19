"""V2.9.7 Kaggle CPU conditional-TE controller: one kernel, strict OOF.
CPU does not contend for T4. This controller NEVER submits to leaderboard.
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
KERNEL="mjw3177109/kas-v297-conditional-cpu-s6e10"
STATE=ROOT/"state/v2_9_7_status.json"
LEDGER=ROOT/"state/v2_9_experiments.json"
REPORT=ROOT/"reports/v2_9_7_conditional_cpu.md"
WORKDIR=ROOT/"competitions"/SLUG/"kaggle_cpu_v297"
OUTDIR=ROOT/"competitions"/SLUG/"results_v297"
FULL_BENCHMARK=0.9607535043719558
MIN_GAIN=.00006
TERMINAL={"skipped_no_cv_gain","candidate_ready_unsubmitted","kernel_failed",
          "output_failed","validation_mismatch","push_failed","kernel_access_error"}


def now():return datetime.now(timezone.utc)


def kaggle(args,timeout=240,check=True):
    p=subprocess.run(args,capture_output=True,text=True,timeout=timeout,cwd=str(ROOT))
    if check and p.returncode:
        raise RuntimeError("Kaggle command failed: "+(p.stderr or p.stdout)[-1200:])
    return p


def verify_contest():
    ladder=read(ROOT/"state/dual_ladder.json")
    if ladder.get("active_competition")!=SLUG:
        raise RuntimeError("Contest focus changed, refusing V297 CPU run")
    deadline=ladder.get("active_deadline")
    if not deadline:raise RuntimeError("Deadline unknown")
    dt=datetime.fromisoformat(deadline.replace("Z","+00:00"))
    if dt.tzinfo is None or now()>=dt:
        raise RuntimeError("Kaggle competition deadline reached")


def save(obj):
    obj["updated_at"]=now().isoformat()
    write(STATE,obj)
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    c=obj.get("comparison") or {}
    m=obj.get("metrics") or {}
    rows=[
        "# V2.9.7 条件目标编码独立 CPU 实验","",
        f"- UTC：{obj['updated_at']}",
        f"- 阶段：{obj.get('stage')}",
        f"- Kaggle Notebook：https://www.kaggle.com/code/{obj.get('kernel') or KERNEL}",
        "- 新模型：LightGBM，服务评分 × 旅客分组的条件 TE，配对评分 × 旅客类型，数值 × 类别 TE。",
        "- 频率特征：严格训练折内 5-fold cross-fit；不从验证集或测试集估计目标。",
        "- 资源：仅 Kaggle CPU（不占 T4），不使用外部标签或伪标签。",
        f"- 独立模型真实三折 OOF：{m.get('full_cv_auc','尚未完成')}",
        f"- 历史三折最佳：{c.get('baseline_auc',FULL_BENCHMARK)}",
        f"- 固定 85% 旧基线 + 15% 独立模型 OOF：{c.get('ensemble_auc','尚未完成')}",
        f"- CV 改进：{c.get('gain','未计算')}",
        f"- 各折配对增量：{c.get('fold_gain','未计算')}",
        f"- Kaggle 新 Public LB：无（此阶段绝不提交）",
        ""]
    if obj.get("error"):rows.extend(["## 错误",str(obj["error"])[-2000:]])
    REPORT.write_text("\n".join(rows)+"\n",encoding="utf-8")


def one(folder,name):
    hits=list(folder.rglob(name))
    if len(hits)!=1:
        raise RuntimeError(f"Require exactly one {name}: {len(hits)}")
    return hits[0]


def verify_outputs(raw,s):
    m=json.loads(one(raw,"v297_metrics.json").read_text(encoding="utf-8"))
    if (m.get("stage")!="full_threefold_complete"
        or m.get("competition")!=SLUG or m.get("validation")!=VALIDATION
        or m.get("folds")!=3 or len(m.get("fold_results",[]))!=3
        or m.get("train_rows")!=N_TRAIN or m.get("test_rows")!=N_TEST
        or m.get("inner_target_folds")!=5 or m.get("inner_frequency_folds")!=5
        or m.get("outer_validation_used_for_early_stopping") is not False
        or m.get("data_source")!="competition_train_only"):
        raise ValueError("Conditional model validation/provenance mismatch")
    with np.load(one(raw,"v297_oof.npz"),allow_pickle=False) as z:
        y=np.asarray(z["target"])
        train_id=np.asarray(z["train_id"])
        test_id=np.asarray(z["test_id"])
        prediction=np.asarray(z["oof_conditional"],dtype=np.float64)
        test_prediction=np.asarray(z["test_conditional"],dtype=np.float64)
    if (len(y)!=N_TRAIN or len(train_id)!=N_TRAIN or len(prediction)!=N_TRAIN
        or len(test_id)!=N_TEST or len(test_prediction)!=N_TEST
        or np.unique(train_id).size!=N_TRAIN or np.unique(test_id).size!=N_TEST
        or not np.isfinite(prediction).all() or not np.isfinite(test_prediction).all()
        or not np.isin(y,[0,1]).all()):
        raise ValueError("Invalid unaligned competition OOF rows")
    if abs(roc_auc_score(y,prediction)-float(m["full_cv_auc"]))>0.000002:
        raise ValueError("Reported conditional CV not reproducible")
    prior={};files={}
    archive=raw.parent/"baseline"
    archive.mkdir(exist_ok=True)
    for name,ver in (("v25","v2_5"),("v28","v2_8")):
        status=read(ROOT/f"state/{ver}_status.json")
        if (status.get("competition")!=SLUG
            or (status.get("metrics") or {}).get("validation")!=VALIDATION
            or not str(status.get("kernel","")).startswith("mjw3177109/")):
            raise ValueError("Historical OOF provenance failed")
        dest=archive/name;dest.mkdir(exist_ok=True)
        kaggle(["kaggle","kernels","output",status["kernel"],"-p",str(dest),"-o","-q"],
               timeout=600)
        prior[name]=status
        files[name]=(one(dest,f"{name}_oof.npz"),pd.read_csv(one(dest,"candidate_submission.csv")))
    with np.load(files["v25"][0],allow_pickle=False) as a, \
         np.load(files["v28"][0],allow_pickle=False) as b:
        labels,oofs,tests=verify_sources(
            a,b,prior["v25"],prior["v28"],files["v25"][1],files["v28"][1])
    if not np.array_equal(labels,y):
        raise ValueError("Historical labels differ from this 3fold run")
    if not np.array_equal(files["v25"][1]["id"].to_numpy(),test_id):
        raise ValueError("Kaggle test ID order mismatch")
    old=.35*rankdata(oofs["v25"])/N_TRAIN+.65*rankdata(oofs["v28"])/N_TRAIN
    old_auc=float(roc_auc_score(labels,old))
    if abs(old_auc-FULL_BENCHMARK)>0.000002:
        raise ValueError("Historical 3fold champion CV mismatch")
    proposed=.85*rankdata(old)/N_TRAIN+.15*rankdata(prediction)/N_TRAIN
    full=float(roc_auc_score(labels,proposed))
    splits=StratifiedKFold(3,shuffle=True,random_state=20261018)
    fold_gain=[]
    for fold,(_,iv) in enumerate(splits.split(np.zeros(len(y),dtype=np.uint8),y)):
        actual=float(roc_auc_score(y[iv],prediction[iv]))
        if abs(actual-float(m["fold_results"][fold]["auc"]))>0.000002:
            raise ValueError("Per-fold AUC mismatch, invalid experiment")
        fold_gain.append(float(roc_auc_score(y[iv],proposed[iv])-
                               roc_auc_score(y[iv],old[iv])))
    c={"baseline_auc":old_auc,"standalone_auc":float(m["full_cv_auc"]),
       "ensemble_auc":full,"gain":full-old_auc,"fold_gain":fold_gain,
       "positive_folds":sum(delta>0 for delta in fold_gain),
       "pre_registered_weights":{"baseline":.85,"conditional":.15}}
    s.update({"metrics":m,"comparison":c})
    passes=c["gain"]>=MIN_GAIN and c["positive_folds"]>=2 and min(fold_gain)>-.00005
    if passes:
        base_test=.35*rankdata(tests["v25"])/N_TEST+.65*rankdata(tests["v28"])/N_TEST
        submit_test=.85*rankdata(base_test)/N_TEST+.15*rankdata(test_prediction)/N_TEST
        candidate=files["v25"][1].copy()
        candidate["satisfaction"]=submit_test
        out=OUTDIR/"v297_candidate_unsubmitted.csv"
        candidate.to_csv(out,index=False)
        s["candidate_sha256"]=hashlib.sha256(out.read_bytes()).hexdigest()
        s["stage"]="candidate_ready_unsubmitted"
    else:
        s["stage"]="skipped_no_cv_gain"
    record_experiment(LEDGER,{
        "competition":SLUG,"candidate":"v297_conditioned_te_cpu",
        "validation":VALIDATION,"seed":20261018,
        "parameters":{"tree":"LGBM47","conditional_te":24,"oof_frequency":6,
                      "pre_registered_weight":.15,"external_teacher":False},
        "status":"completed" if passes else "skipped_no_cv_gain",
        "completed_at":now().isoformat(),"cv":full,
        "cpu_seconds":m.get("elapsed_seconds"),
        "gpu_seconds":0,"kernel_sources":[s.get("kernel")]})
    return s


def run(recover=False):
    old=read(STATE)
    if old.get("stage") in TERMINAL:
        print("V297 already terminal; no duplicate kernel")
        return old
    verify_contest()
    s=old if old else {"competition":SLUG,"stage":"preflight",
                       "kernel":None,"submitted":False}
    if not s.get("kernel"):
        if recover:
            print("No existing CPU kernel to recover")
            return s
        WORKDIR.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/"kaggle_agent/v297_conditional_cpu_train.py",WORKDIR/"main.py")
        write(WORKDIR/"kernel-metadata.json",{
            "id":KERNEL,"title":"kas-v297-conditional-cpu-s6e10",
            "code_file":"main.py","language":"python","kernel_type":"script",
            "is_private":True,"enable_gpu":False,"enable_internet":False,
            "dataset_sources":[],"competition_sources":[SLUG],
            "kernel_sources":[],"model_sources":[]})
        s.update({"kernel":KERNEL,"stage":"pushing_kernel"})
        save(s)
        p=kaggle(["kaggle","kernels","push","-p",str(WORKDIR),"-t","3600"],
                 timeout=300,check=False)
        if p.returncode:
            s.update({"stage":"push_failed","error":(p.stderr or p.stdout)[-1300:]})
            save(s)
            return s
        output=(p.stdout or "")+"\n"+(p.stderr or "")
        match=re.search(r"https?://(?:www\.)?kaggle\.com/code/([A-Za-z0-9_-]+)/([a-z0-9-]+)",output)
        if not match or match.group(1).lower()!="mjw3177109":
            s.update({"stage":"kernel_access_error","error":"Missing account-authenticated Kaggle Notebook URL"})
            save(s)
            return s
        s["kernel"]="/".join(match.groups())
        save(s)
    handle=s.get("kernel")
    if not isinstance(handle,str) or not handle.startswith("mjw3177109/"):
        raise RuntimeError("Unexpected Kaggle kernel owner")
    stage,message=poll_kernel(handle,90 if recover else 300,20)
    s.update({"stage":stage,"kernel_state":stage})
    if stage in ("kernel_running","kernel_queued","kernel_access_error"):
        s["kernel_message"]=message[-900:]
        save(s)
        return s
    raw=OUTDIR/"raw"
    raw.mkdir(parents=True,exist_ok=True)
    p=kaggle(["kaggle","kernels","output",handle,"-p",str(raw),"-o","-q"],
             timeout=600,check=False)
    if stage!="kernel_complete" or p.returncode:
        errors=list(raw.rglob("v297_error.json"))
        s["stage"]="kernel_failed" if stage!="kernel_complete" else "output_failed"
        s["error"]=errors[0].read_text(encoding="utf-8")[-2500:] if errors else (p.stderr or message or p.stdout)[-2200:]
        save(s)
        return s
    try:
        verify_outputs(raw,s)
    except Exception as exc:
        s.update({"stage":"validation_mismatch","error":str(exc)[-2200:]})
        save(s)
        raise
    save(s)
    return s


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--recover-only",action="store_true")
    args=parser.parse_args()
    result=run(recover=args.recover_only)
    print(json.dumps({"stage":result.get("stage"),"comparison":result.get("comparison"),
                      "error":result.get("error"),"kernel":result.get("kernel")},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
