"""V2.9.5 RealMLP pilot controller; first real Kaggle T4 fold only.

Never submits to leaderboard, never updates full 3-fold champion and never
relaunches a Kaggle Notebook after a state exists.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from v291_ledger import read, write
from v293_blend_math import reconstruct, rank, VALIDATION, N_TRAIN
from v23_tpu_pipeline import poll_kernel
from v22_pipeline import kernel_handle

ROOT=Path(__file__).resolve().parents[1]
STATUS=ROOT/"state/v2_9_5_realmlp_status.json"
REPORT=ROOT/"reports/v2_9_5_realmlp_pilot.md"
SLUG="playground-series-s6e10"
KERNEL="mjw3177109/kas-v295-realmlp-pilot-s6e10"
FOLD=0
FINISHED={"pilot_evaluated","pilot_rejected","pilot_failed",
          "kernel_failed","hardware_unverified","output_failed",
          "validation_mismatch","kernel_access_error","push_failed"}


def now():
    return datetime.now(timezone.utc).isoformat()


def call(args,timeout=240,check=True):
    p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if check and p.returncode:
        raise RuntimeError(f"Command {args[:3]} failed: "+(p.stderr or p.stdout)[-1400:])
    return p


def report(status):
    status["updated_at"]=now()
    write(STATUS,status)
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    m=status.get("comparison") or {}
    lines=["# V2.9.5｜RealMLP CUDA 单折探索性实验","",
           f"- UTC：{status['updated_at']}",
           f"- 状态：{status.get('stage','unknown')}",
           f"- Kaggle Notebook：https://www.kaggle.com/code/{KERNEL}",
           "- 独立模型：RealMLP-TD，4 模型内集成、3 固定轮数、T4 GPU。",
           "- **这是原三折协议的第 0 折，不是完整三折 CV，也不是线上分数。**",
           f"- 同折 V2.9.3 参照 AUC：{m.get('baseline_fold_auc','未完成')}",
           f"- 同折 RealMLP AUC：{m.get('realmlp_fold_auc','未完成')}",
           f"- 固定 80/20 融合该折 AUC：{m.get('pilot_blend_auc','未完成')}",
           f"- 相对同折增量：{m.get('paired_delta','未完成')}",
           f"- 实际训练显卡：{(status.get('hardware') or {}).get('device','未核实')}",
           f"- GPU 实测时长：{(status.get('metrics') or {}).get('runtime_seconds','未核实')} 秒",
           "- Kaggle 新提交：无；该试验不产生 submission.csv。",
           "- 第一折单次收益不足以证明模型稳定提分；后续必须另跑剩余两折。",
           "- 不允许把单折 AUC 写入完整实验的 champion CV。",
           ""]
    if status.get("error"):
        lines.extend(["## 错误",str(status["error"])[-2000:]])
    REPORT.write_text("\n".join(lines)+"\n",encoding="utf-8")


def authorized_to_start():
    ladder=read(ROOT/"state/dual_ladder.json")
    if ladder.get("active_competition") != SLUG:
        return False,"competition_changed"
    deadline=ladder.get("active_deadline")
    if not deadline:
        return False,"deadline_missing"
    end=datetime.fromisoformat(deadline.replace("Z","+00:00"))
    if end.tzinfo is None or datetime.now(timezone.utc)>=end:
        return False,"deadline_expired"
    prior=read(ROOT/"state/v2_9_4_status.json")
    # Existing original-teacher branch has first claim on the available T4.
    if prior.get("stage") not in ("skipped_no_cv_gain","candidate_ready_unsubmitted",
            "complete","kernel_failed","gpu_unavailable","artifacts_missing",
            "incomparable_results","output_failed"):
        return False,"waiting_for_v294_to_finish"
    return True,"ok"


def get_one(directory,name):
    matches=list(directory.rglob(name))
    if len(matches)!=1:
        raise RuntimeError(f"Need one {name}; found {len(matches)}")
    return matches[0]


def compare_against_existing(raw):
    """Same fold, same labels and sample order; never compare unlike validation."""
    mp=json.loads(get_one(raw,"v295_metrics.json").read_text(encoding="utf-8"))
    if (mp.get("model")!="RealMLP-TD n_ens=4 3 epochs"
        or mp.get("validation")!=VALIDATION or mp.get("pilot_fold")!=FOLD
        or mp.get("folds_total")!=3 or mp.get("train_rows")!=N_TRAIN
        or mp.get("gpu_verified") is not True
        or mp.get("outer_validation_used_in_training") is not False
        or mp.get("leakage_safe_inner_cross_fit") is not True):
        raise ValueError("RealMLP provenance / validation mismatch")
    archived=raw.parent/"baseline"
    archived.mkdir(exist_ok=True)
    sources=[]
    for ver in ("v2_5","v2_8"):
        state=read(ROOT/f"state/{ver}_status.json")
        if state.get("competition")!=SLUG or (state.get("metrics") or {}).get("validation")!=VALIDATION:
            raise ValueError("Prior OOF validation protocol mismatch")
        handle=state.get("kernel")
        if not isinstance(handle,str) or not handle.startswith("mjw3177109/"):
            raise ValueError("Prior notebook owner is not verified")
        location=archived/ver
        location.mkdir(exist_ok=True)
        call(["kaggle","kernels","output",handle,"-p",str(location),"-o","-q"],timeout=600)
        sources.append(get_one(location,("v25" if ver=="v2_5" else "v28")+"_oof.npz"))
    with np.load(sources[0],allow_pickle=False) as first,\
         np.load(sources[1],allow_pickle=False) as second:
        y25, oof25, _=reconstruct(first,"v25")
        y28, oof28, _=reconstruct(second,"v28")
        if not np.array_equal(y25,y28):
            raise ValueError("Historical OOF labels differ")
    # This re-creates the fixed V2.9.3 35/65 rank ensemble without selecting weights.
    original_baseline=.35*rank(oof25)+.65*rank(oof28)
    if abs(float(roc_auc_score(y25,original_baseline))-0.9607535043719558)>0.000002:
        raise ValueError("Full historical 3-fold benchmark could not be reconstructed")
    indices=list(StratifiedKFold(3,shuffle=True,random_state=20261018).split(
        np.zeros(len(y25),dtype=np.uint8),y25))[FOLD][1]
    with np.load(get_one(raw,"v295_pilot_fold0.npz"),allow_pickle=False) as pilot:
        idx=np.asarray(pilot["validation_indices"])
        labels=np.asarray(pilot["target"])
        neural=np.asarray(pilot["probability"],dtype=np.float64)
        if not np.array_equal(idx,indices) or not np.array_equal(labels,y25[indices]):
            raise ValueError("Neural OOF rows or labels do not match original fold")
    if neural.shape!=(len(idx),) or not np.isfinite(neural).all():
        raise ValueError("Neural prediction lengths or values invalid")
    neural_auc=float(roc_auc_score(labels,neural))
    if abs(neural_auc-float(mp["fold_cv_auc"]))>0.000002:
        raise ValueError("Neural self-reported AUC does not reproduce")
    old_fold=original_baseline[indices]
    prior_auc=float(roc_auc_score(labels,old_fold))
    blend=.8*rank(old_fold)+.2*rank(neural)
    mixed=float(roc_auc_score(labels,blend))
    return mp,{"baseline_fold_auc":prior_auc,
               "realmlp_fold_auc":neural_auc,
               "pilot_blend_auc":mixed,"paired_delta":mixed-prior_auc,
               "out_of_fold_validation_rows":len(idx),
               "single_fold_only":True,
               "next_step":"more_folds_needed" if mixed>prior_auc+0.00005 else "do_not_expand",
               "automatic_submission":False}


def execute(recover=False):
    existing=read(STATUS)
    if existing.get("stage") in FINISHED:
        print("Already terminal; refusing repeated GPU experiment",existing.get("stage"))
        return existing
    if not existing.get("kernel"):
        if recover:
            return {"stage":"not_launched"}
        allowed,reason=authorized_to_start()
        if not allowed:
            status={"stage":"blocked_no_training","reason":reason,
                    "competition":SLUG,"kernel":None}
            report(status)
            return status
        folder=ROOT/"competitions"/SLUG/"kaggle_gpu_v295"
        folder.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/"kaggle_agent/v295_realmlp_pilot_train.py",folder/"main.py")
        write(folder/"kernel-metadata.json", {
            "id":KERNEL,"title":"V2.9.5 RealMLP strict pilot",
            "code_file":"main.py","language":"python","kernel_type":"script",
            "is_private":True,"enable_gpu":True,"machine_shape":"NvidiaTeslaT4",
            "enable_internet":True,"dataset_sources":[],
            "competition_sources":[SLUG],"kernel_sources":[],"model_sources":[]})
        status={"competition":SLUG,"kernel":KERNEL,"stage":"pushing_kernel",
                "submitted":False,"pilot_only":True}
        report(status)
        # If GitHub crashes after the push, its persisted handle prevents a second push.
        result=call(["kaggle","kernels","push","-p",str(folder),"-t","3600",
                     "--accelerator","NvidiaTeslaT4"],timeout=300,check=False)
        if result.returncode:
            status.update({"stage":"push_failed","error":(result.stderr or result.stdout)[-1300:]})
            report(status)
            return status
        try:
            kernel_handle((result.stdout or "")+"\n"+(result.stderr or ""),
                          "mjw3177109",KERNEL.split("/",1)[1])
        except Exception as exc:
            status.update({"stage":"kernel_access_error","error":str(exc)})
            report(status)
            return status
    else:
        if existing.get("kernel")!=KERNEL:
            raise RuntimeError("Stored handle mismatch; refuse accidental kernel overwrite")
        status=existing
    state,details=poll_kernel(KERNEL,90 if recover else 300,20)
    status["stage"]=state
    if state in ("kernel_queued","kernel_running","kernel_access_error"):
        status["last_kernel_message"]=details[-700:]
        report(status)
        return status

    output=ROOT/"competitions"/SLUG/"results_v295"/"raw"
    output.mkdir(parents=True,exist_ok=True)
    p=call(["kaggle","kernels","output",KERNEL,"-p",str(output),"-o","-q"],timeout=600,check=False)
    if state!="kernel_complete" or p.returncode:
        status.update({"stage":"kernel_failed" if state!="kernel_complete" else "output_failed",
                       "error":(p.stderr or details or p.stdout)[-1200:]})
        errors=list(output.rglob("v295_error.json"))
        if errors:status["error"]=errors[0].read_text(encoding="utf-8")[-1600:]
        report(status)
        return status
    hardware=read(get_one(output,"v295_hardware.json"))
    if not hardware.get("ok") or not hardware.get("cuda_available") or not any(
        "T4" in str(x) for x in hardware.get("devices",[])):
        status.update({"stage":"hardware_unverified","error":"T4 evidence unavailable"})
        report(status)
        return status
    status["hardware"]=hardware
    try:
        metrics,comparison=compare_against_existing(output)
        status["metrics"]=metrics
        status["comparison"]=comparison
        status["stage"]="pilot_evaluated" if comparison["paired_delta"]>0.00005 else "pilot_rejected"
    except Exception as exc:
        status.update({"stage":"validation_mismatch","error":str(exc)[-1400:]})
    report(status)
    return status


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--recover-only",action="store_true")
    args=parser.parse_args()
    result=execute(recover=args.recover_only)
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))


if __name__=="__main__":
    main()
