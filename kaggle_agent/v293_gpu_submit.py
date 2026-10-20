"""V2.9.3 real T4 candidate one-shot Kaggle submission with durable Git state.

Use only after the independently verified GPU controller recorded CV gain.
The workflow MUST commit a reservation to GitHub before --submit is invoked.
Do not re-submit on network exceptions; recovery reads Kaggle's own history.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from v291_ledger import read, write, record_experiment
from v293_pipeline import cli, records, matched, count_today, allowed_competition
from v293_blend_math import VALIDATION, N_TRAIN, N_TEST

ROOT=Path(__file__).resolve().parents[1]
COMP="playground-series-s6e10"
GPU_STATE=ROOT/"state/v2_9_3_gpu_status.json"
STATE=ROOT/"state/v2_9_3_gpu_submission.json"
LEDGER=ROOT/"state/v2_9_experiments.json"
REPORT=ROOT/"reports/v2_9_3_gpu_submission.md"
PREFIX="KAS V293 GPU"
GPU_GATE=0.00006
SUBMISSIONS_PER_DAY=5


def now():
    return datetime.now(timezone.utc)


def save(data):
    data["updated_at"]=now().isoformat()
    write(STATE,data)
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    s=(data.get("latest_submission") or {})
    REPORT.write_text("\n".join([
        "# V2.9.3 GPU 安全提交与回收审计","",
        "- UTC: "+str(data["updated_at"]),
        "- 阶段: "+str(data.get("stage")),
        "- Kaggle Notebook: "+str(data.get("kernel_url")),
        "- 经验证真实 GPU CV: "+str(data.get("cv_auc")),
        "- 提升前最佳 CV: "+str(data.get("incumbent_cv")),
        "- CV 提升: "+str(data.get("cv_delta")),
        "- 已预留提交/发送尝试: "+str(data.get("submission_attempted",False)),
        "- Kaggle 已认证提交编号: "+str(s.get("ref","无")),
        "- Public LB: "+str(s.get("publicScore","尚未评分/未提交")),
        "- 原已评分冠军 Public LB: "+str(data.get("incumbent_lb")),
        "- 排名由独立的 Kaggle 已登录账号追踪，不按 CV 推算。",
        "- 严格不重复提交；任何身份不明确或 API 网络异常均停止人工核查。",
        "- Private 最终名次和领奖需要单独核验。"
    ])+"\n",encoding="utf-8")


def only(path,filename):
    hits=list(path.rglob(filename))
    if len(hits)!=1:
        raise RuntimeError(f"Expected one {filename}, found {len(hits)}")
    return hits[0]


def trial_entry(gpu,status,sub=None):
    metrics=gpu["metrics"]
    entry={
        "competition":COMP,"candidate":"v293_gpu_"+str(metrics["best_model"]),
        "validation":VALIDATION,"seed":20261018,
        "parameters":{"variant":"v293_gpu_depth8_service_gap","kernel":gpu["kernel"],
                      "features":metrics["new_methods"]},
        "status":status,"cv":float(metrics["best_cv_auc"]),
        "gpu_seconds":metrics.get("elapsed_seconds"),
        "cpu_seconds":0,
        "kernel_sources":[gpu["kernel"]],
        "completed_at":now().isoformat() if status=="completed" else None,
    }
    if sub is not None:
        if sub.get("ref") is not None:
            entry["submission_id"]=sub["ref"]
        if sub.get("publicScore") not in (None,""):
            entry["public_lb"]=sub["publicScore"]
    return entry


def verified_gpu_candidate():
    gpu=read(GPU_STATE)
    metrics=gpu.get("metrics") or {}
    gate=gpu.get("cv_gate") or {}
    if (gpu.get("competition")!=COMP
        or gpu.get("stage") not in ("candidate_ready_unsubmitted","submission_pending","complete")
        or not isinstance(gpu.get("kernel"),str)
        or not gpu["kernel"].startswith("mjw3177109/kas-v293-gpu-")
        or not (gpu.get("hardware") or {}).get("ok")
        or not metrics.get("gpu_verified")
        or metrics.get("validation")!=VALIDATION
        or int(metrics.get("train_rows",-1))!=N_TRAIN
        or int(metrics.get("test_rows",-1))!=N_TEST
        or int(metrics.get("folds",-1))!=3
        or not all(row.get("cat_device")=="GPU" for row in metrics.get("fold_results",[]))
        or not gate.get("passed")
        or float(gate.get("threshold",0))!=GPU_GATE
        or float(metrics.get("best_cv_auc",0)) < float(gpu.get("incumbent_cv",1))+GPU_GATE):
        raise RuntimeError("No independently verified new Kaggle T4 GPU CV winner")
    return gpu


def prepare(workdir):
    previous=read(STATE)
    if previous.get("stage") in ("prepared","reserved","submitting","submission_pending","complete"):
        print("GPU submission already prepared or attempted; not creating duplicate")
        return
    allowed_competition()
    gpu=verified_gpu_candidate()
    raw=Path(workdir)/"gpu_original_output"
    raw.mkdir(parents=True,exist_ok=True)
    cli(["kaggle","kernels","output",gpu["kernel"],"-p",str(raw),"-o","-q"],timeout=600)
    metric=json.loads(only(raw,"v293_gpu_metrics.json").read_text(encoding="utf-8"))
    if (metric.get("validation")!=VALIDATION
        or abs(float(metric.get("best_cv_auc",-1))-float(gpu["metrics"]["best_cv_auc"]))>0.00000001
        or metric.get("competition")!=COMP
        or not metric.get("gpu_verified")):
        raise RuntimeError("Original Kaggle Notebook metrics mismatch controller state")
    source=only(raw,"candidate_submission.csv")
    df=pd.read_csv(source)
    if (list(df.columns)!=["id","satisfaction"] or len(df)!=N_TEST
            or not df["satisfaction"].between(0,1).all()
            or not df["satisfaction"].notna().all()
            or df["id"].duplicated().any()):
        raise RuntimeError("Candidate output submission invalid")
    dest=Path(workdir)/"candidate_submission.csv"
    dest.write_bytes(source.read_bytes())
    fp=hashlib.sha256(dest.read_bytes()).hexdigest()
    signature=hashlib.sha256(json.dumps({
        "model":gpu["kernel"],"cv":metric["best_cv_auc"],"test_sha":fp
    },sort_keys=True).encode()).hexdigest()[:12]
    state={
        "competition":COMP,"stage":"prepared","kernel_url":gpu.get("kernel_url"),
        "candidate_sha256":fp,"cv_auc":float(metric["best_cv_auc"]),
        "incumbent_cv":float(gpu["incumbent_cv"]),
        "cv_delta":float(metric["best_cv_auc"])-float(gpu["incumbent_cv"]),
        "incumbent_lb":float(gpu["incumbent_lb"]),
        "submission_note":f"{PREFIX} {signature}",
        "submission_attempted":False,"submitted":False,
    }
    save(state)
    record_experiment(LEDGER,trial_entry(gpu,"planned"))
    print("Prepared verified GPU candidate",state["submission_note"])


def check_file(state,workdir):
    path=Path(workdir)/"candidate_submission.csv"
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=state.get("candidate_sha256"):
        raise RuntimeError("GPU candidate file changed after verified CV; refuse submission")
    return path


def reserve(workdir):
    state=read(STATE)
    if state.get("stage")!="prepared" or state.get("submission_attempted"):
        raise RuntimeError("No new pre-verified GPU submission to reserve")
    allowed_competition()
    verified_gpu_candidate()
    check_file(state,workdir)
    entries=records()
    if matched(entries,state["submission_note"]):
        raise RuntimeError("Existing Kaggle submission has same unique candidate fingerprint")
    if count_today(entries,now().date())>=SUBMISSIONS_PER_DAY:
        state["stage"]="quota_wait"
        save(state)
        raise RuntimeError("Kaggle account daily submission quota exhausted")
    state["stage"]="reserved"
    state["submission_attempted"]=True
    state["reserved_at"]=now().isoformat()
    save(state)
    print("Reserved ONE GPU submission. This state must be committed to GitHub before submit.")


def finish(state,matched_row):
    gpu=read(GPU_STATE)
    state["submitted"]=True
    state["latest_submission"]=matched_row
    score=matched_row.get("publicScore")
    state["stage"]="submission_pending" if score in (None,"") else "complete"
    save(state)
    gpu["submitted"]=True
    gpu["latest_submission"]=matched_row
    gpu["stage"]=state["stage"]
    if score not in (None,""):
        gpu["public_lb_delta"]=round(float(score)-float(gpu["incumbent_lb"]),8)
        gpu["champion"]="V2.9.3 GPU "+str((gpu.get("metrics") or {}).get("best_model")) if float(score)>float(gpu["incumbent_lb"]) else gpu.get("champion","V2.5 blend_rank")
    gpu["updated_at"]=now().isoformat()
    write(GPU_STATE,gpu)
    record_experiment(LEDGER,trial_entry(gpu,"completed" if score not in (None,"") else "running",matched_row))


def submit(workdir,recover_only=False):
    state=read(STATE)
    if not state.get("submission_note") or not state.get("submission_attempted"):
        raise RuntimeError("Cannot submit unreserved GPU candidate")
    allowed_competition()
    rows=records()
    found=matched(rows,state["submission_note"])
    if found:
        finish(state,found)
        return
    if recover_only:
        print("No authenticated scored GPU submission to recover; no new submission attempted")
        return
    if state.get("stage")!="reserved":
        raise RuntimeError("Past GPU submission attempt is ambiguous; no automatic retry")
    if os.getenv("V293_GPU_RESERVED_COMMITTED")!="true":
        raise RuntimeError("GitHub reservation was not durably committed")
    path=check_file(state,workdir)
    if count_today(rows,now().date())>=SUBMISSIONS_PER_DAY:
        state["stage"]="quota_wait"
        save(state)
        return
    state["stage"]="submitting"
    save(state)
    try:
        cli(["kaggle","competitions","submit",COMP,"-f",str(path),"-m",
             state["submission_note"],"--wait","600","--poll-interval","30"],timeout=900)
        found=matched(records(),state["submission_note"])
        if not found:
            state["stage"]="submission_unverified"
            state["error"]="Kaggle did not return a matching authenticated submission"
            save(state)
            return
        finish(state,found)
    except Exception as err:
        state["stage"]="submission_ambiguous"
        state["error"]=str(err)[-1200:]+"; DO NOT retry automatically"
        save(state)
        raise


def main():
    ap=argparse.ArgumentParser()
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare",action="store_true")
    group.add_argument("--reserve",action="store_true")
    group.add_argument("--submit",action="store_true")
    group.add_argument("--recover",action="store_true")
    ap.add_argument("--workdir",default=os.environ.get("RUNNER_TEMP") or tempfile.gettempdir())
    args=ap.parse_args()
    workdir=Path(args.workdir)/"kas_v293_gpu_submit"
    workdir.mkdir(parents=True,exist_ok=True)
    if args.prepare: prepare(workdir)
    elif args.reserve: reserve(workdir)
    else: submit(workdir,recover_only=args.recover)


if __name__=="__main__":
    main()
