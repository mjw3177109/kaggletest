"""V2.5 challenger controller: independently run, verify GPU, CV-gate, submit at most once.

Preserves the validated 0.95819 GPU champion and TPU experiment.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from v2_1_pipeline import ROOT, load_json, write_json, run, latest_submission
from v22_pipeline import kernel_handle, validate_submission
from v23_tpu_pipeline import poll_kernel, check_cv
from v24_rank_tracker import score_rank_lines

STATE=ROOT/"state"/"v2_5_status.json"
REPORT=ROOT/"reports"/"v2_5_latest.md"


def now():
    return datetime.now(timezone.utc).isoformat()


def champion_for(slug):
    cpu=load_json(ROOT/"state"/"v2_1_status.json",{}) or {}
    gpu=load_json(ROOT/"state"/"v2_3_gpu_status.json",{}) or {}
    if cpu.get("stage")!="complete" or cpu.get("competition")!=slug:
        raise RuntimeError("Verified completed V2.1 champion required")
    cv=float(cpu["metrics"]["best_cv_auc"])
    lb=float(cpu["latest_submission"]["publicScore"])
    name="V2.1 ensemble"
    if (gpu.get("competition")==slug and gpu.get("stage")=="complete"
            and gpu.get("submitted") and gpu.get("hardware",{}).get("ok")
            and gpu.get("metrics",{}).get("gpu_verified")
            and gpu.get("metrics",{}).get("validation")=="StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)"
            and gpu.get("latest_submission",{}).get("publicScore") not in (None,"")):
        cv=max(cv,float(gpu["metrics"]["best_cv_auc"]))
        glb=float(gpu["latest_submission"]["publicScore"])
        if glb>lb:
            lb=glb
            name=gpu.get("champion") or "V2.3 GPU"
    return {"cv":cv,"public_lb":lb,"model":name,
            "train_rows":int(cpu["metrics"]["rows"])}


def report(status):
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    m=status.get("metrics") or {}
    h=status.get("hardware") or {}
    gate=status.get("cv_gate") or {}
    score=(status.get("latest_submission") or {}).get("publicScore")
    lines=[
        "# V2.5｜研究高分选手后的 GPU 特征工程实验", "",
        f"- 生成时间 UTC：{now()}",
        f"- 比赛：{status.get('competition')}",
        f"- Kaggle Notebook：{status.get('kernel_url')}",
        f"- 运行阶段：{status.get('stage')}",
        f"- 真实 T4：{h.get('ok',False)}；{h.get('devices','尚未验证')}",
        "",
        "## 为什么改进？", "",
        "V2.3 GPU XGBoost 只针对原始特征和少量统计做调参，Public LB 0.95819。",
        "公开高分实验提示，精确数值交叉拟合 Target Encoding、航线距离画像和旅客细分群特征更有价值。",
        "V2.5 新增 5 折内部交叉拟合 TE；每个外层验证折只使用对应训练折生成的特征，禁止目标泄漏。",
        "",
        "## GPU/CPU 两模型候选与 OOF", "",
        f"- 原冠军：{status.get('incumbent_model')}",
        f"- 原冠军 CV：{status.get('incumbent_cv')}",
        f"- 原冠军 Public LB：{status.get('incumbent_lb')}",
        f"- LightGBM + XGBoost GPU 最佳模型：{m.get('best_model','尚未完成')}",
        f"- 新 3-Fold OOF：{m.get('best_cv_auc','尚未完成')}",
        f"- CV 超越差值：{gate.get('delta','暂无')}",
        f"- 门槛：{gate.get('threshold',0.0001)}",
        f"- 达标：{gate.get('passed',False)}",
        f"- 是否已提交：{status.get('submitted',False)}",
        f"- 新 Public LB：{score if score not in (None,'') else '尚未提交/未评分'}",
        f"- 线上冠军：{status.get('champion')}",
        "",
        "## 当前已评分冠军对应的官方排名", "",
        *score_rank_lines(status.get("competition"),
                          score if score not in (None, "") else status.get("incumbent_lb")),
        "",
        "## 同行参考（思路参考，不照搬外部预测文件）", "",
        "- GitHub: https://github.com/alvis1111/s6e10-airline-solution",
        "- Kaggle ROC-AUC 评测：https://www.kaggle.com/competitions/playground-series-s6e10/overview",
        "- 官方排行榜：https://www.kaggle.com/competitions/playground-series-s6e10/leaderboard",
        "",
        "任何外部公开方法或成绩都不是本站实测。上线是否有效，只以本次真实 OOF 和官方 Public LB 判断。",
    ]
    if m.get("models"):
        lines.extend(["","| 候选 | 同3折 OOF AUC |","|---|---:|"])
        for name, score2 in m["models"].items():
            lines.append(f"| {name} | {score2:.8f} |")
    if status.get("stage")=="kernel_queued":
        lines.extend(["","说明：Notebook 还在等待 Kaggle 分配 GPU；并非训练中。"])
    if status.get("error"):
        lines.extend(["","## 错误/状态诊断",str(status["error"])[-5500:]])
    REPORT.write_text("\n".join(lines)+"\n",encoding="utf-8")


def main(recover_only=False):
    baseline=load_json(ROOT/"state"/"v2_1_status.json",{}) or {}
    slug=str(baseline["competition"])
    champ=champion_for(slug)
    prev=load_json(STATE,{}) or {}
    if prev.get("competition")==slug and prev.get("stage") in (
        "complete","skipped_no_cv_gain","submission_pending","awaiting_approval"
    ):
        print("V2.5 already evaluated: refusing duplicate compute/submission")
        return
    status=dict(prev) if prev.get("competition")==slug else {}
    status.update({
        "competition":slug,"version":"V2.5","updated_at":now(),
        "incumbent_cv":champ["cv"],"incumbent_lb":champ["public_lb"],
        "incumbent_model":champ["model"],
        "champion":champ["model"],"submitted":bool(status.get("submitted",False)),
    })
    project=ROOT/"competitions"/slug
    folder=project/"kaggle_gpu_v25"
    results=project/"results_v25"
    folder.mkdir(parents=True,exist_ok=True)
    results.mkdir(parents=True,exist_ok=True)
    username=os.environ.get("KAGGLE_USERNAME") or str(baseline.get("kernel","")).split("/")[0]
    if not username:
        raise RuntimeError("KAGGLE_USERNAME not available")
    book_slug=("kas-v25-gpu-te-"+re.sub("[^a-z0-9-]","-",slug.lower()))[:48].rstrip("-")
    handle=f"{username}/{book_slug}"
    status.setdefault("kernel",handle)
    status.setdefault("kernel_url",f"https://www.kaggle.com/code/{handle}")
    try:
        if not recover_only and status.get("stage") not in ("kernel_queued","kernel_running","kernel_complete"):
            shutil.copy2(ROOT/"kaggle_agent"/"v25_gpu_te_train.py",folder/"main.py")
            write_json(folder/"kernel-metadata.json",{
                "id":handle, "title":book_slug, "code_file":"main.py",
                "language":"python","kernel_type":"script",
                "is_private":True,"enable_gpu":True,"machine_shape":"NvidiaTeslaT4",
                "enable_internet":False,"dataset_sources":[],
                "competition_sources":[slug],"kernel_sources":[],"model_sources":[],
            })
            status["stage"]="pushing_gpu_kernel"
            write_json(STATE,status)
            p=run(["kaggle","kernels","push","-p",str(folder),
                   "-t","3600","--accelerator","NvidiaTeslaT4"],check=False,timeout=300)
            msg=(p.stdout or "")+"\n"+(p.stderr or "")
            (folder/"push_output.txt").write_text(msg,encoding="utf-8")
            if p.returncode:
                status.update({"stage":"push_failed","error":msg[-7000:]})
                return
            status["kernel"]=kernel_handle(msg,username,book_slug)
        elif recover_only and not prev.get("kernel"):
            status.update({"stage":"not_launched","error":"No Kaggle Notebook for read-only recovery"})
            return
        status["kernel_url"]=f"https://www.kaggle.com/code/{status['kernel']}"
        stage,desc=poll_kernel(status["kernel"],90 if recover_only else 600,20)
        (folder/"status_output.txt").write_text(desc,encoding="utf-8")
        status["kernel_state"]=stage
        status["stage"]=stage
        if stage in ("kernel_queued","kernel_running","kernel_access_error"):
            status["error"]=desc[-4000:]
            return
        status.pop("error",None)
        raw=results/"raw"
        raw.mkdir(parents=True,exist_ok=True)
        p=run(["kaggle","kernels","output",status["kernel"],"-p",str(raw),"-o","-q"],check=False,timeout=600)
        hw=next(iter(raw.rglob("v25_hardware.json")),None)
        if hw:
            status["hardware"]=json.loads(hw.read_text(encoding="utf-8"))
            write_json(results/"hardware.json",status["hardware"])
            if not status["hardware"].get("ok"):
                status.update({"stage":"gpu_unavailable","error":"Not an actual NVIDIA T4"})
                return
        if stage!="kernel_complete":
            status.update({"stage":"kernel_failed","error":"Kaggle returned failed status: "+desc[-7000:]})
            return
        if p.returncode or hw is None:
            status.update({"stage":"output_failed","error":(p.stderr or p.stdout or "Missing GPU hardware probe")[-6000:]})
            return
        mp=next(iter(raw.rglob("v25_metrics.json")),None)
        sp=next(iter(raw.rglob("candidate_submission.csv")),None)
        if mp is None or sp is None:
            ep=next(iter(raw.rglob("v25_error.json")),None)
            status.update({"stage":"artifacts_missing","error":ep.read_text(encoding="utf-8")[-6000:] if ep else "Missing metrics or submission"})
            return
        metrics=json.loads(mp.read_text(encoding="utf-8"))
        if (metrics.get("competition")!=slug or not metrics.get("gpu_verified")
                or not metrics.get("leakage_safe_target_encoding")
                or not metrics.get("validation")=="StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)"
                or int(metrics.get("train_rows",-1))!=champ["train_rows"]
                or int(metrics.get("folds",0))!=3
                or not all(str(row.get("xgb_device","")).startswith("cuda") for row in metrics.get("fold_results",[]))):
            status.update({"stage":"incomparable_results","error":"GPU/CV/target encoding evidence invalid"})
            return
        validate_submission(sp,metrics["target"],int(metrics["test_rows"]))
        status["metrics"]=metrics
        write_json(results/"metrics.json",metrics)
        gate=check_cv(metrics["best_cv_auc"],champ["cv"],0.00010)
        status["cv_gate"]=gate
        if not gate["passed"]:
            status["stage"]="skipped_no_cv_gain"
            return
        # No auto-submission if a new competition has become the ladder focus.
        ladder=load_json(ROOT/"state"/"dual_ladder.json",{}) or {}
        if ladder.get("active_competition",slug)!=slug:
            status.update({"stage":"inactive_competition","error":"Contest rotation has already occurred; no old-contest submission"})
            return
        note=f"KAS V2.5 TE {metrics['best_model']} CV={metrics['best_cv_auc']:.6f}"
        latest=latest_submission(slug)
        if not latest or note not in str(latest.get("description","")):
            status["stage"]="submitting"
            write_json(STATE,status)
            p=run(["kaggle","competitions","submit",slug,"-f",str(sp),
                   "-m",note,"--wait","600","--poll-interval","30"],check=False,timeout=900)
            (results/"submit_output.txt").write_text((p.stdout or "")+"\n"+(p.stderr or ""),encoding="utf-8")
            if p.returncode:
                status.update({"stage":"submission_failed","error":(p.stderr or p.stdout)[-6000:]})
                return
            latest=latest_submission(slug)
        if not latest or note not in str(latest.get("description","")):
            status.update({"stage":"submission_unverified","error":"Could not verify Kaggle submitted entry"})
            return
        status["submitted"]=True
        status["latest_submission"]=latest
        write_json(results/"latest_submission.json",latest)
        new=latest.get("publicScore")
        if new in (None,""):
            status["stage"]="submission_pending"
            return
        status["public_lb_delta"]=round(float(new)-champ["public_lb"],8)
        if float(new)>champ["public_lb"]:
            status["champion"]="V2.5 "+str(metrics["best_model"])
        status["stage"]="complete"
    except Exception as e:
        status.update({"stage":"controller_failed","error":repr(e)})
    finally:
        status["updated_at"]=now()
        write_json(STATE,status)
        report(status)
        print(json.dumps(status,ensure_ascii=False,indent=2,default=str),flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--recover-only",action="store_true")
    main(cli.parse_args().recover_only)
