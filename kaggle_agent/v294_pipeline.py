"""V2.9.4 Original Teacher + Route Context challenger controller: independently run, verify GPU, CV-gate, submit at most once.

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

STATE=ROOT/"state"/"v2_9_4_status.json"
REPORT=ROOT/"reports"/"v2_9_4_latest.md"


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
    # V2.5 was a real, officially scored GPU champion; use it as the new bar.
    v25=load_json(ROOT/"state"/"v2_5_status.json",{}) or {}
    m25=v25.get("metrics") or {}
    s25=v25.get("latest_submission") or {}
    if (v25.get("competition")==slug and v25.get("stage")=="complete"
            and v25.get("submitted") and (v25.get("hardware") or {}).get("ok")
            and m25.get("gpu_verified")
            and m25.get("validation")=="StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)"
            and int(m25.get("train_rows",-1))==int(cpu["metrics"]["rows"])
            and s25.get("publicScore") not in (None,"")):
        cv=max(cv,float(m25["best_cv_auc"]))
        lb25=float(s25["publicScore"])
        if lb25>lb:
            lb=lb25
            name=v25.get("champion") or "V2.5 GPU"
    # Include only verified Kaggle submissions (OOF gain alone is not a Public LB).
    v27=load_json(ROOT/"state"/"v2_7_status.json",{}) or {}
    mv27=v27.get("metrics") or {}
    sv27=v27.get("latest_submission") or {}
    if (v27.get("competition")==slug and v27.get("stage")=="complete"
            and v27.get("submitted")
            and mv27.get("validation")=="StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)"
            and int(mv27.get("train_rows",-1))==int(cpu["metrics"]["rows"])
            and sv27.get("publicScore") not in (None,"")):
        cv=max(cv,float(mv27["blended_cv_auc"]))
        if float(sv27["publicScore"])>lb:
            lb=float(sv27["publicScore"])
            name=v27.get("champion") or "V2.7 OOF"
    # Optional TPU challenger: if genuine TPU finished, it must also be beaten.
    tpu=load_json(ROOT/"state"/"v2_3_status.json",{}) or {}
    mt=tpu.get("metrics") or {}
    st=tpu.get("latest_submission") or {}
    if (tpu.get("stage")=="complete" and tpu.get("competition")==slug
            and tpu.get("submitted") and (tpu.get("hardware") or {}).get("actual")=="tpu"
            and mt.get("backend")=="tpu"
            and mt.get("validation")=="StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)"
            and int(mt.get("train_rows",-1))==int(cpu["metrics"]["rows"])
            and st.get("publicScore") not in (None,"")):
        cv=max(cv,float(mt["cv_auc"]))
        if float(st["publicScore"])>lb:
            lb=float(st["publicScore"])
            name=tpu.get("champion") or "V2.3 TPU"
    # GPU challenger must beat the latest *measured* comparable CV, not only
    # Kaggle submissions. A CV-only candidate is not a Public LB champion.
    v28 = load_json(ROOT/"state"/"v2_8_status.json",{}) or {}
    m28 = v28.get("metrics") or {}
    if (v28.get("competition")==slug and v28.get("stage") in
            ("skipped_no_cv_gain","complete")
            and (v28.get("hardware") or {}).get("ok")
            and m28.get("gpu_verified")
            and m28.get("validation")=="StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)"
            and int(m28.get("train_rows",-1))==int(cpu["metrics"]["rows"])):
        cv=max(cv,float(m28["best_cv_auc"]))
    # V2.9.3 CPU blending is an independently evaluated OOF candidate; do
    # not mistake its CV improvement for a new Public score.
    blend = load_json(ROOT/"state"/"v2_9_3_status.json",{}) or {}
    candidates = ((blend.get("result") or {}).get("candidates") or [])
    if blend.get("competition")==slug and blend.get("stage") in ("skipped_no_cv_gain","complete"):
        for row in candidates:
            if isinstance(row.get("cv_auc"),(float,int)):
                cv=max(cv,float(row["cv_auc"]))
    return {"cv":cv,"public_lb":lb,"model":name,
            "train_rows":int(cpu["metrics"]["rows"])}


def report(status):
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    m=status.get("metrics") or {}
    h=status.get("hardware") or {}
    gate=status.get("cv_gate") or {}
    score=(status.get("latest_submission") or {}).get("publicScore")
    lines=[
        "# V2.9.4 Original Teacher + Route Context｜独立原始数据教师 + 航线画像实验", "",
        f"- 生成时间 UTC：{now()}",
        f"- 比赛：{status.get('competition')}",
        f"- Kaggle Notebook：{status.get('kernel_url')}",
        f"- 运行阶段：{status.get('stage')}",
        f"- 真实 T4：{h.get('ok','尚未确认（Kernel 尚未执行）')}；{h.get('devices','尚未验证')}",
        "",
        "## 为什么改进？", "",
        "V2.5 使用无泄漏 Target Encoding + 航线画像获得 Public LB 0.96035，已成为当前正式冠军。",
        "公开高分实验提示，精确数值交叉拟合 Target Encoding、航线距离画像和旅客细分群特征更有价值。",
        "V2.9.4 Original Teacher + Route Context 保留 V2.5 已验证的内层 5 折无泄漏目标编码，新增独立原始标签训练的教师分数、逐折航线/客群概率及 GPU XGBoost，不使用外层验证标签早停。",
        "",
        "## GPU/CPU 两模型候选与 OOF", "",
        f"- 原冠军：{status.get('incumbent_model')}",
        f"- 原冠军 CV：{status.get('incumbent_cv')}",
        f"- 原冠军 Public LB：{status.get('incumbent_lb')}",
        f"- LightGBM CPU + XGBoost GPU 最佳模型：{m.get('best_model','尚未完成')}",
        f"- 新 3-Fold OOF：{m.get('best_cv_auc','尚未完成')}",
        f"- CV 超越差值：{gate.get('delta','暂无')}",
        f"- 门槛：{gate.get('threshold',0.00006)}",
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
    # A scored submission can remain PENDING briefly. Requery Kaggle by its
    # unique submission ref; never train or submit again while awaiting a score.
    if prev.get("competition")==slug and prev.get("stage")=="submission_pending":
        pending=prev.get("latest_submission") or {}
        ref=pending.get("ref")
        if ref is None:
            raise RuntimeError("Pending V2.9.4 Original Teacher + Route Context submission missing Kaggle ref; manual review required")
        actual=latest_submission(slug)
        if actual and str(actual.get("ref"))==str(ref):
            pending=actual
        # Prefer matching ref, not an unrelated newer submission.
        score=pending.get("publicScore")
        result=dict(prev)
        if score not in (None,""):
            result["latest_submission"]=pending
            result["submitted"]=True
            result["public_lb_delta"]=round(float(score)-float(prev["incumbent_lb"]),8)
            if float(score)>float(prev["incumbent_lb"]):
                result["champion"]="V2.9.4 Original Teacher + Route Context "+str((prev.get("metrics") or {}).get("best_model","unknown"))
            result["stage"]="complete"
        result["updated_at"]=now()
        write_json(STATE,result)
        report(result)
        print("V2.9.4 Original Teacher + Route Context existing submission status:", result["stage"])
        return
    if prev.get("competition")==slug and prev.get("stage") in (
        "complete","skipped_no_cv_gain","awaiting_approval"
    ):
        print("V2.9.4 Original Teacher + Route Context already evaluated: refusing duplicate compute/submission")
        return
    status=dict(prev) if prev.get("competition")==slug else {}
    status.update({
        "competition":slug,"version":"V2.9.4 Original Teacher + Route Context","updated_at":now(),
        "incumbent_cv":champ["cv"],"incumbent_lb":champ["public_lb"],
        "incumbent_model":champ["model"],
        "champion":champ["model"],"submitted":bool(status.get("submitted",False)),
    })
    project=ROOT/"competitions"/slug
    folder=project/"kaggle_gpu_v294"
    results=project/"results_v294"
    folder.mkdir(parents=True,exist_ok=True)
    results.mkdir(parents=True,exist_ok=True)
    username=os.environ.get("KAGGLE_USERNAME") or str(baseline.get("kernel","")).split("/")[0]
    if not username:
        raise RuntimeError("KAGGLE_USERNAME not available")
    book_slug=("kas-v294-original-route-"+re.sub("[^a-z0-9-]","-",slug.lower()))[:48].rstrip("-")
    handle=f"{username}/{book_slug}"
    status.setdefault("kernel",handle)
    status.setdefault("kernel_url",f"https://www.kaggle.com/code/{handle}")
    try:
        if not recover_only and status.get("stage") not in ("kernel_queued","kernel_running","kernel_complete"):
            shutil.copy2(ROOT/"kaggle_agent"/"v294_original_teacher_train.py",folder/"main.py")
            write_json(folder/"kernel-metadata.json",{
                "id":handle, "title":book_slug, "code_file":"main.py",
                "language":"python","kernel_type":"script",
                "is_private":True,"enable_gpu":True,"machine_shape":"NvidiaTeslaT4",
                "enable_internet":False,"dataset_sources":["teejmahal20/airline-passenger-satisfaction"],
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
        hw=next(iter(raw.rglob("v294_hardware.json")),None)
        if hw:
            status["hardware"]=json.loads(hw.read_text(encoding="utf-8"))
            write_json(results/"hardware.json",status["hardware"])
            if not status["hardware"].get("ok"):
                status.update({"stage":"gpu_unavailable","error":"Not an actual NVIDIA T4"})
                return
        if stage!="kernel_complete":
            # Kaggle may emit v294_error.json after a failed training attempt.
            err=next(iter(raw.rglob("v294_error.json")),None)
            status.update({
                "stage":"kernel_failed",
                "error":(err.read_text(encoding="utf-8")[-7000:] if err
                         else "Kaggle reported a failed kernel; "+desc[-5000:]),
            })
            logs=run(["kaggle","kernels","logs",status["kernel"]],check=False,timeout=120)
            (folder/"logs.txt").write_text(((logs.stdout or "")+"\n"+(logs.stderr or ""))[-20000:],encoding="utf-8")
            return
        if p.returncode or hw is None:
            status.update({"stage":"output_failed","error":(p.stderr or p.stdout or "Missing GPU hardware probe")[-6000:]})
            return
        mp=next(iter(raw.rglob("v294_metrics.json")),None)
        sp=next(iter(raw.rglob("candidate_submission.csv")),None)
        if mp is None or sp is None:
            ep=next(iter(raw.rglob("v294_error.json")),None)
            status.update({"stage":"artifacts_missing","error":ep.read_text(encoding="utf-8")[-6000:] if ep else "Missing metrics or submission"})
            return
        metrics=json.loads(mp.read_text(encoding="utf-8"))
        if (metrics.get("competition")!=slug or not metrics.get("gpu_verified")
                or not metrics.get("leakage_safe_target_encoding")
                or not (metrics.get("original_teacher") or {}).get("no_competition_target_used")
                or (metrics.get("original_teacher") or {}).get("train_kept",0)<10000
                or not metrics.get("validation")=="StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)"
                or int(metrics.get("train_rows",-1))!=champ["train_rows"]
                or int(metrics.get("folds",0))!=3
                or len(metrics.get("fold_results",[]))!=3
                or metrics.get("outer_valid_used_for_early_stopping") is not False
                or not all(str(row.get("xgb_device","")).startswith("cuda") for row in metrics.get("fold_results",[]))):
            status.update({"stage":"incomparable_results","error":"GPU/CV/target encoding evidence invalid"})
            return
        validate_submission(sp,metrics["target"],int(metrics["test_rows"]))
        status["metrics"]=metrics
        write_json(results/"metrics.json",metrics)
        gate=check_cv(metrics["best_cv_auc"],champ["cv"],0.00006)
        status["cv_gate"]=gate
        if not gate["passed"]:
            status["stage"]="skipped_no_cv_gain"
            return
        # Fail-closed handoff: model genuinely trained on T4 and OOF scored,
        # but Kaggle submission requires a separate durable reservation.
        # The existing V2.5 Public LB champion is never overwritten here.
        status["stage"]="candidate_ready_unsubmitted"
        status["submitted"]=False
        status["safe_submission_pending"]=True
        status["error"]="CV qualified; awaiting independent one-shot submission safety check"
        return
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
