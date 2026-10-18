"""V2.7: pair verified V2.5/V2.6 out-of-fold predictions on CPU.

No new Kaggle training notebook or GPU quota; all input predictions must come
from the owner's previously completed Kaggle kernels. Fail closed on mismatched
rows, unverified CV, daily submission limit or stale competition focus.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from v2_1_pipeline import ROOT, load_json, write_json, run, list_json
from v22_pipeline import validate_submission
from v24_rank_tracker import score_rank_lines
from v27_blend_math import model_oof, paired_fusion

STATE=ROOT/"state"/"v2_7_status.json"
REPORT=ROOT/"reports"/"v2_7_latest.md"
COMPETITION="playground-series-s6e10"
NOTE_PREFIX="KAS V2.7 OOF50"


def now():
    return datetime.now(timezone.utc)


def ready():
    v25=load_json(ROOT/"state"/"v2_5_status.json",{}) or {}
    v26=load_json(ROOT/"state"/"v2_6_status.json",{}) or {}
    if (v25.get("stage")!="complete" or not v25.get("submitted")
            or v26.get("stage") not in ("complete","skipped_no_cv_gain")):
        raise RuntimeError("Both models must have terminal, verified V2.5/V2.6 CV results")
    for d in (v25,v26):
        m=d.get("metrics") or {}
        if (d.get("competition")!=COMPETITION
                or m.get("competition")!=COMPETITION
                or not m.get("gpu_verified")
                or not (d.get("hardware") or {}).get("ok")
                or m.get("validation")!="StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)"
                or int(m.get("train_rows",-1))!=699635
                or int(m.get("test_rows",-1))!=299844
                or not m.get("leakage_safe_target_encoding")
                or m.get("best_model") not in ("lightgbm","xgboost","blend_mean","blend_rank")):
            raise RuntimeError("Incomparable models, invalid GPU verification or CV configuration")
    ladder=load_json(ROOT/"state"/"dual_ladder.json",{}) or {}
    if ladder.get("active_competition") != COMPETITION:
        raise RuntimeError("Competition already rotated away; refuse old contest submission")
    deadline=ladder.get("active_deadline")
    if not deadline:
        raise RuntimeError("No official contest deadline on file")
    cutoff=datetime.fromisoformat(deadline.replace("Z","+00:00"))
    if cutoff.tzinfo is None or now() >= cutoff:
        raise RuntimeError("Competition closed or deadline invalid")
    return v25,v26


def submissions(slug):
    p=run(["kaggle","competitions","submissions",slug,"--page-size","100",
           "--format","json"],check=False,timeout=120)
    if p.returncode:
        raise RuntimeError("Could not verify Kaggle submission quota: "+(p.stderr or p.stdout)[-700:])
    rows=list_json(p.stdout)
    if not isinstance(rows,list):
        raise RuntimeError("Unexpected Kaggle submissions result")
    return rows


def daily_count(rows, today):
    count=0
    for row in rows:
        try:
            dt=datetime.fromisoformat(str(row["date"]).replace("Z","+00:00"))
            if dt.tzinfo is None:
                dt=dt.replace(tzinfo=timezone.utc)
        except (KeyError,ValueError,TypeError):
            # Kaggle listing uncertainty means we must fail closed; don't
            # risk a daily-limit error due to an ambiguous timestamp.
            raise RuntimeError("Submission date unverifiable")
        if dt.astimezone(timezone.utc).date()==today:
            count+=1
    return count


def download_output(handle, directory):
    if not handle or "/" not in str(handle):
        raise RuntimeError("Invalid original Kaggle Notebook handle")
    p=run(["kaggle","kernels","output",handle,"-p",str(directory),"-o","-q"],
          check=False,timeout=600)
    if p.returncode:
        raise RuntimeError("Cannot download own Kaggle predictions: "+(p.stderr or p.stdout)[-1000:])


def only(directory, filename):
    matches=list(directory.rglob(filename))
    if len(matches)!=1:
        raise RuntimeError(f"Expected exactly one Kaggle output {filename}, got {len(matches)}")
    return matches[0]


def score_report(status):
    benchmark=status.get("benchmark_cv")
    metrics=status.get("metrics") or {}
    score=(status.get("latest_submission") or {}).get("publicScore")
    incumbent=status.get("incumbent_lb")
    lines=[
        "# V2.7｜零新增 GPU 算力的双模型 OOF 融合", "",
        f"- 评估时间 UTC：{status.get('updated_at')}",
        f"- 比赛：{COMPETITION}",
        f"- 状态：{status.get('stage')}",
        "- 本轮不创建新的 Kaggle GPU / TPU Notebook，只复用自己已训练模型的 OOF。",
        f"- 训练行数：{metrics.get('train_rows', '待验证')}",
        "- 两个模型同一个 StratifiedKFold(3, seed=20261018)，且标签/样本顺序需逐行完全一致。",
        "- 融合比例：预设 50% V2.5 + 50% V2.6，先排名归一化，不用排行榜反复搜权重。",
        "",
        "## 对比与提交安全门槛", "",
        f"- V2.5 3折 OOF：{metrics.get('v25_oof', '暂无')}",
        f"- V2.6 3折 OOF：{metrics.get('v26_oof', '暂无')}",
        f"- V2.7 融合 OOF：{metrics.get('blended_cv_auc', '暂无')}",
        f"- 当前最强独立 CV 基准：{benchmark if benchmark is not None else '暂无'}",
        f"- 需要超越的阈值：{status.get('required_cv', '暂无')}",
        f"- CV 门槛是否通过：{status.get('cv_gate_passed', False)}",
        f"- 是否提交 Kaggle：{status.get('submitted', False)}",
        f"- 此次正式 Public LB：{score if score not in (None,'') else '尚未提交或评分'}",
        f"- 旧冠军 Public LB：{incumbent if incumbent is not None else '暂无'}",
        "", "## 当前冠军分数对应的官方名次", "",
        *score_rank_lines(COMPETITION,score if score not in (None,"") else incumbent),
        "",
        "Public 榜名次并不是最终 Private 排名。所有新增正式提交还会触发单独的账号认证排行榜检查。",
    ]
    if status.get("error"):
        lines.extend(["", "## 诊断", status["error"]])
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    prev=load_json(STATE,{}) or {}
    if prev.get("stage") in ("complete","skipped_no_cv_gain","submission_recovered"):
        print("V2.7 terminal state exists; refuse duplicate work and submission")
        return
    v25,v26=ready()
    base=load_json(ROOT/"state"/"score_rank_latest.json",{}) or {}
    incumbent=max(
        float(v25["latest_submission"]["publicScore"]),
        float((v26.get("latest_submission") or {}).get("publicScore") or 0),
    )
    cvbenchmark=max(float(v25["metrics"]["best_cv_auc"]),
                    float(v26["metrics"]["best_cv_auc"]))
    status=dict(prev)
    status.update({
        "version":"V2.7","competition":COMPETITION,
        "updated_at":now().isoformat(),"stage":"fetching_own_oof",
        "incumbent_lb":incumbent,"benchmark_cv":cvbenchmark,
        "required_cv":round(cvbenchmark+0.00010,10),
        "submitted":bool(prev.get("submitted",False)),
    })
    try:
        with tempfile.TemporaryDirectory(prefix="kas-v27-") as tmp:
            root=Path(tmp)
            pa=root/"v25";pb=root/"v26"
            pa.mkdir();pb.mkdir()
            download_output(v25["kernel"],pa)
            download_output(v26["kernel"],pb)
            for directory,version in ((pa,"v25"),(pb,"v26")):
                if not only(directory,version+"_oof.npz").is_file():
                    raise RuntimeError("Missing own OOF predictions")
            with np.load(only(pa,"v25_oof.npz")) as a, np.load(only(pb,"v26_oof.npz")) as b:
                ya=np.asarray(a["target"]).copy()
                yb=np.asarray(b["target"]).copy()
                oa=model_oof(a,str(v25["metrics"]["best_model"]))[0]
                ob=model_oof(b,str(v26["metrics"]["best_model"]))[0]
            sa=pd.read_csv(only(pa,"candidate_submission.csv"))
            sb=pd.read_csv(only(pb,"candidate_submission.csv"))
            if (list(sa.columns)!=["id","satisfaction"] or list(sb.columns)!=list(sa.columns)
                    or sa.shape[0]!=299844 or not np.array_equal(sa["id"].values,sb["id"].values)):
                raise RuntimeError("V2.5 and V2.6 test identifiers differ; cannot safely blend")
            result=paired_fusion(ya,yb,oa,ob,sa["satisfaction"],sb["satisfaction"])
            # Verify OOF scoring matches the exact selected CV recorded by Kaggle.
            for key,stat in (("score25",v25),("score26",v26)):
                diff=abs(result[key]-float(stat["metrics"]["best_cv_auc"]))
                if diff>5e-7:
                    raise RuntimeError(f"{key}: cached OOF vs CV mismatch {diff}")
            score=float(result["fused_cv_auc"])
            status["metrics"]={
                "train_rows":len(ya),"test_rows":len(sa),
                "validation":"StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)",
                "fusion":"fixed_50_50_rank_blend","v25_oof":result["score25"],
                "v26_oof":result["score26"],"blended_cv_auc":score,
                "delta_against_champion":score-cvbenchmark,
            }
            status["cv_gate_passed"]=(score>cvbenchmark+0.00010)
            if not status["cv_gate_passed"]:
                status["stage"]="skipped_no_cv_gain"
                return
            d=ready()  # Guard against a competition handoff during download.
            rows=submissions(COMPETITION)
            note=f"{NOTE_PREFIX} CV={score:.6f}"
            matching=next((x for x in rows if str(x.get("description","")).startswith(NOTE_PREFIX)),None)
            if matching is None:
                used=daily_count(rows,now().date())
                status["daily_submissions_before"]=used
                if used>=5:
                    status["stage"]="quota_wait"
                    return
                out=root/"v27_submission.csv"
                scored=sa.copy()
                scored["satisfaction"]=result["fused_test"]
                scored.to_csv(out,index=False)
                validate_submission(out,"satisfaction",299844)
                status["stage"]="submitting"
                write_json(STATE,status)
                p=run(["kaggle","competitions","submit",COMPETITION,
                       "-f",str(out),"-m",note,"--wait","600",
                       "--poll-interval","30"],check=False,timeout=900)
                if p.returncode:
                    status["stage"]="submission_failed"
                    status["error"]=(p.stderr or p.stdout)[-2500:]
                    return
                rows=submissions(COMPETITION)
                matching=next((x for x in rows if str(x.get("description","")).startswith(NOTE_PREFIX)),None)
            if matching is None:
                status["stage"]="submission_unverified"
                status["error"]="Kaggle did not return the submitted V2.7 row"
                return
            status["latest_submission"]=matching
            status["submitted"]=True
            new_score=matching.get("publicScore")
            if new_score in (None,""):
                status["stage"]="submission_pending"
                return
            status["public_lb_delta"]=round(float(new_score)-incumbent,8)
            status["champion"]="V2.7 fixed50 OOF blend" if float(new_score)>incumbent else (
                "V2.6 blend" if float((v26.get("latest_submission") or {}).get("publicScore") or 0)>float(v25["latest_submission"]["publicScore"])
                else "V2.5 blend_rank")
            status["stage"]="complete"
    except Exception as err:
        status["stage"]="failed"
        status["error"]=str(err)[-3500:]
    finally:
        status["updated_at"]=now().isoformat()
        write_json(STATE,status)
        score_report(status)
        print(json.dumps(status,ensure_ascii=False,indent=2,default=str))


if __name__=="__main__":
    main()
