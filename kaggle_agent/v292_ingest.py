"""Import actual completed model status into immutable experiment ledger."""
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
from v291_ledger import read,record_experiment,daily_metrics
ROOT=Path(__file__).resolve().parents[1]
SOURCES=("v2_5","v2_6","v2_7","v2_8")
def ingest(root=ROOT):
    root=Path(root)
    ledger=root/"state/v2_9_experiments.json"
    added=[]
    for version in SOURCES:
        state=read(root/f"state/{version}_status.json")
        metrics=state.get("metrics") or {}
        cv=metrics.get("best_cv_auc",metrics.get("blended_cv_auc"))
        validation=metrics.get("validation")
        if not state.get("competition") or not validation or not isinstance(cv,(int,float)):
            continue
        stage=state.get("stage")
        if stage not in ("complete","skipped_no_cv_gain"):
            continue
        # Legacy status may be refreshed later. Prefer the first result timestamp,
        # and explicitly label it as approximate if the original completion is unknown.
        completed=state.get("completed_at") or state.get("updated_at")
        entry={"competition":state["competition"],"candidate":version,
               "validation":validation,"seed":20261018,"parameters":{"source_version":version},
               "status":"completed" if stage=="complete" else "skipped_no_cv_gain",
               "completed_at":completed,"timestamp_provenance":"completed_at" if state.get("completed_at") else "status_updated_at_proxy",
               "cv":cv,"gpu_seconds":metrics.get("elapsed_seconds") if (state.get("hardware") or {}).get("ok") else None}
        submission=state.get("latest_submission") or {}
        if state.get("submitted") and submission.get("ref") is not None and submission.get("publicScore") not in (None,""):
            entry["submission_id"]=submission["ref"]
            entry["public_lb"]=submission["publicScore"]
        _,created=record_experiment(ledger,entry)
        if created:
            added.append(version)
    return added

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--date",default=datetime.now(timezone.utc).date().isoformat())
    args=p.parse_args()
    added=ingest()
    ladder=read(ROOT/"state/dual_ladder.json")
    competition=ladder.get("active_competition","playground-series-s6e10")
    summary=daily_metrics(ROOT/"state/v2_9_experiments.json",args.date,competition)
    report=ROOT/"reports/v2_9_efficiency_daily.md"
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text("# V2.9 每日精进审计\n\n"+json.dumps({"ingested_versions":added,"competition":competition,"metrics":summary},ensure_ascii=False,indent=2)+"\n\n说明：历史完成时间可能采用状态更新时间代理；GPU 耗时仅包含有证据的报告值。\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False))
if __name__=="__main__":
    main()
