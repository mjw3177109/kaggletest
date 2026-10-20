"""Record only actual completed V2.9.4 Kaggle GPU experiment evidence."""
import json
from datetime import datetime,timezone
from pathlib import Path
from v291_ledger import read,record_experiment,daily_metrics

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/"state/v2_9_4_status.json"
LEDGER=ROOT/"state/v2_9_experiments.json"
REPORT=ROOT/"reports/v2_9_4_experiment_evidence.md"
VALIDATION="StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)"

def main():
    state=read(STATE)
    m=state.get("metrics") or {}
    safe=(state.get("competition")=="playground-series-s6e10"
          and state.get("stage") in ("skipped_no_cv_gain","candidate_ready_unsubmitted","complete")
          and m.get("validation")==VALIDATION
          and len(m.get("fold_results",[]))==3
          and m.get("gpu_verified") is True
          and m.get("outer_valid_used_for_early_stopping") is False
          and (m.get("original_teacher") or {}).get("no_competition_target_used") is True)
    written=False
    if safe:
        cv=m.get("best_cv_auc")
        if not isinstance(cv,(float,int)):
            raise ValueError("GPU experiment completed without numeric CV")
        entry={"competition":"playground-series-s6e10",
               "candidate":"v294_external_teacher_route_xgb",
               "validation":VALIDATION,"seed":20261018,
               "parameters":{"algorithms":["lightgbm","xgboost_cuda"],
                   "original_teacher":"teejmahal20/airline-passenger-satisfaction",
                   "route_category_profile":"outer_fold_only",
                   "frequency_encoding":"outer_fold_only",
                   "outer_early_stopping":False},
               "status":"completed" if state.get("stage") in ("complete","candidate_ready_unsubmitted") else "skipped_no_cv_gain",
               "completed_at":state.get("updated_at"),
               "cv":cv,"gpu_seconds":m.get("elapsed_seconds"),
               "kernel_sources":[state.get("kernel")],
               "gpu_evidence":state.get("hardware"),
               "submission_id":(state.get("latest_submission") or {}).get("ref")}
        _,written=record_experiment(LEDGER,entry)
    now=datetime.now(timezone.utc)
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    report={"checked_at":now.isoformat(),"competition":"playground-series-s6e10",
      "stage":state.get("stage","not_launched"),
      "cv_auc":m.get("best_cv_auc") if safe else None,
      "hardware_confirmed":bool(safe),"ledger_written":written,
      "previous_best_cv":state.get("incumbent_cv"),
      "submission_status":bool(state.get("submitted")),
      "data_lineage":(m.get("original_teacher") if safe else None),
      "daily":daily_metrics(LEDGER,now.date().isoformat(),"playground-series-s6e10")}
    REPORT.write_text("# V2.9.4 真实 GPU 实验审计\n\n"+json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":
    main()
