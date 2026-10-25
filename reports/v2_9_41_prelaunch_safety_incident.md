# V2.9.41 — first-launch preflight safety incident (no training)

- UTC: 2026-10-25 10:37:36.
- First controlled GitHub Actions attempt: #40185504849, conclusion FAILURE.
- Root cause: the 18-test safety suite still asserted that the one-shot workflow was dispatch-only. The authorized first-launch change added a push trigger to that same workflow, so test_workflow_dispatch_only failed before authenticated Kaggle owner-wide preflight, before run reservation, before competition data download, and before fitting.
- The other 17 offline tests passed, including the pinned synthetic spline logistic fit.
- Verified persisted state: state/v2_9_41_status.json remained stage=preregistered_not_started, run_reservation_id=null, training_started=false, submitted=false. No Notebook was created or pushed; no new CV or Kaggle submission.
- Containment: removed the one-shot push trigger to prevent an accidental second launch. Also corrected the report artifact path from v2_9_40_result.md to v2_9_41_result.md and the test-count step label.
- Policy: do not rerun #40185504849 or trigger V2941 training automatically. First recheck actual HEAD, all Actions states, durable reservation absence, owner Kaggle states and official submission evidence; keep fail-closed on ambiguity. A separate, explicitly reviewed launch decision is required. No parameter/feature/weight tuning and no changes to frozen V2941 preregistration.
- Evidence: https://github.com/mjw3177109/kaggletest/actions/runs/40185504849
