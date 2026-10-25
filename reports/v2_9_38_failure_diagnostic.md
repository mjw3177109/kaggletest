# V2.9.38 — terminal verified CatBoost Ordered CPU result (no submission)

Verified on **2026-10-25 05:38:57 UTC**. The sole guarded CPU Actions run **#40164585333** completed SUCCESS on its first attempt. Its exclusive reservation was durably persisted as run ID `40164585333` before official data access or model fitting. All 21 CPU safety tests and pinned CatBoost 1.2.8 Ordered synthetic preflight passed. The CPU run completed three outer folds and independent validation. No Kaggle Notebook was created or pushed, and no official submission was attempted.

## Frozen verified results

- Champion V2915 threefold CV: `0.9612850399332336`.
- V2938 independent CatBoost Ordered threefold CV: `0.9574607832211844`.
- Frozen 85/15 rank blend CV: **`0.9611212529755768`**.
- Paired gain versus champion: **`-0.0001637869576567308`**.
- Fold gains: `[-0.00018233594862138158,-0.0001240216634323943,-0.00018664582521299433]`; **0/3 positive**.
- V2938 source NPZ SHA256: `c07d9f19e0d45eb9706174554cbfaf3d94089217f66da2d1e6ffa61303b509ef`.
- Candidate OOF SHA256: `455dd333ea3f76afcb4088a8475139f4b8ab8fdc3e79cecd1837d5b4bfe4631a`.
- Candidate test SHA256: `b080b9ba5bc36cdb81dcb7db981401089b74712dda13ec53f6e9cbefbe711448`.
- Champion test prediction SHA256: `906185547dcd5f0cac3ac173de644fd9997f20ac863c6745bcb1fd3e9a89e5f7`.
- Environment: Python 3.11.17, catboost 1.2.8, numpy 2.2.6, pandas 2.2.3, scipy 1.15.3, scikit-learn 1.6.1, CPU.
- Immutable evidence: GitHub Actions artifact **12588651984** (`v2938-one-shot-evidence`).

Decision: **`skipped_no_cv_gain`**, not promotable; **do not rerun, retrain, retune or submit V2938**. Lower AUC is observed, but a causal explanation is not established. The 0.961121... CV is not an official Kaggle Public LB score.

The latest persisted authenticated official Public LB remains **0.96099** for submission **57446286**, Public rank **984/2931** as of **2026-10-25 02:44:10 UTC**. That is not a current or final Private rank; a fresh authenticated refresh is still needed.

Evidence: https://github.com/mjw3177109/kaggletest/actions/runs/40164585333 ; https://github.com/mjw3177109/kaggletest/blob/feature/v29-autonomous-evolution/reports/v2_9_38_result.md .
