# V2.9.36 — terminal Kaggle runtime failure (not a CV failure)

Authenticated at **2026-10-25 02:33:43 UTC**: the sole reserved Notebook `mjw3177109/kas-v2936-original37-xgboost-dart-t4-s6e10` returned `KernelWorkerStatus.ERROR` (read-only Actions **#40154887266**). After isolating preregistration tests from the durable state, **19/19 CPU tests passed** (Actions **#40155047807**). Strictly gated `--recover-only` Actions **#40155115026** authenticated the owner-wide Notebook inventory and official V2915 submission, downloaded the existing Notebook's error output, and durably persisted `state/v2_9_36_status.json` as `kernel_failed` at **2026-10-25 02:41:58 UTC**. No second Kaggle Notebook push, training or competition submission occurred.

## Reproduced root cause

The original immutable V2936 Notebook executes `prove_gpu()` **before loading any competition data or fitting any fold**. Its first guard requires `xgboost==3.0.5`. The actual Kaggle runtime reported **xgboost 3.4.1**:

```text
File "/kaggle/src/script.py", line 60, in prove_gpu
RuntimeError: Pinned xgboost==3.0.5 required; got 3.4.1
```

This is an **environment/version mismatch**, not evidence that the frozen XGBoost DART hypothesis underperformed. There is **no V2936 threefold OOF, fold AUC, fused CV, candidate SHA, Public LB or new official submission**. The T4 hardware check did not complete, because the version guard ran first.

## Safety decision and next experiment

V2936 is permanently terminal `kernel_failed`: do **not** create, push, install a new environment into, or retrain the same slug; do not loosen its frozen version assertion or pretend a score exists. Its originally frozen 900-round DART setup remains historical only. The initial recovery attempt #40154942060 failed safely at CPU tests that incorrectly assumed a pristine state; tests were corrected to use synthetic lifecycle states, verified 19/19, then recovery succeeded. This test repair changed no model parameters.

V2937 must be independently preregistered as a **different learning mechanism and feature/regularization route**. It must include a preflight that checks the Kaggle runtime package version *before* any unique Notebook reservation/push; unknown, mismatched or rate-limited API responses fail closed. All original submission and ranking evidence remains unchanged: V2915 official Public LB 0.96099, submission ID 57446286, champion prediction SHA `906185547dcd5f0cac3ac173de644fd9997f20ac863c6745bcb1fd3e9a89e5f7`. Latest persisted authenticated rank 982/2925 (2026-10-25 00:18:51 UTC) is not a new or final Private rank.

Evidence: https://github.com/mjw3177109/kaggletest/actions/runs/40154887266 ; https://github.com/mjw3177109/kaggletest/actions/runs/40155047807 ; https://github.com/mjw3177109/kaggletest/actions/runs/40155115026
