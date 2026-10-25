# V2.9.36 prelaunch safety incident and exact missing-slug recovery

2026-10-25 UTC. First guarded workflow #40149998794 authenticated all owner Notebooks and official champion, but the separate pre-push own-slug status query returned an unclassified CLI error. The pipeline failed closed with `kernel_access_error`; **no Notebook was created, no GPU training began, and no submission occurred**. The GitHub Actions workflow conclusion was success because it persisted the safe terminal state; this is not proof of a successful Notebook launch.

Read-only authenticated diagnostic workflow #40150393528 established:
- `kaggle kernels status mjw3177109/kas-v2936-original37-xgboost-dart-t4-s6e10` exits 1 with exact `Cannot access kernel '...' (Permission 'kernels.get' was denied)` and the CLI's wrong-slug explanation.
- Independent `kaggle kernels list --mine --search kas-v2936-original37-xgboost-dart-t4-s6e10 --page-size 50 -v` returns exactly `Not found`.
- Known V2915 champion Notebook status returns authenticated `KernelWorkerStatus.COMPLETE`.
- No token was logged.

Root cause: Kaggle's CLI expresses a nonexistent owned slug as a permission-denied diagnostic, not a standard 404. The original fail-closed classifier correctly blocked the ambiguous response but could not complete first-time creation.

Minimal repair: accept this **specific** CLI diagnostic only when separately accompanied by exact owned-list absence and authenticated champion COMPLETE witness. All other permission errors, 429, unknown statuses, owner-list ambiguity, concurrent GitHub Actions and active Kaggle training remain blocked. Keep at least 3 seconds between Kaggle calls. Restore the never-created preregistration state only after this evidence; no model parameters or submission thresholds change. Re-run CPU tests before a new guarded first launch. Unique slug still unreserved.
