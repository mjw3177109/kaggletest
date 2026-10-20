# V2.9.11 | New independent CPU passenger-interaction LightGBM

Pre-registered BEFORE new Kaggle kernel creation or CV measurement.

- Contest: playground-series-s6e10, full 3fold StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018), inner 5fold TargetEncoder and 5fold cross-fitted frequency, competition training data only. No external teacher labels in training and no outer validation early stopping.
- Unique notebook: mjw3177109/kas-v2911-segment-contrast-cpu-s6e10. CPU only, GPU/TPU disabled. Push the notebook exactly once and subsequently recover by saved actual canonical kernel ID. Do not create duplicates while queued/running.
- New features: prespecified service-rating *difference* × traveler group and age bucket, difference × cabin and distance bucket, delay/distance context, age/distance context. Use fixed non-target bin edges. Preserve V297's existing 24 conditional keys and add contrast-context keys. 5fold cross-fit protects outer validation and target leakage.
- One stand-alone LightGBM (1100 trees, 39 leaves, lr .035) with fixed params; no training hyperparameter sweep.
- Source provenances and archived sample IDs: strict V2.5/V2.8 canonical source order; V2.9.4 teacher target canonical order; V2.9.6/V2.9.7/new V2.9.11 explicit train_id/test_id. Require 699635 train and 299844 test rows and exact binary target order.
- Current champion candidate V2.9.9: 0.9610561936366048. Fixed new blend weights = old V2.9.3 50%, RealMLP 20%, V2.9.7 conditional TE 15%, V2.9.4 Teacher 5%, new V2.9.11 10%. Weights selected before seeing this new model's CV.
- Promotion gate: candidate 3fold full OOF AUC >=0.9611161936366048, minimum improvement 0.00006 vs V2.9.9, >=2 folds improved, min fold delta > -0.00005. On pass mark candidate_ready_unsubmitted only.
- Persist authentic Kaggle stage, fold AUC, source checksums, model and blend AUC, fold gains, runtime and errors to state and report; update independent ledger on completion. Do not claim GitHub Actions success implies Kaggle training completion.
- No automatic Kaggle leaderboard submissions in this notebook or workflow. Never bypass Kaggle account daily submission limit, duplicate checks or durable one-shot reservation.
- Abort new launches if contest ends, another version-specific job/kernel is active, status is terminal, or owner/competition checks fail.
