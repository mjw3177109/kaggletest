# V2.9.12 | Native categorical CatBoost CPU experiment (PRE-REGISTERED)

This experiment was recorded before creating its notebook and before inspecting any V2.9.12 OOF predictions.

## Standalone design (one unique run)
- Competition: `playground-series-s6e10` (train 699635, test 299844, AUC).
- One unique CPU-only Kaggle notebook: `mjw3177109/kas-v2912-native-cat-cpu-s6e10`.
- Independent CatBoostClassifier CPU; **native ordered categorical statistics** for `Gender`, `Customer Type`, `Type of Travel`, `Class`, `travel_segment`, and four preselected service-pair keys, plus fixed numeric rating summaries, service gap variables, age / distance, delay features.
- **No external teacher labels**, outer validation labels, fold tuning, leaderboard feedback, or Kaggle sample submission labels as features. Categorical processing is internal to CatBoost fit on the outer training fold.
- Validation exactly `StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)`; all predictions full-length OOF aligned on canonical train IDs. No validation early stopping.
- Fixed hyperparameters: CatBoost CPU, 750 iterations, depth 6, learning rate 0.045, L2 7, Bernoulli bootstrap subsample .85, one-hot max size 8, max_ctr_complexity 1, thread_count 4, random seed 20261018+fold. Hard runtime cutoff 3200 seconds prevents partial threefold result acceptance.
- Emit `v2912_oof.npz`: `train_id,target,oof_native_cat,test_id,test_native_cat`; `v2912_metrics.json`; staged notebook runtime progress. No model submissions.

## Fixed blend against proven V2.9.9 (not tuned on new OOF)
- `v293: 0.50, realmlp_v296: 0.20, conditional_v297: 0.15, teacher_v294: 0.05, native_cat_v2912: 0.10`.
- Rebuild V2.9.3 CV = 0.9607535043719558, V2.9.6 CV = 0.9609541991872741, and current V2.9.9 CV = 0.9610561936366048 using archived, fold- and sample-aligned OOF.
- Eligible for **candidate_ready_unsubmitted only** if V2.9.12 threefold blend CV >= 0.9611161936366048 (gain >= 0.00006), >=2 positive fold gains, and worst fold delta > -0.00005 vs V2.9.9.
- Reject mismatched competition, outdated seed, missing fold, duplicate/missing IDs, contradictory model AUC, prior active notebook, missing Kaggle account auth, deadline expired, or comparison errors. Persist only aggregate metrics, evidence hashes, and program logs to GitHub.
- Active notebook stages `kernel_queued/kernel_running` MUST only use `--recover-only`. Never repeat `kernels push` for that ID.
- Controller cannot submit to Kaggle; promotion requires a separate quota/deduplication and durable reservation review.

## Why this adds genuinely new information
Existing XGBoost, LightGBM, and RealMLP use numerical/explicit target encodings. Native CatBoost categorical handling uses a different regularized ordered-statistics and boosting mechanism. Its stand-alone CV may be lower yet it can offer useful independent OOF signal; the fixed ensemble gate is the only basis for acceptance.

## Stop/go
- Wait for V2.9.11 terminal status. Do not overlap new CPU kernel launch with its active job.
- Only push once; successful GitHub Action or a queued Notebook does not establish training completion.
- If the gate fails, record `skipped_no_cv_gain` and do not submit or retry with alternate weights.
