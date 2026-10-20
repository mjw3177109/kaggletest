# V2.9.13 | Demidov-style RealMLP 8-ensemble GPU (pre-registered)

This hypothesis was registered before training or inspecting any V2913 predictions. Unlike V2.9.12 CatBoost and V2.9.11 conditional LightGBM, this tests training-schedule/ensemble effects on a neural tabular learner.

## Fixed protocol
- Contest playground-series-s6e10; **same** 3fold StratifiedKFold shuffle=True seed 20261018, identical 699635/299844 rows and sample-ID order.
- Unique Kaggle notebook `mjw3177109/kas-v2913-realmlp-ens8-t4-s6e10`, T4 GPU mandatory, no automatic Kaggle leaderboard submission.
- Start from V2.9.6's fold-safe features and inner 5fold cross-fitted TE. Keep all original safety checks. Do NOT expose outer fold validation targets to fitting, model selection or early stopping.
- Fixed hyperparameters from public reproducibility research, with no OOF search: RealMLP-TD pytabkit 1.7.3, 3 epochs, stop_epoch=3, n_ens=8 (vs V296 n_ens=4), batch_size 256 (vs 1024), lr .053, wd .015, hidden [512,256,128], dropout .05, `lr_sched=flat_anneal`, `wd_sched=cos_log_15`, `p_drop_sched=invsqrtp1e-3`, `sq_mom=.988`, first-layer lr factor .25, `tfms=one_hot,median_center,robust_scale,smooth_clip,embedding,l2_normalize`, PLR and LS defaults explicitly fixed in the script.
- Memory / GPU budget gate: no CPU fallback, no more than 3200 sec elapsed, full 3fold completeness required. No new Kaggle kernel if an identical job exists or state has kernel_queued/kernel_running; recover-only from existing ID.
- RealMLP new OOF/test predictions saved once with exact train_id/test_id and target; per fold AUC and runtime, real T4 evidence and hash.

## Fixed fusion and stopping rule
- Current best **official** Public LB = 0.96070, submission #57421208, V299 full 3fold CV = 0.9610561936366048.
- Fixed challenger: 85% rank(V299 four-model OOF) + 15% rank(V2913 neural OOF), no weight search.
- Full 3fold contender **>= 0.9611161936366048**, gain >=0.00006 vs V299, >=2 folds improve, worst fold gain >-0.00005, ID/label/fold alignment exact, candidate-test prediction hash recorded.
- If gate passes, stage `candidate_ready_unsubmitted` only; next Kaggle submission needs new written reservation, a fresh authenticated account quota check and duplicate protection. If not, `skipped_no_cv_gain`.
- Avoid using another team's 10fold OOF scores as our 3fold baseline. Avoid training on Kaggle leaderboard feedback. The official Public LB is distinct from CV and final Private leaderboard.

## Attribution / research context
- Inspired by publicly described RealMLP schedules and the public S6E10 independent-reproduction writeup: https://github.com/alvis1111/s6e10-airline-solution/tree/main/independent_repro
- The user's pre-existing V296 pipeline supplies the base implementation; this experiment makes a distinct predefined training-schedule hypothesis. Do not copy unknown-license code verbatim.
