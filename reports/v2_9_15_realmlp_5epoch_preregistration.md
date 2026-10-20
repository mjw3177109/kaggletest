# V2.9.15 — Independent RealMLP long schedule + precommitted complement

Registered before creating the new Kaggle notebook, predicting, or examining V2.9.15 OOF.

## Hypothesis and scope

- RealMLP V2.9.13 8-ensemble 3 epochs achieved standalone 0.9609605572931901; the deployed V2.9.13 four-model+neural ensemble achieved full 3fold 0.96117635576833, Kaggle official Public LB 0.96083 (submission 57423879).
- V2.9.14 independent numeric-dual feature RealMLP achieved standalone 0.9609600385938727 and a fixed 90/10 challenger 0.9612261589523977, gain 0.00004980318406766493 (three positive folds, but under the 0.00006 preset promotion threshold). It was NOT submitted. Those results are known and not new validation.
- New independent hypothesis: RealMLP Ens8 with longer **5-epoch** schedule, lower fixed LR (0.038 vs 0.053) and independent training-model seeds (20261021 + fold) provides a complementary learner without changing the exact outer 3fold splits or labels.
- Unique T4 Notebook: `mjw3177109/kas-v2915-realmlp-5epoch-t4-s6e10`. No relaunch of earlier notebooks.
- Data: competition `playground-series-s6e10` only, 699635 train, 299844 test, full 3fold `StratifiedKFold(n_splits=3, shuffle=True, random_state=20261018)`. Preserve V2.9.13's 37-column fold-safe feature representation and inner 5fold target encoding, without outer-fold validation labels or early stopping.
- One pre-specified model configuration: `RealMLP-TD`, pytabkit==1.7.3, `n_ens=8`, `n_epochs=5`, `stop_epoch=5`, `batch_size=256`, `lr=0.038`, `wd=0.015`, `lr_sched=flat_anneal`, `wd_sched=cos_log_15`, `p_drop_sched=invsqrtp1e-3`, `sq_mom=0.988`, all other V2913 model settings unchanged. Model random seed = 20261021+fold; splitting seed remains 20261018.
- T4 evidence required, no GPU fallback, all three folds complete, capped runtime 3200 seconds. No grid search and no weight sweep.

## Frozen evaluation, registered before V2915 OOF
- Reconstruct and hash-check canonical older V2.9.3, V2.9.6, V2.9.7, V2.9.4, V2.9.13 and V2.9.14 artifacts, including explicit IDs where available and original competition target order for legacy archives.
- V2913 champion OOF must reproduce **0.96117635576833** and candidate-test SHA `f8f1d833ac5ee3cf98f538b415941d8913c1ba84a4508f541d27d81065e25748`.
- Fixed weights against original official V2913 OOF: **80% rank(V2913 champion) + 10% rank(V2914 standalone dual RealMLP) + 10% rank(V2915 standalone 5-epoch RealMLP)**.
- Because inclusion of V2914 was informed by its observed OOF, this is exploratory reuse of the same threefold CV, not independent confirmation. Require the stricter advancement threshold: overall paired CV gain at least **0.00008** vs V2913, i.e. candidate CV >= **0.96125635576833**, at least two positive per-fold gains and worst fold delta > -0.00005. This avoids treating tiny selected OOF gains as reliable.
- No Kaggle leaderboard submit during training; a qualifying V2915 status stays `candidate_ready_unsubmitted`. A second separate explicit SHA/IDs/quota/one-shot reservation must be built to submit. A non-qualifying candidate is `skipped_no_cv_gain`.
- If a Notebook already exists or a launch attempt may have succeeded ambiguously, only recover it. Never create a second GPU kernel for V2915 or bypass Kaggle submission limits.

## Model originality
The implementation extends the user's prior V2913 pipeline and tests a training-duration/seed hypothesis rather than searching new weights against the known OOF. Public RealMLP reproducibility context: https://github.com/alvis1111/s6e10-airline-solution/tree/main/independent_repro
