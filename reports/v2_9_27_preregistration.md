# V2.9.27 — Travel-gated dual RealMLP experts (preregistered, not launched)

Date: 2026-10-24 UTC. **No Kaggle Notebook created or training started.** This preregistration freezes a new hypothesis, not an achieved score.

## Evidence and why V2.9.26 was rejected

V2.9.15 remains the only scored champion: full OOF CV **0.9612850399332336**, official Public LB **0.96099**, official submission **57446286**. V2.9.26 Kaggle Notebook `mjw3177109/kas-v2926-route-profile-realmlp-t4-s6e10` was independently recovered as **COMPLETE** at 2026-10-24T06:17:04Z; threefold OOF standalone AUC **0.9602205569678511**, frozen 80/20 fusion AUC **0.9613072330727896**, gain **+0.0000221931395561**. All three fold gains were positive (+0.000040874, +0.000020369, +0.000003031) but the improvement is below the **+0.00005** minimum; the candidate is permanently **skipped_no_cv_gain** and was never submitted. Frozen output SHA256 `1026cf4aa6990f7eb3fc9e9e5c21209cbe4ae8b6fd42e8398fb9f9a24853e2e2`.

Adding 12 unsupervised exact-distance route aggregates increased V2.9.15's standalone model feature count from 37 to 49, but its new standalone AUC fell from 0.96060130231433 to 0.9602205569678511. This does not prove route features always hurt, but **repeating route aggregation or optimizing its fusion weight is not justified**. V2.9.16's service/delay-interaction RealMLP improved the frozen champion by +0.0000393250953112 (still subthreshold); V2.9.24's Personal-Travel LightGBM specialist added only +0.0000097242712972. V2.9.27 tests a **different model structure** rather than retuning these.

## New falsifiable hypothesis

A **travel-type-gated dual RealMLP-TD mixture-of-experts** may capture nonlinear satisfaction mechanisms that differ between Personal Travel and Business travel, without contaminating validation. For each outer fold, fit **two distinct RealMLP models** only on the corresponding outer-training travel subset. The gating variable is the observed `Type of Travel` feature, available identically in train/test, not a predicted label. Apply each expert only to its matching outer-validation and test rows. Preserve V2.9.16's preregistered 9 service/delay interaction features but **do not** use V2.9.26's 12 route profiles or any new supervised/external feature. TargetEncoder remains inner 5-fold cross-fit on each expert's outer-training subset only. No validation labels or Kaggle test labels in training.

This differs from V2.9.24 (one Personal Travel LightGBM correction), V2.9.26 (single global RealMLP with route aggregates), and V2.9.16 (single global interaction RealMLP). Failure to improve is possible; do not imply otherwise.

## Frozen parameters and safety

- Unique intended owner Notebook: `mjw3177109/kas-v2927-travel-gated-dual-realmlp-t4-s6e10`; **not created** at preregistration.
- Contest: `playground-series-s6e10`; train 699635, test 299844; label `satisfaction`, official ID order verified.
- Outer validation: `StratifiedKFold(n_splits=3, shuffle=True, random_state=20261018)`. Within each outer fold, split training by travel type, and train **one Personal Travel and one Business travel expert**. Check no unexpected third category or null; fail closed if found. Require full 3fold OOF across all 699635 training rows.
- Model: `pytabkit==1.7.3` RealMLP-TD on verified T4, n_ens=8, n_epochs=5, hidden_sizes=[512,256,128], batch_size=256, lr=0.038, wd=0.015, same frozen V2.9.15 optimizer settings; seed `20261027 + 2*fold + expert_index`. No post-hoc hyperparameter search.
- Features: baseline 37 engineered columns plus exactly 9 deterministic V2.9.16 service/delay interactions; **zero** V2.9.26 route profiles. Inner cross-fit target encodings fitted per expert only. No external labels or data.
- Exactly one promotion candidate: **0.85 × global rank(V2.9.15 champion OOF) + 0.15 × global rank(V2.9.27 gated-expert OOF)**; same fixed combination for test. No tuning weights after seeing CV.
- Full OOF gate: CV >= 0.96125635576833 **and** improvement vs champion >= 0.00005 **and** >=2 positive fold gains **and** worst fold gain > -0.00005. Strictly require exact threefold membership, every row covered once, individual subgroup and full AUC, SHA256 and original V2.9.15 source replay/hash.
- Before training: read fresh GitHub Actions across the **whole repository** and Kaggle owner Notebook listing; no other active training/recovery/submission. Persist unique slug before push, recover existing only via `--recover-only`; never repush on ambiguous status. Never submit from evaluator.
- If promotable, use a separately validated durable one-shot submit workflow with prediction file SHA, quota, official submission history dedup, reservation and submission-ID recovery. No Kaggle submit on a CV-only result.
- Scientific caveat: subgroup training shrinks each expert's sample size; stop and document if it hurts. Public leaderboard and Private leaderboard remain separate from CV.

## Implementation gate

Only preregistration and independent state file are authorized by this change. The trainer, verifier, workflow and tests must be reviewed and passed before any Notebook creation. Do not reuse V2.9.26's existing Notebook slug. Preserve the historical terminal record.
