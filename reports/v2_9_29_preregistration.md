# V2.9.29 — full-service ordinal-threshold RealMLP-TD (pre-registration)

Date: 2026-10-24 UTC. **Pre-registered only: no V2929 Notebook, training, or submission has been started.**

## Prior evidence and failed V2.9.28 diagnosis

V2.9.15 remains the **only official submitted champion**: authenticated Kaggle Public LB 0.96099, submission ID 57446286, full threefold OOF CV 0.9612850399332336. Latest authenticated Public rank 962/2870 at 2026-10-24 09:18:27 UTC. Do not treat OOF as official Kaggle LB or Private rank.

V2.9.28 single canonical CPU Notebook `mjw3177109/kas-v2928-service-factor-pca-lgbm-cpu-s6e10` was recovered **once** via `--recover-only` in Actions run 40087152157, Kaggle kernel `COMPLETE`, independent threefold LightGBM CV **0.9597132168847482**, frozen 85/15 rank-fusion CV **0.9612196682704917**. This is **-0.00006537166274189765** relative to champion; fold gains **[-0.00007416573215524025, -0.00007785682917682557, -0.000045846515244196695]** (zero positive folds). The fixed candidate was **rejected without submission** and `state/v2_9_28_status.json` is permanently `skipped_no_cv_gain`. Independent prediction NPZ SHA256: `8e44f204de4f7e67648e081d9c4f48782984f2927bbabe6475a2c9bc1ff1357c`. No duplicate Notebook push or second model training occurred.

The **measured** result is a weaker standalone global LightGBM and negative fixed-blend gain in all three folds. **Hypothesis, not proven cause:** a PCA projection and aggregate service contrasts may compress discrete ordinal thresholds or be poorly matched to the GBDT configuration. There is no evidence of data leakage, invalid CV, or infrastructure failure in the completed report. No LightGBM PCA retune or weight search is authorized.

Historical lessons: V2916 nine interaction features plus RealMLP yielded +0.000039325095311 (below gate); V2917 similar interaction family yielded +0.000015217069944; V2926 fold-local route profile yielded +0.000022193139556; V2927 travel-gated experts +0.000023442538100. V2925 read-only fusion +0.000047332219081 missed the fixed +0.00005 gate. Re-running these is not an experiment.

## Distinct, falsifiable next hypothesis

The best official model is RealMLP-TD, which already encodes **five** of thirteen service ratings as both numeric and categorical. V2929 instead tests **all-service ordinal-threshold representation** without route profiles, PCA, LightGBM, dual experts, sample weighting, or the V2916/V2917 nine aggregate interactions.

**Freeze the exact 73 new label-free columns:**
- Eight additional `dual_ordinal_<canonical_service>` categorical duplicates, for each of the remaining eight service columns not already in the V2915 `DUAL` list. Vocabulary fitted on **outer training rows only**; unknown validation/test categories handled consistently.
- Sixty-five cumulative ordinal indicator columns: for each of the 13 original service scores, five thresholds `>=1`, `>=2`, `>=3`, `>=4`, `>=5`. Fit each service's numeric missing-value median **only on the outer training fold**, then transform training/validation/test using that median; no validation or test labels.
- No service PCA or contrasts, route-distance group counts, pseudo-labels, external labels, test-fitted scaling, target-derived threshold search, or interaction feature selection.

The unchanged V2915 engineer/TE pipeline supplies **37** original columns (21 original + 7 original dual categorical + 3 combo categorical + 3 count + 3 inner-cross-fitted target encoding). V2929 adds 73, yielding **110** model input columns. Assert actual feature schema and exact categorical dtype across outer train/validation/test. If V2915 implementation changes or a count differs, **fail closed**, do not silently alter the pre-registration.

## Frozen model, validation, and candidate

- Competition: `playground-series-s6e10`; canonical 699635 train and 299844 test rows, original train labels only. Unique reserved Notebook: `mjw3177109/kas-v2929-fullservice-ordinal-realmlp-t4-s6e10`. This slug is **not created** by pre-registration.
- Exact outer `StratifiedKFold(n_splits=3, shuffle=True, random_state=20261018)`; identical folds and ID order to V2915. The V2915 3-combo TargetEncoder remains inner 5-fold cross-fit on outer train only. No use of outer validation labels for fitting or early stopping.
- RealMLP-TD `pytabkit==1.7.3`, NVIDIA T4 required, `n_ens=8`, `n_epochs=5`, `stop_epoch=5`, `n_refit=0`, `val_fraction=0`; exact unchanged V2915 hyperparameters, including `hidden_sizes=[512,256,128]`, `lr=0.038`, `wd=0.015`, `p_drop=0.05`. New model seed **20261029 + fold**.
- The **only** fixed candidate: `0.85*rank(V2915 champion OOF) + 0.15*rank(V2929 OOF)`, identically constructed for test predictions. No weight, seed, feature, or fold search after seeing outcomes.
- Promotion only if full threefold CV **>=0.96125635576833**, gain vs V2915 **>=0.00005**, at least **2 positive fold gains**, worst fold **> -0.00005**. Verify fold membership, full OOF and each fold AUC, train/test IDs, original source SHA, new NPZ SHA, candidate OOF/test SHA, real Kaggle T4, environment versions, no label leakage, and official champion test hash `906185547dcd5f0cac3ac173de644fd9997f20ac863c6745bcb1fd3e9a89e5f7`.
- Evaluator **must not submit**. A promotable result requires separate verified one-shot submission workflow with durable reservation, SHA, Kaggle quota, official submission history de-duplication, and submission-ID recovery. Never infer Public LB or top-three ranking from CV.

## Launch gate (NOT satisfied by this preregistration)

First implement isolated trainer, source/hash validator, tests (all 13 service thresholds, unknown category, fold-local imputation, no outer label usage), and a guarded GitHub Actions workflow. Confirm whole-repository no active Actions and no other active Kaggle training; query authenticated Kaggle owner Notebook status/list; persist exact slug on GitHub **before** any push. If a saved slug exists, **only `--recover-only`**; never re-push or create a duplicate. Do not train until tests and safety review pass. Preserve all failed-experiment evidence, do not expose tokens or accept Kaggle rules.
