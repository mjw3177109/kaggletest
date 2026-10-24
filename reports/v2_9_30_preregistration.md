# V2.9.30 — fold-local cross-fitted conditional target-encoding RealMLP (pre-registration)

Date: 2026-10-24 UTC. **Research hypothesis only: no V2930 Notebook created, no GPU training, no Kaggle submission.**

## Measured evidence and failure diagnosis

V2.9.15 remains the only officially submitted champion: threefold CV **0.9612850399332336**, Public LB **0.96099**, official submission **57446286**. Last persisted authenticated Public rank **962/2870**, checked 2026-10-24 09:18:27 UTC; it is not a fresh Private or Public rank.

V2.9.29 is a **real terminal** experiment, Kaggle canonical Notebook `mjw3177109/kas-v2929-fullservice-ordinal-realmlp-t4-s6e10` certified COMPLETE. Its independent 110-feature RealMLP threefold CV is **0.9607552044022232**. Precommitted 85/15 rank fusion with the official V2915 champion achieved **0.9613230501945584**, gain **+0.00003801026132488339**, positive fold gains **[0.000042173126227340596, 0.00003673456857622437, 0.000031834289277887784]**. This is less than the preregistered **+0.00005** improvement threshold, so `skipped_no_cv_gain`, never submitted. Exact raw prediction NPZ SHA256: `845f1995556050ce10c23822557541f4885d1b96ba9ddbe1d8e5fdcedc4e59a8`.

V2929 had one **verification-only** incident: its immutable trainer output used `v2929_baseline_plus_...` as the feature-family string, while the preregistered status called the identical frozen 110-column schema `v2915_baseline_plus_...`. First read-only recovery correctly failed closed at the strict provenance check; the verifier was amended to recognize **only these two exact strings** while still verifying all 110 columns, 3 folds, T4, ID ordering, source SHA, baseline OOF and submission hash. A second **--recover-only** on the same COMPLETE Notebook passed in Actions #40093005552, with no new model run or submission. Incident retained in `recovered_validation_incident`; do not repeat this experiment.

Prior unsuccessful methods: V2916/17 aggregate service interactions; V2918 route-distance TE; V2919R2 TabM; V2920 XGBoost route; V2921 native CatBoost route; V2923 personal weighting; V2926 route profile RealMLP; V2927 travel-gated dual experts; V2928 PCA LightGBM; V2929 all-service ordinal threshold RealMLP. V2925 read-only fusion peaked at +0.000047332219081 but was not promotable. **Do not tune weights or rerun these methods.**

## Distinct V2930 hypothesis: supervised conditional category interactions

Instead of adding unsupervised ordinal thresholds, route profiles, PCA, split experts or sample weighting, test **eight previously unused, cross-fitted supervised conditional category encodings** on top of the exact 37-column V2915 RealMLP pipeline. The extra signals represent combinations of demographics, cabin, trip type and service categories; they are not the original three V2915 target-encoded combinations and do not use Flight Distance groups.

Freeze exactly eight interaction keys (all source columns are pre-existing Kaggle train/test features):

1. `Customer Type | Type of Travel | Class`
2. `Class | Gender | AgeBand`
3. `Type of Travel | AgeBand | Customer Type`
4. `Class | DepartureDelayBand | Customer Type`
5. `Type of Travel | ArrivalDelayBand | Class`
6. `Customer Type | Seat comfort | Type of Travel`
7. `Class | Online boarding | Customer Type`
8. `Type of Travel | Inflight wifi service | Class`

Use **fixed bins**, not target-optimized bins: AgeBand = [0,25), [25,35), [35,45), [45,55), [55,inf); DepartureDelayBand and ArrivalDelayBand = 0, 1–15, 16–60, >60, with negative/missing values separately mapped to `__MISSING__`. Source missing categorical values mapped to `__MISSING__`. No use of `id`, satisfaction labels in key creation, external labels, test labels or pseudo-labels.

For each outer training fold, derive all eight target encodings with **inner 5-fold stratified out-of-fold cross-fit** using only its training labels. For outer validation and test, fit one mapping on that outer training fold only, with fixed smoothing prior weight **50** and a fallback to the outer-training global satisfaction prior for unseen keys. Never use outer validation labels to fit an encoding, train a model, or select features. Do not create raw categorical duplicates of these eight keys: exactly eight numeric TE columns, yielding **37 + 8 = 45** model features. Record actual 45-column schema, categorical dtypes, fold indices and no-leakage assertions. This hypothesis is distinct from the three pre-existing V2915 inner-cross-fitted TE columns, which remain unchanged.

## Frozen experiment and promotion policy

- Unique reserved Notebook slug: `mjw3177109/kas-v2930-conditional-crossfit-te-realmlp-t4-s6e10`. Not yet created.
- Three outer `StratifiedKFold(n_splits=3, shuffle=True, random_state=20261018)`, identical train/test ID order to champion; 699635 train / 299844 test.
- RealMLP-TD on verified NVIDIA T4, `pytabkit==1.7.3`, `n_ens=8`, `n_epochs=5`, `stop_epoch=5`, `n_refit=0`, `val_fraction=0`, `hidden_sizes=[512,256,128]`, `lr=0.038`, `wd=0.015`, `p_drop=0.05`. MODEL_SEED = 20261030 + outer fold.
- Sole precommitted candidate: `0.85*rank(V2915 champion OOF) + 0.15*rank(V2930 new OOF)`, same fixed recipe on test. **No weight, seed, bin, feature, or model selection after viewing results.**
- Promotion requires candidate full threefold CV >= **0.96125635576833**, improvement vs official champion CV >= **0.00005**, >= 2 positive fold gains, worst fold gain strictly > **-0.00005**.
- Verify Kaggle COMPLETE, full 3-fold OOF, fold AUC and membership, train/test IDs, official champion prediction SHA `906185547dcd5f0cac3ac173de644fd9997f20ac863c6745bcb1fd3e9a89e5f7`, all source SHA and new candidate OOF/test SHA, verified T4, environment versions, label isolation. Evaluator NEVER submits.
- Promotion does not imply Kaggle Public LB or Private rank. Any submission requires a separate, verified one-shot workflow with durable reservation, prediction file SHA, quota, official submission-history dedup and submission-ID recovery.

## Hard launch gate

Before any Kaggle creation: review historical TE scripts and previous failure reports; implement independent trainer, verifier, single-Notebook durable reservation/recover-only pipeline, tests and guarded Actions; fix deterministic faults first. Verify whole-repository GitHub Actions and other Kaggle GPU jobs are not active. **At pre-registration time an unrelated V2.8 GPU challenger Actions #40093154210 was active, so launch is explicitly prohibited.** Query exact authenticated owner Notebook status/list; if slug exists, recover it only, never push again. Persist owner reservation remotely before the one authorized push. No competition-rule acceptance or token exposure.
