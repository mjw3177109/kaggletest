# V2.9.28 — fold-local service-factor PCA + global LightGBM (pre-registration)

Date: 2026-10-24 UTC. **Research hypothesis only. No Kaggle Notebook created, no training started, no submission.** This is a genuinely new representation/model, not a retry of V2.9.26 or V2.9.27.

## Evidence and root-cause review

The authenticated champion remains V2.9.15, threefold OOF CV **0.9612850399332336**, official Public LB **0.96099**, official submission **57446286**; last authenticated Public rank **953/2853** at 2026-10-24 04:49:54 UTC. Do not infer a new official score from CV.

V2.9.27 completed its single canonical Notebook `mjw3177109/kas-v2927-travel-gated-dual-realmlp-t4-s6e10` on a real Kaggle T4, 3 folds, 6 independent subgroup models. Standalone full OOF **0.9593005287908923**; frozen 85/15 rank fusion **0.9613084824713334**, gain **+0.0000234425380998**, all 3 fold gains positive but **below the +0.00005 gate**. Permanent `skipped_no_cv_gain`, never submitted. Source prediction archive SHA256 `180cd12af6fcc097725718148946bfcb0a0b6184cf6403cc360b44ef26ffb840`. Personal-travel subgroup AUC 0.8195406930134544; business AUC 0.9568923457818472. Shrinking each expert's sample size and potential between-expert score-scale mismatch are plausible explanations, **not proven causes**.

A recovery validation error was caused by NumPy object-dtype arrays storing travel strings; it was resolved once in Actions #40077588836 **without unpickling, retraining, or repushing the Notebook**. Only numeric OOF/IDs were loaded; opaque gate array headers and independently reconstructed canonical membership + per-fold subgroup counts/AUC were validated. Original object payload bytes were not deserialized, which remains an explicit provenance limitation.

Previous unsuccessful routes: route-distance profile LightGBM (V2918), route XGBoost (V2920), native categorical CatBoost (V2921), Personal Travel sample-weighted LightGBM (V2923), route-augmented RealMLP (V2926), gated dual RealMLP (V2927). No more weight search on these fixed candidates.

## Distinct falsifiable hypothesis

A **single global LightGBM** using a compact, label-free service-quality latent representation may improve the **ranking** without the loss of statistical strength caused by splitting travel groups. This is neither a route-frequency/profile model nor a travel-gated model. It tests PCA/service contrasts, not a repeat of the failed V2926 route features.

External exploratory evidence: a public S6E10 feature-analysis writeup discusses service-score PCA and weighted service contrasts (https://hittmg.hatenablog.jp/entry/2026/10/11/210000). The linked experiments do not establish any gain in our own OOF. A public S6E10 multi-GBDT Notebook reports use of diverse engineered features (https://www.kaggle.com/code/anhadmahajan06/s6e10-are-you-satisfied), but its published score is not our own and is not a valid promotion gate.

## Frozen experiment design

- Unique intended Notebook slug: `mjw3177109/kas-v2928-service-factor-pca-lgbm-cpu-s6e10`; not yet created.
- Competition: `playground-series-s6e10`, train 699635, test 299844, ROC-AUC. Train on **competition training labels only**. No external labels, no pseudo-labels, no Kaggle leaderboard-based tuning, no leaked validation targets.
- Outer folds **exactly** `StratifiedKFold(n_splits=3, shuffle=True, random_state=20261018)`, preserving exact V2915 fold membership and row IDs. No post-hoc split change.
- Feature inputs: baseline original categorical/numeric columns plus the pre-existing V2924 **23 cross-fitted target encodings** fitted only on each outer training fold, but **zero exact-distance route-profile features**. PCA on the fixed 13 service ratings from V2924, fitted with StandardScaler+PCA(n_components=3, random_state=20261028) **only on outer training rows**; transform validation/test with frozen fold scaler/PCA. Eight deterministic service contrasts computed from original rating columns (mean, std, min, max, count of zeros, count >=4, count <=2, mean(digital trio) minus mean(comfort trio)), using training-only imputation medians. Exactly 11 new label-free numeric columns (3 PCA + 8 contrasts).
- One **global** LightGBM CPU model per fold, no separate Personal/Business experts. `n_estimators=1200, learning_rate=0.035, max_depth=8, num_leaves=63, min_child_samples=60, subsample=0.90, colsample_bytree=0.85, reg_lambda=3.0, reg_alpha=0.0, random_state=20261028+fold, n_jobs=4`; no early stopping on the outer validation labels. Pin dependency versions before run and validate actual environment.
- One fixed candidate only: `0.85 * rank(V2915 champion OOF) + 0.15 * rank(V2928 OOF)`, same for test. No weight search, no reuse of a prior Notebook slug.
- Threefold complete OOF; validate every fold, full AUC, feature family, source SHA256, train/test IDs and labels, V2915 frozen prediction hash, no duplicates or missing predictions. Promotion **only** if full CV >= 0.96125635576833 **and** gain over V2915 >= 0.00005 **and** at least 2 positive fold gains **and** worst fold gain > -0.00005.
- Evaluation is no-submit. If genuinely promotable, use separate one-shot durable Kaggle submission reservation, SHA/official history dedup, quota check and official submission-ID recovery. Official Public LB and final Private ranking are separate evidence.

## Implementation/launch gate

This commit authorizes **preregistration only**, not Notebook creation. Next: inspect old LGBM trainer, implement new independent trainer/verifier, add synthetic PCA fold-isolation and anti-submit tests, then review the exact workflow. Before any Notebook creation query **whole-repository active Actions** and authenticated Kaggle owner Notebook listing, persist the unique slug before push, and recover saved slug only with `--recover-only`. Do not start a second training while another Kaggle Notebook is queued/running. No token disclosure, rules acceptance or unsafe pickle.
