# V2.9.38 — CPU CatBoost Ordered boosting, original 37 features (immutable preregistration)

Preregistered on **2026-10-25 UTC**, after verified terminal V2937 `skipped_no_cv_gain`, and **before** implementing a V2938 trainer, downloading competition data, fitting, evaluating OOF, or submitting. This is an experimental hypothesis, not an expected score or Public rank.

## Novelty and exclusions

V2937's LightGBM GOSS original37, V2936's aborted XGBoost DART, V2935's read-only three-source agreement gate, V2921's **33-feature native-categorical route-profile CatBoost**, and other terminal experiments are never rerun. V2938 instead tests **CatBoost Ordered boosting on exactly the champion's original 37 numeric-encoded feature columns (including three inner-fivefold TE features)**, without route profiles, native CatBoost categorical/CTR features, new feature engineering, sample weighting, PCA, pseudo-labels, or external labels. Ordered boosting changes the tree fitting mechanism; whether its rank residual helps V2915 is unknown.

## Frozen input, validation and environment

- Only official Kaggle `playground-series-s6e10` train/test, expected **699635/299844** rows. Champion original **37** feature names/order exactly, with its three outer-train-only inner-5-fold leakage-safe target encodings. Strict `StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)`; model seeds `20261106+fold`. Outer validation rows cannot influence fold-local category vocabularies, missing-value fits, combo counts, or target encodings. No feature search.
- Champion source NPZ SHA256 `46ad63f08537cd8d61c4c8e8698efc20bb3c8fbad4eb0d94ecd7c993b2d894b9`, champion test prediction SHA256 `906185547dcd5f0cac3ac173de644fd9997f20ac863c6745bcb1fd3e9a89e5f7`.
- **GitHub Actions CPU-only**, no Kaggle Notebook push/create. Pin Python 3.11, `catboost==1.2.8`, `numpy==2.2.6`, `pandas==2.2.3`, `scipy==1.15.3`, `scikit-learn==1.6.1`. Install and import-verify all versions before any reservation/data access; unknown or incompatible environment fails closed, no silent package substitution. Preflight a synthetic tiny fit of `boosting_type="Ordered"` before any exclusive reservation; this does not access competition data.
- Frozen `CatBoostClassifier`: `loss_function="Logloss"`, `eval_metric="AUC"`, `boosting_type="Ordered"`, `task_type="CPU"`, `iterations=600`, `depth=6`, `learning_rate=0.035`, `l2_leaf_reg=5.0`, `random_strength=1.0`, `bootstrap_type="Bayesian"`, `bagging_temperature=0.5`, `rsm=0.85`, `border_count=128`, `thread_count=4`, `random_seed=20261106+fold`, `allow_writing_files=False`, `verbose=False`. Treat the same 37 encoded columns as numeric (no native CatBoost CTR). Exactly three outer folds, no early stopping, no seed/weight/parameter tuning.
- Exactly one candidate: `0.85*rank(V2915)+0.15*rank(V2938)`, `rank(x)=scipy.stats.rankdata(x,method="average")/len(x)`, OOF/test ranked separately. No post-hoc threshold, weights or mask search.

## One-shot guard, audit and submission policy

Before training, re-read actual branch HEAD, all five active Actions statuses, V2915 submission/SHA, terminal V2935–V2937, official submission history and authenticated rank evidence, all owner Kaggle Notebook states, and deadline. Kaggle calls at least three seconds apart; 429/unknown fail closed. Persist one exclusive V2938 GitHub run ID + immutable parameter SHA before downloading data or fitting; if already reserved, **read-only recovery only**, never a second fit. Do not create/push a Kaggle Notebook. No raw competition data or tokens in commits/artifacts.

Strictly verify train/test IDs, fold membership, fold-local encoders/TE, feature order, complete 3-fold OOF, full/fold AUC, source/candidate/champion SHA and environment versions. The evaluator must never submit. Only if full blended CV `>=0.96125635576833`, gain vs V2915 `>=+0.00005`, at least 2 folds positive and worst fold `>-0.00005`, may a **separate** durable, quota-checked, deduplicated one-shot official submission be considered; recover official submission ID and Public LB independently. CV is not Kaggle Public LB.

Competition deadline **2026-10-31 23:59 UTC**. Do not accept rules automatically, leak credentials, bypass controls, repeat terminal experiments or promise a top-three result. If V2938 fails, diagnose and preregister a genuinely distinct V2939 before any implementation.
