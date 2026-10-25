# V2.9.36 — original-37 XGBoost DART dropout-boosted trees (immutable preregistration)

Preregistered **2026-10-25 UTC**, after V2935 completed and was terminally rejected, before any V2936 code execution, Notebook creation, fit, OOF evaluation or prediction. Official Kaggle playground-series-s6e10 Public top three is an aspirational target, not a promised outcome.

## Distinct hypothesis and exclusions

V2935 was a **label-free three-model agreement gate**, with threefold CV 0.9613302676316021 and +0.000045227698368566216 vs champion, below the +0.00005 threshold. V2932 attention, V2933 pairwise RankNet and V2934 DeepFM also failed. V2920 was **XGBoost histogram depth6 with route-profile TE23**, CV 0.9612378193529525 blended, negative gain, and is permanently terminal. This V2936 is not a V2920 retry: it tests **dropout-regularized DART boosting with the champion's original 37 features only**, shallower trees, no route profiles, no extra TE or threshold features. Dropout tree ensembles have a distinct regularization mechanism from V2920's standard histogram booster and from recent neural/fusion attempts. This is a hypothesis, not a performance claim.

## Frozen source, feature family and validation

- Source: official competition train/test, original V2915 37 features, including the existing three **outer-fold-train-only inner5fold cross-fit TE** columns. Reconstruct source using the existing V2915 pipeline; verify train/test IDs, target, original feature order, source NPZ SHA256 `46ad63f08537cd8d61c4c8e8698efc20bb3c8fbad4eb0d94ecd7c993b2d894b9` and champion test prediction SHA256 `906185547dcd5f0cac3ac173de644fd9997f20ac863c6745bcb1fd3e9a89e5f7`.
- Exactly the original 37 columns in the original order. Original categorical vocabulary learned from each outer training fold only; missing numeric medians from the same fold; unseen categories mapped to a fixed sentinel. No validation-fold target use for fitting, imputing, encoding or early stopping. No extra feature creation, target labels from external sources, pseudo-labels, PCA, route profiles, service thresholds, new TE, MI pruning, ensemble experts or weight search.
- Three complete outer folds: `StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)`. Independent T4 model seed `20261104+fold`. Strict full OOF, fold membership SHA, fold AUC, train/test IDs, feature order, source SHA, candidate OOF/test SHA, CUDA/T4 and pinned versions required.
- Unique reserved Kaggle Notebook slug **`mjw3177109/kas-v2936-original37-xgboost-dart-t4-s6e10`**. At preregistration it **does not exist and must not be created until preflight succeeds**. Once reserved, never push a second time; only `--recover-only`.

## Fixed model, no search or early stopping

- XGBoost **3.0.5**, `XGBClassifier`, `booster="dart"`, `objective="binary:logistic"`, `eval_metric="auc"`, `tree_method="hist"`, `device="cuda"` on authenticated NVIDIA T4.
- `n_estimators=900`, `learning_rate=0.035`, `max_depth=4`, `min_child_weight=10`, `subsample=0.85`, `colsample_bytree=0.85`, `max_bin=256`, `reg_lambda=5.0`, `reg_alpha=0.0`, `gamma=0.0`.
- DART-only `rate_drop=0.05`, `skip_drop=0.0`, `sample_type="uniform"`, `normalize_type="tree"`. Use all 900 rounds for inference; no dropout at inference. Fixed `n_jobs=4`, no early stopping, no architecture/feature/seed/round/weight sweep. Exactly one candidate.
- Frozen rank blend: `0.85*rank(V2915)+0.15*rank(V2936)` with `scipy.stats.rankdata(method="average")/n`, OOF and test ranked separately. No post-hoc tuning.

## Security, promotion and official evidence

First read actual branch HEAD, repository-wide Actions `in_progress/queued/waiting/requested/pending`, champion state, V2935 terminal state and official Kaggle submissions/Notebook/rank evidence. Check all owner Notebooks with authenticated CLI, minimum three seconds between requests; 429, unknown/cancelling/active states, mismatched SHA, source ID, version, deadline or owner must fail closed. Reserve unique Notebook slug durably on GitHub **before** the only possible push. A recovery must never create, push or retrain. Tests and safety checks precede GPU expenditure.

Evaluator never submits. Only consider a **separate**, durably reserved one-shot official submission if candidate full CV `>=0.96125635576833`, paired gain vs champion `>=+0.00005`, at least two folds positive and worst fold gain strictly `>-0.00005`. Then check candidate test SHA, Kaggle quota, competition deadline **2026-10-31 23:59 UTC**, authenticated official submission history, deduplication and submission ID recovery. Do not accept rules, leak tokens, or bypass protection. CV cannot be represented as Public LB or rank.

No V2936 Notebook or training has started as part of this preregistration.
