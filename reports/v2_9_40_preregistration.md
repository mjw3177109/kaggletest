# V2.9.40 — preregistered fold-safe monotone-constrained histogram boosting (CPU-only)

Preregistered **2026-10-25 UTC**, after the V2939 terminal OOF audit and **before** any V2940 candidate fitting, OOF evaluation, Notebook creation, or Kaggle submission. **Registration only; no training or evaluation is authorized by this document.**

## Distinct scientific hypothesis

V2939's label-free median of the V2915/V2916/V2930 ranks gained only +0.000038375538512269713 and is terminal. V2935's arithmetic agreement gate gained +0.000045227698368566216; V2925's Personal Travel residual fusion gained +0.000047332219081. **Do not search weights, masks, thresholds, or order-statistic variants on these reused OOF labels.** V2937 CPU LightGBM GOSS and V2938 CPU CatBoost Ordered failed their frozen 85/15 gates. V2936 DART never trained because of a pinned dependency mismatch.

V2940 is instead a genuinely different **supervised structural-inductive-bias experiment**: scikit-learn's CPU histogram gradient booster with **explicit domain-motivated monotonicity constraints on four raw ordinal service scores**. This is not V2920's route-profile XGBoost, V2937's GOSS sampling, V2938's Ordered boosting, V2925/V2935/V2939's frozen-source fusion, or any prior RealMLP feature experiment. Hypothesis: smooth monotone service-quality responses might regularize the new learner and yield complementary ranking errors. The hypothesis is unverified and may fail.

## Frozen dataset and leakage protection

- Only official Kaggle playground-series-s6e10 train/test: 699635/299844 rows, original `id` and binary `satisfaction` target.
- Exact V2915 **37 feature names and order**, including original three **inner-fivefold TargetEncoder** columns. No new columns, PCA, route profiles, subgroup weighting, pseudo-labels, external labels or test-label use. Reconstruct the V2915 fold-safe engineer/TE pipeline; outer-validation labels may never enter feature construction or model fitting.
- Outer `StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)`; the same canonical train/test ID order and fold membership as V2915. Train-only categorical vocabularies, numeric medians, and 5fold TE fit; validation/test transform only. Encode categorical inputs to numeric with outer-train-only maps and deterministic unknown/missing codes, and record exact 37-column schema.
- Apply `monotonic_cst=+1` **only** to the raw original features `Inflight wifi service`, `Online boarding`, `Seat comfort`, and `Inflight entertainment`. Every other feature (including their dual categorical copies, all three TE columns, delay features, and count columns) has constraint 0. Build the positional constraint vector from the validated original 37-column schema; fail closed if any of the four raw features is absent, duplicated or nonnumeric. Do not alter these constraints after results.

## Frozen single-model and evaluation contract

- CPU-only Python 3.11 and `scikit-learn==1.6.1`, `numpy==2.2.6`, `pandas==2.2.3`, `scipy==1.15.3`; isolate and verify actual versions and run a synthetic monotone-constraint fit **before** any real training. No GPU/Notebook is planned. Do not silently change pinned versions.
- `sklearn.ensemble.HistGradientBoostingClassifier(loss="log_loss",learning_rate=0.035,max_iter=600,max_leaf_nodes=31,max_depth=6,min_samples_leaf=100,l2_regularization=5.0,max_bins=255,early_stopping=False,random_state=20261107+fold,monotonic_cst=frozen_positional_vector)`. `OMP_NUM_THREADS=4`. No other parameter, seed, feature or model search. Record full environment and synthetic fit evidence.
- One and only candidate: `0.85*rank(V2915 champion)+0.15*rank(V2940 standalone)` with `rank(x)=scipy.stats.rankdata(np.asarray(x,float64),method="average")/len(x)`; independently rank full OOF and test. No calibration or alternate blend.
- Strict independent complete threefold OOF validation: per-fold and full AUC, fold membership SHA, train/test ID and target alignment, 37 feature names/order, inner TE isolation, source NPZ SHA, champion test prediction SHA `906185547dcd5f0cac3ac173de644fd9997f20ac863c6745bcb1fd3e9a89e5f7`, candidate OOF/test SHA and CPU dependency versions. The validator must never submit.

## Hard launch and submission gates

Implement a **new** independent V2940 trainer, scientific validator, durable exclusive GitHub run reservation, safety gate, tests and CPU-only Actions before any real fitting. The safety gate must first check current remote HEAD, **all five repository-wide active Actions statuses**, every Kaggle owner Notebook status, official submission history, rank evidence and deadline. Kaggle authenticated reads at least 3 seconds apart; 429/unknown state fails closed. No duplicate CPU run: after reservation or ambiguous first attempt, only read-only recovery. **Never create/push Kaggle Notebook for V2940.** No competition rule acceptance or token disclosure.

Only if the frozen full blended CV is `>=0.96125635576833`, gain versus V2915 `>=+0.00005`, at least 2/3 folds improve, and worst fold gain is strictly `>-0.00005`, may a separate durable one-shot Kaggle submission workflow be considered. It must independently verify candidate test SHA, quota/deadline, authenticated official history dedup and submission-ID recovery. Evaluator itself **never submits**. Official Public LB is not CV; target Public top-three is aspirational, not guaranteed. Competition deadline **2026-10-31 23:59 UTC**.

If this preregistered method fails or the pinned environment is unavailable, record the cause and do not retune/restart V2940. Pre-register a genuinely distinct future mechanism only after terminal diagnosis.
