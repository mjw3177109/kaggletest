# V2.9.41 — independent fold-safe spline-additive logistic learner (CPU-only), preregistration

**Preregistered 2026-10-25 UTC, before any V2941 model fitting, OOF inspection, Kaggle Notebook creation, or submission.** This document does not authorize a training run; code, offline safety tests, pinned synthetic preflight, owner-wide guards, and an exclusive durable one-shot GitHub run reservation must be implemented and verified first.

## Genuinely different inductive bias

V2940's monotone-constrained histogram tree booster ended at blended CV 0.9611640692055075 (gain -0.00012097072772609962), 0/3 folds positive. V2937 LightGBM GOSS and V2938 CatBoost Ordered were also terminal. V2935/V2939 frozen-source consensus had positive but subthreshold gains. Do not retry any of them or optimize their weights/thresholds.

V2941 tests a **regularized additive spline logistic model**, a different, transparent global additive inductive bias from all previously tried trees, MLPs, and rank-only consensus rules. It can learn smooth nonlinear univariate service-score effects but intentionally no new explicit cross-feature interactions. Whether its residual ranking errors complement V2915 is unknown; no claim of improved Public LB.

## Immutable data and fold isolation

- Official playground-series-s6e10 train 699635 rows, test 299844; original `id`, binary `satisfaction`. Exactly the champion V2915 **37 input feature names/order**, including the three original fold-safe inner-fivefold target-encoded `te_0,te_1,te_2` columns; no extra raw features, labels, pseudo-labels, PCA, route profiles, or test-target access.
- Outer `StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)` and canonical V2915 fold membership, train/test IDs and targets. All imputers, categorical vocabularies, encoders, spline knots, and scalers are fitted on each outer-training fold only. Three protected TEs use `TargetEncoder(target_type="binary",smooth="auto",cv=5,shuffle=True,random_state=20261018+fold)`; outer-validation/test are transform-only. Never use outer-validation labels to fit features or models.
- Frozen 14 categorical input columns: `Gender,Customer Type,Type of Travel,Class,dual_Age,dual_Flight Distance,dual_Inflight wifi service,dual_Online boarding,dual_Seat comfort,dual_Inflight entertainment,dual_Leg room service,combo_0,combo_1,combo_2`. Apply fold-train-only `OneHotEncoder(handle_unknown="infrequent_if_exist",max_categories=32,sparse_output=True)`.
- Frozen 4 raw numeric service inputs: `Inflight wifi service,Online boarding,Seat comfort,Inflight entertainment`. Apply `SplineTransformer(n_knots=5,degree=3,knots="quantile",extrapolation="constant",include_bias=False)` fitted only on outer-train. The remaining **19 numeric inputs** (including three TE columns) use outer-train-only median imputation and `StandardScaler`. Missing raw service values use outer-train medians before splines. The input schema must be exactly 14+4+19=37, no duplicates. Feature expansion from OHE/splines is a fixed transform of those inputs, not new source data.
- Use a single `ColumnTransformer` in the above fixed order, `sparse_threshold=0.3`, with training-only fitting. Assert identical transformed dimensionality/order across validation and test; record categorical vocabulary SHA, spline knots SHA, and canonical 37-input schema SHA.

## Frozen CPU model and candidate

- Isolated Python 3.11; `numpy==2.2.6,pandas==2.2.3,scipy==1.15.3,scikit-learn==1.6.1`. `OMP_NUM_THREADS=4`, CPU-only, synthetic end-to-end transform+fit must pass before official data access. Version mismatch or convergence/NaN issue is terminal fail-closed; never retune after looking at OOF.
- Exactly `LogisticRegression(penalty="l2",C=0.5,solver="lbfgs",max_iter=400,tol=1e-4,fit_intercept=True,class_weight=None)`. Deterministic LBFGS, no seed search or early stopping. Treat `ConvergenceWarning` as a failed frozen run, not a reason to change max_iter.
- Standalone 3fold OOF/test predictions; one and only frozen blend `0.85*rank(V2915)+0.15*rank(V2941)`, `rank(x)=scipy.stats.rankdata(x,method="average")/len(x)`; compute OOF and test ranks independently. No calibration, threshold/weight search, or candidate selection using labels.

## Mandatory safety and promotion gates

Implement a new independent V2941 trainer, validator, CPU-only unit tests, authenticated owner-wide Kaggle state guard and exclusive one-shot GitHub run reservation. Prior to training: verify actual branch HEAD; all repository-wide Actions in_progress/queued/waiting/requested/pending; all owner Notebook states (including historical entries); official submission history and current deadline; pinned dependencies and synthetic fit. Kaggle authenticated requests separated by at least 3 seconds; 429/unknown fails closed. **No Kaggle Notebook creation/push for V2941.** Once a run is reserved or an attempt is ambiguous, read-only recovery only, no second fit or workflow retrigger.

Validate complete 3fold OOF, fold AUC/gains and membership SHA, canonical train/test ID/target order, 37 input feature order, inner TE isolation, train-only categorical/spline/scaler fitting, source NPZ SHA, champion test SHA `906185547dcd5f0cac3ac173de644fd9997f20ac863c6745bcb1fd3e9a89e5f7`, candidate OOF/test SHA, and exact CPU versions. Evaluator **never submits**.

Only if frozen blended full CV >= `0.96125635576833`, gain vs V2915 CV `0.9612850399332336` >= `+0.00005`, >=2 positive folds, and worst fold gain strictly > `-0.00005`, may a separate durable one-shot official submission be considered after quota/deadline, prediction SHA, official submission history dedup and submission-ID recovery checks. Do not infer official Public LB from CV. Deadline **2026-10-31 23:59 UTC**. Do not accept competition rules or disclose credentials. If V2941 fails, record a causal-limits-aware diagnosis before independently preregistering a truly distinct V2942.
