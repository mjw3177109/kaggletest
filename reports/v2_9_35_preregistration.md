# V2.9.35 — agreement-gated, label-free rank consensus (immutable preregistration)

Preregistered 2026-10-24 UTC **after** V2934 was terminally rejected and **before** any V2935 computation. This is a **new read-only CPU evaluation**, not a Notebook training run. No new Notebook, Kaggle prediction submission, new model fit, or hyperparameter search is permitted.

## Motivation and genuinely different hypothesis

V2934 DeepFM standalone CV 0.9576654181929295, fixed blend 0.9611890304161783 (delta -0.00009600951705523997; 0/3 positive folds), terminal `skipped_no_cv_gain`. V2932 attention and V2933 RankNet also failed. Earlier independent V2916 and V2930 RealMLP branches had **positive but individually insufficient** 85/15 gains (+0.000039325095311237845 and +0.000043621614375877193 respectively). V2925 tested a *different* Personal Travel residual fusion and failed by a small margin (+0.000047332219081); do not repeat its specialist correction.

V2935 tests whether **two frozen independent models agreeing on the direction of a ranking correction** are more reliable than their unconditional blend. It is not a new neural architecture, subgroup residual, target encoder, feature family, or optimized ensemble. The gate is entirely label-free and determined from three already-completed predictions.

## Frozen sources and exact one-shot formula

- Champion: V2915 frozen 699635-row threefold OOF and 299844-row test vector, official submission ID `57446286`, official Public LB `0.96099`, CV `0.9612850399332336`, champion test prediction SHA256 `906185547dcd5f0cac3ac173de644fd9997f20ac863c6745bcb1fd3e9a89e5f7`. Champion source NPZ SHA256 `46ad63f08537cd8d61c4c8e8698efc20bb3c8fbad4eb0d94ecd7c993b2d894b9`.
- V2916 original service/delay interaction RealMLP, COMPLETE and terminal, immutable NPZ SHA256 `05facb3098732f1348928c8de62cefae3ecb64b5b1fabdb3a7a84342b7bfe0d7`; standalone CV `0.9605256772889431`.
- V2930 conditional cross-fit TE RealMLP, COMPLETE and terminal, immutable NPZ SHA256 `45e96e595676ec91a1e6c8eea98f975e63c9c0762412b898fe549c0636107362`; standalone CV `0.9606450272778854`.
- Exact original three outer folds: `StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)`; no label access except **final evaluation**. Reconstruct champion from original source hashes and check every train/test ID and target in original order, each source's complete OOF, fold membership, per-fold AUC, Notebook identity, and raw NPZ SHA. No pickle; only numeric arrays with `allow_pickle=False`.
- Define `rank(x) = scipy.stats.rankdata(np.asarray(x,float64),method="average") / len(x)`. Compute ranks independently for OOF and test; **do not** use target labels, fold-specific AUC, score thresholds or subgroup masks in the gate.
- Let `r0=rank(V2915)`, `r16=rank(V2916)`, `r30=rank(V2930)`. Set `agree=((r16-r0)*(r30-r0)>0)`. The sole candidate is `r0 + agree * (0.075*(r16-r0) + 0.075*(r30-r0))`. When models disagree or either delta is zero, preserve the champion rank exactly. This equals `0.85*r0+0.075*r16+0.075*r30` **only** on agreeing rows.
- Freeze weights `0.075/0.075`, strict agreement threshold `>0`, tie rule, no alternative masks, rank methods, seeds, post-hoc tuning, calibration, weight sweep, model fitting, pseudo-labels or external data. Produce immutable OOF/test SHA256 using float64 little-endian bytes. Record train/test agreement fractions **as diagnostics only**, never use them to change the formula.

## Gate, safety and decision

Run only a guarded read-only CPU evaluator that checks the GitHub HEAD and all five repository-wide Actions states, authenticates Kaggle owner and official V2915 submission, checks the three existing source Notebooks COMPLETE, and verifies their artifact SHA/ID/folds before computing. The evaluator may download outputs but **must never push a Notebook, train, or submit**. Fail closed on 429, unknown kernel status, competing Actions/training, missing data, or provenance mismatch. Rate-limit authenticated Kaggle reads at least three seconds apart. No new V2935 Notebook is planned or reserved.

Only consider a **separate** one-shot submission when full candidate OOF CV `>=0.96125635576833`, gain over V2915 `>=+0.00005`, at least two folds positive, and worst fold gain strictly `>-0.00005`. Before any submission require a distinct durable reservation, candidate test SHA, Kaggle quota/deadline check, official submission-history deduplication and ID recovery. Evaluator never submits. Official Public LB and rank require independent Kaggle authentication and cannot be inferred from CV. Competition deadline: 2026-10-31 23:59 UTC.

The previous terminal experiments V2925–V2934 must not be rerun or resubmitted. If this new hypothesis fails, diagnose and record before another distinct preregistration.
