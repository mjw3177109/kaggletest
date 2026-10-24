# V2.9.26 — fold-local route-profile RealMLP-TD, preregistered

Date: 2026-10-24. Status: **preregistered, no Kaggle Notebook yet**.

## Evidence and hypothesis

The champion V2.9.15 official Public LB is 0.96099 (submission 57446286), full 3-fold OOF AUC 0.9612850399332336. Independent V2.9.25 OOF AUC 0.961332372152315 was below its preregistered +0.00005 improvement threshold despite three positive folds. V2.9.23 weighted LightGBM decreased CV; V2.9.24 Personal Travel specialist alone improved only +0.00000972. This new trial is **not** a retune or resubmission of either.

V2.9.26 tests a distinct representation: add **12 label-free per-flight-distance route-profile features** to the V2.9.15 RealMLP-TD feature pipeline. For each outer fold the route statistics are fitted **only on outer training rows**, then applied to validation and test. The original inner 5-fold cross-fitted categorical target encodings remain unchanged. Never access validation targets during feature-fitting or model-fitting. Only competition train labels; no external data/labels or Kaggle leaderboard-based tuning.

## Frozen configuration

- Competition: playground-series-s6e10, 699635 train / 299844 test.
- Only Notebook identifier: `mjw3177109/kas-v2926-route-profile-realmlp-t4-s6e10`.
- StratifiedKFold(n_splits=3, shuffle=True, random_state=20261018), 3 genuinely independent out-of-fold models.
- RealMLP-TD via pytabkit==1.7.3, T4 verified; n_ens=8, 5 epochs, hidden layers [512,256,128], exact V2.9.15 learning parameters; independent seed 20261026, plus fold.
- Added route stats from the proven V2.9.24 formula, with no target/ID features: exact distance group count and group averages/residuals; 12 extra numeric features, no new supervised encoding.
- The **sole precommitted promotion candidate** is `0.80*rank(V2915 champion OOF) + 0.20*rank(V2926 new OOF)`, likewise for canonical test scores. No weight search, no metric-driven post-hoc adaptation.
- Promotion floor full threefold CV >= 0.96125635576833, improvement vs unchanged V2.9.15 champion >= +0.00005, >= 2 positive fold gains, worst fold > -0.00005.
- Pin known-good verifier: numpy==2.4.6, pandas==3.0.6, scipy==1.17.1, scikit-learn==1.9.1. Reconstruct eight V2.9.15 artifact sources, validate exact ID order, source SHA, baseline AUC and committed prediction SHA before evaluation.
- Single-Notebook owner verification: query exact Notebook status and authenticated own Notebook listing; fail closed on ambiguous results. Persist exact slug before/after push. Recover any saved slug with **--recover-only**. Never create duplicate Notebook.
- Full OOF/test integrity only; evaluator does **not** submit. A separate durable one-shot workflow with quota, SHA, history dedup and recovery would be mandatory for eventual Kaggle submit.

## Guardrails

No repeated failed experiment, no manual submit, no top-three claim from CV, no automatic external-dataset ingestion. Do not start a second train while V2.9.26 Notebook is queued/running; release the next experiment only after a verified terminal result. Report evidence and failure accurately.
