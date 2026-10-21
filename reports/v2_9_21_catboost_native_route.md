# V2.9.21 CatBoost native categorical route-profile CPU threefold audit

- UTC: 2026-10-21T16:12:50.972437+00:00
- Stage: skipped_no_cv_gain
- Notebook: https://www.kaggle.com/code/mjw3177109/kas-v2921-catboost-native-route-cpu-s6e10
- Frozen V2915 OOF baseline: 0.9612850399332336
- Verified V2915 official Public LB: 0.96099 (submission 57446286)
- New independent 3fold OOF AUC: 0.959486551014939
- Frozen 85/15 blend 3fold AUC: 0.9612213583225888
- Paired fold deltas: [-8.578299450123783e-05, -3.618550646067309e-05, -7.158977287113899e-05]
- Promotion rule: gain >= 5e-05, >=2 positive folds, worst fold > -0.00005
- Kaggle competition submission: NEVER performed by this evaluator
- Audit note: the original Notebook omitted the summary AUC field. The controller recomputed full OOF ROC-AUC from the ID/target-aligned OOF vector and independently reproduced all three fold AUCs; no retraining or resubmission was performed.
