# V2.9.17 independent feature-interaction RealMLP T4 audit

- UTC: 2026-10-21T14:02:57.826547+00:00
- Stage: skipped_no_cv_gain
- Notebook: https://www.kaggle.com/code/mjw3177109/kas-v2917-interaction-realmlp-ens8-t4-s6e10
- Frozen V2915 OOF baseline: 0.9612850399332336
- Verified V2915 official Public LB: 0.96099 (submission 57446286)
- New independent 3fold OOF AUC: 0.959450624039158
- Frozen 85/15 blend 3fold AUC: 0.9613002570031772
- Paired fold deltas: [2.8428257980550597e-05, 1.9836410025675377e-06, 1.3506105069227559e-05]
- Promotion rule: gain >= 5e-05, >=2 positive folds, worst fold > -0.00005
- Kaggle competition submission: NEVER performed by this evaluator
