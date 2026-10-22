# V2.9.23 Personal-Travel-weighted LightGBM CPU threefold audit

- UTC: 2026-10-22T03:03:24.375060+00:00
- Stage: skipped_no_cv_gain
- Notebook: https://www.kaggle.com/code/mjw3177109/kas-v2923-personal-weight2-lightgbm-cpu-s6e10
- Frozen V2915 OOF baseline: 0.9612850399332336
- Verified V2915 official Public LB: 0.96099 (submission 57446286)
- New independent 3fold OOF AUC: 0.96002755328987
- Frozen 85/15 blend 3fold AUC: 0.96123680234935
- Paired fold deltas: [-6.251114796462787e-05, -2.9369142964030814e-05, -5.1964923908798255e-05]
- Promotion rule: gain >= 5e-05, >=2 positive folds, worst fold > -0.00005
- Kaggle competition submission: NEVER performed by this evaluator
