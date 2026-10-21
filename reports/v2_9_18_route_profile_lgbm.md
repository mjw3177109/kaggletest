# V2.9.18 exact-distance LightGBM CPU threefold audit

- UTC: 2026-10-21T14:22:38.120800+00:00
- Stage: skipped_no_cv_gain
- Notebook: https://www.kaggle.com/code/mjw3177109/kas-v2918-routeprofile-lightgbm-cpu-s6e10
- Frozen V2915 OOF baseline: 0.9612850399332336
- Verified V2915 official Public LB: 0.96099 (submission 57446286)
- New independent 3fold OOF AUC: 0.9602322592841712
- Frozen 85/15 blend 3fold AUC: 0.9612515540913926
- Paired fold deltas: [-5.1493158239046544e-05, -1.5438959213853742e-05, -3.2707224908223864e-05]
- Promotion rule: gain >= 5e-05, >=2 positive folds, worst fold > -0.00005
- Kaggle competition submission: NEVER performed by this evaluator
