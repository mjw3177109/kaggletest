# V2.9.24 Personal Travel specialist residual-correction audit

- UTC: 2026-10-23T01:37:11.231791+00:00
- Stage: validation_mismatch
- Notebook: https://www.kaggle.com/code/mjw3177109/kas-v2924-personal-travel-specialist-cpu-s6e10
- Frozen champion: V2.9.15, CV 0.96128503993323, official Public LB 0.96099 (submission 57446286)
- Specialist: Personal Travel only; LightGBM; 66 features; 3 outer folds; 5-fold inner TargetEncoder
- Personal subgroup OOF AUC: pending
- Fixed residual alpha: 0.1
- Candidate full OOF AUC: pending
- Candidate gain vs champion: pending
- Fold deltas: pending
- Promotion: CV >= 0.96125635576833, gain >= 5e-05, at least 2 positive folds, worst fold > -0.00005
- External labels: none. Submission: never performed by this evaluator.
- Error: Reconstructed V2915 frozen test prediction hash changed
