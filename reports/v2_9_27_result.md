# V2.9.27 fixed travel-gated dual RealMLP threefold OOF

- Updated UTC: 2026-10-24T08:20:37.931426+00:00
- Stage: skipped_no_cv_gain
- Kaggle Notebook: https://www.kaggle.com/code/mjw3177109/kas-v2927-travel-gated-dual-realmlp-t4-s6e10
- Frozen champion CV: 0.9612850399332336
- New independent model full OOF: 0.9593005287908923
- Frozen 85/15 fusion full OOF: 0.9613084824713334
- Candidate gain: 2.3442538099827104e-05
- Paired fold gains: [1.0336810181810208e-05, 2.9806798472975515e-05, 2.9146686000403044e-05]
- CV gain gate: >= 5e-05; official Public LB stays 0.96099 until independently scored.
- No Kaggle competition submission is performed by this evaluator.
- Error: Prior object-dtype gate validation failure preserved in validation_incident; pending one-time safe read-only revalidation
