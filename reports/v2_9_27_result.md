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
- Validation incident: first recover-only audit #40077444649 failed because the Notebook saved two travel-group string columns as object-dtype NPY; safe numpy loading forbids pickle.
- Resolution: one-time --recover-only audit #40077588836 succeeded with no Notebook push, no model retraining, no pickle deserialization, no submission. Numeric OOF/ID and per-fold expert membership/AUC were independently replayed against canonical competition data. The original object-array payload was not read, so direct byte-for-byte group-string equality from those two arrays is not claimed.
- Notebook status: authenticated COMPLETE. Threefold OOF and T4 metadata verified; output SHA256 `180cd12af6fcc097725718148946bfcb0a0b6184cf6403cc360b44ef26ffb840`.
- Scientific interpretation: Personal Travel group AUC 0.8195406930134544; Business travel group AUC 0.9568923457818472; global dual-expert standalone AUC 0.9593005287908923. Smaller per-expert training populations and potential cross-group score-scale mismatch are hypotheses for the weaker standalone performance, not proven causal explanations.
- Permanent decision: no CV gain threshold met (+0.0000234425381 < +0.00005); do not submit or rerun V2.9.27.
