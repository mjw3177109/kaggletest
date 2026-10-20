# V2.9.8 | Fixed three-model fusion (read-only)

## Registered experiment
- Competition: playground-series-s6e10
- Validation: StratifiedKFold(n_splits=3, shuffle=True, random_state=20261018)
- One fixed fusion: rank(V2.9.3 baseline) 65%, rank(V2.9.6 RealMLP) 20%, rank(V2.9.7 conditional TE LightGBM) 15%.
- Do not tune the fixed weights after examining OOF. No model training or Kaggle notebook push, rerun, or submit.
- Prior official scored Public LB: 0.96059, V2.9.6 submission 57408736.
- Baseline 3fold CV: 0.9609541991872741, V2.9.6. Candidate promotion threshold: 0.9610141991872741.

## Required verification
1. Download canonical competition train.csv and test.csv using the authorized Kaggle account inside an ephemeral Actions workspace.
2. Fetch existing Kaggle outputs from the recorded V2.5, V2.8, V2.9.6 and V2.9.7 notebook IDs; record artifact hashes.
3. V2.5 and V2.8 OOF NPZ files omitted train_id, so check the historical scripts' canonical train-file input order and exact target equality with official training CSV. This is source-order provenance, not an explicit ID field in the old files; never claim otherwise.
4. For V2.9.6 and V2.9.7, explicitly align train ID, target and test ID. Reject missing/duplicate/mismatched IDs, different validation protocol, rows or contradictory model metrics.
5. Reconstruct V2.9.3 AUC=0.9607535043719558 and the exact incumbent V2.9.6 AUC=0.9609541991872741 before evaluating the candidate.
6. Promotion requires 3fold AUC gain >=0.00006, two or more folds improved, and worst fold change > -0.00005 compared with incumbent.
7. Store only safe status, signed experimental ledger and report on the branch; never upload competition train/test data, raw OOF or credentials.
8. If passing, mark candidate_ready_unsubmitted only; the protected one-shot submission workflow, quota limit, duplicate guard and durable reservation remain separate.

## Source files
- kaggle_agent/v298_fusion.py
- tests/test_v298_fusion.py
- .github/workflows/kaggle_v298_readonly_fusion.yml
- state/v2_9_8_status.json (written only after actual OOF evaluation)
- reports/v2_9_8_fusion.md (written only after actual OOF evaluation)

## Follow-on candidates
V2.9.9 independent CatBoost cross-interaction features, V2.9.10 subgroup LightGBM residual signal, and V2.9.11 validation seed robustness. Each requires independent pre-registration and no concurrent/repeated Kaggle kernels.
