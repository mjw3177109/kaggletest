# V2.9.10 | Paired OOF bootstrap and subgroup safety audit (not a new CV)

Pre-registered method before running:
- Compare V2.9.6 official incumbent vs V2.9.8 and V2.9.9 using the exact archived, aligned threefold OOF predictions. Do not train, push, rerun, or submit Kaggle notebooks.
- Reconstruct CV 0.9609541991872741 (V296), 0.9610183558267038 (V298), and 0.9610561936366048 (V299), tolerance <=0.000002.
- Stratified paired bootstrap B=120 with random seed 20261020. Preserve each fold's binary target counts separately, resampling the SAME indices for both candidates and incumbent. Record 2.5%, 50%, 97.5% empirical percentile of AUC deltas and fraction of positive draws.
- Subgroup diagnostics for independent categorical slices: Type of Travel, Class, Customer Type, Gender. Only show groups with at least 10000 rows and both classes. Report five worst paired V299 vs V296 AUC deltas; never show individual records.
- No model/weight selection from bootstrap data. These resampled scores are NOT new independently trained CV seeds and MUST NOT be called a new Kaggle leaderboard result.
- Report candidate robustness, not promotion: if lower bootstrap bound <=0 then mark uncertainty and demand extra independent check before any submission. If >0 then positive paired-bootstrap evidence only, not guaranteed Public LB gain.
- No submission, quota usage, or training; run after V299 stage is terminal, deduplicate via state/v2_9_10_robustness_status.json.
