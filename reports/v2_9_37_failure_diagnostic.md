# V2.9.37 — terminal verified CPU GOSS result (no submission)

At **2026-10-25 03:48:15 UTC**, the sole guarded CPU Actions run **#40157906219** finished SUCCESS (first attempt only). Its exclusive reservation was durably written as run ID `40157906219` before data download or fitting. The run installed and checked pinned LightGBM 4.6.0, NumPy 2.2.6, pandas 2.2.3, SciPy 1.15.3 and scikit-learn 1.6.1; completed all three outer folds with 37 original features and strict independent OOF validation. The workflow archived immutable evidence as artifact **12585825871** (ZIP SHA256 `e1b48832c983ddc3508462fb2696dc7bc9d5c484831ed4cd73e679fc9b8ffc90`). It did **not** create a Kaggle Notebook or submit.

## Certified metrics

- Champion V2915 full threefold OOF AUC: `0.9612850399332336`.
- V2937 standalone full OOF AUC: `0.9590691263361465`.
- Frozen 85/15 rank fusion full OOF AUC: `0.9612406974032877`.
- Gain against champion: `-0.00004434252994589549`.
- Fold gains: `[-0.00003999940810672786,-0.00006150032053287013,-0.000033381965860113105]`, **0/3 positive**.
- Source NPZ SHA256: `4de0f81a6e37a45cc37ddcb22e6bc024dfc8fd08a40936138c51d36297de9c5a`.
- Candidate OOF SHA256: `2238033517e642b8f961dea1ec0b0b76a18030996c135e0657b922db2e3d92b4`.
- Candidate test SHA256: `bcaced2693322ea41acfe9f22ab76579bfcc828987ffc86e35f7f43ddd1167b0`.
- Champion test prediction SHA256: `906185547dcd5f0cac3ac173de644fd9997f20ac863c6745bcb1fd3e9a89e5f7`.

The terminal state is `skipped_no_cv_gain`: full CV is below `0.96125635576833`, gain is below `+0.00005`, and no fold improves. **Do not retrain, rerun the one-shot workflow, resubmit, or tune V2937.** Warnings about high-cardinality categorical bins appeared in LightGBM logs; they do not prove the cause of lower AUC. The data and model hypothesis failed the frozen promotion rule; the exact cause is not established.

Official Kaggle Public LB remains **0.96099** for champion submission **57446286**; most recent persisted authenticated Public rank **984/2931** at **2026-10-25 02:44:10 UTC**. No new authenticated leaderboard refresh occurred during the one-shot run. This is not a Private rank.

Evidence: https://github.com/mjw3177109/kaggletest/actions/runs/40157906219 ; https://github.com/mjw3177109/kaggletest/blob/feature/v29-autonomous-evolution/reports/v2_9_37_result.md .
