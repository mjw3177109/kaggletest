# V2.9.35 — frozen three-source consensus gate: terminal OOF diagnostic

Verified **2026-10-25 00:15:15 UTC** by authenticated read-only GitHub Actions **#40145587441**. Offline safety tests **20/20 passed**; the owner-wide Kaggle status gate authenticated the V2915, V2916, V2930 and V2934 Notebooks as COMPLETE. The evaluator downloaded original immutable outputs, replayed the original champion, checked source SHA/ID/target/folds and computed exactly one preregistered candidate. No Notebook was created, trained, pushed, or submitted.

| Metric | Immutable result |
|---|---:|
| V2915 official champion threefold OOF CV | 0.9612850399332336 |
| V2935 agreement-gated threefold OOF CV | **0.9613302676316021** |
| Paired full CV gain | **+0.000045227698368566216** |
| Fold gains | [0.00004092221121243522, 0.0000526677619130389, 0.00004112783554610644] |
| Positive folds | 3/3 |
| Gate agreement OOF / test | 0.8002072509236959 / 0.8315557423193394 |
| Minimum preregistered gain | +0.00005 |
| Decision | **skipped_no_cv_gain**, no submission |

- Frozen V2916 NPZ SHA256: `05facb3098732f1348928c8de62cefae3ecb64b5b1fabdb3a7a84342b7bfe0d7`.
- Frozen V2930 NPZ SHA256: `45e96e595676ec91a1e6c8eea98f975e63c9c0762412b898fe549c0636107362`.
- Champion test prediction SHA256: `906185547dcd5f0cac3ac173de644fd9997f20ac863c6745bcb1fd3e9a89e5f7`.
- V2935 candidate OOF SHA256: `e3b7b0cc288e995393c8ab688bfa7e2683f3eb1643ad05fa8dfd0e18d34d9b32`.
- V2935 candidate test SHA256: `98215a0373ad071f4192ecdecdf9f33a548c5e7fb6e5dab9858770413f4f105c`.
- Official V2915 Public LB remains 0.96099, submission ID 57446286. Latest persisted authenticated rank 979/2917 (2026-10-24 21:45:34 UTC), not a fresh or Private rank.

Interpretation: the strict label-free directional agreement gate was positive in all folds, but the full CV gain missed the fixed +0.00005 promotion threshold by **0.000004772301631433784**. The observed ~80% agreement does not establish a causal explanation for the shortfall; no post-hoc mask/weight tuning is allowed. V2935 is terminal and must not be rerun or submitted.

A distinct next hypothesis should test a different learning mechanism rather than reweighting the same three frozen sources. V2920's histogram XGBoost route-profile TE23 challenger failed; any future tree experiment must explicitly change both the booster regularization and feature family, preregister independently, and never rerun V2920.
