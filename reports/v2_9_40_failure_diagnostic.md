# V2.9.40 — terminal monotone HGB diagnosis (no submission)

Verified from the single GitHub Actions run **#40176220913** (first attempt; completed SUCCESS at 2026-10-25 08:43:26 UTC), its independent threefold validator, persisted `state/v2_9_40_status.json`, and immutable artifact **12592358028**. The exclusive reservation `run_reservation_id=40176220913` was persisted before official data access or fitting. All 24 CPU safety tests and the pinned scikit-learn 1.6.1 synthetic monotonic fit passed. The guarded owner-wide Kaggle preflight and subsequent verification completed. No Kaggle Notebook was created/pushed and no submission was attempted.

## Frozen full-OOF result

- V2915 champion full CV: `0.9612850399332336`; official Public LB `0.96099`; submission `57446286`.
- V2940 standalone CPU monotone HGB CV: **`0.9582210145788145`**.
- Fixed 0.85/0.15 rank blend full CV: **`0.9611640692055075`**.
- Paired gain: **`-0.00012097072772609962`**; fold gains `[-0.00014872118333741913,-0.00008866561659082173,-0.00012699373389479796]`, **0/3 positive**.
- Source NPZ SHA256: `4246eedd53ee680feb493a0a1bf7bb98d7d419a2688508e26f6f8e038cc747a8`.
- Candidate OOF SHA256: `06fc5c0c84d489b8a27918b59b0ecb399a804f7feb20712983ff930b5fad92da`.
- Candidate test SHA256: `41167e482314709a0e6d15f9d798f687666801376cd7767b8419bb37bd31cf35`.
- Champion test prediction SHA256: `906185547dcd5f0cac3ac173de644fd9997f20ac863c6745bcb1fd3e9a89e5f7`.
- CPU pinned versions: Python 3.11.17, scikit-learn 1.6.1, NumPy 2.2.6, pandas 2.2.3, SciPy 1.15.3; 37 input features, original three inner-fivefold TE columns, and four raw service-score monotonic +1 constraints.

**Decision: permanently terminal `skipped_no_cv_gain`, not promotable.** The experiment passed execution/safety checks but failed the frozen CV gain and fold gates. The observed lower CV does **not** establish that monotonicity itself caused the decline; no parameter, constraint, feature, or blend search is authorized on this version. Never rerun/retrain/resubmit V2940.

Most recently persisted authenticated Public rank is **984/2931**, checked 2026-10-25 02:44:10 UTC. This is stale historical evidence, not a newly verified current or Private rank. No official Public score change was observed through a new submission.

Evidence: https://github.com/mjw3177109/kaggletest/actions/runs/40176220913 ; https://github.com/mjw3177109/kaggletest/blob/feature/v29-autonomous-evolution/state/v2_9_40_status.json ; https://github.com/mjw3177109/kaggletest/blob/feature/v29-autonomous-evolution/reports/v2_9_40_result.md .
