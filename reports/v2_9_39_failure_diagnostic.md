# V2.9.39 — terminal median consensus audit (no submission)

Verified **2026-10-25 06:52:53 UTC**. The **sole** guarded read-only CPU GitHub Actions run **#40170332489** finished SUCCESS on its first attempt. Its exclusive run reservation was durably recorded as `40170332489` before source download/evaluation. All 23 offline safety tests and the authenticated owner-wide Kaggle preflight passed. The frozen V2915, V2916, and V2930 complete OOF/test sources were verified. No model was fitted, no Kaggle Notebook was created/pushed, and no submission was attempted.

## Immutable full threefold OOF outcome

- Officially submitted champion V2915 CV: `0.9612850399332336`; official Public LB `0.96099`; submission ID `57446286`.
- Frozen V2939 candidate: `r0 + 0.15*(median(r0,r16,r30)-r0)`, with independently computed average-tie ranks over OOF and test.
- Full candidate OOF CV: **`0.9613234154717458`**.
- Paired gain vs champion: **`+0.000038375538512269713`**.
- Fold AUC: `[0.960841441665085,0.9616188440138298,0.9615340534152381]`.
- Fold gains: `[+0.000034442885625041164,+0.000044273529253402444,+0.000035628966834733156]`; **3/3 positive**, but full gain **below +0.00005**.
- OOF disagreement fraction: `0.19979274907630407`; test disagreement fraction: `0.1684442576806606`. Diagnostics only; these values were not used for tuning.
- Candidate OOF SHA256: `6481e374da5e505a165e8449454004a356bd6cec43b8b36b8ad47cb93a966433`.
- Candidate test SHA256: `c47102e8a45f53b93cf32089ceb46167874dd5c8692c73888d0a23fed12aa3bb`.
- Champion test prediction SHA256: `906185547dcd5f0cac3ac173de644fd9997f20ac863c6745bcb1fd3e9a89e5f7`.
- Source NPZ SHA256: V2915 `46ad63f08537cd8d61c4c8e8698efc20bb3c8fbad4eb0d94ecd7c993b2d894b9`, V2916 `05facb3098732f1348928c8de62cefae3ecb64b5b1fabdb3a7a84342b7bfe0d7`, V2930 `45e96e595676ec91a1e6c8eea98f975e63c9c0762412b898fe549c0636107362`.
- Immutable GitHub artifact **12590345284**, ZIP digest `sha256:6002b0e42e0a26da6d43e3e612714470c8a52aa63daf36731f9c3b28e7af9fed`.

Decision: **`skipped_no_cv_gain`, permanently terminal, no submission**. The candidate passed the minimum CV floor and 3/3 fold checks, but missed the **independent** gain gate by `0.000011624461487730287`. Do not rerun V2939, search median/shrinkage variants, or submit this candidate. The observed difference between the V2935 arithmetic agreement gate and V2939 median does not prove why the latter gained less.

The most recent persisted authenticated Public rank remains **984/2931 at 2026-10-25 02:44:10 UTC**, not a fresh rank or final Private ranking. CV does not establish official Public LB.

Evidence: https://github.com/mjw3177109/kaggletest/actions/runs/40170332489 ; https://github.com/mjw3177109/kaggletest/blob/feature/v29-autonomous-evolution/state/v2_9_39_status.json .
