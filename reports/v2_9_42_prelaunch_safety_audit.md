# V2.9.42 prelaunch fail-closed safety audit — 2026-10-25 UTC

Read-only GitHub verification: branch HEAD `d86b44e4aae8a0de26d7ee886aa3e2b73da6c2f0`; all five repository-wide active Actions statuses were zero. The pinned offline safety run #40190954608 succeeded (21/21), but this is not a real competition fit or an authenticated Kaggle account audit. The historical Public rank 984/2931 at 2026-10-25 02:44:10 UTC remains historical.

## Deterministic defects found before any official data access

1. The dispatch-only V2942 workflow could proceed with `prelaunch_allowed=false`: the preregistration gate checked that the flag was false but did not enforce an independent launch authorization before the owner-wide API gate. A manual dispatch could therefore reach the reservation path without an explicitly approved prelaunch decision. This is a safety gap, not evidence of training.
2. The workflow's terminal result persistence and artifact list referenced `reports/v2_9_41_result.md` instead of `reports/v2_9_42_result.md`, risking incorrect or missing evidence. The safety-test step label also had an obsolete count.

## Minimal repair

- Preserve all frozen V2942 model parameters, source schema, CV thresholds, and `state/v2_9_42_status.json` as-is.
- Require `prelaunch_allowed=true` and a recent (<=15 minutes) explicit owner-Notebook, official-submission and deadline authorization record bound to the champion test SHA **before any owner-wide Kaggle request or reservation**. Current status has `prelaunch_allowed=false` and no such record, so a manual launch fails closed.
- Keep live Kaggle owner-wide and global Actions checks in addition to the authorization evidence. Do not use the attestation in place of live verification.
- Correct V2942 report persistence/artifact paths and add offline regression tests proving no Kaggle checks or reservation when authorization is absent, incomplete or stale.

No Kaggle authenticated owner-wide Notebook or submission query was performed in this audit. No official competition data downloaded, no training, no Kaggle Notebook pushed, and no submission attempted. **V2942 remains preregistered and unlaunched.** V2941's permanent no-retry protection remains intact. This audit is not authorization to launch.

Evidence: https://github.com/mjw3177109/kaggletest/actions/runs/40190954608
