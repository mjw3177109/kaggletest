# V2.9.25 preregistration — dual-axis frozen OOF fusion

Date (UTC): 2026-10-23. **Registration only. No Notebook has been created, no training has started, and no Kaggle submission has been attempted.**

## Why this is a distinct experiment

V2.9.23's weighted LightGBM fusion was negative (CV 0.96123680234935), and V2.9.24's Personal Travel-only LightGBM residual improved the frozen champion by only **0.0000097242712972**, below the required **0.00005** gain. A separate, successful read-only diagnostic (Actions run **39925902649**) pinned the original scientific dependencies, reproduced V2.9.15's frozen test SHA, independently checked V2.9.24's full threefold OOF, and verified all three fold gains positive. V2.9.24's terminal `validation_mismatch` status is intentionally left immutable; the independent audit is recorded in `state/v2_9_24_readonly_audit.json`.

V2.9.16's service/delay interaction RealMLP branch independently achieved CV **0.9613243650285448** (+0.0000393250953 versus V2.9.15), but failed the same gain gate. V2.9.25 preregisters a **new two-axis fusion** of the interaction model and the Personal Travel residual. It tests whether these two previously positive but insufficient improvements are complementary; no combined improvement is assumed.

## Frozen method (no tuning or weight search)

1. Reconstruct the exact V2.9.15 frozen OOF/test prediction vectors and validate SHA **906185547dcd5f0cac3ac173de644fd9997f20ac863c6745bcb1fd3e9a89e5f7**.
2. Read the **existing** V2.9.16 threefold OOF/test predictions; verify canonical IDs, target alignment and artifact SHA **05facb3098732f1348928c8de62cefae3ecb64b5b1fabdb3a7a84342b7bfe0d7**.
3. Read the **existing** V2.9.24 Personal Travel-only OOF/test predictions; verify mask, target alignment, three outer folds and artifact SHA **47b5aa6931ab925e6cb2ec89dfd262c508da1445af8766f69eb8525bc93d9568**.
4. Build the fixed interaction candidate: `0.85 * rank(V2915) + 0.15 * rank(V2916)`. Only within the Personal Travel mask, add `0.15 * (rank(V2924 specialist within mask) - rank(interaction candidate within mask))`. Preserve every non-Personal row unchanged.
5. Compute independent full OOF AUC, all three fold deltas, and candidate SHA. No model fit, new labels, optimization, or repeated experimentation. Reproduce V2.9.15 SHA with **numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, scikit-learn 1.9.1**.

## Promotion and safety gates

Only consider promotion if **full CV >= 0.96125635576833**, **gain versus frozen champion >= 0.00005**, **at least two folds improve**, and **worst fold delta > -0.00005**. The first CV floor is necessary but not sufficient. **No CV score is an official Public LB score.**

Reserved unique Notebook slug: `mjw3177109/kas-v2925-dual-axis-residual-cpu-s6e10`. Its Kaggle existence is **not yet verified**; `state/v2_9_25_status.json` explicitly records no created Notebook. Before any future Notebook creation, check the authenticated owner list and absence of active training/recovery/submission runs. A nonempty or ambiguous owner listing must fail closed. If an existing slug is found, recover that one only; never create a duplicate. Evaluation is CPU-only and never submits. If the promotion gates pass, require a separately verified, persistent, one-shot submission workflow with prediction-file SHA, Kaggle quota check, official history deduplication and submission-ID recovery.

## Audit trail

- Read-only V2.9.24 diagnostic: https://github.com/mjw3177109/kaggletest/actions/runs/39925902649
- Original V2.9.24 status: `state/v2_9_24_status.json` (left unchanged)
- Independent CV and dependency-drift evidence: `state/v2_9_24_readonly_audit.json`
- Official frozen V2.9.15: submission **57446286**, Public LB **0.96099**. No new submission or Private LB has been verified.

**Next:** implement a separate guarded V2.9.25 evaluator only after verifying the owner slug and current workflow state. Do not start a new training job to reproduce existing OOF data.
