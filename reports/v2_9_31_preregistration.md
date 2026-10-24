# V2.9.31 — fold-local supervised feature pruning of V2915 RealMLP (pre-registration)

Date: 2026-10-24 UTC. **Research proposal only: no V2931 Notebook, GPU training or Kaggle submission has been started.**

## Evidence and V2930 failure analysis

The only Kaggle-submitted champion remains V2.9.15, submission **57446286**, authenticated Public LB **0.96099**, full threefold OOF CV **0.9612850399332336**. Last persisted authenticated Public rank **968/2884**, checked **2026-10-24 13:02:35 UTC**; it is not a fresh Private rank.

V2.9.30 unique Kaggle Notebook `mjw3177109/kas-v2930-conditional-crossfit-te-realmlp-t4-s6e10` was authenticated **COMPLETE** through the strictly guarded **--recover-only** GitHub Actions run **#40104158939**, 2026-10-24 14:50 UTC. Independent 45-feature conditional TE RealMLP threefold OOF CV **0.9606450272778854**; the frozen 85/15 rank fusion CV **0.9613286615476094**; improvement **+0.000043621614375877193**, three paired fold gains **[0.00003326460294306166, 0.00005578342216894239, 0.00004087000739994018]**, 3/3 positive. Although all folds improved, the full gain is below the mandatory +0.00005 threshold; therefore `skipped_no_cv_gain`, **never submit or rerun**. Raw prediction NPZ SHA256 **45e96e595676ec91a1e6c8eea98f975e63c9c0762412b898fe549c0636107362**; candidate OOF SHA256 **9233b274dbd475b11623cc33d31fe9ee64d748fc3f2bae621adece4887236153**; candidate test SHA256 **d73d58723cb08d9104435cf64e4666e0518e0b4b36fb1a7501cce644268c7685**. All source/fold/ID/T4/version checks passed; evaluator did not submit.

**Diagnosis:** the new 8 cross-fitted supervised interaction encodings generated small but stable complementary gains, not enough to promote. Their standalone AUC is below the V2915 champion. These observations alone do not prove the added features are harmful or establish a causal mechanism. Past extra-feature attempts (PCA, route profiles, ordinal thresholds, extra target encodings), travel experts and sample weights have not passed promotion. Do not repeat them or tune blend weights based on observed scores.

## Distinct V2931 hypothesis: remove, rather than add, redundant baseline features

Use exactly the **37 original V2915 features** and original three inner-fivefold cross-fitted TE features; add **zero** new features. On each of the three outer training folds, independently rank **only the 34 non-protected original features** using supervised `sklearn.feature_selection.mutual_info_classif`, fitted to that outer training fold's rows and labels only. Protect the three original target-encoding columns, and deterministically retain the **top 26 of the remaining 34**, yielding exactly **29 input features**. Selection must be recomputed independently within each outer fold, never on full competition train, test labels or outer validation labels.

Freeze selection: `n_neighbors=3`, `random_state=20261031`; fit numeric missing-value medians and categorical vocabularies on outer train only, convert categorical variables to integer codes with discrete_features=True, and impute continuous values with train-only medians. Tie-break exact equal MI scores by the canonical original V2915 feature order. Do not optimize number of features, protected columns, seeds or rankings after viewing results. Original V2915 inner-fivefold TE and preprocessing must remain leakage-safe. Record all 37 original names, three protected TE names, the 29 selected names per fold, MI scores, selection parameters and source SHA in the Kaggle metrics. The 29-column model is **a new representation experiment**, not a retune or rerun of failed feature-addition methods.

## Frozen model and promotion

- Unique reserved Notebook slug: `mjw3177109/kas-v2931-foldlocal-mi-pruned-realmlp-t4-s6e10`. **Not yet created.**
- Same competition rows: 699635 train / 299844 test; `StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)`; source train/test IDs identical to V2915.
- RealMLP-TD T4 `pytabkit==1.7.3`, `n_ens=8`, 5 epochs, `hidden_sizes=[512,256,128]`, `lr=0.038`, `wd=0.015`, `p_drop=0.05`, `n_refit=0`, `val_fraction=0`; independent `MODEL_SEED=20261031+fold`.
- One precommitted candidate only: `0.85*rank(V2915 champion OOF)+0.15*rank(V2931 new OOF)`; same on test. No alternative weight, fold, model or feature count selection.
- Promotion only if full candidate 3-fold CV >= **0.96125635576833**, gain over V2915 >= **+0.00005**, >= 2 positive paired folds, worst paired fold gain strictly > **-0.00005**.
- Verify authentic Kaggle COMPLETE, complete 3-fold OOF, fold membership and AUC replay, train/test IDs, new source SHA, frozen champion prediction hash **906185547dcd5f0cac3ac173de644fd9997f20ac863c6745bcb1fd3e9a89e5f7**, candidate OOF/test hashes, T4 hardware and library versions. Evaluator **NEVER** submits.
- Submission, if promoted, requires a **separate** verified one-shot workflow: durable reservation, prediction SHA, quota, official submission history dedup, and submission-ID recovery. CV is not Public LB and never implies rank.

## Hard launch and recovery policy

First review V2915 engineer/TE implementation and V2930 failure report, implement an independent trainer, verifier, single-owner durable reservation/recover-only pipeline, tests and guarded Actions. Fix deterministic errors before GPU spend. Query whole-repository Actions statuses `in_progress/queued/waiting/requested/pending` and authenticated Kaggle owner/active Notebook status before the one authorized push. If canonical slug exists, **--recover-only**; never duplicate Notebook, push or training. Persist canonical owner slug remotely **before** first push. No external labels, pseudo-labels, manual submission, token disclosure, Kaggle rule acceptance or safety bypass. Deadline **2026-10-31 23:59 UTC**.
