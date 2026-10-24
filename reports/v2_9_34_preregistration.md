# V2.9.34 — explicit second-order DeepFM on frozen V2915 features (immutable preregistration)

Date: 2026-10-24 UTC. **Pre-registration only; no V2934 Notebook has been created, no V2934 training, no submission.** This document is committed before implementation and before observing any V2934 score. Do not change the frozen architecture, folds, seed, or fusion after observing a result.

## Verified baseline and V2933 failure

Only submitted champion: V2915 RealMLP, official Kaggle submission **57446286**, authenticated Public LB **0.96099**, full threefold OOF CV **0.9612850399332336**, champion test prediction SHA256 **906185547dcd5f0cac3ac173de644fd9997f20ac863c6745bcb1fd3e9a89e5f7**. Latest authenticated Public rank at preregistration: **977/2915**, checked 2026-10-24 20:15:14 UTC; not Private rank.

V2933's unique Notebook `mjw3177109/kas-v2933-pairwise-ranknet-mlp-t4-s6e10` authenticated **COMPLETE**, recovered with the pre-existing durable slug and no new push, Actions **#40128638628** success. Independent threefold OOF CV **0.957190461399015**; frozen 85/15 candidate CV **0.9611521544050623**; paired gain **-0.00013288552817125598**; fold gains **[-0.00015233592694940956,-0.00012001750662271604,-0.00011193011886700965]**, zero positive folds. Raw source NPZ SHA256 **8ad08e8083ae85d5ab235b3003ab64cda0d89f43f21b074f14c58eb0532be226**; candidate OOF SHA256 **3493c4208e5e07f514a8c28c0b9ecd5ac2d48441c4aff0adfe22714e090fb7ca**; candidate test SHA256 **89cd6aa424070fe4930870926193bbb80a41874efaa5baa868a8b5a9f93f1ab9**. Full fold membership, IDs, source SHA, T4 and environment checked by the strict evaluator. V2933 is permanently `skipped_no_cv_gain`, never retrain, push or submit. The fixed RankNet+BCE setup underperformed, but this does not establish which architectural or loss component caused the gap.

## Genuinely different V2934 hypothesis — explicit pairwise feature interactions

Instead of another generic RealMLP, FT-Transformer, RankNet or supervised feature expansion, test **DeepFM with a factorization-machine second-order term** to model explicit pairwise interactions between all original fields. This changes the inductive bias: an analytically computed FM interaction sum, not a pairwise ranking loss and not a Transformer attention block. Improvement is a hypothesis, not a claim.

**Immutable input and preprocessing.** Use precisely V2915's original 37 ordered features (23 numeric, 14 categorical), including exactly the original 3 inner-5fold cross-fitted TE columns. Outer 3fold `StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)` on the exact 699635 training IDs and 299844 test IDs. All numeric mean/std, categorical vocabulary, missing treatment, and TE mappings must be fit exclusively on the outer training fold; training-row TE is inner-5fold OOF. No new engineered features, PCA, route profiles, extra TE, service thresholds, MI pruning, subgroup experts, external labels, pseudo-labels or test-label access. Preserve canonical column order and numeric/categorical type metadata.

**Frozen DeepFM model.**
- Field embeddings: each of 14 categorical fields has an outer-train-only vocabulary and an 8-dimensional embedding with ID 0 reserved for missing/unknown. Each of 23 standardized numeric fields has a learned 8-dimensional vector multiplied by its standardized value. Thus 37 field vectors of dimension 8 per row.
- First-order term: one trainable scalar coefficient per numeric field, one scalar embedding per categorical field, and one global bias.
- Second-order FM term: `0.5 * sum_k((sum_i v_ik)^2 - sum_i(v_ik^2))`, summed over the 8 embedding dimensions, using the same 37 field vectors. No attention or RankNet pairs.
- Deep term: flatten the 37x8 field vectors, pass through fully connected 128 -> 64, GELU after each hidden layer, dropout 0.10 after each hidden layer, then scalar output. Final logit = first-order + FM + deep.
- Loss: `BCEWithLogitsLoss` on original binary labels; AdamW lr=0.001, weight_decay=0.0001; batch_size=1024, 5 epochs, no early stopping, no checkpoint/seed/hyperparameter search, no ensemble. Model seed = **20261103 + outer_fold**. Authenticated NVIDIA Tesla T4 required; fail closed if unavailable. Record exact PyTorch/CUDA/NumPy/Pandas/scikit-learn versions, runtime, and all source hashes.

**Frozen candidate:** `0.85*rank(V2915 OOF) + 0.15*rank(V2934 OOF)` and same weights on test predictions; no weight search or post-hoc tuning.

## Evidence, safety, and promotion

Implement separate trainer, strict read-only OOF validator, one-owner Kaggle Notebook gate, isolated unit tests, and GitHub Actions only after this preregistration is committed. Validate complete 3fold OOF, each fold AUC and membership hash, exactly aligned train/test IDs and targets, 37 feature names in order, fold-local preprocessing/TE, per-fold seeds, raw NPZ and source SHA, champion test SHA, candidate OOF/test SHA, T4 and dependency versions. Numeric NPZ arrays only; `allow_pickle=False`; no unsafe pickle execution. Evaluator **never submits**.

Promote only for *consideration* of a separate one-shot submission when **full candidate CV >= 0.96125635576833**, **gain vs champion >= +0.00005**, **at least two positive fold gains**, and **worst fold gain strictly > -0.00005**. A later one-shot submission requires a distinct durable reservation, exact candidate test SHA, Kaggle quota and official submission-history dedup, submission ID recovery, and deadline/rules check. CV is never official Public LB or rank.

**Sole reserved, not created Notebook slug:** `mjw3177109/kas-v2934-explicit-secondorder-deepfm-t4-s6e10`. Before its single creation, re-read HEAD, all five active Actions statuses across the entire repository, authenticated Kaggle owner and all other running/queued kernels, current scored submission and competition deadline **2026-10-31 23:59 UTC**. Persist a unique durable owner reservation before the one permitted push. If a slug exists, owner is ambiguous, or any gate fails: recover-only or stop; never repush, retrain old experiments, bypass protection, auto-accept rules or expose tokens. If the new method fails, diagnose and record before proposing another genuinely different experiment.
