# V2.9.14 — Dual-representation RealMLP Ens8 (independent 3fold, preregistered)

This one-shot independent training design is registered before V2.9.14 sees OOF/validation labels or exists as a Kaggle Notebook.

## Objective and separation
- Competition: `playground-series-s6e10`, threefold `StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)`, canonical 699635 train and 299844 test rows, ROC AUC.
- Current **official** champion V2.9.13: Public LB **0.96083**, Kaggle submission **57423879** (2026-10-20T05:30:37 UTC). Current CV champion V2.9.13 = **0.96117635576833**.
- Different neural feature representation vs V2913: original numerical predictors retain float copies AND gain categorical copies (train-fold-fitted vocabulary), original categorical inputs are retained, fold-train-only count encoding for four categorical columns and `Leg room service`, and predeclared categorical combos plus inner 5fold TE on combo keys. No external data or test-label information. All fit maps trained strictly on outer training fold; no outer fold validation label used by the learner or its early-stopping.
- One fixed RealMLP-TD model (pytabkit==1.7.3) per fold with exactly the successful V2913 Ens8 schedule, max T4 GPU session 3200 seconds, 3 epochs, batch_size 256, no early stopping, n_ens=8, `flat_anneal` LR and `cos_log_15` WD. Full OOF and test prediction required.
- **Unique** notebook `mjw3177109/kas-v2914-dual-numeric-realmlp-t4-s6e10`; T4 required. Only one initial push; subsequent operations recover original ID only. Authenticate positive existence and owner list; fail closed on ambiguous Kaggle errors. No Kaggle competition submission by this notebook/controller.

## Single frozen blend
- Reconstruct official V2913 fixed challenger with Kaggle archived source hashes and exact sample IDs: `v299 = .60*v293 + .20*v296 + .15*v297 + .05*v294` (each component individually rank normalized), then `v2913=.85*rank(v299)+.15*rank(neural_ens8)`.
- Challenging candidate `v2914 = .90*rank(v2913)+.10*rank(new_dual_RealMLP)`. All weights chosen now, BEFORE seeing any V2914 OOF.
- Promotion requirements: complete threefold overall CV **>=0.96123635576833**, gain >=0.00006 vs V2913, >=2 folds improved, worst fold improvement > -0.00005. Reproduce V299 CV 0.9610561936366048 and V2913 CV 0.96117635576833; match V2913 prior candidate test-prediction SHA.
- Mark `candidate_ready_unsubmitted` only after meeting gate, otherwise `skipped_no_cv_gain`. Only an independently reserved, SHA-verified, quota-checked one-shot workflow can later submit any promoted candidate. Never auto-submit from training.
- On Kaggle GPU unavailable/queued/failed, preserve exact stage and evidence; no duplicate training or invented CV.

## Provenance and inspiration
Independent representation hypothesis motivated by public RealMLP dual-numeric representation experiments (https://github.com/alvis1111/s6e10-airline-solution/tree/main/independent_repro), implemented using this repository's existing V2913 source and original Kaggle-safe cross-fitting. Other teams' CV / LB are NOT used as our fold-wise evidence.
