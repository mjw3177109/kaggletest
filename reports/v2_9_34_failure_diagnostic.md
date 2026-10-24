# V2.9.34 — DeepFM threefold failure diagnostic (immutable result)

Verified 2026-10-24 23:31:44 UTC by authenticated unique Notebook `mjw3177109/kas-v2934-explicit-secondorder-deepfm-t4-s6e10` and strictly read-only recovery Actions #40142698555. The recovery reused the durable owner and did not push a Notebook, retrain or submit.

- Champion V2915 full OOF CV: **0.9612850399332336**; official Public LB **0.96099**, submission **57446286**.
- DeepFM standalone OOF CV: **0.9576654181929295**.
- Frozen 85/15 rank-fusion OOF CV: **0.9611890304161783**; delta **-0.00009600951705523997**.
- Paired fold gains: **[-0.00011359596350091206, -0.00008333370608548663, -0.00008974255795546338]**. Positive folds **0/3**.
- Full threefold membership, IDs, ordered 37 features, original fold-local TE, model seeds, T4 and versions, source SHA and candidate hashes were checked by the strict evaluator. Kaggle Notebook status: **COMPLETE**. Terminal status: **skipped_no_cv_gain**.
- Source NPZ SHA256: `e0a3b343c4aad51f9ca80130e0f7f41dc69298b625778686db37a7cf9ef07d79`; trainer source SHA256: `55e5f7264d2f67e106d991d186eb93b4187256ba8a3270827e18c49855bfc2a2`.
- Runtime: 619.9 seconds on authenticated Tesla T4. Evaluation only: no official submission or LB change.

Interpretation: the frozen explicit second-order DeepFM inductive bias did **not** add beneficial rank diversity under the pre-registered 85/15 fusion. This does not prove the cause (optimization, model capacity, feature interactions or calibration). V2932 attention and V2933 RankNet also had negative gains; avoid rerunning or post-hoc tuning these terminal directions.

Next experiment must be separately pre-registered and must not refit, push, or resubmit any terminal model. A new read-only consensus gate over already independently validated OOF artifacts is a distinct *fusion decision rule* rather than another DeepFM/RankNet/Transformer training run. CV cannot be presented as official Public LB or rank.
