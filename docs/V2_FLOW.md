# Kaggle Auto Scientist V2

V2 is the first end-to-end competition runner.

## Flow

```text
Kaggle auth
  -> list open Playground / Getting Started competitions
  -> rank/select one competition
  -> fetch description/evaluation/data pages
  -> check whether the account has joined/accepted rules
  -> download competition data
  -> profile train/test/sample_submission
  -> infer single target column
  -> generate reproducible sklearn baseline
  -> generate Kaggle kernel metadata
  -> push and run the kernel on Kaggle
  -> poll kernel status
  -> download submission.csv + metrics.json
  -> submit to the competition
  -> fetch latest submission/score
  -> write analysis, state, code and score back to GitHub
```

## Human-only boundary

The automation never clicks or accepts competition rules for the user. If Kaggle reports `userHasEntered=false`, the run stops with `stage=rules_required`. Open the selected competition, click **Join Competition**, accept the rules, then rerun the V2 workflow.

## Current baseline scope

V2 intentionally targets conventional tabular competitions with `train.csv`, `test.csv`, `sample_submission.csv`, and one train-only target column. More complex CV/NLP/time-series/code competitions should be handled by later adapters instead of forcing a bad generic baseline.
