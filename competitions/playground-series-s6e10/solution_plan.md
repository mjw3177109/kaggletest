# Solution Plan — playground-series-s6e10

## Run 001: reproducible tabular baseline

1. Identify `train.csv`, `test.csv`, and `sample_submission.csv` from the Kaggle competition source.
2. Infer the target as the column present in train but absent from test.
3. Exclude submission ID columns from model features.
4. Impute numeric and categorical missing values independently.
5. Ordinal-encode categorical columns with unknown-category handling.
6. Use HistGradientBoosting for a dependency-light baseline that is available in the Kaggle sklearn image.
7. Hold out validation data and record ROC-AUC for binary classification, accuracy for multiclass, or RMSE for regression.
8. Refit on full training data and write `submission.csv`.
9. V2 pushes the script to Kaggle, waits for completion, downloads output, and submits automatically when rules have already been accepted.

## Next iterations after V2 baseline

- CatBoost with native categorical handling.
- LightGBM / XGBoost comparison.
- Cross-validation and out-of-fold predictions.
- Feature engineering guided by leakage-safe validation.
- Seed/model ensembling only when CV and leaderboard movement agree.
