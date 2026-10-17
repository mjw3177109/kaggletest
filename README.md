# Kaggle Auto Scientist

This repository is an automated Kaggle competition lab.

## Current version: V2

V2 can authenticate with Kaggle, find currently-open competitions, automatically prefer Playground / Getting Started tasks, inspect competition pages, download data after you have accepted the competition rules, profile the dataset, generate a baseline, run it as a Kaggle Kernel, submit `submission.csv`, collect the latest score, and write the run artifacts back to this GitHub repository.

### Required GitHub Secret

`KAGGLE_API_TOKEN`

Optional: `KAGGLE_USERNAME`. If omitted, the V2 kernel owner falls back to the GitHub repository owner. Add `KAGGLE_USERNAME` only if your Kaggle username is different from your GitHub username.

## Run V2

Open **Actions → Kaggle Auto Scientist V2 → Run workflow**.

The workflow is also triggered automatically when V2 code/config changes.

### If the run stops at `rules_required`

This is expected for a competition you have not joined yet. Open `reports/v2_latest.md`, follow the selected Kaggle competition link, click **Join Competition**, accept the competition rules yourself, then rerun the workflow.

After rules are accepted, V2 continues automatically:

```text
select competition
-> download data
-> analyze data
-> generate baseline
-> Kaggle Kernel training
-> produce submission.csv
-> submit
-> collect score
-> write results back to GitHub
```

## Output

- `state/v2_candidates.json` — ranked open competitions
- `state/v2_selection.json` — selected competition
- `state/v2_status.json` — current V2 stage/status
- `reports/v2_latest.md` — human-readable latest run report
- `competitions/<slug>/problem_analysis.md`
- `competitions/<slug>/data_analysis.md`
- `competitions/<slug>/solution_plan.md`
- `competitions/<slug>/kaggle/main.py`
- `competitions/<slug>/kaggle/kernel-metadata.json`
- `competitions/<slug>/results/metrics.json`
- `competitions/<slug>/results/latest_submission.json`

Raw competition data, credentials, models, and large `submission.csv` files are intentionally excluded from Git.
