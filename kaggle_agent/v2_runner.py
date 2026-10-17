"""Stable entrypoint for Kaggle Auto Scientist V2.

Kaggle competition sources are not guaranteed to mount at
/kaggle/input/<competition-slug>. Patch the generated baseline so it scans the
whole /kaggle/input tree for train.csv/test.csv/sample_submission.csv.
"""

import v2_pipeline as pipeline

OLD_ROOT = 'ROOT = Path("/kaggle/input") / COMPETITION'
NEW_ROOT = 'ROOT = Path("/kaggle/input")'

if OLD_ROOT not in pipeline.BASELINE:
    raise RuntimeError("V2 baseline template changed; input-root compatibility patch must be reviewed")

pipeline.BASELINE = pipeline.BASELINE.replace(OLD_ROOT, NEW_ROOT, 1)

if __name__ == "__main__":
    pipeline.main()
