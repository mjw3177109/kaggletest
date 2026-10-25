# V2.9.36 frozen explicit second-order DART original37 threefold OOF

- UTC: 2026-10-25T02:41:58.765908+00:00
- Stage: kernel_failed
- Unique Kaggle Notebook: https://www.kaggle.com/code/mjw3177109/kas-v2936-original37-xgboost-dart-t4-s6e10
- Official V2915 champion CV: 0.9612850399332336
- Independent DART threefold CV: pending
- Fixed 85/15 candidate CV: pending
- Gain: pending
- Fold gains: pending
- Evaluation only: no Kaggle submission. CV is not Public LB.
- Error: {
  "traceback": "Traceback (most recent call last):\n  File \"/kaggle/src/script.py\", line 245, in <module>\n    main()\n    ~~~~^^\n  File \"/kaggle/src/script.py\", line 188, in main\n    hardware=prove_gpu()\n  File \"/kaggle/src/script.py\", line 60, in prove_gpu\n    raise RuntimeError(\"Pinned xgboost==3.0.5 required; got \"+xgb.__version__)\nRuntimeError: Pinned xgboost==3.0.5 required; got 3.4.1\n"
}
