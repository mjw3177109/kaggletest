# V2.9.19R repaired self-contained TabM-D k8 T4 threefold audit

- UTC: 2026-10-21T14:36:11.520214+00:00
- Stage: kernel_failed
- Notebook: https://www.kaggle.com/code/mjw3177109/kas-v2919r-tabm-selfcontained-t4-s6e10
- Frozen V2915 OOF baseline: 0.9612850399332336
- Verified V2915 official Public LB: 0.96099 (submission 57446286)
- New independent 3fold OOF AUC: pending
- Frozen 85/15 blend 3fold AUC: pending
- Paired fold deltas: pending
- Promotion rule: gain >= 5e-05, >=2 positive folds, worst fold > -0.00005
- Kaggle competition submission: NEVER performed by this evaluator
- Error: {
  "traceback": "Traceback (most recent call last):\n  File \"/kaggle/src/script.py\", line 325, in <module>\n    main()\n    ~~~~^^\n  File \"/kaggle/src/script.py\", line 281, in main\n    pv,pt,elapsed,features=fit_one_fold(train,test,y,itr,iva,fold)\n                           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/kaggle/src/script.py\", line 224, in fit_one_fold\n    xtr, xval, xtest = matrices(\n                       ~~~~~~~~^\n        train.iloc[itr], train.iloc[iva], test, targets[itr], fold,\n        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n    )\n    ^\n  File \"/kaggle/src/script.py\", line 195, in matrices\n    key_train, key_val, key_test = (target_keys(frame, names) for frame in (fit, holdout, test))\n    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/kaggle/src/script.py\", line 195, in <genexpr>\n    key_train, key_val, key_test = (target_keys(frame, names) for frame in (fit, holdout, test))\n                                    ~~~~~~~~~~~^^^^^^^^^^^^^^\n  File \"/kaggle/src/script.py\", line 118, in target_keys\n    if len(z.columns) != TE_COUNT or z.isna().any().any():\n                         ^^^^^^^^\nNameError: name 'TE_COUNT' is not defined\n"
}
