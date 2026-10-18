"""Offline V2.7 checks: identical row order, fixed weight and no submission spam."""
from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"kaggle_agent"))
from v27_blend_math import model_oof, paired_fusion
from v27_oof_blender import daily_count, safe_to_make_first_submission


class V27Tests(unittest.TestCase):
    def samples(self):
        y=np.tile([0,1],50)
        rng=np.random.default_rng(140)
        a=np.where(y==1,0.6,0.3)+rng.normal(0,.12,len(y))
        b=np.where(y==1,0.55,0.35)+rng.normal(0,.13,len(y))
        return y,np.clip(a,0,1),np.clip(b,0,1)

    def test_saved_rank_model_is_reproduced(self):
        y,a,b=self.samples()
        data={"oof_lgb":a,"oof_xgb":b}
        selected,_=model_oof(data,"blend_rank")
        self.assertTrue(np.allclose(selected,0.5*(rankdata(a)+rankdata(b))/len(a)))
        for kind in ("lightgbm","xgboost","blend_mean","blend_rank"):
            self.assertEqual(len(model_oof(data,kind)[0]),len(y))

    def test_fixed_oof_fusion_is_valid(self):
        y,a,b=self.samples()
        result=paired_fusion(y,y,a,b,a[:28],b[:28])
        self.assertAlmostEqual(result["score25"],roc_auc_score(y,a))
        self.assertAlmostEqual(result["score26"],roc_auc_score(y,b))
        self.assertTrue(0<=result["fused_cv_auc"]<=1)
        self.assertEqual(len(result["fused_test"]),28)
        self.assertAlmostEqual(float(result["fused_test"].mean()),0.5+0.5/28)

    def test_label_mismatch_refused(self):
        y,a,b=self.samples()
        mutated=y.copy();mutated[0]=1-mutated[0]
        with self.assertRaisesRegex(ValueError,"OOF targets differ"):
            paired_fusion(y,mutated,a,b,a[:28],b[:28])

    def test_mismatched_prediction_rows_refused(self):
        y,a,b=self.samples()
        with self.assertRaises(ValueError):
            paired_fusion(y,y,a[:-1],b,a[:28],b[:28])
        with self.assertRaises(ValueError):
            paired_fusion(y,y,a,b,a[:28],b[:27])

    def test_invalid_weights_never_searched(self):
        y,a,b=self.samples()
        result=paired_fusion(y,y,a,b,a[:20],b[:20])
        expected=(rankdata(a)+rankdata(b))/(2*len(a))
        self.assertAlmostEqual(result["fused_cv_auc"],roc_auc_score(y,expected))

    def test_never_repeat_unknown_or_existing_submission(self):
        self.assertTrue(safe_to_make_first_submission({},None))
        self.assertFalse(safe_to_make_first_submission({},{"ref":456}))
        for stage in ("submitting","submission_pending","submission_failed",
                      "submission_unverified","submission_ambiguous"):
            self.assertFalse(safe_to_make_first_submission({"stage":stage},None))
        self.assertFalse(safe_to_make_first_submission({"submission_attempted":True},None))
        self.assertFalse(safe_to_make_first_submission({"submitted":True},None))
        self.assertTrue(safe_to_make_first_submission({"stage":"quota_wait"},None))

    def test_utc_daily_quota(self):
        rows=[
            {"date":"2026-10-18T00:01:00+00:00"},
            {"date":"2026-10-17T23:59:00+00:00"},
            {"date":"2026-10-18T16:00:00+08:00"},
        ]
        self.assertEqual(daily_count(rows,date(2026,10,18)),2)
        with self.assertRaisesRegex(RuntimeError,"date unverifiable"):
            daily_count([{"date":"unknown"}],date(2026,10,18))


if __name__=="__main__":
    unittest.main()
