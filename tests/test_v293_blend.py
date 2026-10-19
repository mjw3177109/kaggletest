"""Offline safety and arithmetic tests for the real V2.9.3 experiment."""
import hashlib
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"kaggle_agent"))
import v293_blend_math as m
import v293_pipeline as p
from v291_ledger import read, write


class FusionTests(unittest.TestCase):
    def test_fixed_candidates_gate_and_test_file_shape(self):
        old_train,old_test=m.N_TRAIN,m.N_TEST
        m.N_TRAIN,m.N_TEST=100,24
        try:
            rng=np.random.default_rng(42)
            y=np.array([0,1]*50)
            good=y*0.7+rng.random(100)*0.3
            oofs={"v25":good,"v28":good+np.linspace(0,.0001,100),
                  "xgb":good,"cat":good}
            tests={"v25":np.linspace(.01,.99,24),"v28":np.linspace(.05,.95,24)}
            result,pred=m.evaluate(y,oofs,tests,benchmark=.95)
            self.assertEqual(result["stage"],"candidate_ready")
            self.assertEqual(len(pred),24)
            self.assertEqual(len(result["candidates"]),4)
            self.assertEqual(sum(row.get("status")=="unavailable_test_predictions"
                                 for row in result["candidates"]),2)
            skipped,empty=m.evaluate(y,oofs,tests,benchmark=1.0)
            self.assertEqual(skipped["stage"],"skipped_no_cv_gain")
            self.assertIsNone(empty)
        finally:
            m.N_TRAIN,m.N_TEST=old_train,old_test

    def test_reconstruct_and_cross_source_comparability(self):
        old_train,old_test=m.N_TRAIN,m.N_TEST
        m.N_TRAIN,m.N_TEST=80,14
        try:
            y=np.array([0,1]*40)
            a=y*.75+np.linspace(.01,.05,80)
            b=y*.7+np.linspace(.02,.06,80)
            npz25={"target":y,"oof_lgb":a,"oof_xgb":b}
            npz28={"target":y.copy(),"oof_lgb":a,"oof_cat":b}
            label="StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)"
            score=m.__dict__["roc_auc_score"](y,(m.rank(a)+m.rank(b))/2)
            def status(stage,submitted=False):
                return {"competition":"playground-series-s6e10","stage":stage,
                        "submitted":submitted,"hardware":{"ok":True},
                        "metrics":{"validation":label,"best_model":"blend_rank",
                                   "gpu_verified":True,"leakage_safe_target_encoding":True,
                                   "train_rows":80,"test_rows":14,"best_cv_auc":score}}
            frame=pd.DataFrame({"id":np.arange(14),"satisfaction":np.linspace(.1,.9,14)})
            yy,oofs,tests=m.verify_sources(npz25,npz28,status("complete",True),
                                           status("skipped_no_cv_gain"),frame,frame.copy())
            self.assertTrue(np.array_equal(yy,y))
            self.assertEqual(set(tests),{"v25","v28"})
            wrong=frame.copy()
            wrong["id"]=wrong["id"]+1
            with self.assertRaisesRegex(ValueError,"identifiers"):
                m.verify_sources(npz25,npz28,status("complete",True),
                                 status("skipped_no_cv_gain"),frame,wrong)
        finally:
            m.N_TRAIN,m.N_TEST=old_train,old_test


class SubmissionSafetyTests(unittest.TestCase):
    def test_requires_persisted_reservation_and_verifies_csv_digest(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            work=root/"tmp"/"kas_v293"
            work.mkdir(parents=True)
            data=b"id,satisfaction\n1,0.7\n"
            (work/"v293_submission.csv").write_bytes(data)
            original=(p.STATE,p.ROOT,p.LEDGER,p.REPORT)
            p.ROOT=root
            p.STATE=root/"state/v2_9_3_status.json"
            p.LEDGER=root/"state/v2_9_experiments.json"
            p.REPORT=root/"reports/v2_9_3_latest.md"
            try:
                write(p.STATE,{"stage":"candidate_ready","submitted":False,
                      "incumbent_lb":.96,"benchmark_cv":.9600,
                      "candidate_file_sha256":hashlib.sha256(data).hexdigest(),
                      "submission_note":"KAS V293 BLEND test",
                      "result":{"stage":"candidate_ready","cv_auc":.961,"candidate":"test",
                                "weights":{"v25":.6,"v28":.4}}})
                today=datetime.now(timezone.utc).isoformat()
                with patch.object(p,"allowed_competition"),\
                     patch.object(p,"baseline",return_value=({}, {}, .96, .95)),\
                     patch.object(p,"records",return_value=[
                       {"date":today,"description":"prior","ref":100}]):
                    p.do_reserve(work)
                    self.assertTrue(read(p.STATE)["submission_attempted"])
                    self.assertEqual(read(p.STATE)["stage"],"submission_reserved")
                    with self.assertRaisesRegex(RuntimeError,"Uncommitted"):
                        p.do_submit(work)
                    self.assertEqual(read(p.STATE)["stage"],"submission_reserved")
            finally:
                p.STATE,p.ROOT,p.LEDGER,p.REPORT=original

    def test_daily_quota_fails_closed(self):
        day=datetime(2026,10,19,tzinfo=timezone.utc).date()
        with self.assertRaisesRegex(RuntimeError,"date unknown"):
            p.count_today([{"date":None}],day)
        dates=[{"date":"2026-10-19T12:00:00+00:00"}]*6
        self.assertEqual(p.count_today(dates,day),6)


if __name__=="__main__":
    unittest.main()
