"""Guard against premature RealMLP submission and quota/duplicate errors."""
import sys
import tempfile
import unittest
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import v296_submit as sub
from v291_ledger import read,write


class RealMLPSubmissionTests(unittest.TestCase):
    def test_gate_only_complete_cv_with_robust_folds(self):
        sample={"stage":"candidate_ready_unsubmitted","submitted":False,
           "submission_attempted":False,"hardware":{"ok":True},
           "metrics":{"gpu_verified":True},
           "comparison":{"pre_registered_weights":{"v293":.8,"realmlp":.2},
                         "delta":.000198,"positive_folds":3,
                         "fold_gain":[.00020,.00015,.00025]}}
        with patch.object(sub,"competition_open"):
            sub.verify_gate(sample)
            sample["comparison"]["fold_gain"]=[.0002,.0003,-.0001]
            with self.assertRaisesRegex(RuntimeError,"gate"):
                sub.verify_gate(sample)
            sample["comparison"]["fold_gain"]=[.0002,.0003,.00025]
            sample["submission_attempted"]=True
            with self.assertRaisesRegex(RuntimeError,"gate"):
                sub.verify_gate(sample)

    def test_uncommitted_reservation_refuses_submission(self):
        with tempfile.TemporaryDirectory() as td:
            prior=sub.STATUS
            sub.STATUS=Path(td)/"status.json"
            try:
                write(sub.STATUS,{"submission_note":"KAS V296 REALMLP test",
                                  "submission_attempted":True,"stage":"submission_reserved"})
                with patch.object(sub,"competition_open"),patch.object(sub,"history",return_value=[]):
                    with self.assertRaisesRegex(RuntimeError,"durably committed"):
                        sub.submit()
            finally:
                sub.STATUS=prior

    def test_requires_explicit_dates_for_quota(self):
        day=datetime.now(timezone.utc).isoformat()
        with patch.object(sub,"now",return_value=datetime.now(timezone.utc)):
            self.assertTrue(sub.quota([{"date":day}]))
            with self.assertRaisesRegex(RuntimeError,"date missing"):
                sub.quota([{"date":None}])


if __name__=="__main__":
    unittest.main()
