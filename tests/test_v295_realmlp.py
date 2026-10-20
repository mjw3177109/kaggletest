"""V2.9.5 offline validation: fold isolation, non-submission and resource gate."""
import ast
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import v295_realmlp_pilot_train as train
import v295_realmlp_pilot as pilot
from v291_ledger import write


class StrictRealMLPTests(unittest.TestCase):
    def sample_data(self):
        n=36
        frame=pd.DataFrame({
            "Gender":["Male","Female"]*(n//2),
            "Customer Type":["Loyal Customer","disloyal Customer"]*(n//2),
            "Type of Travel":["Business travel","Personal Travel"]*(n//2),
            "Class":["Business","Eco"]*(n//2),
            "Age":np.arange(n)+20,
            "Flight Distance":np.repeat([300,400,500],12),
            "Inflight wifi service":np.arange(n)%6,
            "Online boarding":np.arange(n)%6,
            "Seat comfort":np.arange(n)%6,
            "Inflight entertainment":np.arange(n)%6,
            "Leg room service":np.arange(n)%6,
            "Departure Delay in Minutes":np.arange(n)%3,
        })
        return frame

    def test_engineering_is_train_only_and_val_values_do_not_update_counts(self):
        raw=self.sample_data()
        left=raw.iloc[:30].copy()
        right=raw.iloc[30:].copy()
        right["Class"]="unseen_class"
        xa,xv,te_train,te_val=train.engineer(left,right)
        self.assertEqual(len(xa),30)
        self.assertEqual(len(xv),6)
        self.assertTrue(pd.api.types.is_categorical_dtype(xa["Class"]))
        self.assertTrue(xv["Class"].isna().all())
        self.assertTrue((xv["count_combo_0"]>=0).all())
        self.assertEqual(xa.shape[1],xv.shape[1])
        self.assertEqual(te_train.shape[1],len(train.PAIRS))
        self.assertEqual(te_val.shape[1],len(train.PAIRS))

    def test_gpu_only_no_submit_and_outer_val_not_passed_to_fit(self):
        source=(ROOT/"kaggle_agent/v295_realmlp_pilot_train.py").read_text(encoding="utf-8")
        controller=(ROOT/"kaggle_agent/v295_realmlp_pilot.py").read_text(encoding="utf-8")
        ast.parse(source)
        ast.parse(controller)
        self.assertIn('model.fit(xa, pd.Series(ytr))',source)
        self.assertIn('val_fraction=0.0',source)
        self.assertIn('use_early_stopping=False',source)
        self.assertIn('sklearn',controller)
        self.assertNotIn('kaggle", "competitions", "submit"',controller)
        self.assertNotIn('"kaggle","competitions","submit"',controller)
        self.assertIn('refusing CPU fallback',source)
        self.assertIn('StratifiedKFold(FOLDS,shuffle=True,random_state=SEED)',source)

    def test_gate_fails_closed_when_previous_gpu_queued(self):
        with tempfile.TemporaryDirectory() as d:
            previous=pilot.ROOT
            pilot.ROOT=Path(d)
            try:
                write(pilot.ROOT/"state/dual_ladder.json",{
                    "active_competition":pilot.SLUG,
                    "active_deadline":"2026-10-31T23:59:00Z"})
                write(pilot.ROOT/"state/v2_9_4_status.json",{"stage":"kernel_queued"})
                self.assertEqual(pilot.authorized_to_start(),
                                 (False,"waiting_for_v294_to_finish"))
                write(pilot.ROOT/"state/v2_9_4_status.json",{"stage":"skipped_no_cv_gain"})
                with patch.object(pilot, "now", return_value="2026-10-19T00:00:00Z"):
                    ok,reason=pilot.authorized_to_start()
                self.assertTrue(ok,reason)
            finally:
                pilot.ROOT=previous


if __name__=="__main__":
    unittest.main()
