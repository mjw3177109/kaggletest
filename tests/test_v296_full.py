"""V2.9.6 strict CV, fixed ensemble and no double Kaggle submissions."""
import ast
import sys
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import v296_pipeline as c
from v291_ledger import write


class FullRealMLPTests(unittest.TestCase):
    def test_notebook_source_does_true_full_threefold(self):
        s=(ROOT/"kaggle_agent/v296_realmlp_full_train.py").read_text(encoding="utf-8")
        ast.parse(s)
        self.assertIn('for fold,(itr,iva) in enumerate(fold_splits):',s)
        self.assertIn('oof=np.full(len(y),np.nan',s)
        self.assertIn('np.savez_compressed(OUTPUT/"v296_realmlp_oof.npz"',s)
        self.assertIn('test_prediction+=pt/FOLDS',s)
        self.assertIn('model.fit(xa, pd.Series(ytr))',s)
        self.assertIn('enc.fit_transform(te_train, ytr)',s)
        self.assertIn('use_early_stopping=False',s)
        self.assertNotIn('model.fit(xa, pd.Series(ytr), xv',s)
        self.assertNotIn('"kaggle","competitions","submit"',s)

    def test_controller_uses_real_validation_and_no_submit(self):
        s=(ROOT/"kaggle_agent/v296_pipeline.py").read_text(encoding="utf-8")
        ast.parse(s)
        self.assertIn('"validation_mismatch"',s)
        self.assertIn('TRIAL_GAIN=0.00006',s)
        self.assertIn('0.8*rankdata(base)',s.replace('.8*rankdata(base)','0.8*rankdata(base)'))
        self.assertIn('min(fold_gain)>-0.00005',s)
        self.assertIn('realmlp_oof.npz',s)
        self.assertIn('ready_unsubmitted',s)
        self.assertNotIn('"kaggle","competitions","submit"',s)

    def test_requires_real_pilot_and_competition_active(self):
        prior=c.ROOT
        with tempfile.TemporaryDirectory() as d:
            c.ROOT=Path(d)
            try:
                write(c.ROOT/"state/dual_ladder.json",{
                    "active_competition":c.SLUG,
                    "active_deadline":"2026-10-31T23:59:00Z"})
                write(c.ROOT/"state/v2_9_4_status.json",{"stage":"skipped_no_cv_gain"})
                with self.assertRaisesRegex(RuntimeError,"positive"):
                    c.eligible()
                write(c.ROOT/"state/v2_9_5_realmlp_status.json",{
                    "stage":"pilot_evaluated","hardware":{"ok":True},
                    "comparison":{"paired_delta":0.0001982,"single_fold_only":True}})
                with patch.object(c,"now",return_value=datetime(2026,10,19,13,tzinfo=timezone.utc)):
                    c.eligible()
                write(c.ROOT/"state/v2_9_4_status.json",{"stage":"kernel_running"})
                with patch.object(c,"now",return_value=datetime(2026,10,19,13,tzinfo=timezone.utc)):
                    with self.assertRaisesRegex(RuntimeError,"not terminal"):
                        c.eligible()
            finally:
                c.ROOT=prior


if __name__=="__main__":
    unittest.main()
