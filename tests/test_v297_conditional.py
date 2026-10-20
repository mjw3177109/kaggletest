"""V2.9.7 conditional model: isolation, registration and Kaggle safety."""
import ast
import sys
import tempfile
import unittest
from pathlib import Path
from datetime import datetime,timezone
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import v297_conditional_cpu_train as tr
import v297_pipeline as p
from v291_ledger import read,write


def example(n=120):
    i=np.arange(n)
    frame=pd.DataFrame({
        "id":i,"Gender":np.where(i%2==0,"Male","Female"),
        "Customer Type":np.where(i%3==0,"Loyal Customer","disloyal Customer"),
        "Type of Travel":np.where(i%2==0,"Business travel","Personal Travel"),
        "Class":np.where(i%5==0,"Business","Eco"),
        "Age":i%55+20,"Flight Distance":i%21+200,
        "Departure Delay in Minutes":i%6,
        "Arrival Delay in Minutes":i%7,
    })
    for k,c in enumerate(tr.SERVICES):frame[c]=((i+k)%6).astype(int)
    return frame


class ConditionalEncodingTests(unittest.TestCase):
    def test_nontrivial_key_count_and_pair_context(self):
        a=example()
        result=tr.keys(a)
        self.assertEqual(result.shape,(len(a),24))
        self.assertIn("pair_1",result)
        self.assertIn("segment_Online boarding",result)
        self.assertTrue(result["pair_1"].str.count(r"\|").eq(3).all())

    def test_fold_only_frequency_never_uses_validation(self):
        training=tr.keys(example(100))
        validation=tr.keys(example(12).assign(**{"Type of Travel":["UNKNOWN"]*12}))
        y=np.array([0,1]*50)
        a,b,c=tr.split_frequencies(training,validation,validation.copy(),y,0)
        self.assertEqual(len(a),100)
        self.assertEqual(len(b),12)
        self.assertEqual(len(c.columns),len(tr.FREQUENCY_COLUMNS))
        self.assertTrue(np.isfinite(a.to_numpy()).all())
        self.assertTrue((b.to_numpy()==0).all())
        # A validation-only category must not affect the training-row counts.
        second=validation.copy()
        second["pair_0"]="BRAND_NEW|VALUE"
        aa,bb,cc=tr.split_frequencies(training,second,validation.copy(),y,0)
        pd.testing.assert_frame_equal(a,aa)

    def test_matrices_do_not_leak_val_targets(self):
        full=example(105)
        y=np.array([0,1]*50)
        a,b,c=tr.matrices(full.iloc[:100],full.iloc[100:],full.iloc[100:].copy(),y,0)
        self.assertEqual(a.shape[0],100)
        self.assertEqual(b.shape[0],5)
        self.assertEqual(c.shape[0],5)
        self.assertEqual(a.columns.tolist(),b.columns.tolist())
        self.assertTrue(np.isfinite(a.to_numpy()).all())

    def test_source_has_no_unsafe_submit_or_outer_fold_es(self):
        source=(ROOT/"kaggle_agent/v297_conditional_cpu_train.py").read_text(encoding="utf-8")
        control=(ROOT/"kaggle_agent/v297_pipeline.py").read_text(encoding="utf-8")
        ast.parse(source)
        ast.parse(control)
        self.assertIn("fit_transform(kt,labels)",source)
        self.assertIn("m.fit(xtr,y[fit])",source)
        self.assertIn('"inner_frequency_folds":5',source)
        self.assertNotIn("eval_set=[(xva",source)
        self.assertNotIn('"kaggle","competitions","submit"',source)
        self.assertNotIn('"kaggle","competitions","submit"',control)

    def test_recovery_refuses_create_and_competition_change(self):
        original=(p.ROOT,p.STATE,p.REPORT)
        with tempfile.TemporaryDirectory() as d:
            p.ROOT=Path(d)
            p.STATE=p.ROOT/"state/v2_9_7_status.json"
            p.REPORT=p.ROOT/"reports/v2_9_7_conditional_cpu.md"
            try:
                write(p.ROOT/"state/dual_ladder.json",{
                    "active_competition":"different-contest",
                    "active_deadline":"2026-10-31T23:59:00Z"})
                with self.assertRaisesRegex(RuntimeError,"Contest focus changed"):
                    p.verify_contest()
                write(p.ROOT/"state/dual_ladder.json",{
                    "active_competition":p.SLUG,
                    "active_deadline":"2026-10-31T23:59:00Z"})
                with patch.object(p,"now",return_value=datetime(2026,10,20,tzinfo=timezone.utc)):
                    self.assertEqual(p.run(recover=True).get("stage"),"preflight")
            finally:
                p.ROOT,p.STATE,p.REPORT=original

if __name__=="__main__":
    unittest.main()
