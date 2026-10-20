"""V2.9.10 paired-bootstrap reproducibility and no-remote-mutation tests."""
import ast
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import v210_robustness_audit as m

class PairedAuditTests(unittest.TestCase):
    def test_reproducible_paired_bootstrap(self):
        y=np.array([0,1]*90,dtype=np.int8)
        base=np.arange(len(y),dtype=float)/len(y)
        challenger=.1*base+.9*y
        x=m.matched_bootstrap(y,base,{"better":challenger},replicates=30,seed=999)
        z=m.matched_bootstrap(y,base,{"better":challenger},replicates=30,seed=999)
        self.assertEqual(x,z)
        self.assertGreater(x["better"]["lower_025"],0)
        self.assertEqual(x["better"]["positive_fraction"],1.0)

    def test_equal_prediction_must_not_falsely_improve(self):
        y=np.array([0,1]*60,dtype=np.int8)
        predictions=np.linspace(.1,.9,len(y))
        a=m.matched_bootstrap(y,predictions,{"same":predictions},replicates=20,seed=9)
        self.assertAlmostEqual(a["same"]["lower_025"],0)
        self.assertEqual(a["same"]["positive_fraction"],0)

    def test_subgroup_aggregate_only(self):
        n=120
        y=np.array([0,1]*60,dtype=np.int8)
        frame=pd.DataFrame({
            "Type of Travel":["Business"]*60+["Personal"]*60,
            "Class":["Eco"]*n,
            "Customer Type":["Loyal"]*n,
            "Gender":["Male"]*n,
        })
        previous=np.linspace(0,1,n)
        candidate=.3*previous+.7*y
        results=m.subgroup_audit(frame,y,previous,candidate,min_group=10)
        self.assertEqual(len(results),5)
        self.assertTrue(all(v["rows"]>=10 for v in results))
        self.assertTrue(all("id" not in row for row in results))

    def test_no_training_or_submission_controls(self):
        source=(ROOT/"kaggle_agent/v210_robustness_audit.py").read_text(encoding="utf-8")
        ast.parse(source)
        self.assertNotIn('"push"',source)
        self.assertNotIn('"submit"',source)
        self.assertIn("REPLICATES=120",source)
        self.assertIn("SEED=20261020",source)

if __name__=="__main__":
    unittest.main()
