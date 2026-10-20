"""V2.9.9 fixed-teacher OOF math and no-submit regression tests."""
import ast
import sys
import unittest
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import v299_teacher_fusion as m

class TeacherComplementSafety(unittest.TestCase):
    def test_pre_registered_weights_and_source(self):
        self.assertEqual(m.WEIGHTS,{"v293":.60,"realmlp":.20,
                                     "conditional":.15,"teacher_v294":.05})
        self.assertAlmostEqual(sum(m.WEIGHTS.values()),1.0)
        self.assertEqual(m.MIN_GAIN,.00010)
        source=(ROOT/"kaggle_agent/v299_teacher_fusion.py").read_text(encoding="utf-8")
        ast.parse(source)
        self.assertNotIn('"push"',source)
        self.assertNotIn('"submit"',source)
        self.assertIn('"kernels","output"',source)

    def test_four_model_positive_complement(self):
        rng=np.random.default_rng(23)
        n=180
        y=np.array([0,1]*90,dtype=np.int8)
        base=np.clip(.2+y*.4+rng.normal(0,.3,n),0,1)
        real=np.clip(.2+y*.4+rng.normal(0,.23,n),0,1)
        cond=np.clip(.1+y*.75+rng.normal(0,.08,n),0,1)
        teacher=np.clip(.05+y*.9+rng.normal(0,.05,n),0,1)
        old=.8*m.p.rank(base)+.2*m.p.rank(real)
        result=m.fuse_fixed(y,base,real,cond,teacher,float(roc_auc_score(y,old)))
        self.assertEqual(result["stage"],"candidate_ready_unsubmitted")
        self.assertEqual(result["positive_folds"],3)
        self.assertTrue(all(g>0 for g in result["fold_gain"]))
        self.assertGreaterEqual(result["gain"],m.MIN_GAIN)
        self.assertFalse(result["submitted"])

    def test_redundant_signal_does_not_pass(self):
        y=np.array([0,1]*90,dtype=np.int8)
        base=np.linspace(0,1,len(y))
        old=float(roc_auc_score(y,m.p.rank(base)))
        result=m.fuse_fixed(y,base,base,base,base,old)
        self.assertEqual(result["stage"],"skipped_no_cv_gain")
        self.assertLess(result["gain"],m.MIN_GAIN)
        self.assertFalse(result["submitted"])

    def test_invalid_incumbent_blocks(self):
        y=np.array([0,1]*90,dtype=np.int8)
        base=np.linspace(0,1,len(y))
        with self.assertRaisesRegex(ValueError,"Official V296"):
            m.fuse_fixed(y,base,base,base,base,.5)

if __name__=="__main__":
    unittest.main()
