"""V2.9.11 CPU interaction encoding, strict comparison and no-duplicate safeguards."""
import ast
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import v2911_segment_cpu_train as train
import v2911_pipeline as run

def example(n=120):
    i=np.arange(n)
    frame=pd.DataFrame({
        "id":i,"Gender":np.where(i%2==0,"Male","Female"),
        "Customer Type":np.where(i%3==0,"Loyal Customer","disloyal Customer"),
        "Type of Travel":np.where(i%2==0,"Business travel","Personal Travel"),
        "Class":np.where(i%5==0,"Business","Eco"),
        "Age":i%55+20,"Flight Distance":i%700+200,
        "Departure Delay in Minutes":i%45,
        "Arrival Delay in Minutes":i%46,
    })
    for k,col in enumerate(train.SERVICES):
        frame[col]=((i+k)%6).astype(int)
    return frame

class SegmentContrastSafety(unittest.TestCase):
    def test_new_keys_and_cpu_frequency_schema(self):
        a=train.keys(example(120))
        self.assertEqual(a.shape,(120,34))
        self.assertEqual(len(train.FREQUENCY_COLUMNS),10)
        self.assertIn("gap_0",a.columns)
        self.assertIn("gap_class_0",a.columns)
        self.assertIn("delay_distance_context",a.columns)
        self.assertIn("age_distance_context",a.columns)
        self.assertTrue(a.notna().all().all())
        self.assertEqual(len(set(a.columns)),34)

    def test_target_leakage_guard(self):
        src=(ROOT/"kaggle_agent/v2911_segment_cpu_train.py").read_text(encoding="utf-8")
        controller=(ROOT/"kaggle_agent/v2911_pipeline.py").read_text(encoding="utf-8")
        ast.parse(src);ast.parse(controller)
        self.assertIn("fit_transform(kt,labels)",src)
        self.assertIn("m.fit(xtr,y[fit])",src)
        self.assertNotIn("eval_set=[(xva",src)
        self.assertNotIn('"submit"',src)
        self.assertNotIn('"submit"',controller)
        self.assertIn('"enable_gpu":False',controller)
        self.assertIn("--recover-only",controller)

    def test_fixed_model_math(self):
        rng=np.random.default_rng(8)
        y=np.array([0,1]*90,dtype=np.int8)
        base=np.clip(.2+y*.4+rng.normal(0,.30,len(y)),0,1)
        real=np.clip(.2+y*.4+rng.normal(0,.23,len(y)),0,1)
        cond=np.clip(.2+y*.4+rng.normal(0,.21,len(y)),0,1)
        t=np.clip(.2+y*.4+rng.normal(0,.24,len(y)),0,1)
        new=np.clip(.1+y*.8+rng.normal(0,.08,len(y)),0,1)
        rank=run.audit.rank
        old=.60*rank(base)+.20*rank(real)+.15*rank(cond)+.05*rank(t)
        with patch.object(run,"BASELINE",roc_auc_score(y,old)):
            result=run.combine(y,base,real,cond,t,new)
        self.assertAlmostEqual(sum(result["fixed_weights"].values()),1)
        self.assertEqual(len(result["fold_gain"]),3)
        self.assertGreater(result["gain"],0)
        self.assertEqual(result["positive_folds"],3)

    def test_recovery_never_pushes_missing_kernel(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(run,"STATE",Path(d)/"v2911.json"),patch.object(run,"eligible"):
                result=run.run(recover=True)
                self.assertEqual(result.get("stage"),"preflight")

if __name__=="__main__":
    unittest.main()
