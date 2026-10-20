"""V2.9.13 RealMLP 8-ensemble test safety, no Kaggle GPU needed."""
from __future__ import annotations
import ast
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import v2913_realmlp_ens8_train as model
import v2913_pipeline as pipeline

def fixture(n=32):
    i=np.arange(n)
    f=pd.DataFrame({
        "id":i,"Gender":np.where(i%2,"Male","Female"),
        "Customer Type":np.where(i%3,"Loyal Customer","disloyal Customer"),
        "Type of Travel":np.where(i%2,"Business travel","Personal Travel"),
        "Class":np.where(i%4,"Eco","Business"),
        "Age":i%55+18,"Flight Distance":(i%1500)+150,
        "Departure Delay in Minutes":i%33,
        "Arrival Delay in Minutes":i%25,"satisfaction":i%2
    })
    for field in ("Inflight wifi service","Online boarding","Seat comfort",
                  "Inflight entertainment","Leg room service"):
        f[field]=(i%5).astype(float)
    return f

class RealMLPEns8Safety(unittest.TestCase):
    def test_feature_matrix_is_fold_train_only_and_label_free(self):
        src=fixture(32)
        train=src.iloc[:21]
        val=src.iloc[21:]
        a,b,k,z=model.engineer(train,val)
        self.assertEqual(list(a.columns),list(b.columns))
        self.assertEqual(len(a),21)
        self.assertEqual(len(b),11)
        self.assertEqual(k.shape,(21,3))
        self.assertEqual(z.shape,(11,3))
        self.assertNotIn("satisfaction",a.columns)
        self.assertNotIn("id",a.columns)
        before=a.copy()
        modified=train.copy()
        modified["satisfaction"]=1-modified["satisfaction"]
        again,_,_,_=model.engineer(modified,val)
        pd.testing.assert_frame_equal(before,again)

    def test_fixed_config_and_cv_critical_guards(self):
        src=(ROOT/"kaggle_agent/v2913_realmlp_ens8_train.py").read_text()
        controller=(ROOT/"kaggle_agent/v2913_pipeline.py").read_text()
        ast.parse(src);ast.parse(controller)
        for value in ('n_ens=8','n_epochs=3','stop_epoch=3','batch_size=256',
                      'lr_sched="flat_anneal"','wd_sched="cos_log_15"',
                      'p_drop_sched="invsqrtp1e-3"'):
            self.assertIn(value,src)
        self.assertIn("model.fit(xa, pd.Series(ytr))",src)
        self.assertNotIn("model.fit(xa, pd.Series(ytr), xv",src)
        self.assertIn('"enable_gpu":True',controller)
        self.assertIn('NvidiaTeslaT4',controller)
        self.assertNotIn('"submit"',controller)
        self.assertNotIn('"submit"',src)

    def test_fixed_blend_and_folds(self):
        rng=np.random.default_rng(77)
        y=np.array([0,1]*90)
        p=[np.clip(.2+.45*y+rng.normal(0,x,len(y)),0,1)
           for x in (.30,.24,.23,.25,.08)]
        a,b,c,d,new=p
        r=pipeline.audit.rank
        old=.60*r(a)+.20*r(b)+.15*r(c)+.05*r(d)
        score=float(roc_auc_score(y,old))
        with patch.object(pipeline,"BASELINE",score):
            result=pipeline.compare(y,a,b,c,d,new)
        self.assertEqual(len(result["fold_gain"]),3)
        self.assertEqual(result["positive_folds"],3)
        self.assertGreater(result["gain"],0)
        self.assertAlmostEqual(sum(result["fixed_weights"].values()),1.0)

    def test_recovery_without_canonical_notebook_refuses_push(self):
        with tempfile.TemporaryDirectory() as d:
            with (patch.object(pipeline,"STATE",Path(d)/"status.json"),
                  patch.object(pipeline,"preflight"),
                  patch.object(pipeline,"cmd") as cli):
                result=pipeline.run(recover_only=True)
                self.assertEqual(result["stage"],"preflight")
                cli.assert_not_called()

    def test_known_owned_kernel_positive_control(self):
        responses=[
          SimpleNamespace(returncode=1,stdout="",stderr="kernels.get denied"),
          SimpleNamespace(returncode=0,stdout="Not found",stderr=""),
          SimpleNamespace(returncode=0,stdout="KernelWorkerStatus.COMPLETE",stderr=""),
        ]
        with patch.object(pipeline,"cmd",side_effect=responses) as client:
            self.assertFalse(pipeline.existing_or_absent())
            self.assertEqual(client.call_count,3)

    def test_ambiguous_owner_list_never_pushes(self):
        responses=[
          SimpleNamespace(returncode=1,stdout="",stderr="permission denied"),
          SimpleNamespace(returncode=1,stdout="",stderr="network error")
        ]
        with patch.object(pipeline,"cmd",side_effect=responses):
            with self.assertRaisesRegex(RuntimeError,"Cannot verify owned"):
                pipeline.existing_or_absent()

if __name__=="__main__":
    unittest.main()
