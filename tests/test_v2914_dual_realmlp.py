"""V2914 independent RealMLP dual numeric feature, OOF and one-shot safety."""
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
import v2914_dual_realmlp_train as model
import v2914_pipeline as controller

def dataset(n=70):
    i=np.arange(n)
    rows=pd.DataFrame({
        "id":i,"Gender":np.where(i%2,"Male","Female"),
        "Customer Type":np.where(i%3,"Loyal Customer","disloyal Customer"),
        "Type of Travel":np.where(i%2,"Business travel","Personal Travel"),
        "Class":np.where(i%4,"Business","Eco"),
        "Age":i%60+18,"Flight Distance":i%1600+150,
        "Departure Delay in Minutes":i%42,
        "Arrival Delay in Minutes":i%46,
        "satisfaction":i%2})
    for j,c in enumerate(("Inflight wifi service","Online boarding","Seat comfort",
                          "Inflight entertainment","Leg room service")):
        rows[c]=(i+j)%6
    return rows

class DualMLPSafety(unittest.TestCase):
    def test_fold_train_only_feature_generation(self):
        src=dataset()
        train,validation=src.iloc[:47],src.iloc[47:]
        x,v,ta,tv=model.engineer(train,validation)
        self.assertEqual(len(x),47)
        self.assertEqual(len(v),23)
        self.assertEqual(list(x.columns),list(v.columns))
        self.assertEqual(ta.shape,(47,3))
        self.assertEqual(tv.shape,(23,3))
        self.assertIn("dual_Age",x)
        self.assertIn("dual_Flight Distance",x)
        self.assertIn("dual_Arrival Delay in Minutes",x)
        self.assertIn("count_original_Gender",x)
        self.assertIn("count_original_Leg room service",x)
        self.assertNotIn("satisfaction",x)
        self.assertNotIn("id",x)
        self.assertGreater(len(x.columns),37)
        assert pd.api.types.is_categorical_dtype(x["dual_Age"].dtype)
        same=train.copy()
        same["satisfaction"]=1-same["satisfaction"]
        left,_,_,_=model.engineer(same,validation)
        pd.testing.assert_frame_equal(x,left)

    def test_frozen_math_and_source_preflight(self):
        with patch.object(controller.audit,"check_competition"):
            controller.preflight()
        rng=np.random.default_rng(54)
        y=np.array([0,1]*90,dtype=np.int8)
        old=np.clip(.2+y*.45+rng.normal(0,.23,len(y)),0,1)
        v2913=np.clip(.2+y*.50+rng.normal(0,.20,len(y)),0,1)
        new=np.clip(.1+y*.75+rng.normal(0,.10,len(y)),0,1)
        baseline=.85*controller.audit.rank(old)+.15*controller.audit.rank(v2913)
        with patch.object(controller,"BASELINE",roc_auc_score(y,baseline)):
            actual=controller.frozen_blend(y,old,v2913,new)
        self.assertGreater(actual["gain"],0)
        self.assertEqual(actual["positive_folds"],3)
        self.assertAlmostEqual(sum(actual["fixed_weights"].values()),1.0)

    def test_exact_cv_utility_prevents_partial_folds(self):
        src=(ROOT/"kaggle_agent/v2914_dual_realmlp_train.py").read_text()
        ctrl=(ROOT/"kaggle_agent/v2914_pipeline.py").read_text()
        ast.parse(src);ast.parse(ctrl)
        self.assertIn("n_ens=8",src)
        self.assertIn("n_epochs=3",src)
        self.assertIn("TargetEncoder(",src)
        self.assertIn("fit_transform(te_train, ytr)",src)
        self.assertNotIn("model.fit(xa, pd.Series(ytr), xv",src)
        self.assertIn('"enable_gpu":True',ctrl)
        self.assertIn("NvidiaTeslaT4",ctrl)
        self.assertNotIn('"submit"',ctrl)

    def test_recovery_must_not_create_or_push(self):
        with tempfile.TemporaryDirectory() as folder:
            with (patch.object(controller,"STATE",Path(folder)/"empty.json"),
                  patch.object(controller,"preflight"),
                  patch.object(controller,"call") as cli):
                status=controller.run(recover_only=True)
                self.assertEqual(status["stage"],"preflight")
                cli.assert_not_called()

    def test_owner_exact_id_proof(self):
        attempts=[SimpleNamespace(returncode=1,stdout="",stderr="kernel permission denied"),
                  SimpleNamespace(returncode=0,stdout="Not found",stderr=""),
                  SimpleNamespace(returncode=0,stdout="KernelWorkerStatus.COMPLETE",stderr="")]
        with patch.object(controller,"call",side_effect=attempts) as cli:
            self.assertFalse(controller.existing_or_absent())
            self.assertEqual(cli.call_count,3)

    def test_non_auth_notebook_list_is_fail_closed(self):
        failures=[SimpleNamespace(returncode=1,stdout="",stderr="denied"),
                  SimpleNamespace(returncode=1,stdout="",stderr="offline")]
        with patch.object(controller,"call",side_effect=failures):
            with self.assertRaisesRegex(RuntimeError,"owner-owned"):
                controller.existing_or_absent()

if __name__=="__main__":
    unittest.main()
