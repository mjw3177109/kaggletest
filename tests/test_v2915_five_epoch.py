"""Regression tests for unique V2.9.15 RealMLP 5epoch T4 challenger."""
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
import v2915_realmlp_5epoch_train as trainer
import v2915_pipeline as pipeline

def frame(n=76):
    i=np.arange(n)
    result=pd.DataFrame({
        "id":i,"Gender":np.where(i%2,"Male","Female"),
        "Customer Type":np.where(i%3,"Loyal Customer","disloyal Customer"),
        "Type of Travel":np.where(i%2,"Business travel","Personal Travel"),
        "Class":np.where(i%4,"Business","Eco"),
        "Age":i%60+18,"Flight Distance":i%1700+200,
        "Departure Delay in Minutes":i%45,
        "Arrival Delay in Minutes":i%48,
        "satisfaction":i%2})
    for offset,key in enumerate(("Inflight wifi service","Online boarding",
                                  "Seat comfort","Inflight entertainment",
                                  "Leg room service")):
        result[key]=(i+offset)%6
    return result

class FiveEpochSafety(unittest.TestCase):
    def test_fold_safe_feature_engineer(self):
        data=frame()
        train,test=data.iloc[:52],data.iloc[52:]
        x,v,k,z=trainer.engineer(train,test)
        self.assertEqual(len(x),52)
        self.assertEqual(len(v),24)
        self.assertEqual(list(x.columns),list(v.columns))
        self.assertEqual(k.shape,(52,3))
        self.assertEqual(z.shape,(24,3))
        self.assertNotIn("satisfaction",x.columns)
        self.assertNotIn("id",x.columns)
        changed=train.copy()
        changed["satisfaction"]=1-changed["satisfaction"]
        xx,_,_,_=trainer.engineer(changed,test)
        pd.testing.assert_frame_equal(x,xx)

    def test_fixed_training_and_oof_split_seed_independent(self):
        source=(ROOT/"kaggle_agent/v2915_realmlp_5epoch_train.py").read_text()
        program=(ROOT/"kaggle_agent/v2915_pipeline.py").read_text()
        ast.parse(source);ast.parse(program)
        self.assertEqual(trainer.SEED,20261018)
        self.assertEqual(trainer.MODEL_SEED,20261021)
        for key in ("n_ens=8","n_epochs=5","stop_epoch=5","batch_size=256",
                    "lr=0.038","random_state=MODEL_SEED+fold",
                    "random_state=SEED)","oof_realmlp_5epoch",
                    "pred_realmlp_5epoch"):
            self.assertIn(key,source)
        self.assertIn('"enable_gpu":True',program)
        self.assertIn('NvidiaTeslaT4',program)
        self.assertNotIn('"submit"',program)

    def test_exact_fixed_fusion_and_fold_rules(self):
        rng=np.random.default_rng(71)
        y=np.array([0,1]*90,dtype=np.int8)
        old=np.clip(.2+.45*y+rng.normal(0,.27,len(y)),0,1)
        dual=np.clip(.2+.45*y+rng.normal(0,.15,len(y)),0,1)
        new=np.clip(.2+.45*y+rng.normal(0,.11,len(y)),0,1)
        with patch.object(pipeline,"OFFICIAL_BASELINE",float(roc_auc_score(y,old))):
            r=pipeline.evaluate_blend(y,old,dual,new)
        self.assertGreater(r["gain"],0)
        self.assertEqual(r["positive_folds"],3)
        self.assertEqual(r["fixed_weights"],{
            "v2913_rank":.80,"v2914_dual_rank":.10,
            "v2915_five_epoch_rank":.10})
        self.assertEqual(len(r["fold_gain"]),3)
        self.assertEqual(pipeline.MIN_GAIN,.00008)

    def test_invalid_incumbent_rejected(self):
        y=np.array([0,1]*90,dtype=np.int8)
        base=np.linspace(.01,.99,len(y))
        with patch.object(pipeline,"OFFICIAL_BASELINE",.99):
            with self.assertRaisesRegex(ValueError,"official champion CV"):
                pipeline.evaluate_blend(y,base,base,base)

    def test_recovery_requires_known_kernel_no_push(self):
        with tempfile.TemporaryDirectory() as folder:
            with (patch.object(pipeline,"STATE",Path(folder)/"absent.json"),
                  patch.object(pipeline,"gate"),
                  patch.object(pipeline,"cli") as cli):
                r=pipeline.run(recover_only=True)
                self.assertEqual(r["stage"],"preflight")
                cli.assert_not_called()

    def test_unknown_kernel_fails_closed(self):
        replies=[
            SimpleNamespace(returncode=1,stdout="",stderr="permission denied"),
            SimpleNamespace(returncode=1,stdout="",stderr="network unavailable")
        ]
        with patch.object(pipeline,"cli",side_effect=replies):
            with self.assertRaisesRegex(RuntimeError,"owner-authenticated list failed"):
                pipeline.existing_or_absent()

    def test_known_owner_listing_absence_positive_control(self):
        replies=[
            SimpleNamespace(returncode=1,stdout="",stderr="not found"),
            SimpleNamespace(returncode=0,stdout="Not found",stderr=""),
            SimpleNamespace(returncode=0,stdout="KernelWorkerStatus.COMPLETE",stderr="")
        ]
        with patch.object(pipeline,"cli",side_effect=replies) as cli:
            self.assertFalse(pipeline.existing_or_absent())
            self.assertEqual(cli.call_count,3)

    def test_strict_t4_cv_metadata(self):
        m={"competition":pipeline.SLUG,"stage":"three_folds_complete",
            "validation":pipeline.VALIDATION,"folds":3,
            "fold_results":[{"fold":i} for i in range(3)],
            "train_rows":pipeline.N_TRAIN,"test_rows":pipeline.N_TEST,
            "model":"RealMLP-TD n_ens=8 5 epochs independent seeded",
            "model_seed":20261021,"epochs":5,"gpu_verified":True,
            "hardware":{"ok":True,"devices":["Tesla T4, 15360 MiB"]},
            "outer_validation_used_in_training":False,
            "leakage_safe_inner_cross_fit":True}
        pipeline.validate_metrics(m)
        m["outer_validation_used_in_training"]=True
        with self.assertRaisesRegex(ValueError,"leakage evidence mismatch"):
            pipeline.validate_metrics(m)

if __name__=="__main__":
    unittest.main()
