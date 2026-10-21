"""V2.9.17 guardrails for independent interaction-augmented RealMLP-TD."""
from __future__ import annotations
import ast
import json
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
import v2917_realmlp_interactions_train as trainer
import v2917_pipeline as pipeline

def frame(n=90):
    i=np.arange(n)
    f=pd.DataFrame({
        "id":i,
        "Gender":np.where(i%2,"Male","Female"),
        "Customer Type":np.where(i%3,"Loyal Customer","disloyal Customer"),
        "Type of Travel":np.where(i%2,"Business travel","Personal Travel"),
        "Class":np.where(i%4,"Business","Eco"),
        "Age":i%60+18,
        "Flight Distance":i%1700+200,
        "Departure Delay in Minutes":i%45,
        "Arrival Delay in Minutes":i%48,
        "satisfaction":i%2})
    for offset,key in enumerate(("Inflight wifi service","Online boarding",
            "Seat comfort","Inflight entertainment","Leg room service")):
        f[key]=(i+offset)%6
    return f

class InteractionsOOFSafety(unittest.TestCase):
    def test_features_are_fold_safe_and_label_independent(self):
        data=frame()
        train,val=data.iloc[:60],data.iloc[60:]
        x,v,key,hold=trainer.engineer(train,val)
        self.assertEqual(len(x),60)
        self.assertEqual(len(v),30)
        self.assertEqual(key.shape,(60,3))
        self.assertEqual(hold.shape,(30,3))
        self.assertEqual(list(x.columns),list(v.columns))
        self.assertNotIn("satisfaction",x)
        self.assertNotIn("id",x)
        extra=[c for c in x if c.startswith("interaction_")]
        self.assertEqual(len(extra),9)
        self.assertTrue(np.isfinite(x[extra].to_numpy(dtype="float32")).all())
        self.assertTrue(np.isfinite(v[extra].to_numpy(dtype="float32")).all())
        changed=train.copy()
        changed["satisfaction"]=1-changed["satisfaction"]
        x2,v2,k2,h2=trainer.engineer(changed,val)
        pd.testing.assert_frame_equal(x,x2)
        pd.testing.assert_frame_equal(v,v2)
        pd.testing.assert_frame_equal(key,k2)
        pd.testing.assert_frame_equal(hold,h2)

    def test_preregistered_single_model_and_weighted_candidate(self):
        source=(ROOT/"kaggle_agent/v2917_realmlp_interactions_train.py").read_text()
        control=(ROOT/"kaggle_agent/v2917_pipeline.py").read_text()
        ast.parse(source);ast.parse(control)
        self.assertEqual(trainer.SEED,20261018)
        self.assertEqual(trainer.MODEL_SEED,20261025)
        self.assertEqual(trainer.FOLDS,3)
        self.assertEqual(pipeline.MODEL_SEED,trainer.MODEL_SEED)
        self.assertEqual(pipeline.WEIGHTS,{"v2915_rank":.85,"v2917_rank":.15})
        self.assertAlmostEqual(pipeline.BASELINE,.9612850399332336)
        self.assertAlmostEqual(pipeline.MIN_GAIN,.00005)
        for token in ("n_ens=8","n_epochs=7","stop_epoch=7",
                      "random_state=MODEL_SEED+fold",
                      "random_state=SEED)",
                      "oof_realmlp_interactions","pred_realmlp_interactions",
                      "feature_family"):
            self.assertIn(token,source)
        self.assertIn('"enable_gpu":True',control)
        self.assertIn('"machine_shape":"NvidiaTeslaT4"',control)
        self.assertNotIn('"competitions","submit"',control)
        self.assertNotIn('["kaggle","kernels","push"',source)

    def test_new_challenger_requires_terminal_predecessor(self):
        controller=(ROOT/"kaggle_agent/v2917_pipeline.py").read_text()
        self.assertIn("state/v2_9_16_status.json",controller)
        self.assertIn("authentic terminal V2916 failed-promotion",controller)
        self.assertEqual(pipeline.MODEL_SEED,20261025)
        self.assertEqual(pipeline.MIN_GAIN,.00005)

    def test_fixed_blend_uses_independent_full_oof(self):
        rng=np.random.default_rng(16)
        y=np.array([0,1]*90,dtype=np.int8)
        old=np.clip(.20+.43*y+rng.normal(0,.28,len(y)),0,1)
        newcomer=np.clip(.20+.48*y+rng.normal(0,.11,len(y)),0,1)
        baseline=roc_auc_score(y,old)
        with patch.object(pipeline,"BASELINE",baseline):
            result=pipeline.blend(y,old,newcomer)
        self.assertEqual(result["fixed_weights"],pipeline.WEIGHTS)
        self.assertEqual(len(result["fold_gain"]),3)
        self.assertGreater(result["candidate_auc"],result["baseline_auc"])
        self.assertGreaterEqual(result["positive_folds"],2)
        with self.assertRaisesRegex(ValueError,"lengths differ"):
            pipeline.blend(y,old,newcomer[:-1])

    def test_recovery_will_not_create_unsaved_kernel(self):
        with tempfile.TemporaryDirectory() as folder:
            with (patch.object(pipeline,"STATE",Path(folder)/"not-created.json"),
                  patch.object(pipeline,"preflight"),
                  patch.object(pipeline,"cmd") as cli):
                result=pipeline.run(recover_only=True)
                self.assertEqual(result["stage"],"preflight")
                cli.assert_not_called()

    def test_existing_kernel_ambiguity_fails_closed(self):
        replies=[SimpleNamespace(returncode=1,stdout="",stderr="not found"),
                 SimpleNamespace(returncode=1,stdout="",stderr="no auth")]
        with patch.object(pipeline,"cmd",side_effect=replies):
            with self.assertRaisesRegex(RuntimeError,"owner list unavailable"):
                pipeline.existing_or_absent()

    def test_strict_gpu_cv_metadata(self):
        details={"competition":pipeline.SLUG,"stage":"three_folds_complete",
                 "model":"RealMLP-TD n_ens=8 7 epochs feature-interaction independent seeded",
                 "validation":pipeline.VALIDATION,"folds":3,
                 "fold_results":[{"fold":i} for i in range(3)],
                 "train_rows":pipeline.N_TRAIN,"test_rows":pipeline.N_TEST,
                 "model_seed":20261025,"epochs":7,
                 "feature_family":"service_and_delay_interactions_v1",
                 "derived_feature_count":9,
                 "gpu_verified":True,
                 "hardware":{"ok":True,"devices":["Tesla T4, 15360 MiB"]},
                 "outer_validation_used_in_training":False,
                 "leakage_safe_inner_cross_fit":True}
        pipeline.validate_metrics(details)
        details["outer_validation_used_in_training"]=True
        with self.assertRaisesRegex(ValueError,"evidence mismatch"):
            pipeline.validate_metrics(details)

if __name__=="__main__":
    unittest.main()
