"""V2919 TabM-GPU regression: distinct architecture, fixed OOF, no duplicate notebook."""
from __future__ import annotations
import ast
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import v2919_tabm_route_train as trainer
import v2919_pipeline as pipeline
import v2918_route_profile_lgbm_train as features


class TabMIndependentPilotTests(unittest.TestCase):
    def test_locked_architecture_and_shared_no_leak_features(self):
        train_file=(ROOT/"kaggle_agent/v2919_tabm_route_train.py").read_text()
        control=(ROOT/"kaggle_agent/v2919_pipeline.py").read_text()
        engineering=(ROOT/"kaggle_agent/v2918_route_profile_lgbm_train.py").read_text()
        ast.parse(train_file);ast.parse(control);ast.parse(engineering)
        self.assertEqual(trainer.MODEL_SEED,20261027)
        self.assertEqual(trainer.FOLDS,3)
        self.assertEqual(trainer.TABM_K,8)
        self.assertEqual(trainer.EPOCHS,8)
        self.assertEqual(pipeline.MODEL_SEED,trainer.MODEL_SEED)
        self.assertEqual(pipeline.BASELINE,.9612850399332336)
        self.assertEqual(pipeline.WEIGHTS,{"v2915_rank":.85,"v2919_rank":.15})
        self.assertEqual(pipeline.MIN_GAIN,.00005)
        for token in ("TabM_D_Classifier","val_fraction=0.10","tabm_k=TABM_K",
                      "n_epochs=EPOCHS","model.fit(xtr, pd.Series(targets[itr]))",
                      "v2919_tabm_oof.npz","inner_target_folds",
                      "outer_validation_used_in_training"):
            self.assertIn(token,train_file)
        self.assertIn('feature_engineering.py',control)
        self.assertIn('state/v2_9_17_status.json',control)
        self.assertIn('cannot overlap unfinished V2917 training',control)
        self.assertIn('"enable_gpu":True',control)
        self.assertIn('"machine_shape":"NvidiaTeslaT4"',control)
        self.assertNotIn('"competitions","submit"',control)
        self.assertNotIn("eval_set=",train_file)
        self.assertIn("route_features",engineering)
        self.assertTrue(trainer.route is features)

    def test_locked_full_oof_metrics_hardware_and_fold_metadata(self):
        v={"competition":pipeline.SLUG,"stage":"three_folds_complete",
           "model":trainer.MODEL,"validation":pipeline.VALIDATION,
           "folds":3,"fold_results":[{"fold":i} for i in range(3)],
           "train_rows":pipeline.N_TRAIN,"test_rows":pipeline.N_TEST,
           "model_seed":20261027,"epochs":8,"tabm_k":8,
           "feature_family":"exact_distance_profile_te23_v1",
           "route_profile_feature_count":12,
           "target_encoding_columns":23,"inner_target_folds":5,
           "internal_validation_fraction":0.10,
           "cpu_only":False,"gpu_verified":True,
           "hardware":{"ok":True,"devices":["Tesla T4, 15360 MiB"]},
           "data_source":"competition_train_only","external_labels":False,
           "submitted":False,"leakage_safe_inner_cross_fit":True,
           "outer_validation_used_in_training":False}
        pipeline.validate_metrics(v)
        for key,val in [
            ("model","RealMLP"),("epochs",7),("gpu_verified",False),
            ("tabm_k",32),("inner_target_folds",0),
            ("external_labels",True),("outer_validation_used_in_training",True),
            ("internal_validation_fraction",.20)
        ]:
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError,"provenance"):
                    pipeline.validate_metrics({**v,key:val})

    def test_recovery_without_kernel_is_never_a_push(self):
        with tempfile.TemporaryDirectory() as directory:
            with (patch.object(pipeline,"STATE",Path(directory)/"empty.json"),
                  patch.object(pipeline,"preflight"),
                  patch.object(pipeline,"cmd") as cli):
                result=pipeline.run(recover_only=True)
                self.assertEqual(result["stage"],"preflight")
                cli.assert_not_called()

    def test_uncertain_existing_kaggle_owner_fails_closed(self):
        responses=[SimpleNamespace(returncode=1,stdout="",stderr="missing"),
                   SimpleNamespace(returncode=1,stdout="",stderr="no auth")]
        with patch.object(pipeline,"cmd",side_effect=responses):
            with self.assertRaisesRegex(RuntimeError,"owner list unavailable"):
                pipeline.existing_or_absent()

    def test_fixed_champion_fusion_independent_and_no_weight_search(self):
        rng=np.random.default_rng(19)
        y=np.array([0,1]*100,dtype=np.int8)
        old=np.clip(.2+.4*y+rng.normal(0,.19,len(y)),0,1)
        new=np.clip(.2+.4*y+rng.normal(0,.11,len(y)),0,1)
        auc=roc_auc_score(y,old)
        with patch.object(pipeline,"BASELINE",auc):
            result=pipeline.blend(y,old,new)
        self.assertEqual(result["fixed_weights"],pipeline.WEIGHTS)
        self.assertEqual(len(result["fold_gain"]),3)
        self.assertTrue(np.isfinite(result["candidate_auc"]))
        with self.assertRaisesRegex(ValueError,"lengths differ"):
            pipeline.blend(y,old,new[:-1])


if __name__=="__main__":
    unittest.main()
