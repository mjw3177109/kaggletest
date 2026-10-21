"""V2920 independent CPU exact-route LightGBM regression and leakage gates."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_agent"))
import v2920_route_profile_xgb_train as trainer
import v2920_pipeline as pipeline


def frame(n=100, offset=0):
    i = np.arange(n) + offset
    f = pd.DataFrame({
        "id":i,
        "Gender":np.where(i%2, "Male", "Female"),
        "Customer Type":np.where(i%3, "Loyal Customer", "disloyal Customer"),
        "Type of Travel":np.where(i%2, "Business travel", "Personal Travel"),
        "Class":np.where(i%3, "Business", "Eco"),
        "Age":i%59+18,
        "Flight Distance":200+(i%7)*101,
        "Departure Delay in Minutes":i%23,
        "Arrival Delay in Minutes":i%27,
        "satisfaction":i%2,
    })
    for j, name in enumerate(trainer.SERVICES):
        f[name] = (i+j)%6
    return f


class RouteProfileOOFTests(unittest.TestCase):
    def test_exact_feature_schema_and_fold_only_route_profile(self):
        fit, hold, test = frame(100), frame(24, 100), frame(23, 200)
        self.assertEqual(len(trainer.feature_names(fit)), 21)
        train, val, tst = trainer.route_features(fit, hold, test)
        self.assertEqual(train.shape, (100, 12))
        self.assertEqual(val.shape, (24, 12))
        self.assertTrue(train.columns.equals(val.columns))
        self.assertTrue(val.columns.equals(tst.columns))
        self.assertTrue(np.isfinite(train.to_numpy()).all())
        modified = hold.copy()
        modified["Inflight wifi service"] = 0
        changed_fit, changed_val, _ = trainer.route_features(fit, modified, test)
        pd.testing.assert_frame_equal(train, changed_fit)
        self.assertFalse(val.equals(changed_val))
        unseen = hold.copy()
        unseen["Flight Distance"] = 999999
        _, u, _ = trainer.route_features(fit, unseen, test)
        self.assertTrue(np.allclose(u["route_profile_count_log"].to_numpy(), 0.0))

    def test_all_21_original_fields_inner_target_encoding(self):
        fit, hold, test = frame(100), frame(24, 100), frame(23, 200)
        keys = trainer.target_keys(fit, trainer.feature_names(fit))
        self.assertEqual(keys.shape, (100, 23))
        self.assertFalse(any("satisfaction" in col or "raw_id" == col for col in keys))
        y = fit["satisfaction"].to_numpy()
        tr, va, ts = trainer.matrices(fit, hold, test, y, fold=0)
        self.assertEqual(tr.shape, (100, 56))
        self.assertEqual(va.shape, (24, 56))
        self.assertEqual(ts.shape, (23, 56))
        self.assertTrue(tr.columns.equals(va.columns))
        self.assertTrue(va.columns.equals(ts.columns))
        self.assertFalse(tr.isna().any().any())
        changed = fit.copy()
        changed["satisfaction"] = 1-changed["satisfaction"]
        self.assertTrue(trainer.target_keys(fit, trainer.feature_names(fit)).equals(
            trainer.target_keys(changed, trainer.feature_names(changed))))
        a, _, _ = trainer.route_features(fit, hold, test)
        b, _, _ = trainer.route_features(changed, hold, test)
        pd.testing.assert_frame_equal(a,b)

    def test_single_model_locked_and_never_submits(self):
        training = (ROOT/"kaggle_agent/v2920_route_profile_xgb_train.py").read_text()
        control = (ROOT/"kaggle_agent/v2920_pipeline.py").read_text()
        ast.parse(training); ast.parse(control)
        self.assertEqual(trainer.MODEL_SEED, 20261028)
        self.assertEqual(trainer.SEED, 20261018)
        self.assertEqual(trainer.FOLDS, 3)
        self.assertEqual(trainer.INNER_FOLDS, 5)
        self.assertEqual(pipeline.MODEL_SEED, trainer.MODEL_SEED)
        self.assertEqual(pipeline.WEIGHTS, {"v2915_rank":.85,"v2920_rank":.15})
        self.assertEqual(pipeline.BASELINE, .9612850399332336)
        self.assertEqual(pipeline.MIN_GAIN, .00005)
        for term in ('n_estimators=1000','encoder.fit_transform(key_train',
                     'model.fit(xtr, y[itr])','random_state=SEED',
                     'v2920_route_xgb_oof.npz','route_profile_feature_count',
                     '"external_labels":False'):
            self.assertIn(term, training)
        self.assertIn('xgb.XGBClassifier(',training)
        self.assertIn('tree_method="hist"',training)
        self.assertIn('max_depth=6',training)
        self.assertIn('state/v2_9_18_status.json',control)
        self.assertIn('terminal negative V2918 route',control)
        self.assertNotIn("eval_set=", training)
        self.assertNotIn('"competitions","submit"', control)
        self.assertIn('"enable_gpu":False', control)
        self.assertNotIn("--accelerator", control)
        self.assertNotIn('"NvidiaTeslaT4"', control)

    def test_validation_metadata_fails_closed(self):
        data = {"competition":pipeline.SLUG,
                "stage":"three_folds_complete","model":trainer.MODEL,
                "validation":pipeline.VALIDATION,"folds":3,
                "fold_results":[{"fold":i} for i in range(3)],
                "train_rows":pipeline.N_TRAIN,"test_rows":pipeline.N_TEST,
                "model_seed":20261028,
                "feature_family":trainer.FEATURE_FAMILY,
                "route_profile_feature_count":12,
                "target_encoding_columns":23,"inner_target_folds":5,
                "cpu_only":True,"gpu_verified":False,
                "data_source":"competition_train_only","external_labels":False,
                "submitted":False,"leakage_safe_inner_cross_fit":True,
                "outer_validation_used_in_training":False}
        pipeline.validate_metrics(data)
        for key, other in [
            ("model_seed", 99), ("inner_target_folds", 0),
            ("gpu_verified", True), ("external_labels", True),
            ("outer_validation_used_in_training", True),
            ("route_profile_feature_count", 11),
        ]:
            bad = {**data,key:other}
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "provenance"):
                    pipeline.validate_metrics(bad)

    def test_recovery_never_creates_missing_kernel(self):
        with tempfile.TemporaryDirectory() as folder:
            with (patch.object(pipeline, "STATE",Path(folder)/"not-present.json"),
                  patch.object(pipeline,"preflight"),
                  patch.object(pipeline,"cmd") as cli):
                result = pipeline.run(recover_only=True)
                self.assertEqual(result["stage"],"preflight")
                cli.assert_not_called()

    def test_authenticated_kernel_list_is_fail_closed(self):
        responses = [SimpleNamespace(returncode=1,stdout="",stderr="missing"),
                     SimpleNamespace(returncode=1,stdout="",stderr="no token")]
        with patch.object(pipeline,"cmd",side_effect=responses):
            with self.assertRaisesRegex(RuntimeError,"owner list unavailable"):
                pipeline.existing_or_absent()

    def test_fixed_blend_only_full_oof(self):
        rng = np.random.default_rng(18)
        y = np.array([0,1]*90,dtype=np.int8)
        incumbent = np.clip(.2+.44*y+rng.normal(0,.22,len(y)),0,1)
        newcomer = np.clip(.2+.42*y+rng.normal(0,.11,len(y)),0,1)
        baseline = roc_auc_score(y, incumbent)
        with patch.object(pipeline,"BASELINE",baseline):
            result = pipeline.blend(y,incumbent,newcomer)
        self.assertEqual(result["fixed_weights"],pipeline.WEIGHTS)
        self.assertEqual(len(result["fold_gain"]),3)
        self.assertTrue(np.isfinite(result["candidate_auc"]))
        with self.assertRaisesRegex(ValueError,"lengths differ"):
            pipeline.blend(y,incumbent,newcomer[:-1])


if __name__ == "__main__":
    unittest.main()
