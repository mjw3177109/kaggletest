"""V2936 immutable XGBoost DART safety suite; no GPU or Kaggle access."""
import ast
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import v2936_dart_train as m
import v2936_pipeline as p
import v2936_actions_gate as g

class FrozenDart(unittest.TestCase):
    def test_01_preregistration_immutable(self):
        s=p.read(p.STATUS)
        self.assertTrue(p.verify_prereg(s))
        self.assertEqual(s["stage"],"preregistered_not_started")
        self.assertEqual(s["notebook_created"],False)
    def test_02_parameters_match(self):
        self.assertEqual(m.PARAMS,p.frozen_params()["model_parameters"])
        self.assertEqual(m.PARAMS["booster"],"dart")
        self.assertEqual(m.PARAMS["n_estimators"],900)
        self.assertEqual(m.PARAMS["device"],"cuda")
    def test_03_37_features_and_te(self):
        self.assertEqual(len(m.CANONICAL),37)
        self.assertEqual(m.CANONICAL,p.CANONICAL)
        self.assertEqual(m.CANONICAL[-3:],["te_0","te_1","te_2"])
        self.assertEqual(len(m.CATEGORICAL),14)
    def test_04_outer_train_only_category(self):
        n=16
        rng=np.random.default_rng(20261104)
        df={}
        for col in m.CANONICAL:
            if col in m.CATEGORICAL:
                df[col]=pd.Categorical(["A","B"]*8,categories=["A","B"])
            else:
                df[col]=rng.normal(size=n).astype("float32")
        xa=pd.DataFrame(df)[m.CANONICAL]
        xv=xa.iloc[:4].copy()
        xt=xa.iloc[4:8].copy()
        for col in m.CATEGORICAL:
            xv[col]=pd.Categorical(["UNKNOWN"]*4,categories=["A","B"])
        frames,sha=m.fold_matrix(xa,xv,xt)
        self.assertEqual(list(frames[0].columns),m.CANONICAL)
        self.assertTrue(all((frames[1][col]==-1).all() for col in m.CATEGORICAL))
        self.assertEqual(len(sha),64)
    def test_05_unknown_category_schema_rejected(self):
        n=4
        df={col:(pd.Categorical(["A"]*n,categories=["A"]) if col in m.CATEGORICAL
                 else np.ones(n,dtype="float32")) for col in m.CANONICAL}
        xa=pd.DataFrame(df)[m.CANONICAL]
        xv=xa.copy()
        xv["Gender"]=pd.Categorical(["B"]*n,categories=["B"])
        with self.assertRaises(ValueError):
            m.fold_matrix(xa,xv,xa)
    def test_06_fold_index_hash(self):
        y=np.array([0,1]*15)
        splits=list(p.StratifiedKFold(3,shuffle=True,random_state=p.SEED).split(np.zeros(len(y)),y))
        self.assertEqual(len(splits),3)
        self.assertEqual(sorted(np.concatenate([v for _,v in splits]).tolist()),list(range(len(y))))
    def test_07_gate_active_actions_complete(self):
        self.assertEqual(g.ACTIVE,("in_progress","queued","waiting","requested","pending"))
        self.assertGreaterEqual(g._MIN_KAGGLE_CALL_GAP_SECONDS,3)
    def test_08_no_submit_or_kernel_push_in_evaluator(self):
        for name in ("v2936_dart_train.py","v2936_actions_gate.py"):
            txt=(ROOT/"kaggle_agent"/name).read_text()
            self.assertNotIn("competitions submit",txt)
            self.assertNotIn('"submit"',txt)
        txt=(ROOT/"kaggle_agent/v2936_pipeline.py").read_text()
        self.assertNotIn('["kaggle","competitions","submit"',txt)
    def test_09_recover_only_refuses_unreserved(self):
        with patch.object(p,"preflight",return_value=p.read(p.STATUS)):
            with patch.object(p,"read",return_value=p.read(p.STATUS)):
                result=p.run(recover_only=True)
        self.assertEqual(result["stage"],"preregistered_not_started")
    def test_10_terminal_never_reruns(self):
        s=p.read(p.STATUS)
        s["stage"]="skipped_no_cv_gain"
        with patch.object(p,"read",return_value=s):
            result=p.run()
        self.assertEqual(result["stage"],"skipped_no_cv_gain")
    def test_11_guard_preflight_frozen(self):
        with patch.object(g,"read",side_effect=lambda f:p.read(p.ROOT/f)):
            with patch.object(g,"datetime") as clock:
                from datetime import datetime,timezone
                clock.now.return_value=datetime(2026,10,25,tzinfo=timezone.utc)
                clock.fromisoformat=datetime.fromisoformat
                mode=g.assert_preregistration()
        self.assertEqual(mode,"run")
    def test_12_immutable_legacy_sources(self):
        s=p.read(p.STATUS)
        self.assertEqual(s["previous_v2935"]["stage"],"skipped_no_cv_gain")
        self.assertEqual(s["champion_test_prediction_sha256"],"906185547dcd5f0cac3ac173de644fd9997f20ac863c6745bcb1fd3e9a89e5f7")
    def test_13_inference_all_trees(self):
        source=(ROOT/"kaggle_agent/v2936_dart_train.py").read_text()
        self.assertIn("iteration_range=(0,PARAMS[\"n_estimators\"])",source)
        self.assertNotIn("early_stopping_rounds",source)
    def test_14_no_validation_target_fit(self):
        source=(ROOT/"kaggle_agent/v2936_dart_train.py").read_text()
        self.assertIn("enc.fit_transform(te_train,y[itr])",source)
        self.assertIn("enc.transform(te_val)",source)
        self.assertNotIn("y[iva]",source.split("model.fit(")[0])
    def test_16_owner_rate_limit_fails_closed(self):
        import v2936_actions_gate as safety
        with patch.object(safety,"kaggle",side_effect=RuntimeError("rate_limited")):
            with self.assertRaises(RuntimeError):
                p.checked_owner_exists()
    def test_17_owner_status_present_never_push(self):
        import v2936_actions_gate as safety
        with patch.object(safety,"kaggle",return_value="KernelWorkerStatus.QUEUED"):
            self.assertTrue(p.checked_owner_exists())
    def test_18_exact_denial_and_owner_absence(self):
        from types import SimpleNamespace
        import v2936_actions_gate as safety
        err=("Cannot access kernel '"+p.KERNEL+
             "' (Permission 'kernels.get' was denied). The most likely cause is a wrong kernel slug.")
        with patch.object(p.subprocess,"run",return_value=SimpleNamespace(returncode=1,stderr=err,stdout="")):
            with patch("time.sleep"):
                with patch.object(safety,"kaggle",side_effect=["Not found","KernelWorkerStatus.COMPLETE"]) as call:
                    self.assertFalse(p.checked_owner_exists())
                    self.assertEqual(call.call_count,2)
    def test_19_unknown_denial_stays_blocked(self):
        from types import SimpleNamespace
        import v2936_actions_gate as safety
        with patch.object(p.subprocess,"run",return_value=SimpleNamespace(returncode=1,stderr="HTTP 429 rate limit",stdout="")):
            with patch.object(safety,"kaggle",side_effect=AssertionError("must not query owner list")):
                with self.assertRaises(RuntimeError):
                    p.checked_owner_exists()
    def test_15_own_slug_only(self):
        self.assertEqual(p.KERNEL,g.KERNEL)
        self.assertEqual(p.KERNEL,"mjw3177109/kas-v2936-original37-xgboost-dart-t4-s6e10")

if __name__=="__main__":
    unittest.main()
