"""V2.9.12 native CatBoost feature/provenance and unique-kernel safety tests."""
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
import v2912_native_cat_cpu_train as train
import v2912_pipeline as pipe

def frame(n=84):
    i=np.arange(n)
    x=pd.DataFrame({"id":i,"Gender":np.where(i%2,"Male","Female"),
        "Customer Type":np.where(i%3,"Loyal Customer","disloyal Customer"),
        "Type of Travel":np.where(i%2,"Business travel","Personal Travel"),
        "Class":np.where(i%5,"Business","Eco"),
        "Age":i%60+18,"Flight Distance":i%1000+200,
        "Departure Delay in Minutes":i%44,"Arrival Delay in Minutes":i%47,
        "satisfaction":(i%2).astype(np.int8)})
    for k,col in enumerate(train.SERVICES):
        x[col]=((i+k)%6).astype(int)
    return x

class NativeCatSafety(unittest.TestCase):
    def test_feature_shapes_and_cat_types(self):
        f=frame()
        features,cats=train.features(f)
        self.assertEqual(len(features),len(f))
        self.assertEqual(len(cats),9)
        self.assertIn("traveler_segment",cats)
        self.assertEqual(len(set(features.columns)),len(features.columns))
        self.assertNotIn("satisfaction",features)
        self.assertNotIn("id",features)
        self.assertTrue(np.isfinite(features.drop(columns=cats).to_numpy()).all())
        for name in cats:
            self.assertTrue(pd.api.types.is_string_dtype(features[name]))
        self.assertEqual(train.PARAMS["task_type"],"CPU")
        self.assertEqual(train.PARAMS["iterations"],750)

    def test_label_changes_do_not_modify_features(self):
        f=frame()
        a,_=train.features(f)
        f["satisfaction"]=1-f["satisfaction"]
        b,_=train.features(f)
        pd.testing.assert_frame_equal(a,b)

    def test_native_cat_tiny_fit_is_cpu_only(self):
        from catboost import CatBoostClassifier
        f=frame(84)
        y=f["satisfaction"].to_numpy()
        X,cats=train.features(f)
        model=CatBoostClassifier(iterations=6,depth=3,
                  loss_function="Logloss",task_type="CPU",
                  random_seed=3,allow_writing_files=False,
                  verbose=False,thread_count=2)
        model.fit(X.iloc[:65],y[:65],cat_features=cats,verbose=False)
        pred=model.predict_proba(X.iloc[65:])[:,1]
        self.assertEqual(len(pred),19)
        self.assertTrue(np.isfinite(pred).all())
        self.assertEqual(model.get_all_params().get("task_type"),"CPU")

    def test_fixed_blend_gate_and_reconstruction(self):
        rng=np.random.default_rng(43)
        y=np.array([0,1]*90,dtype=np.int8)
        arr=[]
        for noise in (.3,.24,.22,.25,.08):
            arr.append(np.clip(.2+y*.5+rng.normal(0,noise,len(y)),0,1))
        b,r,c,t,new=arr
        rank=pipe.audit.rank
        old=.60*rank(b)+.20*rank(r)+.15*rank(c)+.05*rank(t)
        with patch.object(pipe,"BASELINE",float(roc_auc_score(y,old))):
            m=pipe.combine(y,b,r,c,t,new)
        self.assertEqual(m["positive_folds"],3)
        self.assertGreater(m["gain"],0)
        self.assertAlmostEqual(sum(m["fixed_weights"].values()),1.0)
        self.assertEqual(len(m["fold_gain"]),3)

    def test_recovery_must_not_push_or_create(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(pipe,"STATE",Path(tmp)/"status.json"),patch.object(pipe,"eligible"):
                with patch.object(pipe,"kaggle") as kaggle:
                    result=pipe.run(recover=True)
                    self.assertEqual(result["stage"],"preflight")
                    kaggle.assert_not_called()

    def test_probe_failure_refuses_new_kernel_push(self):
        with tempfile.TemporaryDirectory() as tmp:
            with (patch.object(pipe,"STATE",Path(tmp)/"status.json"),
                  patch.object(pipe,"REPORT",Path(tmp)/"report.md"),
                  patch.object(pipe,"eligible"),
                  patch.object(pipe,"kaggle",return_value=SimpleNamespace(
                      returncode=1,stderr="403 denied",stdout="")) as kaggle):
                result=pipe.run(recover=False)
                self.assertEqual(result["stage"],"kernel_access_error")
                self.assertEqual(kaggle.call_count,2)
                self.assertEqual(kaggle.call_args_list[0].args[0][2],"status")
                self.assertEqual(kaggle.call_args_list[1].args[0][2],"list")

    def test_known_owner_mine_list_proves_missing_id(self):
        probes=[
            SimpleNamespace(returncode=1,stderr="Permission 'kernels.get' was denied",stdout=""),
            SimpleNamespace(returncode=0,stderr="",stdout="Not found\\n"),
            SimpleNamespace(returncode=0,stderr="",stdout=
                'mjw3177109/kas-v2911-segment-contrast-cpu-s6e10 has status "KernelWorkerStatus.COMPLETE"'),
        ]
        with patch.object(pipe,"kaggle",side_effect=probes) as client:
            self.assertFalse(pipe.existing_or_absent())
            self.assertEqual(client.call_count,3)

    def test_existing_kernel_never_pushes_or_probes_again(self):
        with patch.object(pipe,"kaggle",return_value=SimpleNamespace(
                returncode=0,stderr="",stdout='KernelWorkerStatus.QUEUED')) as client:
            self.assertTrue(pipe.existing_or_absent())
            self.assertEqual(client.call_count,1)

    def test_unverified_list_fails_closed(self):
        results=[
            SimpleNamespace(returncode=1,stderr="permission denied",stdout=""),
            SimpleNamespace(returncode=0,stderr="",stdout="Not found\\n"),
            SimpleNamespace(returncode=1,stderr="offline",stdout=""),
        ]
        with patch.object(pipe,"kaggle",side_effect=results):
            with self.assertRaisesRegex(RuntimeError,"owner control failed"):
                pipe.existing_or_absent()

    def test_script_has_no_submission(self):
        source=(ROOT/"kaggle_agent/v2912_pipeline.py").read_text(encoding="utf-8")
        trainer=(ROOT/"kaggle_agent/v2912_native_cat_cpu_train.py").read_text(encoding="utf-8")
        ast.parse(source);ast.parse(trainer)
        self.assertNotIn('"submit"',source)
        self.assertNotIn('"submit"',trainer)
        self.assertIn('"enable_gpu":False',source)
        self.assertIn("preexisting_kernel_recovered",source)
        self.assertNotEqual(pipe.STATE.name,"v2_9_11_status.json")
        self.assertEqual(pipe.KERNEL,"mjw3177109/kas-v2912-native-cat-cpu-s6e10")

if __name__=="__main__":unittest.main()
