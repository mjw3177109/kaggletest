"""V2921 native CatBoost feature, provenance, recovery, and promotion gates."""
from __future__ import annotations
import ast,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import v2921_catboost_native_route_train as trainer
import v2921_pipeline as pipeline


def frame(n=110,offset=0):
    i=np.arange(n)+offset
    out=pd.DataFrame({
        "id":i,
        "Gender":np.where(i%2,"Male","Female"),
        "Customer Type":np.where(i%3,"Loyal Customer","disloyal Customer"),
        "Type of Travel":np.where(i%2,"Business travel","Personal Travel"),
        "Class":np.where(i%3,"Business",np.where(i%3==1,"Eco","Eco Plus")),
        "Age":i%59+18,
        "Flight Distance":200+(i%7)*101,
        "Departure Delay in Minutes":i%23,
        "Arrival Delay in Minutes":i%27,
        "satisfaction":i%2,
    })
    for j,name in enumerate(trainer.SERVICES):
        out[name]=(i+j)%6
    return out


class NativeCatBoostTests(unittest.TestCase):
    def test_raw_categorical_features_and_fold_local_route_profiles(self):
        train,hold,test=frame(80),frame(20,80),frame(10,100)
        xtr,xval,xtest=trainer.build_features(train,hold,test)
        self.assertEqual([x.shape for x in (xtr,xval,xtest)],[(80,33),(20,33),(10,33)])
        self.assertEqual(list(trainer.CATS),["Gender","Customer Type","Type of Travel","Class"])
        self.assertTrue(xtr.columns.equals(xval.columns))
        self.assertTrue(xval.columns.equals(xtest.columns))
        for col in trainer.CATS:
            self.assertTrue(all(isinstance(v,str) for v in xtr[col].tolist()))
        self.assertNotIn("id",xtr.columns)
        self.assertNotIn("satisfaction",xtr.columns)
        self.assertEqual(sum(c.startswith("route_profile_") for c in xtr.columns),12)
        numeric=xtr.drop(columns=list(trainer.CATS)).to_numpy(dtype=np.float32)
        self.assertTrue(np.isfinite(numeric).all())
        changed=hold.copy()
        changed["Inflight wifi service"]=0
        a,_,_=trainer.route_features(train,hold,test)
        b,_,_=trainer.route_features(train,changed,test)
        pd.testing.assert_frame_equal(a,b)
        unseen=hold.copy()
        unseen["Flight Distance"]=999999
        _,u,_=trainer.route_features(train,unseen,test)
        self.assertTrue(np.allclose(u["route_profile_count_log"].to_numpy(),0.0))

    def test_feature_generation_does_not_depend_on_target_labels(self):
        train,hold,test=frame(80),frame(20,80),frame(10,100)
        a=trainer.build_features(train,hold,test)
        changed=train.copy()
        changed["satisfaction"]=1-changed["satisfaction"]
        b=trainer.build_features(changed,hold,test)
        for left,right in zip(a,b):
            pd.testing.assert_frame_equal(left,right)

    def test_locked_catboost_native_architecture_and_cpu_only(self):
        training=(ROOT/"kaggle_agent/v2921_catboost_native_route_train.py").read_text()
        control=(ROOT/"kaggle_agent/v2921_pipeline.py").read_text()
        ast.parse(training);ast.parse(control)
        self.assertEqual(trainer.MODEL_SEED,20261029)
        self.assertEqual(trainer.SEED,20261018)
        self.assertEqual(trainer.FOLDS,3)
        self.assertEqual(trainer.MODEL,"CatBoost native-categorical route-profile 3fold seeded")
        for token in ("iterations=1000","depth=7","learning_rate=0.04",
                      'cat_features=list(CATS)','task_type="CPU"',
                      "model.fit(xtr,y[itr],cat_features=list(CATS))",
                      "v2921_catboost_native_route_oof.npz",
                      '"target_encoding_columns":0','"native_catboost_ctr":True'):
            self.assertIn(token,training)
        self.assertNotIn("TargetEncoder",training)
        self.assertNotIn("eval_set=",training)
        self.assertNotIn('"competitions","submit"',control)
        self.assertIn('"enable_gpu":False',control)
        self.assertIn('state/v2_9_20_status.json',control)
        self.assertIn("terminal negative V2920 XGBoost OOF",control)

    def test_metadata_requires_native_catboost_not_te_or_gpu(self):
        m={
            "competition":pipeline.SLUG,"stage":"three_folds_complete","model":trainer.MODEL,
            "validation":pipeline.VALIDATION,"folds":3,"fold_results":[{"fold":i} for i in range(3)],
            "train_rows":pipeline.N_TRAIN,"test_rows":pipeline.N_TEST,
            "model_seed":20261029,"feature_family":trainer.FEATURE_FAMILY,
            "route_profile_feature_count":12,"target_encoding_columns":0,"inner_target_folds":0,
            "native_catboost_ctr":True,"native_categorical_columns":list(trainer.CATS),
            "total_feature_count":33,"original_predictor_count":21,"iterations":1000,"depth":7,
            "learning_rate":0.04,"cpu_only":True,"gpu_verified":False,
            "data_source":"competition_train_only","external_labels":False,"submitted":False,
            "leakage_safe_inner_cross_fit":False,"outer_validation_used_in_training":False,
        }
        pipeline.validate_metrics(m)
        for key,value in [("native_catboost_ctr",False),("target_encoding_columns",23),
                          ("cpu_only",False),("gpu_verified",True),
                          ("outer_validation_used_in_training",True),("depth",9)]:
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError,"provenance"):
                    pipeline.validate_metrics({**m,key:value})

    def test_only_exact_missing_auc_metadata_error_is_recoverable(self):
        recoverable={"stage":"validation_mismatch","kernel":pipeline.KERNEL,
                     "error":"'full_oof_auc'"}
        self.assertTrue(pipeline.recoverable_missing_full_auc(recoverable))
        for bad in [
            {**recoverable,"error":"CatBoost recorded full OOF AUC mismatch"},
            {**recoverable,"kernel":"someone-else/kernel"},
            {**recoverable,"stage":"kernel_failed"},
        ]:
            self.assertFalse(pipeline.recoverable_missing_full_auc(bad))
        source=(ROOT/"kaggle_agent/v2921_pipeline.py").read_text()
        self.assertIn("full_oof_auc_recomputed_by_auditor",source)
        self.assertIn("recompute_missing_full_oof_auc_from_aligned_oof",source)

    def test_recovery_never_pushes_an_unrecorded_notebook(self):
        with tempfile.TemporaryDirectory() as folder:
            with (patch.object(pipeline,"STATE",Path(folder)/"empty.json"),
                  patch.object(pipeline,"preflight"),
                  patch.object(pipeline,"cmd") as cli):
                result=pipeline.run(recover_only=True)
                self.assertEqual(result["stage"],"preflight")
                cli.assert_not_called()

    def test_fixed_blend_has_no_posthoc_weight_search(self):
        rng=np.random.default_rng(21)
        y=np.array([0,1]*100,dtype=np.int8)
        old=np.clip(.2+.4*y+rng.normal(0,.19,len(y)),0,1)
        new=np.clip(.2+.4*y+rng.normal(0,.11,len(y)),0,1)
        baseline=roc_auc_score(y,old)
        with patch.object(pipeline,"BASELINE",baseline):
            result=pipeline.blend(y,old,new)
        self.assertEqual(result["fixed_weights"],pipeline.WEIGHTS)
        self.assertEqual(pipeline.WEIGHTS,{"v2915_rank":.85,"v2921_rank":.15})
        self.assertEqual(len(result["fold_gain"]),3)
        with self.assertRaisesRegex(ValueError,"lengths differ"):
            pipeline.blend(y,old,new[:-1])


if __name__=="__main__":
    unittest.main()
