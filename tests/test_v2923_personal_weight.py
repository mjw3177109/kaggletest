from __future__ import annotations
import ast,sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import v2923_personal_weight_lgbm_train as trainer
import v2923_pipeline as pipeline

class PersonalTravelWeightTests(unittest.TestCase):
    def test_preregistered_weighting_and_no_submission(self):
        source=(ROOT/"kaggle_agent/v2923_personal_weight_lgbm_train.py").read_text()
        control=(ROOT/"kaggle_agent/v2923_pipeline.py").read_text()
        ast.parse(source);ast.parse(control)
        self.assertEqual(trainer.MODEL_SEED,20261030)
        self.assertEqual(trainer.FOLDS,3)
        self.assertEqual(trainer.INNER_FOLDS,5)
        self.assertEqual(trainer.FEATURE_FAMILY,"exact_distance_profile_te23_personal_weight2_v1")
        self.assertIn('sample_weight=np.where(fit_rows["Type of Travel"]',source)
        self.assertIn("sample_weight=sample_weight",source)
        self.assertIn('"sample_weight_personal_travel":2.0',control)
        self.assertIn('v2_9_22_subgroup_stability.md',control)
        self.assertNotIn('"competitions","submit"',control)
        self.assertIn('"enable_gpu":False',control)

    def test_feature_pipeline_remains_fold_safe(self):
        i=np.arange(110)
        frame=pd.DataFrame({
          "id":i,"Gender":np.where(i%2,"Male","Female"),
          "Customer Type":np.where(i%3,"Loyal Customer","disloyal Customer"),
          "Type of Travel":np.where(i%2,"Business travel","Personal Travel"),
          "Class":np.where(i%3,"Business","Eco"),"Age":i%59+18,
          "Flight Distance":200+(i%7)*101,
          "Departure Delay in Minutes":i%23,"Arrival Delay in Minutes":i%27,
          "satisfaction":i%2})
        for j,c in enumerate(trainer.SERVICES):frame[c]=(i+j)%6
        fit,hold,test=frame.iloc[:80].copy(),frame.iloc[80:100].copy(),frame.iloc[100:].copy()
        y=fit["satisfaction"].to_numpy()
        a,b,c=trainer.matrices(fit,hold,test,y,fold=0)
        self.assertEqual([x.shape for x in (a,b,c)],[(80,56),(20,56),(10,56)])
        self.assertTrue(a.columns.equals(b.columns) and b.columns.equals(c.columns))
        self.assertFalse("satisfaction" in a.columns or "id" in a.columns)
        weights=np.where(fit["Type of Travel"].astype("string").fillna("__NA__").to_numpy()=="Personal Travel",2.0,1.0)
        self.assertEqual(set(np.unique(weights)),{1.0,2.0})

    def test_metadata_requires_exact_weight_policy(self):
        m={"competition":pipeline.SLUG,"stage":"three_folds_complete",
           "model":trainer.MODEL,"validation":pipeline.VALIDATION,"folds":3,
           "fold_results":[{"fold":i} for i in range(3)],
           "train_rows":pipeline.N_TRAIN,"test_rows":pipeline.N_TEST,
           "model_seed":20261030,"feature_family":trainer.FEATURE_FAMILY,
           "route_profile_feature_count":12,"target_encoding_columns":23,
           "inner_target_folds":5,"sample_weight_policy":{"Personal Travel":2.0,"all_other_rows":1.0},
           "weighting_source":"Type of Travel feature only; no validation labels",
           "cpu_only":True,"gpu_verified":False,"data_source":"competition_train_only",
           "external_labels":False,"submitted":False,"leakage_safe_inner_cross_fit":True,
           "outer_validation_used_in_training":False}
        pipeline.validate_metrics(m)
        with self.assertRaisesRegex(ValueError,"provenance"):
            pipeline.validate_metrics({**m,"sample_weight_policy":{"Personal Travel":3.0,"all_other_rows":1.0}})

if __name__=="__main__":unittest.main()
