from __future__ import annotations
import sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import v2922_subgroup_stability as audit


def synthetic(n=200):
    i=np.arange(n)
    return pd.DataFrame({
        "id":1000+i,"satisfaction":(i%3==0).astype(int),
        "Class":np.where(i%2,"Business","Eco"),
        "Type of Travel":np.where(i%4,"Business travel","Personal Travel"),
        "Customer Type":np.where(i%5,"Loyal Customer","disloyal Customer"),
        "Gender":np.where(i%2,"Male","Female"),
        "Flight Distance":200+(i%17)*71,
        "Age":18+(i%58),
        "Departure Delay in Minutes":i%90,
        "Arrival Delay in Minutes":i%110,
    })


class SubgroupAuditTests(unittest.TestCase):
    def test_oof_id_and_target_alignment_survives_shuffled_artifact(self):
        train=synthetic()
        order=np.random.default_rng(22).permutation(len(train))
        ids=train["id"].to_numpy()[order]
        target=train["satisfaction"].to_numpy()[order]
        pred=(.15+.7*target+.0005*np.arange(len(train)))[order]
        y,aligned=audit.align_predictions(train,ids,target,pred)
        self.assertTrue(np.array_equal(y,train["satisfaction"].to_numpy()))
        self.assertEqual(len(aligned),len(train))
        self.assertTrue(np.isfinite(aligned).all())

    def test_wrong_targets_duplicate_ids_and_nonfinite_predictions_fail_closed(self):
        train=synthetic()
        ids=train["id"].to_numpy()
        y=train["satisfaction"].to_numpy()
        pred=np.linspace(.1,.9,len(train))
        wrong=y.copy();wrong[0]=1-wrong[0]
        with self.assertRaisesRegex(ValueError,"target vector"):
            audit.align_predictions(train,ids,wrong,pred)
        duplicate=ids.copy();duplicate[0]=duplicate[1]
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            audit.align_predictions(train,duplicate,y,pred)
        bad=pred.copy();bad[0]=np.nan
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            audit.align_predictions(train,ids,y,bad)

    def test_subgroup_reports_are_descriptive_and_cover_expected_groups(self):
        train=synthetic()
        y=train["satisfaction"].to_numpy(dtype=np.int8)
        rng=np.random.default_rng(3)
        pred=np.clip(.15+.7*y+rng.normal(0,.1,len(y)),0,1)
        out=audit.subgroup_metrics(train,y,pred,min_rows=10)
        self.assertIn("Class",out)
        self.assertIn("Flight Distance quintile",out)
        self.assertIn("Departure Delay in Minutes band",out)
        self.assertIn("Age band",out)
        self.assertIn("Business",out["Class"])
        self.assertIn("auc",out["Class"]["Business"])
        self.assertTrue(0<=out["Class"]["Business"]["auc"]<=1)

    def test_audit_is_read_only_and_does_not_search_blend_weights(self):
        code=(ROOT/"kaggle_agent/v2922_subgroup_stability.py").read_text()
        self.assertIn('"training_performed":False',code)
        self.assertIn('"submission_performed":False',code)
        self.assertIn('"blend_weight_search":False',code)
        self.assertIn("kaggle kernels output",code)
        self.assertNotIn("kaggle competitions submit",code)
        self.assertNotIn("CatBoostClassifier(",code)


if __name__=="__main__":
    unittest.main()
