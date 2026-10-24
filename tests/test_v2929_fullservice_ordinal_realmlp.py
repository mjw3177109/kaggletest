"""V2929 isolated preregistration, fold-local thresholds, recovery and anti-submit tests."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import v2929_pipeline as p
import v2929_fullservice_ordinal_realmlp_train as m

def frame(n,seed=1):
    rng=np.random.default_rng(seed)
    f=pd.DataFrame(index=np.arange(n))
    for col in m.SERVICES:
        f[col]=rng.integers(0,6,n).astype(float)
    f["Gender"]=np.where(np.arange(n)%2,"Female","Male")
    f["Customer Type"]=np.where(np.arange(n)%3,"Loyal Customer","disloyal Customer")
    f["Type of Travel"]=np.where(np.arange(n)%2,"Business travel","Personal Travel")
    f["Class"]=np.where(np.arange(n)%3,"Eco","Business")
    f["Flight Distance"]=rng.integers(100,5000,n)
    f["Age"]=rng.integers(18,80,n)
    f["Departure Delay in Minutes"]=rng.integers(0,100,n)
    f["Arrival Delay in Minutes"]=rng.integers(0,100,n)
    return f

class V2929Safety(unittest.TestCase):
    def test_preregistered_owner_and_immutable_features(self):
        s=p.read(p.STATUS)
        self.assertTrue(p.verify_prereg(s))
        if s["stage"]=="preregistered_not_started":
            self.assertEqual(s["kernel"],"")
            self.assertFalse(s["notebook_created"])
            self.assertFalse(s["training_started"])
        else:
            self.assertEqual(s["kernel"],p.KERNEL)
            self.assertFalse(s["submitted"])
            self.assertFalse(s["submission_attempted"])
        self.assertEqual(s["blend"],{"v2915_rank":.85,"v2929_rank":.15})
        self.assertEqual(s["feature_count"],110)
        self.assertEqual(m.MODEL_SEED,20261029)
        self.assertEqual(len(m.SERVICES),13)
        self.assertEqual(len(m.ADDITIONAL),8)
        self.assertEqual(m.THRESHOLDS,(1,2,3,4,5))
        self.assertEqual(p.KERNEL,"mjw3177109/kas-v2929-fullservice-ordinal-realmlp-t4-s6e10")

    def test_ordinal_feature_schema_and_unknown_category(self):
        fit=frame(45,1);hold=frame(9,2)
        fit["Gate location"]=1.0
        hold["Gate location"]=9.0
        a,b,_,_=m.engineer(fit,hold)
        self.assertEqual(a.shape[1],34)
        a,b=m.augment_service(fit,hold,a,b)
        self.assertEqual(a.shape[1],107)
        self.assertEqual(b.shape[1],107)
        self.assertTrue(a.columns.equals(b.columns))
        ordcols=[x for x in a if x.startswith("ordinal_ge")]
        catcols=[x for x in a if x.startswith("dual_ordinal_")]
        self.assertEqual(len(ordcols),65)
        self.assertEqual(len(catcols),8)
        self.assertEqual(len(set(a.columns)),107)
        self.assertTrue(all(str(a[x].dtype)=="category" for x in catcols))
        self.assertTrue(pd.isna(b["dual_ordinal_gate_location"].iloc[0]))
        self.assertEqual(float(b["ordinal_ge5_gate_location"].iloc[0]),1.0)

    def test_missing_medians_only_outer_fit(self):
        fit=frame(40,3);hold=frame(9,4)
        fit["Seat comfort"]=2.0
        hold["Seat comfort"]=np.nan
        a,b,_,_=m.engineer(fit,hold)
        a,b=m.augment_service(fit,hold,a,b)
        self.assertTrue((b["ordinal_ge2_seat_comfort"]==1).all())
        self.assertTrue((b["ordinal_ge3_seat_comfort"]==0).all())
        bad=fit.copy()
        bad["Seat comfort"]=np.nan
        a2,b2,_,_=m.engineer(bad,hold)
        with self.assertRaisesRegex(ValueError,"all-missing"):
            m.augment_service(bad,hold,a2,b2)

    def test_holdout_does_not_change_outer_fit_features(self):
        fit=frame(40,6);hold=frame(7,7);test=frame(6,8)
        a,b,_,_=m.engineer(fit,hold)
        a,b=m.augment_service(fit,hold,a,b)
        c,d,_,_=m.engineer(fit,test)
        c,d=m.augment_service(fit,test,c,d)
        changed=hold.copy()
        changed["Inflight wifi service"]=10000
        a2,b2,_,_=m.engineer(fit,changed)
        a2,b2=m.augment_service(fit,changed,a2,b2)
        pd.testing.assert_frame_equal(a,a2)
        pd.testing.assert_frame_equal(a,c)
        self.assertTrue(a.columns.equals(d.columns))
        self.assertFalse(np.array_equal(b["ordinal_ge5_inflight_wifi_service"],b2["ordinal_ge5_inflight_wifi_service"]))

    def test_no_submit_recovery_only_and_fold_oof(self):
        src=(ROOT/"kaggle_agent/v2929_pipeline.py").read_text(encoding="utf-8")
        train=(ROOT/"kaggle_agent/v2929_fullservice_ordinal_realmlp_train.py").read_text(encoding="utf-8")
        for token in ('"--recover-only"',"durably_reserve_unique_kernel(s)","allow_pickle=False"):
            self.assertIn(token,src)
        self.assertNotIn('kaggle competitions submit',src)
        self.assertNotIn('allow_pickle=True',src)
        self.assertIn("model.fit(xa, pd.Series(ytr))",train)
        self.assertIn("TargetEncoder(target_type=\"binary\", smooth=\"auto\", cv=5",train)
        self.assertIn("StratifiedKFold(FOLDS,shuffle=True,random_state=SEED)",train)
        self.assertIn("if len(details)!=3",train)

if __name__=="__main__":
    unittest.main()
