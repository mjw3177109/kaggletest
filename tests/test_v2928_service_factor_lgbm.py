"""V2928 CPU LightGBM fold-local PCA, cross-fit and anti-duplicate safety tests."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import v2928_pipeline as p
import v2928_service_factor_lgbm_train as m


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


class V2928SafeExperiment(unittest.TestCase):
    def test_preregistered_params_and_slug(self):
        state=p.read(p.STATUS)
        self.assertTrue(p.verify_prereg(state))
        self.assertEqual(state["model_parameters"],p.frozen_params())
        self.assertEqual(state["model_parameters"],m.MODEL_PARAMETERS)
        self.assertEqual(p.KERNEL,"mjw3177109/kas-v2928-service-factor-pca-lgbm-cpu-s6e10")
        self.assertFalse(state["notebook_created"])
        self.assertFalse(state["training_started"])

    def test_pca_fits_outer_training_only(self):
        fit=frame(50,1);hold=frame(10,2);test=frame(8,3)
        a,b,c=m.service_factor_features(fit,hold,test)
        self.assertEqual([z.shape for z in (a,b,c)],[(50,11),(10,11),(8,11)])
        self.assertTrue(a.columns.equals(b.columns) and b.columns.equals(c.columns))
        self.assertTrue(all(np.isfinite(z.to_numpy()).all() for z in (a,b,c)))
        changed=hold.copy()
        changed.loc[:,"Inflight wifi service"]=1000
        a2,b2,c2=m.service_factor_features(fit,changed,test)
        pd.testing.assert_frame_equal(a,a2)
        pd.testing.assert_frame_equal(c,c2)
        self.assertFalse(np.array_equal(b.to_numpy(),b2.to_numpy()))
        self.assertEqual(len(set(a.columns)),11)

    def test_outer_training_imputation_not_validation(self):
        fit=frame(50,4);hold=frame(10,5);test=frame(8,6)
        fit.loc[:,"Seat comfort"]=np.nan
        with self.assertRaisesRegex(ValueError,"All-missing"):
            m.service_factor_features(fit,hold,test)

    def test_crossfit_23_te_plus_11_features(self):
        fit=frame(70,11);hold=frame(12,12);test=frame(7,13)
        y=np.asarray([0,1]*35,dtype=np.int8)
        a,b,c=m.matrices(fit,hold,test,y,0)
        self.assertEqual([z.shape for z in (a,b,c)],[(70,55),(12,55),(7,55)])
        self.assertEqual(len([x for x in a.columns if x.startswith("te_")]),23)
        self.assertEqual(len([x for x in a.columns if x.startswith("service_")]),11)
        self.assertTrue(all(np.isfinite(z.to_numpy()).all() for z in (a,b,c)))

    def test_no_submit_and_recover_only(self):
        source=(ROOT/"kaggle_agent"/"v2928_pipeline.py").read_text(encoding="utf-8")
        self.assertIn('"--recover-only"',source)
        self.assertIn('durably_reserve_unique_kernel(s)',source)
        self.assertIn('if recover_only:',source)
        self.assertIn('allow_pickle=False',source)
        self.assertNotIn('"competitions","submit"',source)
        self.assertNotIn("kaggle competitions submit",source)
        self.assertNotIn("allow_pickle=True",source)


if __name__=="__main__":
    unittest.main()
