"""V2926 invariant gates; no remote Kaggle write in tests."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import v2926_pipeline as p
import v2926_route_realmlp_train as m


class V2926SafeTraining(unittest.TestCase):
    def test_frozen_registration_matches_distinct_model(self):
        state=p.read(p.STATUS)
        self.assertTrue(p.verify_prereg(state))
        self.assertEqual(p.NEW_WEIGHT,.20)
        self.assertEqual(p.MIN_GAIN,.00005)
        self.assertEqual(p.KERNEL,"mjw3177109/kas-v2926-route-profile-realmlp-t4-s6e10")
        self.assertNotEqual(m.MODEL_SEED,20261021)
        self.assertEqual(m.MODEL_SEED,20261026)

    def test_route_features_fit_only_on_outer_training(self):
        def frame(distances, rating):
            result=pd.DataFrame({"Flight Distance":distances,"Class":["Eco"]*len(distances)})
            for column in m.SERVICES:
                result[column]=rating
            return result
        fit=frame([40,40,100],2)
        validation=frame([40,999],5)
        test=frame([100,999],1)
        a,b,c=m.route_features(fit,validation,test)
        self.assertEqual([len(z.columns) for z in (a,b,c)],[12,12,12])
        self.assertTrue(a.columns.equals(b.columns) and b.columns.equals(c.columns))
        self.assertTrue(all(np.isfinite(z.to_numpy(dtype=np.float64)).all() for z in (a,b,c)))
        changed=validation.copy()
        changed.loc[:,"Inflight wifi service"]=1
        a2,b2,c2=m.route_features(fit,changed,test)
        pd.testing.assert_frame_equal(a,a2)
        pd.testing.assert_frame_equal(c,c2)
        self.assertTrue(np.allclose(a["route_profile_rating_mean_mean"].to_numpy(),2))
        self.assertAlmostEqual(float(b["route_profile_inflight_wifi_service_residual"].iloc[0]),3)

    def test_fixed_rank_fusion_no_optimization(self):
        base=np.array([.1,.7,.3,.9,.4],dtype=float)
        other=np.array([.6,.2,.5,.8,.3],dtype=float)
        old_rank=p.v24.v23.audit.rank
        candidate=.8*old_rank(base)+.2*old_rank(other)
        self.assertEqual(len(candidate),5)
        self.assertTrue(np.isfinite(candidate).all())

    def test_explicit_no_submit_and_recovery_only(self):
        source=(ROOT/"kaggle_agent"/"v2926_pipeline.py").read_text(encoding="utf-8")
        for token in ('"--recover-only"', '"kernels", "push"', '"submitted": False'):
            self.assertIn(token,source)
        self.assertNotIn('"competitions", "submit"',source)
        self.assertNotIn("kaggle competitions submit",source)
        self.assertIn("if recover_only:",source)
        self.assertIn("Kaggle owner list failed",source)


if __name__=="__main__":
    unittest.main()
