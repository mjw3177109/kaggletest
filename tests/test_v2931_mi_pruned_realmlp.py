"""V2931 immutable fold-local MI selection, safety and no-submit invariants."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import v2931_pipeline as p
import v2931_mi_pruned_realmlp_train as m

def frames(n=120):
    rng=np.random.default_rng(20261031)
    base=pd.DataFrame({f"feature_{i:02d}":rng.normal(size=n).astype("float32") for i in range(33)})
    base.insert(0,"category",pd.Categorical(np.where(np.arange(n)%2,"A","B")))
    for i in range(3):
        base[f"te_{i}"]=rng.random(n).astype("float32")
    return base,base.iloc[:12].copy(),base.iloc[12:19].copy()

class V2931Safety(unittest.TestCase):
    def test_frozen_preregistration(self):
        state=p.read(p.STATUS)
        self.assertTrue(p.verify_prereg(state))
        self.assertEqual(p.KERNEL,"mjw3177109/kas-v2931-foldlocal-mi-pruned-realmlp-t4-s6e10")
        self.assertEqual(m.FEATURE_FAMILY,p.FEATURE_FAMILY)
        self.assertEqual(m.MODEL_SEED,20261031)
        self.assertEqual(p.NEW_WEIGHT,.15)
        self.assertEqual(state["selected_total_features"],29)
        self.assertFalse(state["submitted"])

    def test_mi_tie_break_keeps_canonical_order_and_protected_te(self):
        train,val,test=frames()
        y=np.array([0,1]*60,dtype=np.uint8)
        with patch("sklearn.feature_selection.mutual_info_classif",return_value=np.zeros(34)) as mi:
            a,b,c,ev=m.select_fold_features(train,val,test,y)
        self.assertEqual(a.shape,(120,29))
        self.assertEqual(b.shape,(12,29))
        self.assertEqual(c.shape,(7,29))
        eligible=ev["eligible_features"]
        expected=[col for col in train.columns if col in set(eligible[:26])|set(["te_0","te_1","te_2"])]
        self.assertEqual(ev["selected_features"],expected)
        self.assertEqual(list(a.columns),expected)
        self.assertEqual(len(ev["removed_features"]),8)
        self.assertEqual(ev["protected_target_encoded_features"],["te_0","te_1","te_2"])
        self.assertEqual(ev["discrete_feature_mask"],[True]+[False]*33)
        self.assertEqual(mi.call_args.kwargs["n_neighbors"],3)
        self.assertEqual(mi.call_args.kwargs["random_state"],20261031)
        np.testing.assert_array_equal(mi.call_args.args[1],y)

    def test_heldout_changes_do_not_change_mi_ranking(self):
        train,val,test=frames()
        y=np.array([0,1]*60,dtype=np.uint8)
        with patch("sklearn.feature_selection.mutual_info_classif",return_value=np.arange(34,dtype=float)):
            a,b,c,ev=m.select_fold_features(train,val,test,y)
            val2=val.copy()
            val2["feature_00"]=1e6
            a2,b2,c2,ev2=m.select_fold_features(train,val2,test,y)
        pd.testing.assert_frame_equal(a,a2)
        pd.testing.assert_frame_equal(c,c2)
        self.assertEqual(ev,ev2)

    def test_wrong_schema_and_nonfinite_mi_fail_closed(self):
        train,val,test=frames()
        y=np.array([0,1]*60,dtype=np.uint8)
        with self.assertRaises(ValueError):
            m.select_fold_features(train.drop(columns="te_2"),val,test,y)
        with patch("sklearn.feature_selection.mutual_info_classif",return_value=np.full(34,np.nan)):
            with self.assertRaises(ValueError):
                m.select_fold_features(train,val,test,y)

    def test_source_preserves_original_te_and_training_isolation(self):
        trainer=(ROOT/"kaggle_agent/v2931_mi_pruned_realmlp_train.py").read_text()
        pipe=(ROOT/"kaggle_agent/v2931_pipeline.py").read_text()
        gate=(ROOT/"kaggle_agent/v2931_actions_gate.py").read_text()
        self.assertIn('TargetEncoder(target_type="binary", smooth="auto", cv=5',trainer)
        self.assertIn('model.fit(xa, pd.Series(ytr))',trainer)
        self.assertIn('select_fold_features(xa, xv, xt, ytr)',trainer)
        self.assertIn("allow_pickle=False",pipe)
        self.assertIn("durably_reserve_unique_kernel(s)",pipe)
        self.assertIn("if recover_only:",pipe)
        self.assertIn("assert_no_other_actions()",gate)
        self.assertIn('read("state/v2_9_30_status.json")',gate)
        self.assertNotIn("kaggle competitions submit",pipe)
        self.assertNotIn("allow_pickle=True",pipe)

    def test_frozen_champion_and_no_submit_status(self):
        state=p.read(p.STATUS)
        official=p.read(ROOT/"state/v2_9_15_submission.json")
        previous=p.read(ROOT/"state/v2_9_30_status.json")
        self.assertEqual(state["official_submission_id"],official["submission"]["ref"])
        self.assertEqual(official["prediction_hash"],"906185547dcd5f0cac3ac173de644fd9997f20ac863c6745bcb1fd3e9a89e5f7")
        self.assertEqual(previous["stage"],"skipped_no_cv_gain")
        self.assertEqual(previous["source_artifacts"]["v2930"]["sha256"],"45e96e595676ec91a1e6c8eea98f975e63c9c0762412b898fe549c0636107362")
        self.assertEqual(state["submission_policy"],"evaluation_only_no_submission")
        self.assertFalse(state["submission_attempted"])

if __name__=="__main__":
    unittest.main()
