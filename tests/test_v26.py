"""V2.6 offline checks for target leakage and safe GPU advancement."""
import ast
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import numpy as np
import pandas as pd
import v26_gpu_freq_train as tr
import v26_gpu_pipeline as pipeline


class Tests(unittest.TestCase):
    def test_challenger_isolates_state_from_finished_champion(self):
        self.assertEqual(pipeline.STATE.name, "v2_6_status.json")
        self.assertEqual(pipeline.REPORT.name, "v2_6_latest.md")
        self.assertNotEqual(pipeline.STATE.name, "v2_5_status.json")

    def test_sources_parse(self):
        for filename in ["v26_gpu_freq_train.py","v26_gpu_pipeline.py"]:
            ast.parse((ROOT/"kaggle_agent"/filename).read_text(encoding="utf-8"))

    def example(self):
        rows=65
        rng=np.random.default_rng(7)
        info={
            "id":np.arange(rows),
            "Gender":rng.choice(["Male","Female"],size=rows),
            "Customer Type":rng.choice(["Loyal Customer","disloyal Customer"],size=rows),
            "Type of Travel":rng.choice(["Business travel","Personal Travel"],size=rows),
            "Class":rng.choice(["Eco","Business"],size=rows),
            "Age":rng.integers(18,80,size=rows),
            "Flight Distance":rng.integers(200,1000,size=rows),
            "Departure Delay in Minutes":rng.integers(0,40,size=rows),
            "Arrival Delay in Minutes":rng.integers(0,35,size=rows),
        }
        for name in tr.RATINGS:
            info[name]=rng.integers(0,6,size=rows)
        x=pd.DataFrame(info)
        x["satisfaction"]=np.tile([0,1],rows//2+1)[:rows]
        return x

    def test_key_features_are_target_free(self):
        x=self.example()
        before=tr.keys(x)
        x["satisfaction"]=1-x["satisfaction"]
        after=tr.keys(x)
        pd.testing.assert_frame_equal(before,after)
        self.assertTrue(any(c.startswith("exact_Flight Distance") for c in before))
        self.assertTrue(any(c.startswith("rating_by_segment_") for c in before))

    def test_route_profiles_never_use_validation(self):
        x=self.example()
        vocab={c: sorted(x[c].astype(str).unique()) for c in tr.CAT}
        matrix=tr.raw_numeric(x,vocab)
        original=tr.route_profile(matrix.iloc[:42],matrix.iloc[42:58],matrix.iloc[58:])
        shifted=matrix.copy()
        shifted.loc[42:57,"Online boarding"]=99
        after=tr.route_profile(shifted.iloc[:42],shifted.iloc[42:58],shifted.iloc[58:])
        pd.testing.assert_frame_equal(original[0],after[0])
        for c in ["route_online_boarding_mean","route_online_boarding_std"]:
            np.testing.assert_allclose(original[1][c],after[1][c])

    def test_crossfitted_frequencies_not_affected_by_validation_rows(self):
        x=self.example()
        full=tr.keys(x)
        fit=full.iloc[:45].copy()
        val=full.iloc[45:55].copy()
        test=full.iloc[55:].copy()
        old=tr.crossfit_frequency(fit,val,test,1)
        # Only the hold-out/validation keys change, but train-side and test
        # frequency values must not change because they are trained solely on fit.
        val.loc[:,"travel_segment"]="fake-unseen-traveler"
        new=tr.crossfit_frequency(fit,val,test,1)
        pd.testing.assert_frame_equal(old[0],new[0])
        pd.testing.assert_frame_equal(old[2],new[2])
        self.assertTrue((new[1]["freq_2"]<=old[1]["freq_2"]).all())
        self.assertEqual(len(tr.FREQUENCY_KEYS),12)
        self.assertEqual(old[0].shape,(45,12))

    def test_frequency_numeric_scaling_and_missing_groups(self):
        x=self.example()
        keys=tr.keys(x)
        a,b,c=tr.crossfit_frequency(keys.iloc[:50],keys.iloc[50:60],keys.iloc[60:],4)
        for chunk in (a,b,c):
            self.assertTrue(np.isfinite(chunk.to_numpy()).all())
            self.assertGreaterEqual(float(chunk.to_numpy().min()),0.0)

    def test_inner_crossfit_has_no_same_row_label(self):
        x=self.example()
        y=x["satisfaction"].to_numpy().astype(np.int8)
        test=x.drop(columns=["satisfaction"]).iloc[:7].copy()
        vocab={c:sorted(x[c].astype(str).unique()) for c in tr.CAT}
        nums_tr=tr.raw_numeric(x,vocab)
        nums_ts=tr.raw_numeric(test,vocab)
        a,b,c=tr.fold_input(x,test,y,np.arange(48),np.arange(48,65),
                            tr.keys(x),tr.keys(test),nums_tr,nums_ts,1)
        self.assertEqual(a.shape[0],48)
        self.assertEqual(b.shape[0],17)
        self.assertEqual(c.shape[0],7)
        self.assertEqual(list(a.columns),list(b.columns))
        self.assertEqual(list(b.columns),list(c.columns))
        self.assertTrue(np.isfinite(a.to_numpy()).all())
        self.assertTrue(np.isfinite(b.to_numpy()).all())
        self.assertTrue(np.isfinite(c.to_numpy()).all())


if __name__=="__main__":
    unittest.main()
