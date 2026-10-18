"""V2.5 offline checks for target leakage and safe GPU advancement."""
import ast
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import numpy as np
import pandas as pd
import v25_gpu_te_train as tr


class Tests(unittest.TestCase):
    def test_sources_parse(self):
        for filename in ["v25_gpu_te_train.py","v25_gpu_pipeline.py"]:
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
