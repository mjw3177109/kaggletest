"""V2937 CPU-only safety tests. No Kaggle access, training run, or submission."""
import hashlib, json, sys, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import v2937_goss_train as m

class FrozenGoss(unittest.TestCase):
    def test_01_frozen_preregistration(self):
        s=json.loads((ROOT/"state/v2_9_37_status.json").read_text())
        self.assertEqual(s["version"],"V2.9.37")
        self.assertEqual(s["model_parameters"],m.PARAMS)
        self.assertEqual(s["model_seed"],m.MODEL_SEED)
        self.assertEqual(s["seed"],m.SEED)
        self.assertEqual(s["baseline_features"],37)
        self.assertEqual(s["protected_target_encoded_features"],3)
        self.assertFalse(s["submitted"])
        self.assertFalse(s["submission_attempted"])
    def test_02_canonical_37_order(self):
        self.assertEqual(len(m.CANONICAL),37)
        self.assertEqual(m.CANONICAL[-3:],["te_0","te_1","te_2"])
        self.assertEqual(len(m.CATEGORICAL),14)
        self.assertEqual(len(set(m.CANONICAL)),37)
    def test_03_goss_only(self):
        self.assertEqual(m.PARAMS["data_sample_strategy"],"goss")
        self.assertEqual(m.PARAMS["boosting_type"],"gbdt")
        self.assertEqual(m.PARAMS["n_estimators"],800)
        self.assertEqual(m.PARAMS["top_rate"],0.2)
        self.assertEqual(m.PARAMS["other_rate"],0.1)
        self.assertEqual(m.PARAMS["n_jobs"],4)
    def test_04_pinned_versions(self):
        self.assertEqual(m.VERSIONS,{"lightgbm":"4.6.0","numpy":"2.2.6",
          "pandas":"2.2.3","scipy":"1.15.3","sklearn":"1.6.1"})
    def test_05_no_reservation_fails_closed(self):
        with patch.dict("os.environ",{"GITHUB_RUN_ID":"","V2937_RESERVED_RUN_ID":""}):
            with self.assertRaises(RuntimeError):m.prove_environment()
    def test_06_fold_isolation_and_membership(self):
        from sklearn.model_selection import StratifiedKFold
        y=np.array([0,1]*15)
        folds=list(StratifiedKFold(3,shuffle=True,random_state=m.SEED).split(np.zeros(len(y)),y))
        self.assertEqual(sorted(np.concatenate([v for _,v in folds]).tolist()),list(range(len(y))))
        for tr,va in folds:
            self.assertFalse(set(tr)&set(va))
    def test_07_unknown_categories(self):
        df={}
        for col in m.CANONICAL:
            if col in m.CATEGORICAL:
                df[col]=pd.Categorical(["A","B"]*4,categories=["A","B"])
            else:
                df[col]=np.ones(8,dtype="float32")
        xa=pd.DataFrame(df)[m.CANONICAL]
        xv=xa.iloc[:4].copy()
        for col in m.CATEGORICAL:
            xv[col]=pd.Categorical(["UNKNOWN"]*4,categories=["A","B"])
        frames,digest=m.fold_matrix(xa,xv,xa)
        self.assertEqual(list(frames[0].columns),m.CANONICAL)
        self.assertTrue(all((frames[1][col]==-1).all() for col in m.CATEGORICAL))
        self.assertEqual(len(digest),64)
    def test_08_category_vocab_mismatch_blocked(self):
        df={}
        for col in m.CANONICAL:
            df[col]=(pd.Categorical(["A"]*4,categories=["A"]) if col in m.CATEGORICAL
                     else np.ones(4,dtype="float32"))
        xa=pd.DataFrame(df)[m.CANONICAL]
        xv=xa.copy()
        xv["Gender"]=pd.Categorical(["B"]*4,categories=["B"])
        with self.assertRaises(ValueError):m.fold_matrix(xa,xv,xa)
    def test_09_outer_train_only_te(self):
        source=(ROOT/"kaggle_agent/v2937_goss_train.py").read_text()
        self.assertIn("enc.fit_transform(te_train,y[itr])",source)
        self.assertIn("enc.transform(te_val)",source)
        self.assertIn("enc.transform(te_test)",source)
        self.assertNotIn("y[iva]",source.split("model.fit(")[0])
    def test_10_never_submit_or_push(self):
        source=(ROOT/"kaggle_agent/v2937_goss_train.py").read_text()
        for forbidden in ("competitions submit","kernels push","subprocess.run","early_stopping_rounds"):
            self.assertNotIn(forbidden,source)
    def test_11_binary_data_checks(self):
        source=(ROOT/"kaggle_agent/v2937_goss_train.py").read_text()
        self.assertIn("np.isin(y,[0,1])",source)
        self.assertIn("Duplicate official IDs",source)
        self.assertIn("Original 21 raw train columns changed",source)
    def test_12_all_fold_provenance(self):
        source=(ROOT/"kaggle_agent/v2937_goss_train.py").read_text()
        for key in ("validation_index_sha256","category_vocabulary_sha256",
                    "source_code_sha256","environment_versions","train_id","test_id"):
            self.assertIn(key,source)

if __name__=="__main__":unittest.main()
