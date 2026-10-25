"""V2940 CPU-only safety tests. No Kaggle access, training run, or submission."""
import hashlib, json, sys, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import v2940_monotone_train as m

class FrozenMonotoneHGB(unittest.TestCase):
    def test_01_frozen_preregistration(self):
        s=json.loads((ROOT/"state/v2_9_40_status.json").read_text())
        self.assertEqual(s["version"],"V2.9.40")
        self.assertEqual(s["model_parameters"],m.PARAMS)
        self.assertEqual(s["model_seed"],m.MODEL_SEED)
        self.assertEqual(s["outer_seed"],m.SEED)
        self.assertEqual(s["baseline_features"],37)
        self.assertEqual(s["protected_target_encoded_features"],3)
        self.assertFalse(s["submitted"])
        self.assertFalse(s["submission_attempted"])
    def test_02_canonical_37_order(self):
        self.assertEqual(len(m.CANONICAL),37)
        self.assertEqual(m.CANONICAL[-3:],["te_0","te_1","te_2"])
        self.assertEqual(len(m.CATEGORICAL),14)
        self.assertEqual(len(set(m.CANONICAL)),37)
    def test_03_frozen_monotone_model(self):
        self.assertEqual(m.PARAMS,{"loss":"log_loss","learning_rate":0.035,
          "max_iter":600,"max_leaf_nodes":31,"max_depth":6,
          "min_samples_leaf":100,"l2_regularization":5.0,
          "max_bins":255,"early_stopping":False})
        self.assertEqual(m.MODEL_SEED,20261107)
        self.assertEqual(m.frozen_monotonic_vector(m.CANONICAL).count(1),4)
        self.assertEqual(tuple(m.MONOTONE),("Inflight wifi service","Online boarding",
                                             "Seat comfort","Inflight entertainment"))
    def test_04_pinned_versions(self):
        self.assertEqual(m.VERSIONS,{"numpy":"2.2.6",
          "pandas":"2.2.3","scipy":"1.15.3","sklearn":"1.6.1"})
    def test_05_no_reservation_fails_closed(self):
        with patch.dict("os.environ",{"GITHUB_RUN_ID":"","V2940_RESERVED_RUN_ID":""}):
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
        source=(ROOT/"kaggle_agent/v2940_monotone_train.py").read_text()
        self.assertIn("enc.fit_transform(te_train,y[itr])",source)
        self.assertIn("enc.transform(te_val)",source)
        self.assertIn("enc.transform(te_test)",source)
        self.assertNotIn("y[iva]",source.split("model.fit(")[0])
    def test_10_never_submit_or_push(self):
        source=(ROOT/"kaggle_agent/v2940_monotone_train.py").read_text()
        for forbidden in ("competitions submit","kernels push","subprocess.run","early_stopping_rounds"):
            self.assertNotIn(forbidden,source)
    def test_11_binary_data_checks(self):
        source=(ROOT/"kaggle_agent/v2940_monotone_train.py").read_text()
        self.assertIn("np.isin(y,[0,1])",source)
        self.assertIn("Duplicate official IDs",source)
        self.assertIn("Original 21 raw train columns changed",source)
    def test_12_all_fold_provenance(self):
        source=(ROOT/"kaggle_agent/v2940_monotone_train.py").read_text()
        for key in ("validation_index_sha256","category_vocabulary_sha256",
                    "source_code_sha256","environment_versions","train_id","test_id"):
            self.assertIn(key,source)

    def test_13_gate_frozen_prereg_and_branch(self):
        import v2940_actions_gate as g
        with patch.dict("os.environ",{"GITHUB_REF_NAME":g.BRANCH,"GITHUB_RUN_ID":"123456"}):
            s=g.assert_preregistration()
        self.assertEqual(s["stage"],"preregistered_not_started")
        self.assertEqual(s["model_parameters"],g.FROZEN)
    def test_14_gate_fails_on_existing_reservation(self):
        import v2940_actions_gate as g
        original=g.read
        def altered(name):
            s=original(name)
            if name=="state/v2_9_40_status.json":
                s["run_reservation_id"]="another-run"
            return s
        with patch.dict("os.environ",{"GITHUB_REF_NAME":g.BRANCH,"GITHUB_RUN_ID":"123456"}):
            with patch.object(g,"read",side_effect=altered):
                with self.assertRaises(RuntimeError):g.assert_preregistration()
    def test_15_owner_scan_includes_v2936(self):
        import v2940_actions_gate as g
        with patch.object(g.inherited,"assert_kaggle_idle_and_official") as scan:
            with patch.object(g.inherited,"kaggle",return_value="KernelWorkerStatus.ERROR") as status:
                g.assert_owner_idle()
        scan.assert_called_once()
        self.assertEqual(g.inherited.KERNEL,"__V2940_NO_KERNEL_TO_EXCLUDE__")
        self.assertEqual(status.call_args.args[0],["kernels","status",g.TERMINAL36])
    def test_16_owner_unknown_fails_closed(self):
        import v2940_actions_gate as g
        with patch.object(g.inherited,"assert_kaggle_idle_and_official",side_effect=RuntimeError("429")):
            with self.assertRaises(RuntimeError):g.assert_owner_idle()
    def test_17_reservation_fingerprint_changes_on_code(self):
        import v2940_actions_gate as g
        s=g.read("state/v2_9_40_status.json")
        original=g.fingerprint(s)
        self.assertEqual(len(original),64)
        s["model_seed"]+=1
        self.assertNotEqual(original,g.fingerprint(s))
    def test_18_validator_never_submits(self):
        txt=(ROOT/"kaggle_agent/v2940_pipeline.py").read_text()
        for forbidden in ("competitions submit","kernels push","early_stopping_rounds"):
            self.assertNotIn(forbidden,txt)
        self.assertIn("old.baseline_arrays(td,champion)",txt)
        self.assertIn("rate_limited_kaggle_subprocess()",txt)
        self.assertIn("validation_index_sha256",txt)
    def test_19_no_training_in_gate(self):
        txt=(ROOT/"kaggle_agent/v2940_actions_gate.py").read_text()
        self.assertNotIn("model.fit(",txt)
        self.assertNotIn("competitions submit",txt)
        self.assertNotIn("kernels push",txt)
        self.assertIn("assert_no_other_actions",txt)
        self.assertIn("assert_kaggle_idle_and_official",txt)
    def test_20_reservation_cannot_start_unreserved(self):
        import v2940_actions_gate as g
        with patch.dict("os.environ",{"GITHUB_RUN_ID":"123456"}):
            with self.assertRaises(RuntimeError):g.verify_reserved()

    def test_21_synthetic_monotone_cpu_fit(self):
        import v2940_actions_gate as g
        g.prove_cpu_synthetic_preflight()

    def test_22_only_four_raw_service_constraints(self):
        vector=m.frozen_monotonic_vector(m.CANONICAL)
        for name,value in zip(m.CANONICAL,vector):
            self.assertEqual(value,int(name in m.MONOTONE))
        self.assertEqual(len(vector),37)
        self.assertEqual(sum(vector),4)
        self.assertTrue(all(vector[m.CANONICAL.index("dual_"+x)]==0 for x in m.MONOTONE))
    def test_23_reordered_or_missing_features_fail_closed(self):
        bad=m.CANONICAL.copy()
        bad[0],bad[1]=bad[1],bad[0]
        with self.assertRaises(RuntimeError):m.frozen_monotonic_vector(bad)
        with self.assertRaises(RuntimeError):m.frozen_monotonic_vector(m.CANONICAL[:-1])
    def test_24_frozen_provenance_in_validator(self):
        txt=(ROOT/"kaggle_agent/v2940_pipeline.py").read_text()
        self.assertIn("metrics.get(\"monotonic_vector\")",txt)
        self.assertIn("info.get(\"monotonic_vector\")",txt)
        self.assertIn("train.frozen_monotonic_vector(train.CANONICAL)",txt)
if __name__=="__main__":unittest.main()
