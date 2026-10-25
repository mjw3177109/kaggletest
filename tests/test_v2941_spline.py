"""V2941 offline safety: no Kaggle credentials, fitting official data, or submission."""
import json,sys,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import v2941_spline_train as m
class TestV2941(unittest.TestCase):
    def test_frozen_status(self):
        s=json.loads((ROOT/"state/v2_9_41_status.json").read_text())
        self.assertEqual(s["model_parameters"],m.PARAMS)
        self.assertEqual(s["stage"],"preregistered_not_started")
    def test_schema(self):
        self.assertEqual((len(m.CANONICAL),len(m.CAT_ORDER),len(m.SERVICE),len(m.NUM_ORDER)),(37,14,4,19))
        self.assertEqual(len(m.frozen_schema(m.CANONICAL)),64)
    def test_no_reorder(self):
        a=m.CANONICAL.copy();a.reverse()
        with self.assertRaises(RuntimeError):m.frozen_schema(a)
    def test_pinned_versions(self):
        self.assertEqual(m.VERSIONS,{"numpy":"2.2.6","pandas":"2.2.3","scipy":"1.15.3","sklearn":"1.6.1"})
    def test_reservation_required(self):
        with patch.dict("os.environ",{"GITHUB_RUN_ID":"","V2941_RESERVED_RUN_ID":""}):
            with self.assertRaises(RuntimeError):m.prove_environment()
    def test_preprocess(self):
        p=m.preprocessor()
        self.assertEqual(p.sparse_threshold,0.3)
        self.assertEqual([x[0] for x in p.transformers],["categorical","spline","numeric"])
    def test_synthetic(self):m.synthetic_preflight()
    def test_fold_local(self):
        s=(ROOT/"kaggle_agent/v2941_spline_train.py").read_text()
        for x in ("enc.fit_transform(te_train,y[itr])","enc.transform(te_val)","enc.transform(te_test)","prep.fit_transform(a)","prep.transform(b)","prep.transform(c)"):
            self.assertIn(x,s)
    def test_no_submission(self):
        for path in ("kaggle_agent/v2941_spline_train.py","kaggle_agent/v2941_pipeline.py"):
            s=(ROOT/path).read_text()
            for x in ("competitions submit","kernels push","early_stopping_rounds"):self.assertNotIn(x,s)
    def test_gate(self):
        import v2941_actions_gate as g
        with patch.dict("os.environ",{"GITHUB_REF_NAME":g.BRANCH,"GITHUB_RUN_ID":"12345"}):
            self.assertEqual(g.assert_preregistration()["stage"],"preregistered_not_started")
    def test_duplicate_run_block(self):
        import v2941_actions_gate as g
        old=g.read
        def alter(name):
            s=old(name)
            if name=="state/v2_9_41_status.json":s["run_reservation_id"]="prior"
            return s
        with patch.dict("os.environ",{"GITHUB_REF_NAME":g.BRANCH,"GITHUB_RUN_ID":"12345"}):
            with patch.object(g,"read",side_effect=alter):
                with self.assertRaises(RuntimeError):g.assert_preregistration()
    def test_owner_429(self):
        import v2941_actions_gate as g
        with patch.object(g.inherited,"assert_kaggle_idle_and_official",side_effect=RuntimeError("429")):
            with self.assertRaises(RuntimeError):g.assert_owner_idle()
    def test_fingerprint(self):
        import v2941_actions_gate as g
        s=g.read("state/v2_9_41_status.json")
        a=g.fingerprint(s);s["outer_seed"]+=1
        self.assertNotEqual(a,g.fingerprint(s))
    def test_never_train_in_gate(self):
        s=(ROOT/"kaggle_agent/v2941_actions_gate.py").read_text()
        self.assertNotIn("model.fit(",s)
        self.assertIn("assert_no_other_actions",s)
    def test_no_unreserved_verify(self):
        import v2941_actions_gate as g
        with patch.dict("os.environ",{"GITHUB_RUN_ID":"12345"}):
            with self.assertRaises(RuntimeError):g.verify_reserved()
    def test_validator_provenance(self):
        s=(ROOT/"kaggle_agent/v2941_pipeline.py").read_text()
        for x in ("old.baseline_arrays(td,champion)","spline_knots_sha256","validation_index_sha256","rate_limited_kaggle_subprocess"):self.assertIn(x,s)
    def test_workflow_dispatch_only(self):
        s=(ROOT/".github/workflows/kaggle_v2941_cpu_one_shot.yml").read_text()
        self.assertIn("workflow_dispatch:",s)
        self.assertNotIn("  push:",s)
    def test_original_data_guards(self):
        s=(ROOT/"kaggle_agent/v2941_spline_train.py").read_text()
        for x in ("Duplicate official IDs","Original 21 raw train columns changed","np.isin(y,[0,1])"):self.assertIn(x,s)
if __name__=="__main__":unittest.main()
