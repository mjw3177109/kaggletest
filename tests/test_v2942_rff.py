"""V2942 offline safety: no Kaggle credentials, fitting official data, or submission."""
import json,sys,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import v2942_rff_train as m
class TestV2942(unittest.TestCase):
    def test_frozen_status(self):
        s=json.loads((ROOT/"state/v2_9_42_status.json").read_text())
        self.assertEqual(s["model_parameters"],m.PARAMS)
        self.assertFalse(s["prelaunch_allowed"])
        self.assertEqual(s["stage"],"preregistered_not_started")
    def test_schema(self):
        self.assertEqual((len(m.CANONICAL),len(m.CAT_ORDER),len(m.NUM_ORDER)),(37,14,23))
        self.assertEqual(len(m.frozen_schema(m.CANONICAL)),64)
    def test_no_reorder(self):
        a=m.CANONICAL.copy();a.reverse()
        with self.assertRaises(RuntimeError):m.frozen_schema(a)
    def test_pinned_versions(self):
        self.assertEqual(m.VERSIONS,{"numpy":"2.2.6","pandas":"2.2.3","scipy":"1.15.3","sklearn":"1.6.1"})
    def test_reservation_required(self):
        with patch.dict("os.environ",{"GITHUB_RUN_ID":"","V2942_RESERVED_RUN_ID":""}):
            with self.assertRaises(RuntimeError):m.prove_environment()
    def test_preprocess(self):
        p=m.preprocessor(0)
        self.assertEqual(p.sparse_threshold,1.0)
        self.assertEqual([x[0] for x in p.transformers],["categorical","numeric_linear","numeric_rff"])
    def test_synthetic(self):m.synthetic_preflight()
    def test_fold_local(self):
        s=(ROOT/"kaggle_agent/v2942_rff_train.py").read_text()
        for x in ("enc.fit_transform(te_train,y[itr])","enc.transform(te_val)","enc.transform(te_test)","prep.fit_transform(a)","prep.transform(b)","prep.transform(c)"):
            self.assertIn(x,s)
    def test_no_submission(self):
        for path in ("kaggle_agent/v2942_rff_train.py","kaggle_agent/v2942_pipeline.py"):
            s=(ROOT/path).read_text()
            for x in ("competitions submit","kernels push","early_stopping_rounds"):self.assertNotIn(x,s)
    def test_gate(self):
        import v2942_actions_gate as g
        with patch.dict("os.environ",{"GITHUB_REF_NAME":g.BRANCH,"GITHUB_RUN_ID":"12345"}):
            self.assertEqual(g.assert_preregistration()["stage"],"preregistered_not_started")
    def test_duplicate_run_block(self):
        import v2942_actions_gate as g
        old=g.read
        def alter(name):
            s=old(name)
            if name=="state/v2_9_42_status.json":s["run_reservation_id"]="prior"
            return s
        with patch.dict("os.environ",{"GITHUB_REF_NAME":g.BRANCH,"GITHUB_RUN_ID":"12345"}):
            with patch.object(g,"read",side_effect=alter):
                with self.assertRaises(RuntimeError):g.assert_preregistration()
    def test_owner_429(self):
        import v2942_actions_gate as g
        with patch.object(g.inherited,"assert_kaggle_idle_and_official",side_effect=RuntimeError("429")):
            with self.assertRaises(RuntimeError):g.assert_owner_idle()
    def test_fingerprint(self):
        import v2942_actions_gate as g
        s=g.read("state/v2_9_42_status.json")
        a=g.fingerprint(s);s["outer_seed"]+=1
        self.assertNotEqual(a,g.fingerprint(s))
    def test_never_train_in_gate(self):
        s=(ROOT/"kaggle_agent/v2942_actions_gate.py").read_text()
        self.assertNotIn("model.fit(",s)
        self.assertIn("assert_no_other_actions",s)
    def test_no_unreserved_verify(self):
        import v2942_actions_gate as g
        with patch.dict("os.environ",{"GITHUB_RUN_ID":"12345"}):
            with self.assertRaises(RuntimeError):g.verify_reserved()
    def test_validator_provenance(self):
        s=(ROOT/"kaggle_agent/v2942_pipeline.py").read_text()
        for x in ("old.baseline_arrays(td,champion)","rff_weights_sha256","validation_index_sha256","rate_limited_kaggle_subprocess"):self.assertIn(x,s)
    def test_workflow_dispatch_only(self):
        s=(ROOT/".github/workflows/kaggle_v2942_cpu_one_shot.yml").read_text()
        self.assertIn("workflow_dispatch:",s)
        self.assertNotIn("  push:",s)
    def test_rff_seed_and_gamma(self):
        p=m.preprocessor(2)
        r=p.transformers[2][1].named_steps["rff"]
        self.assertEqual(r.random_state,20261111)
        self.assertEqual(r.n_components,128)
        self.assertAlmostEqual(r.gamma,1/23)
    def test_v2941_terminal(self):
        s=json.loads((ROOT/"state/v2_9_41_status.json").read_text())
        self.assertTrue(s["retry_prohibited"])
        self.assertEqual(s["stage"],"blocked_prelaunch_no_training")
    def test_synthetic_full_fold_with_inner_te(self):
        import numpy as np
        import pandas as pd
        rng=np.random.default_rng(2942)
        rows=300
        raw={}
        for col in m.CANONICAL[:21]:
            if col in m.CATS:
                raw[col]=rng.choice(["A","B","C"],size=rows)
            else:
                raw[col]=rng.normal(size=rows)
        frame=pd.DataFrame(raw)
        frame.insert(0,"id",np.arange(rows,dtype=np.int64))
        y=np.array([0,1]*150,dtype=np.int8)
        itr=np.arange(200,dtype=np.int64)
        iva=np.arange(200,300,dtype=np.int64)
        pv,pt,info=m.fit_fold(frame,frame.iloc[:24].copy(),y,itr,iva,0)
        self.assertEqual(pv.shape,(100,))
        self.assertEqual(pt.shape,(24,))
        self.assertEqual(info["features"],37)
        self.assertEqual(info["feature_order"],m.CANONICAL)
        for key in ("rff_weights_sha256","numeric_linear_median_sha256",
                    "numeric_linear_scaler_sha256","numeric_rff_median_sha256",
                    "numeric_rff_scaler_sha256","validation_index_sha256"):
            self.assertEqual(len(info[key]),64)
    def test_prelaunch_permission_blocks_before_kaggle(self):
        import v2942_actions_gate as g
        with patch.dict("os.environ",{"GITHUB_REF_NAME":g.BRANCH,"GITHUB_RUN_ID":"safe-test"}):
            with patch.object(g,"assert_global_idle") as actions, patch.object(g,"assert_owner_idle") as owner:
                with self.assertRaisesRegex(RuntimeError,"launch blocked"):
                    g.reserve()
                actions.assert_not_called()
                owner.assert_not_called()
    def test_approval_without_authenticated_evidence_rejected(self):
        import v2942_actions_gate as g
        s=g.read("state/v2_9_42_status.json")
        s["prelaunch_allowed"]=True
        with self.assertRaisesRegex(RuntimeError,"evidence incomplete"):
            g.assert_launch_authorized(s)
    def test_expired_launch_approval_rejected(self):
        import v2942_actions_gate as g
        s=g.read("state/v2_9_42_status.json")
        s["prelaunch_allowed"]=True
        s["launch_authorization"]={
            "authenticated_owner_notebooks_verified":True,
            "official_submissions_verified":True,
            "official_deadline_verified":True,
            "champion_test_sha256":g.CHAMPION_SHA,
            "verified_at_utc":"2026-01-01T00:00:00Z"}
        with self.assertRaisesRegex(RuntimeError,"stale"):
            g.assert_launch_authorized(s)
    def test_workflow_persists_v2942_report_only(self):
        workflow=(ROOT/".github/workflows/kaggle_v2942_cpu_one_shot.yml").read_text()
        self.assertIn("reports/v2_9_42_result.md",workflow)
        self.assertNotIn("reports/v2_9_41_result.md",workflow)
    def test_original_data_guards(self):
        s=(ROOT/"kaggle_agent/v2942_rff_train.py").read_text()
        for x in ("Duplicate official IDs","Original 21 raw train columns changed","np.isin(y,[0,1])"):self.assertIn(x,s)
if __name__=="__main__":unittest.main()
