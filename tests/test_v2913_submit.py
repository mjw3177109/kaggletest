"""Safe one-shot V2913 submission: verify immutable CV, quota, reserve and recovery."""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import v2913_submit as m

class V2913SubmissionSafety(unittest.TestCase):
    def test_real_triple_fold_gpu_baseline_and_candidate(self):
        with patch.object(m.audit,"check_competition"):
            r=m.verified_cv()
        self.assertEqual(r["stage"],"candidate_ready_unsubmitted")
        self.assertAlmostEqual(r["comparison"]["candidate_auc"],.96117635576833)
        self.assertEqual(r["comparison"]["positive_folds"],3)
        self.assertEqual(r["metrics"]["model"],"RealMLP-TD n_ens=8 3 epochs scheduled")
        self.assertTrue(r["metrics"]["hardware"]["ok"])
        self.assertFalse(r["submitted"])

    def test_source_set_strict(self):
        with self.assertRaisesRegex(RuntimeError,"provenance set"):
            m.same_sources({"a":{"sha256":"x"}},{"b":{"sha256":"x"}})
        with self.assertRaisesRegex(RuntimeError,"checksum mismatch"):
            m.same_sources({"a":{"sha256":"x"}},{"a":{"sha256":"y"}})

    def test_no_repeated_reservation(self):
        with tempfile.TemporaryDirectory() as temp:
            state=Path(temp)/"status.json"
            state.write_text('{"stage":"submission_reserved","submission_attempted":true}')
            with patch.object(m,"STATE",state),patch.object(m,"replay_file") as reproduce:
                with self.assertRaisesRegex(RuntimeError,"reservation already exists"):
                    m.reserve()
                reproduce.assert_not_called()

    def test_candidate_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/"submission.csv"
            file.write_text("id,satisfaction\n1,0.49\n")
            hash_good=hashlib.sha256(file.read_bytes()).hexdigest()
            with patch.object(m,"OUTPUT",file):
                self.assertEqual(m.locked_candidate({"candidate_file_sha256":hash_good}),file)
                file.write_text("id,satisfaction\n1,0.60\n")
                with self.assertRaisesRegex(RuntimeError,"SHA mismatch"):
                    m.locked_candidate({"candidate_file_sha256":hash_good})

    def test_submitting_without_committed_state_never_uses_kaggle(self):
        with tempfile.TemporaryDirectory() as temp:
            status=Path(temp)/"status.json"
            status.write_text(json.dumps({"stage":"submission_reserved",
                "submitted":False,"submission_attempted":True}))
            with (patch.object(m,"STATE",status),
                  patch.dict(os.environ,{"V2913_RESERVED_COMMITTED":"false"}),
                  patch.object(m.kaggle_guard,"cli") as cli):
                with self.assertRaisesRegex(RuntimeError,"not durably persisted"):
                    m.submit()
                cli.assert_not_called()

    def test_recover_is_readonly_if_no_matching_record(self):
        with tempfile.TemporaryDirectory() as temp:
            status=Path(temp)/"status.json"
            status.write_text(json.dumps({"stage":"submission_ambiguous",
                "submission_note":"KAS V2913 ENS8 deadbeef",
                "submitted":False,"submission_attempted":True}))
            with (patch.object(m,"STATE",status),
                  patch.object(m.kaggle_guard,"history",return_value=[]),
                  patch.object(m.kaggle_guard,"cli") as cli):
                m.recover()
                cli.assert_not_called()
                self.assertEqual(json.loads(status.read_text())["stage"],"submission_ambiguous")

    def test_fixed_submission_name_and_no_auto_gpu_retrain(self):
        self.assertEqual(m.NOTE_PREFIX,"KAS V2913 ENS8 ")
        self.assertAlmostEqual(m.EXPECTED_CV,.96117635576833)
        self.assertEqual(m.note("1"*64),"KAS V2913 ENS8 "+"1"*16)
        source=(ROOT/"kaggle_agent/v2913_submit.py").read_text()
        self.assertIn('kaggle_guard.quota(rows)',source)
        self.assertIn('s["stage"]="submitting"',source)
        self.assertIn('"--wait","600"',source)
        self.assertNotIn('"kernels","push"',source)

if __name__=="__main__":
    unittest.main()
