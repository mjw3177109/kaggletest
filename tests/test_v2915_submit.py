"""V2915 submission must be conditional, deterministic and impossible to retry blindly."""
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
import v2915_submit as m

class OneShotFiveEpochSafety(unittest.TestCase):
    def test_live_not_ready_candidate_must_not_submit(self):
        with patch.object(m.old,"check_competition"):
            with self.assertRaisesRegex(RuntimeError,"V2915 strict full-OOF"):
                m.qualified()

    def test_precommitted_threshold_and_exact_unique_note(self):
        self.assertEqual(m.MIN_GAIN,.00008)
        self.assertAlmostEqual(m.PREVIOUS_AUC,.96117635576833)
        self.assertEqual(m.NOTE_PREFIX,"KAS V2915 5E ")
        self.assertEqual(m.note("9"*64),"KAS V2915 5E "+"9"*16)

    def test_archived_provenance_mismatch_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError,"set mismatch"):
            m.check_provenance({"v25":{}},{"v28":{}})
        with self.assertRaisesRegex(RuntimeError,"SHA or notebook changed"):
            m.check_provenance({"v25":{"sha256":"a"}},{"v25":{"sha256":"b"}})

    def test_duplicate_reservation_refused_before_kaggle_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"state.json"
            path.write_text('{"stage":"submission_reserved","submission_attempted":true}')
            with patch.object(m,"STATE",path),patch.object(m,"score_file") as replay:
                with self.assertRaisesRegex(RuntimeError,"reservation exists"):
                    m.reserve()
                replay.assert_not_called()

    def test_only_signed_csv_is_eligible(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"candidate.csv"
            path.write_text("id,satisfaction\n123,0.42\n")
            good=hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(m,"OUTPUT",path):
                self.assertEqual(m.signed_csv({"candidate_file_sha256":good}),path)
                path.write_text("id,satisfaction\n123,0.99\n")
                with self.assertRaisesRegex(RuntimeError,"SHA256 mismatch"):
                    m.signed_csv({"candidate_file_sha256":good})

    def test_no_committed_reservation_means_no_kaggle_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"state.json"
            path.write_text(json.dumps({"stage":"submission_reserved",
                "submitted":False,"submission_attempted":True}))
            with (patch.object(m,"STATE",path),
                  patch.dict(os.environ,{"V2915_RESERVED_COMMITTED":"false"}),
                  patch.object(m.guard,"cli") as cli):
                with self.assertRaisesRegex(RuntimeError,"was not persisted"):
                    m.submit_once()
                cli.assert_not_called()

    def test_ambiguous_recovery_does_not_resubmit(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"state.json"
            path.write_text(json.dumps({"stage":"submission_ambiguous",
                "submission_attempted":True,"submitted":False,
                "submission_note":"KAS V2915 5E deadbeef"}))
            with (patch.object(m,"STATE",path),
                  patch.object(m.guard,"history",return_value=[]),
                  patch.object(m.guard,"cli") as cli):
                m.recover()
                cli.assert_not_called()
                self.assertEqual(json.loads(path.read_text())["stage"],
                                 "submission_ambiguous")

    def test_no_kernel_push_inside_submission_controller(self):
        contents=(ROOT/"kaggle_agent/v2915_submit.py").read_text()
        self.assertNotIn('"kernels","push"',contents)
        self.assertIn('guard.quota(rows)',contents)
        self.assertIn('s["stage"]="submitting"',contents)

if __name__=="__main__":
    unittest.main()
