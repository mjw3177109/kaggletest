"""V299 one-shot Kaggle submission regression checks; no live Kaggle writes."""
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
import v299_submit as submit

class ReservationSafety(unittest.TestCase):
    def test_live_gate_accepts_exact_v299_record(self):
        with patch.object(submit.base,"check_competition"):
            v=submit.verified_cv()
        self.assertAlmostEqual(v["candidate_cv"],submit.EXPECTED_AUC,places=7)
        self.assertEqual(v["stage"],"candidate_ready_unsubmitted")
        self.assertTrue(all(x>0 for x in v["fold_gain"]))

    def test_duplicate_reservation_refused_without_kaggle_access(self):
        with (tempfile.TemporaryDirectory() as d):
            path=Path(d)/"reservation.json"
            path.write_text('{"stage":"submission_reserved","submission_attempted":true}')
            with patch.object(submit,"STATE",path),patch.object(submit,"prepare_file") as build:
                with self.assertRaisesRegex(RuntimeError,"Existing submission state"):
                    submit.reserve()
                build.assert_not_called()

    def test_candidate_refuses_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/"test.csv"
            out.write_text("id,satisfaction\n1,0.5\n",encoding="utf-8")
            good=hashlib.sha256(out.read_bytes()).hexdigest()
            with patch.object(submit,"OUTPUT",out):
                self.assertEqual(submit.candidate({"candidate_file_sha256":good}),out)
                out.write_text("id,satisfaction\n1,0.1\n",encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError,"does not match"):
                    submit.candidate({"candidate_file_sha256":good})

    def test_no_reservation_means_no_recovery_action(self):
        with tempfile.TemporaryDirectory() as d:
            with (patch.object(submit,"STATE",Path(d)/"missing.json"),
                  patch.object(submit,"authenticated_history") as history,
                  patch.object(submit.protection,"cli") as cli):
                submit.recover()
                history.assert_not_called()
                cli.assert_not_called()

    def test_existing_note_readonly_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"reservation.json"
            path.write_text(json.dumps({
                "stage":"submission_ambiguous","submission_attempted":True,
                "submission_note":"KAS V299 FIXED4 deadbeef", "submitted":False}))
            with (patch.object(submit,"STATE",path),
                  patch.object(submit,"authenticated_history",return_value=[
                      {"description":"OTHER","ref":42}]),
                  patch.object(submit.protection,"cli") as cli):
                submit.recover()
                cli.assert_not_called()
                self.assertEqual(json.loads(path.read_text())["stage"],"submission_ambiguous")

    def test_submit_requires_committed_exact_reservation(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"reservation.json"
            path.write_text(json.dumps({
                "stage":"submission_reserved","submission_attempted":True,
                "submission_note":"KAS V299 FIXED4 deadbeef"}))
            with (patch.object(submit,"STATE",path),
                  patch.dict(os.environ,{"V299_RESERVED_COMMITTED":"false"}),
                  patch.object(submit.protection,"cli") as cli):
                with self.assertRaisesRegex(RuntimeError,"durably committed"):
                    submit.submit()
                cli.assert_not_called()

    def test_exact_fixed_candidate_and_source_hash(self):
        self.assertEqual(submit.NOTE_PREFIX,"KAS V299 FIXED4 ")
        self.assertAlmostEqual(submit.EXPECTED_AUC,.9610561936366048)
        self.assertEqual(len(submit.note_hash("f"*64)),len(submit.NOTE_PREFIX)+16)
        with self.assertRaisesRegex(RuntimeError,"provenance set"):
            submit.compare_artifacts({"v25":{}},{"v28":{}})

if __name__=="__main__":
    unittest.main()
