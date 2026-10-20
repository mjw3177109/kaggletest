"""Rank must match the newest *scored* V299 Kaggle submission, not older v2.5."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import v24_rank_tracker as rank

SLUG="playground-series-s6e10"

def save(folder,name,obj):
    path=Path(folder)/"state"/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj))

class NewChampRankTests(unittest.TestCase):
    def test_real_branch_current_champion(self):
        champion=rank.candidates_for(SLUG)
        self.assertIsNotNone(champion)
        self.assertGreaterEqual(champion["public_score"],.96083)
        self.assertIn(champion["source"],(
            "v2_9_13_submission.json","v2_9_15_submission.json"))
        if champion["source"]=="v2_9_13_submission.json":
            self.assertEqual(champion["submission_id"],57423879)

    def test_pending_candidate_does_not_override_official_score(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(rank,"ROOT",Path(tmp)):
            save(tmp,"v2_9_9_status.json",{
                "stage":"candidate_ready_unsubmitted","candidate_cv":.9610562})
            save(tmp,"v2_9_9_submission.json",{
                "competition":SLUG,"stage":"submission_reserved","submitted":False,
                "official_lb":.99,"submission_note":"demo",
                "submission":{"ref":57421208,"status":"SubmissionStatus.COMPLETE",
                  "description":"demo","publicScore":".99"}})
            self.assertIsNone(rank.candidates_for(SLUG))

    def test_scored_v299_wins_over_scored_v296(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(rank,"ROOT",Path(tmp)):
            save(tmp,"v2_9_6_status.json",{
                "competition":SLUG,"stage":"complete","submitted":True,
                "latest_submission":{"ref":57408736,"status":"SubmissionStatus.COMPLETE",
                  "date":"2026-10-19T20:35:00","publicScore":".96059"}})
            save(tmp,"v2_9_9_status.json",{
                "stage":"candidate_ready_unsubmitted","candidate_cv":.9610561936366048})
            save(tmp,"v2_9_9_submission.json",{
                "competition":SLUG,"stage":"complete","submitted":True,
                "official_lb":.96070,"submission_note":"KAS V299 demo",
                "submission":{"ref":57421208,"status":"SubmissionStatus.COMPLETE",
                  "description":"KAS V299 demo","publicScore":".96070",
                  "date":"2026-10-20T04:07:27"}})
            x=rank.candidates_for(SLUG)
            self.assertEqual(x["source"],"v2_9_9_submission.json")
            self.assertEqual(x["submission_id"],57421208)

    def test_completed_v2913_beats_prior_official(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(rank,"ROOT",Path(tmp)):
            save(tmp,"v2_9_13_status.json",{
                "stage":"candidate_ready_unsubmitted",
                "comparison":{"candidate_auc":.96117635576833}})
            save(tmp,"v2_9_13_submission.json",{
                "competition":SLUG,"stage":"complete","submitted":True,
                "official_lb":.96083,"submission_note":"KAS V2913 ENS8 demo",
                "submission":{"ref":57423879,"status":"SubmissionStatus.COMPLETE",
                  "description":"KAS V2913 ENS8 demo","publicScore":".96083"}})
            save(tmp,"v2_9_9_status.json",{
                "stage":"candidate_ready_unsubmitted","candidate_cv":.9610561936366048})
            save(tmp,"v2_9_9_submission.json",{
                "competition":SLUG,"stage":"complete","submitted":True,
                "official_lb":.96070,"submission_note":"KAS V299 demo",
                "submission":{"ref":57421208,"status":"SubmissionStatus.COMPLETE",
                  "description":"KAS V299 demo","publicScore":".96070"}})
            found=rank.candidates_for(SLUG)
            self.assertEqual(found["submission_id"],57423879)

    def test_pending_v2913_cannot_replace_scored_v299(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(rank,"ROOT",Path(tmp)):
            save(tmp,"v2_9_13_status.json",{
                "stage":"candidate_ready_unsubmitted",
                "comparison":{"candidate_auc":.96117635576833}})
            save(tmp,"v2_9_13_submission.json",{
                "competition":SLUG,"stage":"submission_reserved","submitted":False,
                "official_lb":.99999,"submission_note":"PREMATURE",
                "submission":{"ref":57423879,"status":"SubmissionStatus.COMPLETE",
                  "description":"PREMATURE","publicScore":".99999"}})
            save(tmp,"v2_9_9_status.json",{
                "stage":"candidate_ready_unsubmitted","candidate_cv":.9610561936366048})
            save(tmp,"v2_9_9_submission.json",{
                "competition":SLUG,"stage":"complete","submitted":True,
                "official_lb":.96070,"submission_note":"VALID",
                "submission":{"ref":57421208,"status":"SubmissionStatus.COMPLETE",
                  "description":"VALID","publicScore":".96070"}})
            self.assertEqual(rank.candidates_for(SLUG)["submission_id"],57421208)

    def test_wrong_description_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(rank,"ROOT",Path(tmp)):
            save(tmp,"v2_9_9_status.json",{
                "stage":"candidate_ready_unsubmitted","candidate_cv":.9610561936366048})
            save(tmp,"v2_9_9_submission.json",{
                "competition":SLUG,"stage":"complete","submitted":True,
                "official_lb":.96070,"submission_note":"AUTHENTIC",
                "submission":{"ref":57421208,"status":"SubmissionStatus.COMPLETE",
                  "description":"OTHER","publicScore":".96070"}})
            self.assertIsNone(rank.candidates_for(SLUG))

if __name__=="__main__":
    unittest.main()
