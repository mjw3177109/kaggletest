"""Future ranked V2915 score must be official, SHA-bound and non-reserved."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import v24_rank_tracker as ranked

SLUG="playground-series-s6e10"
SHA="abcd1234"+"e"*56
NOTE="KAS V2915 5E "+SHA[:16]

def save(folder,name,data):
    f=Path(folder)/"state"/name
    f.parent.mkdir(parents=True,exist_ok=True)
    f.write_text(json.dumps(data))

def experiment():
    return {"stage":"candidate_ready_unsubmitted","submitted":False,
            "candidate_test_hash":SHA,
            "comparison":{"candidate_auc":.96126000,"gain":.00008365}}

def official(stage="complete",score=".96090",note=NOTE):
    return {"competition":SLUG,"stage":stage,"submitted":stage=="complete",
            "submission_attempted":True,"prediction_hash":SHA,
            "submission_note":note,"official_lb":float(score),
            "submission":{"ref":57430001,"status":"SubmissionStatus.COMPLETE",
                          "description":note,"publicScore":score}}

class V2915RankSafety(unittest.TestCase):
    def test_completed_authenticated_submission_wins(self):
        with tempfile.TemporaryDirectory() as d,patch.object(ranked,"ROOT",Path(d)):
            save(d,"v2_9_15_status.json",experiment())
            save(d,"v2_9_15_submission.json",official())
            result=ranked.candidates_for(SLUG)
            self.assertEqual(result["submission_id"],57430001)
            self.assertEqual(result["source"],"v2_9_15_submission.json")
            self.assertAlmostEqual(result["public_score"],.96090)

    def test_reserved_unscored_candidate_must_not_appear(self):
        with tempfile.TemporaryDirectory() as d,patch.object(ranked,"ROOT",Path(d)):
            save(d,"v2_9_15_status.json",experiment())
            save(d,"v2_9_15_submission.json",official(stage="submission_reserved"))
            self.assertIsNone(ranked.candidates_for(SLUG))

    def test_score_downgrade_preserves_best_scored_prior(self):
        with tempfile.TemporaryDirectory() as d,patch.object(ranked,"ROOT",Path(d)):
            save(d,"v2_9_15_status.json",experiment())
            save(d,"v2_9_15_submission.json",official(score=".96010"))
            save(d,"v2_9_13_status.json",{
                "stage":"candidate_ready_unsubmitted",
                "comparison":{"candidate_auc":.96117635576833}})
            save(d,"v2_9_13_submission.json",{
                "competition":SLUG,"stage":"complete","submitted":True,
                "official_lb":.96083,"submission_note":"KAS V2913 ENS8 demo",
                "submission":{"ref":57423879,"status":"SubmissionStatus.COMPLETE",
                 "description":"KAS V2913 ENS8 demo","publicScore":".96083"}})
            result=ranked.candidates_for(SLUG)
            self.assertEqual(result["submission_id"],57423879)

    def test_sha_and_description_tampering_refused(self):
        with tempfile.TemporaryDirectory() as d,patch.object(ranked,"ROOT",Path(d)):
            save(d,"v2_9_15_status.json",experiment())
            bad=official(note="INVENTED")
            save(d,"v2_9_15_submission.json",bad)
            self.assertIsNone(ranked.candidates_for(SLUG))
            good=official()
            good["prediction_hash"]="f"*64
            save(d,"v2_9_15_submission.json",good)
            self.assertIsNone(ranked.candidates_for(SLUG))


    def test_checked_in_v2915_official_score_is_an_eligible_rank_source(self):
        # Keep public-score candidate selection tied to a real completed Kaggle
        # submission, even when Actions-generated state commits do not emit
        # follow-up push workflow events.
        path=ROOT/"state/v2_9_15_submission.json"
        if not path.is_file():
            return
        record=json.loads(path.read_text())
        if record.get("stage")!="complete" or record.get("submitted") is not True:
            return
        entry=record.get("submission") or {}
        if entry.get("status")!="SubmissionStatus.COMPLETE":
            return
        score=float(entry["publicScore"])
        winner=ranked.candidates_for(SLUG)
        self.assertIsNotNone(winner)
        self.assertGreaterEqual(winner["public_score"],score)
        if abs(winner["public_score"]-score)<1e-12:
            self.assertEqual(winner["submission_id"],entry["ref"])
            self.assertEqual(winner["source"],"v2_9_15_submission.json")

if __name__=="__main__":
    unittest.main()
