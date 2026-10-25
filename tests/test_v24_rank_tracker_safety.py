"""Offline safety regression for Kaggle rank polling; NEVER contacts Kaggle."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_agent"))
import v24_rank_tracker as r

class RankPacingSafety(unittest.TestCase):
    def test_cli_calls_are_spaced_at_least_three_seconds(self):
        mock_result = SimpleNamespace(returncode=0, stdout="[]", stderr="")
        with patch.object(r, "_last_kaggle_request_at", None), \
             patch.object(r.time, "monotonic", side_effect=[100.0,100.0,101.0,103.1]), \
             patch.object(r.time, "sleep") as sleep, \
             patch.object(r.subprocess, "run", return_value=mock_result) as run:
            r.rate_limited_kaggle_subprocess(["kaggle", "competitions", "list"], timeout=120)
            r.rate_limited_kaggle_subprocess(["kaggle", "competitions", "list"], timeout=120)
            self.assertEqual(run.call_count, 2)
            sleep.assert_called_once()
            self.assertGreaterEqual(sleep.call_args.args[0], 2.0)

    def test_non_kaggle_command_refused(self):
        with self.assertRaises(RuntimeError):
            r.rate_limited_kaggle_subprocess(["echo", "unsafe"], timeout=120)

    def test_429_aborts_without_fallback(self):
        with patch.object(r, "kaggle_json", side_effect=RuntimeError("HTTP 429 Too Many Requests")), \
             patch.object(r, "leaderboard_lookup") as fallback:
            with self.assertRaisesRegex(RuntimeError, "429"):
                r.best_rank("playground-series-s6e10", "mjw3177109")
            fallback.assert_not_called()

    def test_unrecognized_payload_is_not_silent_empty_list(self):
        fake = SimpleNamespace(returncode=0, stdout='{"unexpected":"shape"}', stderr="")
        with patch.object(r, "rate_limited_kaggle_subprocess", return_value=fake):
            with self.assertRaisesRegex(RuntimeError, "Unrecognized Kaggle"):
                r.kaggle_json(["--search", "playground-series-s6e10"])

    def test_unverified_refresh_preserves_existing_trusted_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "state").mkdir()
            (root / "state" / "dual_ladder.json").write_text(json.dumps({
                "active_competition": "playground-series-s6e10"}))
            (root / "state" / "v2_1_status.json").write_text("{}")
            previous = {"rank":984, "rank_verified":True,
                        "checked_at":"2026-10-25T02:44:10Z"}
            path = root / "state" / "score_rank_latest.json"
            path.write_text(json.dumps(previous))
            bad = {"rank":None,"rank_verified":False,"rank_source":"no_public_rank",
                   "score":{"submission_id":57446286}}
            with patch.object(r, "ROOT", root), patch.object(r, "STATE", path), \
                 patch.object(r, "REPORT", root / "reports" / "score_rank_latest.md"), \
                 patch.dict("os.environ", {"KAGGLE_API_TOKEN":"offline-placeholder",
                                          "KAGGLE_USERNAME":"mjw3177109"}), \
                 patch.object(r, "create_snapshot", return_value=bad):
                with self.assertRaisesRegex(RuntimeError, "preserve last trusted"):
                    r.main()
                self.assertEqual(json.loads(path.read_text()), previous)

    def test_rank_workflow_cannot_auto_poll_on_push(self):
        workflow = (ROOT / ".github/workflows/kaggle_v299_rank_refresh.yml").read_text()
        self.assertIn("workflow_dispatch:", workflow)
        self.assertNotIn("  push:", workflow)

    def test_both_kaggle_calls_use_serialized_wrapper(self):
        import inspect
        self.assertIn("rate_limited_kaggle_subprocess(", inspect.getsource(r.kaggle_json))
        self.assertIn("rate_limited_kaggle_subprocess(", inspect.getsource(r.leaderboard_lookup))

if __name__ == "__main__":
    unittest.main()
