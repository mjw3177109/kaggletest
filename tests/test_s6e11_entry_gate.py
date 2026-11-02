"""Offline tests: never touch Kaggle, secrets, training or submission."""
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from kaggle_agent import s6e11_entry_gate as gate

TARGET = {
    "slug": gate.SLUG,
    "deadline": "2026-11-30T23:59:00Z",
    "userHasEntered": False,
}


class S6E11GateTests(unittest.TestCase):
    def setUp(self):
        clock = patch.object(gate, "datetime")
        mocked = clock.start()
        mocked.now.return_value = datetime(2026, 11, 2, tzinfo=timezone.utc)
        self.addCleanup(clock.stop)

    def test_workflow_has_bounded_success_only_fallback(self):
        workflow = (Path(__file__).resolve().parents[1] /
                    ".github/workflows/kaggle_s6e11_entry_gate.yml").read_text(encoding="utf-8")
        self.assertIn('cron: "23 * * * *"', workflow)
        self.assertIn('"Kaggle Scout"', workflow)
        self.assertIn('"Kaggle Dual GPU-TPU Top3 or Deadline Ladder"', workflow)
        self.assertIn("github.event.workflow_run.conclusion == 'success'", workflow)
        self.assertIn("cancel-in-progress: false", workflow)
        self.assertIn("ref: main", workflow)

    def test_not_joined_stays_blocked(self):
        result = gate.verify([TARGET], [], True)
        self.assertEqual(result["stage"], "blocked_awaiting_user_rule_acceptance")
        self.assertFalse(result["training_allowed"])
        self.assertFalse(result["submission_allowed"])

    def test_joined_requires_both_authenticated_sources(self):
        joined = dict(TARGET, userHasEntered=True)
        result = gate.verify([joined], [joined], True)
        self.assertEqual(result["stage"], "joined_pending_official_rules_review")
        self.assertTrue(result["userHasEntered"])
        self.assertFalse(result["training_allowed"])

    def test_disagreement_fails_closed(self):
        joined = dict(TARGET, userHasEntered=True)
        with self.assertRaises(RuntimeError):
            gate.verify([joined], [], True)
        with self.assertRaises(RuntimeError):
            gate.verify([TARGET], [joined], True)
        with self.assertRaises(RuntimeError):
            gate.verify([TARGET], [], False)

    def test_missing_and_ambiguous_target_fail_closed(self):
        with self.assertRaises(RuntimeError):
            gate.verify([], [], True)
        with self.assertRaises(RuntimeError):
            gate.verify([TARGET, TARGET], [], True)

    def test_invalid_deadline_fails_closed(self):
        with self.assertRaises(RuntimeError):
            gate.verify([dict(TARGET, deadline="")], [], True)

    @patch.object(gate.time, "sleep")
    @patch.object(gate.scout, "run")
    def test_pagination_is_bounded_and_rate_limited(self, fake_run, fake_sleep):
        fake_run.return_value = '[{"ref":"https://www.kaggle.com/competitions/playground-series-s6e11","deadline":"2026-11-30T23:59:00Z","userHasEntered":false}]'
        rows, exhaustive = gate.rows_for("entered", "all", 5)
        self.assertTrue(exhaustive)
        self.assertEqual(rows[0]["slug"], gate.SLUG)
        self.assertEqual(fake_run.call_count, 1)
        fake_sleep.assert_called_once_with(3)


if __name__ == "__main__":
    unittest.main()
