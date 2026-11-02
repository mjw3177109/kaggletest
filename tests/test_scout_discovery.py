"""Offline regression checks for Kaggle discovery; no credentials or network."""
import json
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from kaggle_agent import scout


class ScoutDiscoveryTests(unittest.TestCase):
    def test_cli_json_and_iso_deadline(self):
        rows = scout.parse_competitions(json.dumps([{
            "ref": "https://www.kaggle.com/competitions/playground-series-s6e11",
            "deadline": "2026-11-30T23:59:00Z",
            "category": "Playground",
            "userHasEntered": False,
        }]))
        self.assertEqual(rows[0]["slug"], "playground-series-s6e11")
        self.assertTrue(scout.is_open(rows[0]["deadline"], datetime(2026, 11, 2, tzinfo=timezone.utc)))
        self.assertGreater(scout.score(rows[0], ["playground", "featured"], datetime(2026, 11, 2, tzinfo=timezone.utc)), 40)

    def test_empty_and_legacy_table(self):
        self.assertEqual(scout.parse_competitions("[]"), [])
        rows = scout.parse_competitions(
            "ref  deadline  category  reward  teamCount  userHasEntered  userRank\n"
            "https://www.kaggle.com/competitions/example-contest  2026-11-30 23:59:00  Playground  3 Swag  100  False  0"
        )
        self.assertEqual(rows[0]["slug"], "example-contest")

    @patch.object(scout.time, "sleep")
    @patch.object(scout, "run")
    def test_category_discovery_and_entered_override(self, fake_run, fake_sleep):
        def respond(args):
            group = args[args.index("--group") + 1]
            category = args[args.index("--category") + 1]
            if category == "playground":
                return json.dumps([{"ref": "https://www.kaggle.com/competitions/playground-series-s6e11",
                                    "deadline": "2026-11-30T23:59:00Z", "category": "Playground",
                                    "userHasEntered": False}])
            if group == "entered":
                return json.dumps([{"ref": "https://www.kaggle.com/competitions/playground-series-s6e11",
                                    "deadline": "2026-11-30T23:59:00Z", "category": "Playground",
                                    "userHasEntered": True, "userRank": 0}])
            return "[]"
        fake_run.side_effect = respond
        rows = scout.discover_competitions(page_size=100, max_pages=1)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["userHasEntered"])
        self.assertEqual(fake_run.call_count, 5)
        self.assertEqual(fake_sleep.call_count, 4)


if __name__ == "__main__":
    unittest.main()
