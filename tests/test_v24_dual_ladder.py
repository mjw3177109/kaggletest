"""Pure, offline unit tests for public-rank-gated contest switching."""
import json
import sys
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle_agent"))
import v24_dual_ladder as ladder


class LadderTests(unittest.TestCase):
    def setUp(self):
        self.when = datetime(2026, 10, 18, 8, tzinfo=timezone.utc)

    def row(self, slug, *, joined=False, category="Playground", hours=120):
        return {
            "ref": "https://www.kaggle.com/competitions/" + slug,
            "deadline": (self.when + timedelta(hours=hours)).isoformat(),
            "category": category,
            "userHasEntered": joined,
        }

    def test_public_rank_zero_never_promotes(self):
        for raw in [0, None, -1, "no-rank", 3.4, True]:
            self.assertIsNone(ladder.safe_rank({"userRank": raw}))
        self.assertEqual(ladder.safe_rank({"userRank": "3"}), 3)

    def test_unverified_rank_no_switch(self):
        d=ladder.decide({}, "playground-series-s6e10", {
            "userHasEntered": True, "userRank": 0
        }, [self.row("playground-next", joined=True)], self.when)
        self.assertEqual(d["status"], "rank_unverified")
        self.assertNotIn("next_competition", d)

    def test_public_top3_is_not_prize(self):
        d=ladder.decide({}, "playground-series-s6e10", {
            "userHasEntered": True, "userRank": 3
        }, [self.row("playground-next", joined=True)], self.when)
        self.assertEqual(d["status"], "next_entered_requires_task_check")
        self.assertEqual(d["next_competition"]["competition"], "playground-next")
        self.assertFalse(d["public_top3_achieved"]["official_prize_confirmed"])

    def test_not_entered_needs_explicit_rules(self):
        d=ladder.decide({}, "playground-series-s6e10", {
            "userHasEntered": True, "userRank": 2
        }, [self.row("playground-next")], self.when)
        self.assertEqual(d["status"], "next_awaits_rule_acceptance")

    def test_missing_current_entry_cannot_switch(self):
        d=ladder.decide({}, "playground-series-s6e10", None, [
            self.row("playground-next", joined=True)], self.when)
        self.assertEqual(d["status"], "rank_unverified")

    def test_only_valid_eligible_contests(self):
        raw=[
            self.row("almost-closed", joined=True, hours=5),
            self.row("arc-prize-2026", joined=True, category="Featured"),
            self.row("playground-next", joined=True, hours=95),
            self.row("playground-other", joined=False, hours=200),
        ]
        choices=ladder.next_candidates(raw, "playground-series-s6e10", set(), self.when)
        self.assertEqual([c["competition"] for c in choices], ["playground-next", "playground-other"])

    def test_older_top3_never_leaks_into_new_competition(self):
        old={"active_competition": "new-competition",
             "public_top3_achieved": {"competition": "previous", "rank": 1}}
        result=ladder.decide(old,"new-competition",{"userHasEntered":True,"userRank":99},[],self.when)
        self.assertEqual(result["status"],"working_toward_top3")
        self.assertNotEqual(result["rank"],1)

if __name__ == "__main__":
    unittest.main()
