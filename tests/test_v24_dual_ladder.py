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
        self.assertEqual(d["status"], "monitoring_deadline")
        self.assertIsNone(d["next_competition"])

    def test_public_top3_is_not_prize(self):
        d=ladder.decide({}, "playground-series-s6e10", {
            "userHasEntered": True, "userRank": 3
        }, [self.row("playground-next", joined=True)], self.when)
        self.assertEqual(d["status"], "rotated_waiting_task_preflight")
        self.assertEqual(d["active_competition"], "playground-next")
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
        self.assertEqual(d["status"], "monitoring_deadline")

    def test_near_submission_deadline_rotates_without_rank(self):
        old={"active_competition": "playground-series-s6e10",
             "active_deadline": (self.when + timedelta(hours=12)).isoformat()}
        d=ladder.decide(old, "playground-series-s6e10", {
            "ref": "playground-series-s6e10", "userHasEntered": True,
            "userRank": 0, "deadline": (self.when + timedelta(hours=12)).isoformat()
        }, [self.row("playground-next", joined=True)], self.when)
        self.assertEqual(d["status"], "rotated_waiting_task_preflight")
        self.assertIn("submission_deadline_window", d["handoff_reasons"])
        self.assertEqual(d["active_competition"], "playground-next")
        self.assertFalse(d["next_training_enabled"])

    def test_contest_ended_rotates_even_if_entered_listing_missing(self):
        old={"active_competition": "previous-competition",
             "active_deadline": (self.when - timedelta(minutes=1)).isoformat()}
        d=ladder.decide(old, "previous-competition", None,
                        [self.row("playground-next", joined=True)], self.when)
        self.assertEqual(d["active_competition"], "playground-next")
        self.assertIn("competition_ended", d["handoff_reasons"])
        self.assertIsNone(d.get("public_top3_achieved"))

    def test_not_entered_deadline_trigger_does_not_fake_join(self):
        old={"active_competition": "old-contest",
             "active_deadline": (self.when - timedelta(days=2)).isoformat()}
        d=ladder.decide(old, "old-contest", None,
                        [self.row("playground-next", joined=False)], self.when)
        self.assertEqual(d["status"], "next_awaits_rule_acceptance")
        self.assertEqual(d["active_competition"], "old-contest")
        self.assertIsNone(d["rank"])

    def test_missing_both_deadline_and_rank_fails_closed(self):
        d=ladder.decide({}, "unknown-future-contest", None,
                        [self.row("playground-next", joined=True)], self.when)
        self.assertEqual(d["status"], "rank_and_deadline_unverified")
        self.assertEqual(d["active_competition"], "unknown-future-contest")

    def test_entering_deadline_at_exact_48_hours(self):
        deadline=self.when + timedelta(hours=48)
        self.assertEqual(ladder.handoff_reasons(
            {"userHasEntered": True, "userRank": 100}, deadline, self.when
        ), ["submission_deadline_window"])

    def test_next_competition_needs_more_than_handoff_window(self):
        candidates=[
            self.row("expires-tomorrow", joined=True, hours=24),
            self.row("playground-eligible", joined=True, hours=96),
        ]
        items=ladder.next_candidates(candidates, "previous-competition", set(), self.when)
        self.assertEqual([x["competition"] for x in items], ["playground-eligible"])

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
