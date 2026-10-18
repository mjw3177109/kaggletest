"""Offline tests: never invent user rank by sorting scores or matching display names."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"kaggle_agent"))
import v24_rank_tracker as r


class ScoreRankTests(unittest.TestCase):
    def test_official_account_rank_and_team_count(self):
        data=[{"ref":"playground-series-s6e10","userHasEntered":True,
               "userRank":124,"teamCount":587}]
        out=r.direct_rank(data,"playground-series-s6e10")
        self.assertEqual(out["rank"],124)
        self.assertEqual(out["teams"],587)
        self.assertEqual(out["source"],"kaggle_authenticated_userRank")

    def test_zero_rank_unverified(self):
        for zero in (None,0,"0","??"):
            result=r.direct_rank([{"ref":"playground-series-s6e10",
                                   "userHasEntered":True,"teamCount":999,
                                   "userRank":zero}],"playground-series-s6e10")
            self.assertIsNone(result["rank"])

    def test_not_joined_not_verified(self):
        result=r.direct_rank([{"ref":"playground-series-s6e10",
                               "userHasEntered":False,"userRank":1}],
                             "playground-series-s6e10")
        self.assertIsNone(result["rank"])

    def test_invalid_rank_greater_than_team_count(self):
        result=r.direct_rank([{"ref":"playground-series-s6e10",
                               "userHasEntered":True,"userRank":205,"teamCount":200}],
                             "playground-series-s6e10")
        self.assertIsNone(result["rank"])

    def test_leaderboard_exact_member_username(self):
        sample=("Rank,TeamName,TeamMembers,Score\n"
                "1,SomeOther,mjw3177109,0.98\n"
                "2,mjw3177109,wronguser,0.97\n"
                "3,AlsoOther,another,0.96\n")
        result=r.read_leaderboard_csv(sample,"mjw3177109")
        self.assertEqual(result["rank"],1)
        self.assertEqual(result["teams"],3)
        self.assertEqual(result["source"],"leaderboard_verified_member_username")

    def test_same_team_name_not_proof(self):
        sample=("Rank,TeamName,Score\n"
                "1,mjw3177109,0.99\n"
                "2,Other,0.98\n")
        result=r.read_leaderboard_csv(sample,"mjw3177109")
        self.assertIsNone(result["rank"])
        self.assertEqual(result["teams"],2)
        self.assertEqual(result["unverified_candidate_rank"],1)

    def test_no_guessed_rank(self):
        snap={"competition":"playground-series-s6e10",
              "rank":None,"ranked_teams":300,
              "rank_source":"no_public_rank",
              "score":{"public_score":0.95819,"model":"GPU depth7"},
              "checked_at":"2026-10-18T07:00:00Z",
              "leaderboard_url":"https://www.kaggle.com/competitions/playground-series-s6e10/leaderboard"}
        self.assertIn("官方排名暂未核实",r.render(snap))
        self.assertNotIn("第 1 名",r.render(snap))

    def test_score_change_invalidates_previous_rank(self):
        # Do not assert rank of a *new* GPU submission from an old score.
        previous_state=r.STATE
        try:
            from tempfile import TemporaryDirectory
            with TemporaryDirectory() as directory:
                r.STATE=Path(directory)/"score.json"
                r.save(r.STATE,{
                    "competition":"playground-series-s6e10",
                    "score":{"public_score":0.95819},
                    "rank":1596,"rank_verified":True,"ranked_teams":2221,
                    "checked_at":"2026-10-18T08:04:40+00:00",
                    "rank_source":"kaggle_authenticated_userRank",
                })
                stale=r.score_rank_lines("playground-series-s6e10",0.9584)
                self.assertIn("未核实",stale[0])
                current=r.score_rank_lines("playground-series-s6e10",0.95819)
                self.assertIn("第 1596 名",current[0])
                self.assertIn("2221",current[0])
                other=r.score_rank_lines("a-different-contest",0.95819)
                self.assertIn("未核实",other[0])
        finally:
            r.STATE=previous_state

    def test_successful_rank_print(self):
        snap={"competition":"playground-series-s6e10",
              "rank":23,"ranked_teams":480,
              "rank_source":"kaggle_authenticated_userRank",
              "score":{"public_score":0.95819,"model":"GPU depth7"},
              "checked_at":"2026-10-18T07:00:00Z",
              "leaderboard_url":"https://www.kaggle.com/competitions/playground-series-s6e10/leaderboard"}
        self.assertIn("第 23 名 / 共 480 支队伍",r.render(snap))


if __name__=="__main__":
    unittest.main()
