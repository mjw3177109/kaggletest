"""Safety regression: rank output must be attached only to the matching score."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"kaggle_agent"))
from v24_score_report_sync import refresh


class TestRankingReportSync(unittest.TestCase):
    def test_replaces_old_rank_without_touching_model_metrics(self):
        doc="# Run\n\n- CV 0.96066\n\n## Rank\n\n- rank unknown\n\n## Other\n\n- model provenance\n"
        result=refresh(doc,"## Rank",["- LB: 0.96035","- Rank: 958 / 2224","- Verified: now"])
        self.assertEqual(result.count("## Rank"),1)
        self.assertIn("CV 0.96066",result)
        self.assertIn("Rank: 958 / 2224",result)
        self.assertNotIn("rank unknown",result)
        self.assertIn("## Other\n\n- model provenance",result)

    def test_adds_section_to_older_report(self):
        doc="# Old GPU Model\n\nScore 0.95819\n"
        result=refresh(doc,"## Public ranking",["- Ranking not verified"])
        self.assertIn("Score 0.95819",result)
        self.assertIn("## Public ranking",result)

    def test_idempotent_and_same_report(self):
        doc="# Model\n\n## Current\n\nold rank\n\n## Notes\n\ntext\n"
        one=refresh(doc,"## Current",["rank verified: 958"])
        two=refresh(one,"## Current",["rank verified: 958"])
        self.assertEqual(one,two)


if __name__=="__main__":
    unittest.main()
