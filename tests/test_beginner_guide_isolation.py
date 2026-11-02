"""Regression: a new competition must not inherit another contest's results."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_agent import beginner_guide


class BeginnerGuideContestIsolationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "state").mkdir()
        for slug in ("playground-series-s6e10", "playground-series-s6e11"):
            (self.root / "competitions" / slug).mkdir(parents=True)
        self.write("state/v2_status.json", {
            "competition": "playground-series-s6e10", "stage": "complete",
            "metrics": {"validation_score": 0.956928},
            "latest_submission": {"publicScore": "0.95738"}
        })
        self.write("state/v2_1_status.json", {
            "competition": "playground-series-s6e10", "stage": "complete",
            "public_score_delta": 0.00036,
            "latest_submission": {"publicScore": "0.95774"}
        })
        self.write("state/s6e11_entry_gate.json", {
            "competition": "playground-series-s6e11",
            "stage": "blocked_awaiting_user_rule_acceptance",
            "userHasEntered": False, "training_allowed": False,
            "submission_allowed": False
        })

    def write(self, path, data):
        file = self.root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(data), encoding="utf-8")

    def render(self, slug):
        with patch.object(beginner_guide, "ROOT", self.root):
            return beginner_guide.render_guide(slug)

    def test_unjoined_s6e11_does_not_inherit_s6e10_cv_or_submission(self):
        guide = self.render("playground-series-s6e11")
        self.assertIn("尚未加入比赛", guide)
        self.assertIn("当前模型：`尚未训练`", guide)
        self.assertIn("Kaggle Public LB：`暂无`", guide)
        self.assertIn("本地验证分数（CV）：`暂无`", guide)
        self.assertIn("暂无 V2.1 线上对比", guide)
        self.assertNotIn("0.95774", guide)
        self.assertNotIn("0.956928", guide)
        self.assertNotIn("已经完成训练、提交和结果回写**", guide)

    def test_s6e10_keeps_its_own_verified_history(self):
        guide = self.render("playground-series-s6e10")
        self.assertIn("已经完成训练、提交和结果回写", guide)
        self.assertIn("0.95774", guide)
        self.assertIn("0.956928", guide)

    def test_unknown_contest_is_untrained_not_complete(self):
        slug = "another-competition"
        (self.root / "competitions" / slug).mkdir()
        guide = self.render(slug)
        self.assertIn("尚未开始", guide)
        self.assertIn("尚未训练", guide)
        self.assertNotIn("0.95774", guide)


if __name__ == "__main__":
    unittest.main()
