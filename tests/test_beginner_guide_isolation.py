"""Regression: a new competition must not inherit another contest's results."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_agent import beginner_guide, v25_guide, v26_guide, v27_guide, v28_guide


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


    def test_legacy_experiment_chapters_do_not_leak_to_s6e11(self):
        """Old airline-specific experiment claims must stay in S6E10."""
        for version in ("5", "6", "7", "8"):
            self.write(f"state/v2_{version}_status.json", {
                "competition": "playground-series-s6e10",
                "stage": "complete",
                "metrics": {"best_cv_auc": 0.96035},
                "submitted": True,
            })
        for module in (v25_guide, v26_guide, v28_guide):
            with patch.object(module, "ROOT", self.root):
                new_chapter = module.section("playground-series-s6e11")
                old_chapter = module.section("playground-series-s6e10")
            self.assertIn("该比赛尚未启动", new_chapter)
            self.assertNotIn("Flight Distance", new_chapter)
            self.assertNotIn("0.95819", new_chapter)
            self.assertNotIn("航线画像", new_chapter)
            self.assertIn("0.96035", old_chapter)

        for slug in ("playground-series-s6e10", "playground-series-s6e11"):
            path = self.root / "competitions" / slug / "小白说明书.md"
            path.write_text(self.render(slug), encoding="utf-8")
        with patch.object(v27_guide, "ROOT", self.root):
            v27_guide.main()
        new_guide = (self.root / "competitions/playground-series-s6e11/小白说明书.md").read_text(encoding="utf-8")
        old_guide = (self.root / "competitions/playground-series-s6e10/小白说明书.md").read_text(encoding="utf-8")
        self.assertIn("该比赛尚未启动 V2.7 实验", new_guide)
        self.assertNotIn("50/50 排名融合", new_guide)
        self.assertIn("50/50 排名融合", old_guide)


if __name__ == "__main__":
    unittest.main()
