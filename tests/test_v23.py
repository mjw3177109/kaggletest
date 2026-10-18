"""Offline V2.3 tests; no Kaggle secret, data or TPU required."""
import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_agent"))

from v23_tpu_pipeline import check_cv


class TestV23TPU(unittest.TestCase):
    def test_sources_parse(self):
        for name in ("v23_tpu_train.py", "v23_tpu_pipeline.py", "v23_tpu_guide.py"):
            ast.parse((ROOT / "kaggle_agent" / name).read_text(encoding="utf-8"), filename=name)

    def test_submission_gate(self):
        self.assertTrue(check_cv(0.95850, 0.95827867, 0.00010)["passed"])
        self.assertFalse(check_cv(0.95830, 0.95827867, 0.00010)["passed"])
        self.assertFalse(check_cv(0.95800, 0.95827867, 0.00010)["passed"])

    def test_hardware_verification_mandatory(self):
        source = (ROOT / "kaggle_agent" / "v23_tpu_train.py").read_text(encoding="utf-8")
        self.assertIn('jax.default_backend() == "tpu"', source)
        self.assertIn('v23_hardware.json', source)
        self.assertIn('No', "No CPU fallback is permitted")
        pipeline = (ROOT / "kaggle_agent" / "v23_tpu_pipeline.py").read_text(encoding="utf-8")
        self.assertIn('"--accelerator", accelerator', pipeline)
        self.assertIn('"machine_shape": accelerator', pipeline)
        self.assertIn('hardware.get("actual") != "tpu"', pipeline)

    def test_trees_untouched(self):
        source = (ROOT / "kaggle_agent" / "v23_tpu_pipeline.py").read_text(encoding="utf-8")
        self.assertNotIn("v2_1_pipeline.py", source)
        self.assertNotIn("v2_1_status.json", "unintended update logic")


if __name__ == "__main__":
    unittest.main()
