"""V2.3 GPU unit tests run without allocating Kaggle GPU resources."""
import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_agent"))
from v23_tpu_pipeline import check_cv


class GPUBranchTests(unittest.TestCase):
    def test_source_parsing(self):
        for path in ("v23_gpu_train.py", "v23_gpu_pipeline.py"):
            ast.parse((ROOT / "kaggle_agent" / path).read_text(encoding="utf-8"))

    def test_cv_rejects_non_improvement(self):
        self.assertFalse(check_cv(.95827, .9582786744608702, .00010)["passed"])
        self.assertTrue(check_cv(.9585, .9582786744608702, .00010)["passed"])

    def test_nvidia_and_cuda_are_verified(self):
        script = (ROOT / "kaggle_agent" / "v23_gpu_train.py").read_text(encoding="utf-8")
        pipeline = (ROOT / "kaggle_agent" / "v23_gpu_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("nvidia-smi", script)
        self.assertIn('"device", "")', script)
        self.assertIn('device="cuda"', script)
        self.assertIn('gpu_was_used(model)', script)
        self.assertIn('"NvidiaTeslaT4"', pipeline)
        self.assertIn('not m.get("gpu_verified")', pipeline)
        self.assertIn('str(f.get("device", "")).startswith("cuda")', pipeline)
        self.assertNotIn('write_json(ROOT / "state" / "v2_1_status.json"', pipeline)

if __name__ == "__main__":
    unittest.main()
