"""V2.2 offline safety tests. No Kaggle token or competition data needed."""
import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_agent"))

from v22_pipeline import model_gate, kernel_handle


class V22Tests(unittest.TestCase):
    def test_source_compiles(self):
        for filename in ("v22_train.py", "v22_pipeline.py", "v22_guide.py"):
            src = (ROOT / "kaggle_agent" / filename).read_text(encoding="utf-8")
            ast.parse(src, filename)

    def test_cv_gate(self):
        self.assertTrue(model_gate(.95850, .95827867, .00008)["passed"])
        self.assertFalse(model_gate(.95830, .95827867, .00008)["passed"])
        self.assertFalse(model_gate(.95810, .95827867, .00008)["passed"])
        self.assertFalse(model_gate(.95835867, .95827867, .00008)["passed"])

    def test_real_kernel_slug(self):
        msg = "Kernel version 1 successfully pushed. Please check progress at https://www.kaggle.com/code/mjw3177109/kas-v22-playground-series-s6e10"
        self.assertEqual(
            kernel_handle(msg, "mjw3177109", "kas-v22-playground-series-s6e10"),
            "mjw3177109/kas-v22-playground-series-s6e10",
        )
        with self.assertRaises(ValueError):
            kernel_handle(msg, "mjw3177109", "kas-v2-2-playground-series-s6e10")
        with self.assertRaises(ValueError):
            kernel_handle("Push succeeded but no URL", "mjw3177109", "anything")


if __name__ == "__main__":
    unittest.main()
