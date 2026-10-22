import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_agent"))

import v2924_pipeline as pipeline
import v2924_personal_travel_specialist_train as model


class V2924PersonalTravelTests(unittest.TestCase):
    def test_residual_correction_changes_only_personal_travel_rows(self):
        base = np.array([0.10, 0.30, 0.50, 0.70, 0.90], dtype=np.float64)
        mask = np.array([False, True, False, True, False])
        specialist = np.array([0.80, 0.20], dtype=np.float64)
        result = pipeline.residual_correct(base, specialist, mask, alpha=0.10)
        np.testing.assert_array_equal(result[~mask], base[~mask])
        self.assertTrue(np.isfinite(result).all())
        self.assertFalse(np.array_equal(result[mask], base[mask]))

    def test_residual_correction_rejects_misaligned_inputs(self):
        with self.assertRaises(ValueError):
            pipeline.residual_correct(
                np.array([0.1, 0.2, 0.3]),
                np.array([0.7]),
                np.array([True, True, False]),
            )

    def test_v2923_must_be_confirmed_terminal_before_advancing(self):
        good = {
            "stage": "skipped_no_cv_gain",
            "kernel_state": "kernel_complete",
            "kernel_message": 'has status "KernelWorkerStatus.COMPLETE"',
        }
        self.assertTrue(pipeline.preregistered_v2923_is_terminal(good))
        running = dict(good)
        running["kernel_message"] = 'has status "KernelWorkerStatus.RUNNING"'
        self.assertFalse(pipeline.preregistered_v2923_is_terminal(running))
        ambiguous = dict(good)
        ambiguous["kernel_message"] = "status unknown"
        self.assertFalse(pipeline.preregistered_v2923_is_terminal(ambiguous))

    def test_residual_policy_is_fixed_and_no_auto_submit(self):
        self.assertEqual(pipeline.ALPHA, 0.10)
        self.assertEqual(pipeline.CV_FLOOR, 0.96125635576833)
        self.assertEqual(pipeline.MIN_GAIN, 0.00005)
        source = (ROOT / "kaggle_agent" / "v2924_pipeline.py").read_text(encoding="utf-8")
        self.assertIn('"submitted": False', source)
        self.assertNotIn("kaggle competitions submit", source)

    def test_interaction_features_have_ten_finite_columns(self):
        columns = list(model.SERVICES) + [
            "Departure Delay in Minutes", "Arrival Delay in Minutes",
            "Flight Distance", "Age",
        ]
        frame = pd.DataFrame({name: [1, 3, 5] for name in columns})
        result = model.interaction_features(frame)
        self.assertEqual(result.shape, (3, 10))
        self.assertTrue(np.isfinite(result.to_numpy(dtype=np.float64)).all())

    def test_target_keys_are_fold_safe_23_columns(self):
        values = {
            "Gender": ["Female", "Male", "Female"],
            "Customer Type": ["Loyal Customer", "disloyal Customer", "Loyal Customer"],
            "Age": [30, 44, 21],
            "Type of Travel": ["Personal Travel", "Business travel", "Personal Travel"],
            "Class": ["Eco", "Business", "Eco"],
            "Flight Distance": [500, 800, 500],
            "Departure Delay in Minutes": [0, 4, 2],
            "Arrival Delay in Minutes": [0, 7, 3],
        }
        for name in model.SERVICES:
            values[name] = [3, 4, 2]
        frame = pd.DataFrame(values)
        names = list(frame.columns)
        result = model.target_keys(frame, names)
        self.assertEqual(result.shape, (3, 23))
        self.assertFalse(any("satisfaction" in c.lower() or c.lower() == "id" for c in result.columns))
        self.assertFalse(result.isna().any().any())


if __name__ == "__main__":
    unittest.main()
