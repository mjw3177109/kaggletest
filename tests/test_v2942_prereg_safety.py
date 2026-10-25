"""V2942 preregistration-only guard tests; no Kaggle credentials, data or model fitting."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))

class TestV2942Preregistration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = load("state/v2_9_42_status.json")
        cls.v41 = load("state/v2_9_41_status.json")
        cls.report = (ROOT / "reports/v2_9_42_preregistration.md").read_text(encoding="utf-8")

    def test_no_execution_or_submission(self):
        s = self.s
        self.assertEqual(s["stage"], "preregistered_not_started")
        self.assertFalse(s["training_started"])
        self.assertFalse(s["notebook_created"])
        self.assertFalse(s["submitted"])
        self.assertFalse(s["submission_attempted"])
        self.assertIsNone(s["run_reservation_id"])
        self.assertFalse(s["prelaunch_allowed"])

    def test_v2941_failed_attempt_never_retrigger(self):
        s = self.v41
        self.assertEqual(s["stage"], "blocked_prelaunch_no_training")
        self.assertEqual(s["first_launch_run_id"], "40185504849")
        self.assertTrue(s["retry_prohibited"])
        self.assertIsNone(s["run_reservation_id"])
        self.assertFalse(s["training_started"])
        self.assertFalse(s["submitted"])

    def test_frozen_source_and_outer_folds(self):
        s = self.s
        self.assertEqual((s["train_rows"], s["test_rows"]), (699635, 299844))
        self.assertEqual((s["baseline_features"], s["categorical_inputs"], s["numeric_inputs"]), (37, 14, 23))
        self.assertEqual((s["outer_folds"], s["outer_seed"], s["inner_target_folds"]), (3, 20261018, 5))
        self.assertEqual(s["protected_target_encoded_features"], 3)
        self.assertTrue(s["target_encoder"]["train_inner_oof_only"])

    def test_frozen_fourier_and_logistic(self):
        s = self.s
        self.assertAlmostEqual(s["rff"]["gamma"], 1 / 23)
        self.assertEqual(s["rff"]["n_components"], 128)
        self.assertEqual(s["rff"]["random_state"], "20261109+fold")
        self.assertEqual(s["rff"]["fit_on"], "outer_train_only")
        self.assertEqual(s["column_transformer"]["branches"], ["categorical", "numeric_linear", "numeric_rff"])
        self.assertEqual(s["model_parameters"], {
            "penalty": "l2", "C": 0.25, "solver": "lbfgs", "max_iter": 300,
            "tol": 0.0001, "fit_intercept": True, "class_weight": None,
        })

    def test_frozen_cpu_dependencies(self):
        self.assertEqual(self.s["environment"], {
            "python": "3.11", "numpy": "2.2.6", "pandas": "2.2.3",
            "scipy": "1.15.3", "sklearn": "1.6.1",
            "device": "cpu", "omp_num_threads": 4,
        })

    def test_blend_and_promotion(self):
        s = self.s
        self.assertEqual(s["blend"]["v2915_rank"], 0.85)
        self.assertEqual(s["blend"]["v2942_rank"], 0.15)
        self.assertEqual(s["blend"]["rank_method"], "average")
        self.assertTrue(s["blend"]["oof_and_test_ranked_separately"])
        self.assertEqual(s["promotion"], {
            "min_cv": 0.96125635576833, "min_gain_vs_v2915": 0.00005,
            "min_positive_folds": 2, "worst_fold_strict_gt": -0.00005,
        })

    def test_champion_integrity(self):
        s = self.s
        self.assertEqual(s["official_submission_id"], 57446286)
        self.assertEqual(s["champion_test_prediction_sha256"],
            "906185547dcd5f0cac3ac173de644fd9997f20ac863c6745bcb1fd3e9a89e5f7")
        self.assertEqual(s["baseline_cv"], 0.9612850399332336)

    def test_no_fresh_authenticated_rank_claim(self):
        self.assertFalse(self.s["kaggle_authenticated_owner_state_currently_verified"])
        self.assertEqual(self.s["last_persisted_authenticated_public_rank"]["freshness"], "historical_not_current")

    def test_preregistration_no_launch_permission(self):
        self.assertIn("not authorization to launch", self.report)
        self.assertIn(">=3 seconds", self.report)
        self.assertIn("429", self.report)
        self.assertIn("No implicit Kaggle rule acceptance", self.report)
        self.assertEqual(self.s["submission_policy"], "evaluation_only_no_submission")

if __name__ == "__main__":
    unittest.main()
