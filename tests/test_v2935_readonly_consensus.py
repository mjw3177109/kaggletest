"""Offline-only V2935 scientific/safety regression tests; no network or Kaggle API."""
from __future__ import annotations

import inspect
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_agent"))
import v2935_actions_gate as gate
import v2935_readonly_consensus as model


class TestV2935(unittest.TestCase):
    def test_preregistered_immutable_and_official(self):
        s = gate.preregistration()
        self.assertEqual(s["stage"], "preregistered_not_started")
        self.assertFalse(s["submitted"])
        self.assertTrue(s["evaluation_only"])

    def test_agree_positive_shift(self):
        a = np.array([0., 1., 2., 3., 4.])
        b = np.array([1., 0., 2., 3., 4.])
        c = np.array([2., 0., 1., 3., 4.])
        out, mask = model.agreement_gate(a, b, c)
        self.assertTrue(mask[0])
        self.assertTrue(mask[1])
        self.assertAlmostEqual(out[0], 0.2 + 0.075 * (0.4 - 0.2) + 0.075 * (0.6 - 0.2))
        self.assertAlmostEqual(out[1], 0.4 + 0.075 * (0.2 - 0.4) + 0.075 * (0.2 - 0.4))

    def test_disagree_preserves_champion(self):
        a = np.array([0., 1., 2., 3.])
        b = np.array([1., 0., 2., 3.])
        c = np.array([0., 2., 1., 3.])
        out, mask = model.agreement_gate(a, b, c)
        base = rankdata(a, method="average") / len(a)
        np.testing.assert_array_equal(out[~mask], base[~mask])

    def test_ties_preserve_exactly(self):
        a = np.array([1., 1., 2., 3.])
        b = np.array([1., 1., 2., 3.])
        c = np.array([2., 1., 1., 3.])
        out, mask = model.agreement_gate(a, b, c)
        self.assertFalse(mask.any())
        np.testing.assert_array_equal(out, rankdata(a, method="average") / 4)

    def test_unchanged_models_no_change(self):
        a = np.array([0.1, 0.2, 0.3])
        out, mask = model.agreement_gate(a, a, a)
        self.assertFalse(mask.any())
        np.testing.assert_array_equal(out, rankdata(a) / len(a))

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            model.agreement_gate([1., np.nan], [1., 2.], [1., 2.])

    def test_inf_rejected(self):
        with self.assertRaises(ValueError):
            model.agreement_gate([1., 2.], [1., np.inf], [1., 2.])

    def test_wrong_length_rejected(self):
        with self.assertRaises(ValueError):
            model.agreement_gate([1., 2.], [1.], [1., 2.])

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            model.agreement_gate([], [], [])

    def test_repeatability(self):
        rng = np.random.default_rng(35)
        a, b, c = (rng.normal(size=1000) for _ in range(3))
        x, m = model.agreement_gate(a, b, c)
        y, n = model.agreement_gate(a, b, c)
        np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(m, n)

    def test_frozen_formula_no_label_argument(self):
        self.assertEqual(list(inspect.signature(model.agreement_gate).parameters),
                         ["champion", "source16", "source30"])

    def test_candidate_hash_little_endian(self):
        import hashlib
        values = np.array([0.25, 0.5, 0.75])
        self.assertEqual(model.digest(values),
                         hashlib.sha256(values.astype("<f8").tobytes()).hexdigest())

    def test_source_sha_locked(self):
        self.assertEqual(gate.SOURCES["v2930_npz"],
                         "45e96e595676ec91a1e6c8eea98f975e63c9c0762412b898fe549c0636107362")
        self.assertEqual(gate.SOURCES["v2916_npz"],
                         "05facb3098732f1348928c8de62cefae3ecb64b5b1fabdb3a7a84342b7bfe0d7")

    def test_prereg_mutation_rejected(self):
        original = gate.common.read
        def tamper(path):
            obj = original(path)
            if path == "state/v2_9_35_status.json":
                obj["fixed_weights"]["v2916_when_agree"] = 0.08
            return obj
        with patch.object(gate.common, "read", side_effect=tamper):
            with self.assertRaises(RuntimeError):
                gate.preregistration()

    def test_champion_sha_mutation_rejected(self):
        original = gate.common.read
        def tamper(path):
            obj = original(path)
            if path == "state/v2_9_15_submission.json":
                obj["prediction_hash"] = "0" * 64
            return obj
        with patch.object(gate.common, "read", side_effect=tamper):
            with self.assertRaises(RuntimeError):
                gate.preregistration()

    def test_readonly_cli_code(self):
        src = inspect.getsource(model)
        self.assertNotIn('"kernels", "push"', src)
        self.assertNotIn('"competitions", "submit"', src)
        self.assertNotIn('"kernels", "create"', src)

    def test_terminal_never_evaluated(self):
        with patch.object(model, "read", return_value={"stage": "skipped_no_cv_gain"}):
            with self.assertRaises(RuntimeError):
                model.evaluate()

    def test_non_kaggle_subprocess_untouched(self):
        sentinel = object()
        with patch.object(model, "ORIGINAL_RUN", return_value=sentinel) as run:
            self.assertIs(model.paced_run(["git", "status"]), sentinel)
            run.assert_called_once()

    def test_kaggle_failure_closed(self):
        class Result:
            returncode = 1
            stderr = "429 Too Many Requests"
            stdout = ""
        with patch.object(model, "ORIGINAL_RUN", return_value=Result()):
            with patch.object(model, "LAST_CALL", None):
                with self.assertRaisesRegex(RuntimeError, "rate_limited"):
                    model.paced_run(["kaggle", "kernels", "status", "example"])

    def test_no_new_notebook_in_prereg(self):
        state = json.loads((ROOT / "state/v2_9_35_status.json").read_text())
        self.assertIsNone(state["kernel"])
        self.assertFalse(state["notebook_created"])
        self.assertFalse(state["training_started"])


if __name__ == "__main__":
    unittest.main()
