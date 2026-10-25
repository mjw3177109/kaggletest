"""V2939 offline-only rank-median scientific/safety tests; no network or Kaggle API."""
from __future__ import annotations
import hashlib
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
import v2939_actions_gate as gate
import v2939_readonly_median as model

class TestV2939(unittest.TestCase):
    def test_preregistration_is_pristine(self):
        s = gate.preregistration()
        self.assertEqual(s["stage"], "preregistered_not_started")
        self.assertIsNone(s["run_reservation_id"])
        self.assertFalse(s["training_started"])
        self.assertFalse(s["submitted"])

    def test_frozen_median_formula(self):
        self.assertEqual(gate.FROZEN["candidate_formula"],
                         "r0+0.15*(median(r0,r16,r30)-r0)")

    def test_label_free_signature(self):
        self.assertEqual(list(inspect.signature(model.median_consensus).parameters),
                         ["champion", "source16", "source30"])

    def test_median_moves_toward_nearer_source(self):
        a = np.array([0., 1., 2., 3., 4.])
        b = np.array([1., 0., 2., 3., 4.])
        c = np.array([2., 0., 1., 3., 4.])
        out, disagree = model.median_consensus(a, b, c)
        r0 = rankdata(a) / len(a)
        r16 = rankdata(b) / len(a)
        r30 = rankdata(c) / len(a)
        self.assertFalse(disagree[0])
        self.assertAlmostEqual(out[0], r0[0] + 0.15 * (
            np.median([r0[0], r16[0], r30[0]]) - r0[0]))
        self.assertTrue(np.isfinite(out).all())

    def test_disagree_preserves_champion(self):
        a = np.array([0., 1., 2., 3.])
        b = np.array([1., 0., 2., 3.])
        c = np.array([0., 2., 1., 3.])
        out, disagree = model.median_consensus(a, b, c)
        np.testing.assert_array_equal(out[disagree], rankdata(a)[disagree] / 4)

    def test_identical_sources_preserve_champion(self):
        a = np.array([0.1, 0.2, 0.3])
        out, disagree = model.median_consensus(a, a, a)
        self.assertTrue(disagree.all())
        np.testing.assert_array_equal(out, rankdata(a) / 3)

    def test_ties_preserve_champion(self):
        a = np.array([1., 1., 2., 3.])
        out, disagree = model.median_consensus(a, a, a)
        self.assertTrue(disagree.all())
        np.testing.assert_array_equal(out, rankdata(a, method="average") / 4)

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            model.median_consensus([1., np.nan], [1., 2.], [1., 2.])

    def test_inf_rejected(self):
        with self.assertRaises(ValueError):
            model.median_consensus([1., 2.], [1., np.inf], [1., 2.])

    def test_wrong_length_rejected(self):
        with self.assertRaises(ValueError):
            model.median_consensus([1., 2.], [1.], [1., 2.])

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            model.median_consensus([], [], [])

    def test_repeatability(self):
        rng = np.random.default_rng(2939)
        a, b, c = (rng.normal(size=1000) for _ in range(3))
        x, maskx = model.median_consensus(a, b, c)
        y, masky = model.median_consensus(a, b, c)
        np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(maskx, masky)

    def test_median_bounded_between_sources(self):
        rng = np.random.default_rng(42)
        a, b, c = (rng.normal(size=500) for _ in range(3))
        out, _ = model.median_consensus(a, b, c)
        r0 = rankdata(a) / len(a)
        r16 = rankdata(b) / len(a)
        r30 = rankdata(c) / len(a)
        lo = np.minimum.reduce((r0, r16, r30))
        hi = np.maximum.reduce((r0, r16, r30))
        self.assertTrue(np.all(out >= lo - 1e-12))
        self.assertTrue(np.all(out <= hi + 1e-12))

    def test_candidate_not_old_arithmetic_gate(self):
        a = np.array([0., 1., 2., 3., 4.])
        b = np.array([1., 0., 2., 3., 4.])
        c = np.array([2., 0., 1., 3., 4.])
        median, _ = model.median_consensus(a, b, c)
        arithmetic, _ = model.old.agreement_gate(a, b, c)
        self.assertNotEqual(float(median[0]), float(arithmetic[0]))

    def test_source_sha_locked(self):
        self.assertEqual(gate.SOURCES["v2930_npz"],
                         "45e96e595676ec91a1e6c8eea98f975e63c9c0762412b898fe549c0636107362")
        self.assertEqual(gate.SOURCES["v2916_npz"],
                         "05facb3098732f1348928c8de62cefae3ecb64b5b1fabdb3a7a84342b7bfe0d7")

    def test_champion_sha_locked(self):
        self.assertEqual(gate.CHAMPION_SHA,
                         "906185547dcd5f0cac3ac173de644fd9997f20ac863c6745bcb1fd3e9a89e5f7")

    def test_candidate_hash_little_endian(self):
        a = np.array([0.25, 0.5, 0.75])
        self.assertEqual(model.old.digest(a),
                         hashlib.sha256(a.astype("<f8").tobytes()).hexdigest())

    def test_prereg_mutation_rejected(self):
        original = gate.source_gate.common.read
        def mutate(path):
            obj = original(path)
            if path == "state/v2_9_39_status.json":
                obj["fixed_shrinkage"] = 0.2
            return obj
        with patch.object(gate.source_gate.common, "read", side_effect=mutate):
            with self.assertRaises(RuntimeError):
                gate.preregistration()

    def test_terminal_re_evaluation_rejected(self):
        with patch.object(gate, "preregistration", side_effect=RuntimeError("terminal")):
            with self.assertRaises(RuntimeError):
                model.evaluate()

    def test_no_kernel_push_or_submission(self):
        source = inspect.getsource(model) + inspect.getsource(gate)
        self.assertNotIn('["kaggle", "kernels", "push"', source)
        self.assertNotIn('["kaggle", "competitions", "submit"', source)
        self.assertNotIn('["kaggle", "kernels", "create"', source)

    def test_pinned_previous_terminal(self):
        state = json.loads((ROOT / "state/v2_9_39_status.json").read_text())
        self.assertEqual(state["previous_v2938"]["stage"], "skipped_no_cv_gain")
        self.assertEqual(state["previous_v2938"]["run_id"], 40164585333)
        self.assertFalse(state["previous_v2938"]["submitted"])

if __name__ == "__main__":
    unittest.main()
