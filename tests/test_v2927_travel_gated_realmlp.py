"""V2927 safety and fold-local expert tests; never contact Kaggle."""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_agent"))
import v2927_pipeline as p
import v2927_travel_gated_realmlp_train as m


class V2927FrozenSafety(unittest.TestCase):
    def test_preregistration_immutable_and_notebook_unique(self):
        state = p.read(p.STATUS)
        self.assertTrue(p.verify_prereg(state))
        self.assertEqual(state["stage"], "preregistered_not_started")
        self.assertFalse(state["notebook_created"])
        self.assertFalse(state["training_started"])
        self.assertEqual(p.KERNEL, "mjw3177109/kas-v2927-travel-gated-dual-realmlp-t4-s6e10")
        self.assertEqual(p.NEW_WEIGHT, 0.15)
        self.assertEqual(p.MIN_GAIN, 0.00005)
        self.assertEqual(p.MODEL_SEED, 20261027)
        self.assertEqual(m.EXPERTS, ("Personal Travel", "Business travel"))
        self.assertEqual(m.FOLDS, 3)
        self.assertEqual(m.MODEL_SEED, 20261027)
        self.assertEqual(m.SEED, 20261018)

    def test_fail_closed_when_prereg_or_owner_changes(self):
        original = p.read(p.STATUS)
        for key, value in (
            ("intended_kernel", "mjw3177109/wrong"),
            ("blend", {"v2915_rank": 0.8, "v2927_rank": 0.2}),
            ("submitted", True),
            ("route_feature_count", 12),
            ("kernel", "mjw3177109/another"),
        ):
            bad = dict(original, **{key: value})
            with self.assertRaises(RuntimeError, msg=key):
                p.verify_prereg(bad)

    def test_travel_gate_exact_two_categories(self):
        frame = pd.DataFrame({"Type of Travel": ["Personal Travel", "Business travel"]})
        self.assertEqual(m.travel_mask(frame, "Personal Travel").tolist(), [True, False])
        self.assertEqual(m.travel_mask(frame, "Business travel").tolist(), [False, True])
        for values in (["Personal Travel", None], ["Business travel", "Other"]):
            with self.assertRaises(ValueError):
                m.travel_mask(pd.DataFrame({"Type of Travel": values}), "Personal Travel")

    def test_expert_trains_only_on_matching_outer_train_rows(self):
        # 800 synthetic rows, two balanced travel groups; no GPU and no Kaggle access.
        train = pd.DataFrame({
            "Type of Travel": ["Personal Travel"] * 400 + ["Business travel"] * 400,
            "id": np.arange(800),
        })
        test = pd.DataFrame({
            "Type of Travel": ["Personal Travel"] * 120 + ["Business travel"] * 120,
            "id": np.arange(240) + 800,
        })
        targets = np.arange(800) % 2
        seen = []
        class FakeModel:
            def fit(self, x, y):
                seen.append(("fit", len(x), len(y)))
                return self
            def predict_proba(self, x):
                return np.tile(np.array([[0.5, 0.5]], dtype=float), (len(x), 1))
        def fake_engineer(fit, holdout):
            cols = [f"f{i}" for i in range(43)]
            xa = pd.DataFrame(0.0, index=fit.index, columns=cols)
            xv = pd.DataFrame(0.0, index=holdout.index, columns=cols)
            tr = pd.DataFrame({f"k{i}": fit["Type of Travel"].astype(str).to_numpy()
                               for i in range(3)}, index=fit.index)
            va = pd.DataFrame({f"k{i}": holdout["Type of Travel"].astype(str).to_numpy()
                               for i in range(3)}, index=holdout.index)
            return xa, xv, tr, va
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None))
        with (patch.object(m, "engineer", side_effect=fake_engineer),
              patch.object(m, "make_model", return_value=FakeModel()) as model,
              patch.dict(sys.modules, {"torch": fake_torch})):
            val_idx, test_idx, pv, pt, result = m.fit_expert(
                train, test, targets, np.arange(600), np.arange(600, 800),
                fold=0, expert_index=1, group="Business travel")
        self.assertEqual(len(val_idx), 100)
        self.assertTrue(np.all(val_idx >= 600))
        self.assertTrue(np.all(val_idx < 800))
        self.assertEqual(len(test_idx), 120)
        self.assertTrue(np.all(test_idx >= 120))
        self.assertEqual(result["train_rows"], 200)
        self.assertEqual(result["validation_rows"], 100)
        self.assertEqual(result["features"], 46)
        self.assertEqual(result["model_seed"], 20261028)
        self.assertEqual(seen, [("fit", 200, 200)])
        self.assertEqual(model.call_count, 1)
        self.assertTrue(np.isfinite(pv).all() and np.isfinite(pt).all())

    def test_evaluator_no_submit_and_recover_only_guard(self):
        source = (ROOT / "kaggle_agent/v2927_pipeline.py").read_text(encoding="utf-8")
        for token in ('"--recover-only"', '"kernels", "push"', '"submitted": False',
                      'if recover_only:', "Kaggle owner list failed",
                      'verify_metrics(metrics, y, new_oof, train, test)'):
            self.assertIn(token, source)
        self.assertNotIn('"competitions", "submit"', source)
        self.assertNotIn("kaggle competitions submit", source)
        self.assertNotIn('STATUS = ROOT / "state/v2_9_26_status.json"', source)
        self.assertNotIn("0.855", source)


if __name__ == "__main__":
    unittest.main()
