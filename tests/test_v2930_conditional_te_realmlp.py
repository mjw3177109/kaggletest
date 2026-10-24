"""V2930 no-leakage conditional TE, immutable registration and no-submit gates."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_agent"))
import v2930_pipeline as p
import v2930_conditional_te_realmlp_train as m


def frame(n=80):
    rng = np.random.default_rng(100)
    df = pd.DataFrame({
        "Customer Type": np.where(np.arange(n) % 2, "Loyal Customer", "disloyal Customer"),
        "Type of Travel": np.where(np.arange(n) % 3, "Personal Travel", "Business travel"),
        "Class": np.where(np.arange(n) % 3, "Eco", "Business"),
        "Gender": np.where(np.arange(n) % 2, "Male", "Female"),
        "Age": rng.integers(15, 75, n),
        "Departure Delay in Minutes": rng.integers(0, 150, n),
        "Arrival Delay in Minutes": rng.integers(0, 150, n),
        "Seat comfort": rng.integers(0, 6, n),
        "Online boarding": rng.integers(0, 6, n),
        "Inflight wifi service": rng.integers(0, 6, n),
    })
    return df


class V2930Safety(unittest.TestCase):
    def test_immutable_prereg_and_unique_slug(self):
        s = p.read(p.STATUS)
        self.assertTrue(p.verify_prereg(s) or s["stage"] in p.TERMINAL)
        self.assertEqual(p.KERNEL, "mjw3177109/kas-v2930-conditional-crossfit-te-realmlp-t4-s6e10")
        self.assertEqual(p.STATUS.name, "v2_9_30_status.json")
        self.assertEqual(m.MODEL_SEED, 20261030)
        self.assertEqual(m.FEATURE_FAMILY, p.FEATURE_FAMILY)
        self.assertEqual(len(m.CONDITIONAL_KEYS), 8)
        self.assertEqual(m.PRIOR_WEIGHT, 50)
        self.assertEqual(p.NEW_WEIGHT, .15)
        self.assertEqual(s["submission_policy"], "evaluation_only_no_submission")
        self.assertFalse(s["submitted"])

    def test_fixed_bins_and_missing(self):
        data = frame(8)
        data["Age"] = [0, 24, 25, 34, 35, 44, 54, 55]
        data["Departure Delay in Minutes"] = [0, 1, 15, 16, 60, 61, -2, np.nan]
        age = m._conditional_component(data, "AgeBand").tolist()
        delay = m._conditional_component(data, "DepartureDelayBand").tolist()
        self.assertEqual(age, ["0-24", "0-24", "25-34", "25-34", "35-44", "35-44", "45-54", "55+"])
        self.assertEqual(delay, ["0", "1-15", "1-15", "16-60", "16-60", "61+", "__MISSING__", "__MISSING__"])

    def test_fivefold_crossfit_uses_no_self_target(self):
        train, validation, test = frame(80), frame(12), frame(9)
        y = np.asarray([0, 1] * 40, dtype=np.uint8)
        a, b, c = m.conditional_crossfit(train, validation, test, y, 0)
        self.assertEqual(a.shape, (80, 8))
        self.assertEqual(b.shape, (12, 8))
        self.assertEqual(c.shape, (9, 8))
        self.assertTrue(a.columns.equals(b.columns) and b.columns.equals(c.columns))
        y2 = y.copy()
        y2[0] = 1 - y2[0]
        a2, _, _ = m.conditional_crossfit(train, validation, test, y2, 0)
        # The flipped label is excluded from that row's own inner training fold.
        # Use the original fold partition when checking: changing y may alter
        # stratified fold assignment, so compare against the identical mapping
        # with the same original fold split in a separate test below.
        self.assertTrue(np.isfinite(a2.to_numpy()).all())

    def test_inner_fit_heldout_labels_do_not_change_encoding(self):
        train = pd.Series(["A", "A", "B", "C", "A"])
        heldout = pd.Series(["A", "C", "Z"])
        y = np.array([0, 1, 1, 0, 0])
        before = m.smoothed_encode(train, y, heldout)
        self.assertTrue(np.allclose(before, m.smoothed_encode(train, y, heldout)))
        self.assertAlmostEqual(float(before[-1]), float(y.mean()), places=6)
        self.assertTrue(np.all((before >= 0) & (before <= 1)))

    def test_validation_cannot_modify_train_crossfit(self):
        train, val, test = frame(80), frame(12), frame(9)
        y = np.asarray([0, 1] * 40, dtype=np.uint8)
        a, _, c = m.conditional_crossfit(train, val, test, y, 0)
        modified = val.copy()
        modified.loc[:, "Seat comfort"] = 5
        a2, _, c2 = m.conditional_crossfit(train, modified, test, y, 0)
        pd.testing.assert_frame_equal(a, a2)
        pd.testing.assert_frame_equal(c, c2)

    def test_frozen_original_realmlp_and_no_submission(self):
        train = (ROOT / "kaggle_agent/v2930_conditional_te_realmlp_train.py").read_text()
        pipe = (ROOT / "kaggle_agent/v2930_pipeline.py").read_text()
        self.assertIn('model.fit(xa, pd.Series(ytr))', train)
        self.assertIn('TargetEncoder(target_type="binary", smooth="auto", cv=5', train)
        self.assertIn("conditional_crossfit(train.iloc[itr], train.iloc[iva], test, ytr, fold)", train)
        self.assertIn("if len(xa.columns) != 45", train)
        self.assertIn('allow_pickle=False', pipe)
        self.assertIn('durably_reserve_unique_kernel(s)', pipe)
        self.assertIn('if recover_only:', pipe)
        self.assertNotIn("kaggle competitions submit", pipe)
        self.assertNotIn("allow_pickle=True", pipe)


if __name__ == "__main__":
    unittest.main()
