"""V2.9.3 GPU regression gates: distinct isolated run and no unsafe submit."""
import ast
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import v293_gpu_pipeline as p

class GPUChallengerTests(unittest.TestCase):
    def test_trainer_features_are_label_free_and_cv_is_stable(self):
        text=(ROOT/"kaggle_agent/v293_gpu_train.py").read_text(encoding="utf-8")
        ast.parse(text)
        self.assertIn('SEED = 20261018',text)
        self.assertIn('TargetEncoder(target_type="binary"',text)
        self.assertIn('full.fit_transform(tr_key, y[outer_train])',text)
        self.assertIn('task_type="GPU", devices="0"',text)
        self.assertIn('depth=8',text)
        self.assertIn('__wifi_booking_gap',text)
        self.assertIn('__distance_per_age',text)
        self.assertIn('v293_gpu_metrics.json',text)
        self.assertNotIn('v28_metrics.json',text)

    def test_real_controller_never_issues_kaggle_submit_before_reservation(self):
        text=(ROOT/"kaggle_agent/v293_gpu_pipeline.py").read_text(encoding="utf-8")
        ast.parse(text)
        self.assertIn('candidate_ready_unsubmitted',text)
        self.assertIn('v2_9_3_gpu_status.json',text)
        self.assertIn('v293_gpu_train.py',text)
        self.assertNotIn('"kaggle","competitions","submit"',text)
        self.assertNotIn('"kaggle", "competitions", "submit"',text)

    def test_incumbent_must_include_new_verified_cpu_oof_best(self):
        def fake_load(path,default=None):
            name=Path(path).name
            if name=="v2_1_status.json":
                return {"stage":"complete","competition":"playground-series-s6e10",
                        "metrics":{"best_cv_auc":.959,"rows":699635},
                        "latest_submission":{"publicScore":"0.959"}}
            if name=="v2_5_status.json":
                return {"stage":"complete","competition":"playground-series-s6e10",
                        "submitted":True,"hardware":{"ok":True},
                        "metrics":{"best_cv_auc":.9606599604603964,"train_rows":699635,
                                   "gpu_verified":True,"validation":"StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)"},
                        "latest_submission":{"publicScore":"0.96035"}}
            if name=="v2_8_status.json":
                return {"stage":"skipped_no_cv_gain","competition":"playground-series-s6e10",
                        "hardware":{"ok":True},
                        "metrics":{"best_cv_auc":.9607259172925492,"train_rows":699635,
                                   "gpu_verified":True,"validation":"StratifiedKFold(n_splits=3,shuffle=True,random_state=20261018)"}}
            if name=="v2_9_3_status.json":
                return {"stage":"skipped_no_cv_gain","competition":"playground-series-s6e10",
                        "result":{"candidates":[{"cv_auc":.9607535043719558}]}}
            return {}
        with patch.object(p,"load_json",side_effect=fake_load):
            baseline=p.champion_for("playground-series-s6e10")
        self.assertAlmostEqual(baseline["cv"],.9607535043719558,places=10)
        self.assertAlmostEqual(baseline["public_lb"],.96035,places=9)
if __name__=="__main__":
    unittest.main()
