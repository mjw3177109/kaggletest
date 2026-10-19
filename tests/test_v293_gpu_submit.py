"""V2.9.3 GPU submission: end-to-end offline safety tests."""
import json
import sys
import tempfile
import unittest
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"kaggle_agent"))
import v293_gpu_submit as m
from v291_ledger import read,write

class GPUOneShotSafety(unittest.TestCase):
    def test_prepare_and_persisted_reservation_without_live_submit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            originals=(m.ROOT,m.GPU_STATE,m.STATE,m.LEDGER,m.REPORT,m.N_TEST,m.N_TRAIN)
            m.ROOT=root
            m.GPU_STATE=root/"state/v2_9_3_gpu_status.json"
            m.STATE=root/"state/v2_9_3_gpu_submission.json"
            m.LEDGER=root/"state/v2_9_experiments.json"
            m.REPORT=root/"reports/v2_9_3_gpu_submission.md"
            m.N_TEST=3
            m.N_TRAIN=4
            metrics={"best_model":"blend_rank","best_cv_auc":.961,
                     "competition":m.COMP,"validation":m.VALIDATION,
                     "train_rows":4,"test_rows":3,"folds":3,
                     "gpu_verified":True,
                     "fold_results":[{"cat_device":"GPU"}]*3,
                     "new_methods":["feature_test"],"elapsed_seconds":120}
            gpu={"competition":m.COMP,
                 "stage":"candidate_ready_unsubmitted",
                 "kernel":"mjw3177109/kas-v293-gpu-test",
                 "kernel_url":"https://www.kaggle.com/code/mjw3177109/kas-v293-gpu-test",
                 "hardware":{"ok":True},
                 "metrics":metrics,
                 "incumbent_cv":.9608,"incumbent_lb":.96035,
                 "cv_gate":{"passed":True,"threshold":.00006}}
            write(m.GPU_STATE,gpu)
            def fake_cli(command,timeout=120):
                location=Path(command[command.index("-p")+1])
                location.mkdir(parents=True,exist_ok=True)
                (location/"v293_gpu_metrics.json").write_text(json.dumps(metrics))
                (location/"candidate_submission.csv").write_text("id,satisfaction\n1,0.2\n2,0.5\n3,0.8\n")
                return ""
            try:
                tmp=root/"tmp"
                day=datetime.now(timezone.utc).isoformat()
                with patch.object(m,"allowed_competition"),patch.object(m,"cli",side_effect=fake_cli):
                    m.prepare(tmp)
                self.assertEqual(read(m.STATE)["stage"],"prepared")
                with patch.object(m,"allowed_competition"),patch.object(m,"records",
                          return_value=[{"date":day,"description":"prior","ref":1}]):
                    m.reserve(tmp)
                self.assertTrue(read(m.STATE)["submission_attempted"])
                self.assertEqual(read(m.STATE)["stage"],"reserved")
                with patch.object(m,"allowed_competition"),patch.object(m,"records",return_value=[]):
                    with self.assertRaisesRegex(RuntimeError,"durably committed"):
                        m.submit(tmp)
                self.assertEqual(read(m.STATE)["stage"],"reserved")
            finally:
                m.ROOT,m.GPU_STATE,m.STATE,m.LEDGER,m.REPORT,m.N_TEST,m.N_TRAIN=originals

    def test_cv_gate_refuses_nonwinning_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            original=m.GPU_STATE
            m.GPU_STATE=Path(d)/"gpu.json"
            try:
                write(m.GPU_STATE,{"stage":"kernel_queued","competition":m.COMP})
                with self.assertRaisesRegex(RuntimeError,"No independently verified"):
                    m.verified_gpu_candidate()
            finally:
                m.GPU_STATE=original

if __name__=="__main__":
    unittest.main()
