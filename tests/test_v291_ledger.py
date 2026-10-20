import sys
import tempfile
import unittest
from datetime import datetime,timezone,timedelta
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"kaggle_agent"))
from v291_ledger import record_experiment,queue_age,daily_metrics

class LedgerTests(unittest.TestCase):
    def test_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"ledger.json"
            x=dict(competition="c",candidate="f",validation="folds",seed=42,parameters={},status="completed",completed_at="2026-10-19T00:00:00Z",cv=.96,gpu_seconds=3600,submission_id=1)
            a,created=record_experiment(path,x)
            b,again=record_experiment(path,x)
            self.assertTrue(created)
            self.assertFalse(again)
            self.assertEqual(a["experiment_id"],b["experiment_id"])
            s=daily_metrics(path,"2026-10-19","c")
            self.assertEqual(s["new_independent_experiments"],1)
            self.assertEqual(s["new_submission_count"],1)
    def test_queue_breaker(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"queue.json"
            t=datetime(2026,10,19,tzinfo=timezone.utc)
            self.assertFalse(queue_age(p,"k",True,t)["tripped"])
            self.assertTrue(queue_age(p,"k",True,t+timedelta(hours=7))["tripped"])
            self.assertFalse(queue_age(p,"k",False,t+timedelta(hours=8))["tripped"])
if __name__=="__main__":
    unittest.main()
