import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"kaggle_agent"))
import v292_dispatch as m

class DispatcherTests(unittest.TestCase):
    def test_failure_classification(self):
        self.assertEqual(m.classify_failure({"error":"CUDA OOM","status":"failed"}),"memory")
        self.assertEqual(m.classify_failure({"status":"skipped_no_cv_gain"}),"no_gain")
    def test_selector_skips_completed_and_validation_failures(self):
        options=[{"kind":"model","candidate":"a"},{"kind":"feature","candidate":"b"}]
        rows=[{"candidate":"a","status":"completed"},{"candidate":"b","status":"failed","error":"validation leakage"}]
        self.assertIsNone(m.choose(rows,options))
    def test_budget_blocks_when_deadline_missing(self):
        ok,reason=m.eligible({"active_competition":"playground-series-s6e10"},__import__("datetime").datetime.now(__import__("datetime").timezone.utc),{},[])
        self.assertFalse(ok)
        self.assertEqual(reason,"deadline_unverified")
if __name__=="__main__":unittest.main()
