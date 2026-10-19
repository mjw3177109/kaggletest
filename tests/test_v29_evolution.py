import sys
import unittest
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"kaggle_agent"))
import v29_evolution as m

class EvolutionTests(unittest.TestCase):
    def test_invalid_queue_timestamp_is_unknown(self):
        self.assertIsNone(m.stamp(None))
    def test_candidate_pool_is_unique(self):
        self.assertEqual(len(set(x[1] for x in m.CANDIDATES)),len(m.CANDIDATES))
    def test_proposal_never_submits(self):
        result=m.evaluate(datetime(2026,10,19,tzinfo=timezone.utc))
        self.assertEqual(result["mode"],"proposal_only_no_training_or_submission")
        self.assertTrue(result["safety"]["no_automatic_submission"])
if __name__=="__main__":
    unittest.main()
