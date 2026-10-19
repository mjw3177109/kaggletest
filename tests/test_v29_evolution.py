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
    def test_queue_age_reads_persisted_first_observation(self):
        import tempfile
        from datetime import timedelta
        with tempfile.TemporaryDirectory() as tmp:
            previous=m.STATE
            m.STATE=Path(tmp)
            try:
                now=datetime(2026,10,19,12,tzinfo=timezone.utc)
                m.write(m.STATE/"v2_3_status.json",{
                    "stage":"kernel_queued",
                    "competition":"playground-series-s6e10",
                    "kernel":"owner/kas-tpu"
                })
                m.write(m.STATE/"dual_ladder.json",{"active_competition":"playground-series-s6e10"})
                m.write(m.STATE/"v2_9_queue_memory.json",{
                    "kernel":"owner/kas-tpu",
                    "queued":True,
                    "first_seen_at":(now-timedelta(hours=7)).isoformat()
                })
                result=m.evaluate(now)
                self.assertEqual(result["tpu"]["queued_hours"],7.0)
                self.assertTrue(result["tpu"]["breaker_tripped"])
                self.assertEqual(result["tpu"]["age_basis"],"first_observed_at_or_verified_queued_since")
            finally:
                m.STATE=previous

    def test_proposal_never_submits(self):
        result=m.evaluate(datetime(2026,10,19,tzinfo=timezone.utc))
        self.assertEqual(result["mode"],"proposal_only_no_training_or_submission")
        self.assertTrue(result["safety"]["no_automatic_submission"])
if __name__=="__main__":
    unittest.main()
