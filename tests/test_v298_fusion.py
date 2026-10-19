"""V2.9.8 OOF fusion regression and Kaggle safety tests."""
import ast
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import v298_fusion as p

class SafeFusion(unittest.TestCase):
    def test_explicit_id_join_and_duplicate_rejection(self):
        canonical=np.array([10,20,30,40])
        source=np.array([40,10,30,20])
        pred=np.array([.4,.1,.3,.2])
        np.testing.assert_allclose(p.align(pred,source,canonical,"example"),[.1,.2,.3,.4])
        with self.assertRaisesRegex(ValueError,"ID set mismatch"):
            p.align(pred,[40,10,99,20],canonical,"example")
        with self.assertRaisesRegex(ValueError,"Invalid example IDs"):
            p.align(pred,[10,10,30,40],canonical,"example")

    def test_official_target_and_order(self):
        train=pd.DataFrame({"id":[9,8,7],"satisfaction":["neutral","positive","neutral"]})
        test=pd.DataFrame({"id":[3,2]})
        with patch.object(p,"N_TRAIN",3),patch.object(p,"N_TEST",2):
            tid,sid,y=p.validate_official(train,test)
        np.testing.assert_array_equal(tid,[9,8,7])
        np.testing.assert_array_equal(sid,[3,2])
        np.testing.assert_array_equal(y,[0,1,0])

    def test_no_kernel_push_or_submit(self):
        source=(ROOT/"kaggle_agent/v298_fusion.py").read_text(encoding="utf-8")
        ast.parse(source)
        self.assertNotIn('"push"',source)
        self.assertNotIn('"submit"',source)
        self.assertIn("candidate_ready_unsubmitted",source)
        self.assertIn("EXPECTED_V296=.9609541991872741",source)

    def test_full_candidate_with_scrambled_new_model_ids(self):
        rng=np.random.default_rng(12)
        n,m=120,22
        y=np.array([0,1]*(n//2),dtype=np.int8)
        tid=np.arange(1000,1000+n)
        sid=np.arange(5000,5000+m)
        train=pd.DataFrame({"id":tid,"satisfaction":y})
        test=pd.DataFrame({"id":sid})
        a=np.clip(.2+y*.4+rng.normal(0,.32,n),0,1)
        b=np.clip(.2+y*.4+rng.normal(0,.30,n),0,1)
        real=np.clip(.2+y*.4+rng.normal(0,.20,n),0,1)
        cond=np.clip(.1+y*.7+rng.normal(0,.10,n),0,1)
        perm=rng.permutation(n);q=rng.permutation(m)
        archive={
            "v25":{"target":y.copy()},"v28":{"target":y.copy()},
            "v25_submission":pd.DataFrame({"id":sid,"satisfaction":np.linspace(.01,.99,m)}),
            "v28_submission":pd.DataFrame({"id":sid,"satisfaction":np.linspace(.99,.01,m)}),
            "v296":{"train_id":tid[perm],"target":y[perm],"oof_realmlp":real[perm],
                    "test_id":sid[q],"pred_realmlp":np.linspace(.1,.9,m)[q]},
            "v297":{"train_id":tid[perm],"target":y[perm],"oof_conditional":cond[perm],
                    "test_id":sid[q],"test_conditional":np.linspace(.2,.8,m)[q]},
        }
        base=.35*rankdata(a)/n+.65*rankdata(b)/n
        benchmark=.8*rankdata(base)/n+.2*rankdata(real)/n
        states={"v25":{},"v28":{},
                "v296":{"metrics":{"full_oof_auc":roc_auc_score(y,real)}},
                "v297":{"metrics":{"full_cv_auc":roc_auc_score(y,cond)}}}
        with (patch.object(p,"N_TRAIN",n),patch.object(p,"N_TEST",m),
              patch.object(p,"EXPECTED_V293",roc_auc_score(y,base)),
              patch.object(p,"EXPECTED_V296",roc_auc_score(y,benchmark)),
              patch.object(p,"check_provenance"),
              patch.object(p,"verify_sources",return_value=(y,{"v25":a,"v28":b},
                     {"v25":np.linspace(.01,.99,m),"v28":np.linspace(.99,.01,m)}))):
            result=p.evaluate(train,test,archive,states,audit_legacy_order=True)
            self.assertEqual(result["stage"],"candidate_ready_unsubmitted")
            self.assertGreater(result["gain"],0)
            self.assertEqual(result["positive_folds"],3)
            self.assertIn("candidate_test_hash",result)
            archive["v297"]["train_id"]=tid+9999
            with self.assertRaisesRegex(ValueError,"mismatch"):
                p.evaluate(train,test,archive,states,audit_legacy_order=True)

if __name__=="__main__":
    unittest.main()
