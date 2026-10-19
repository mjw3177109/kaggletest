import ast
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import v294_original_teacher_train as model

class V294StrictValidation(unittest.TestCase):
    def test_teacher_source_and_noncompetition_labels(self):
        source=(ROOT/"kaggle_agent/v294_original_teacher_train.py").read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("teejmahal20",source)
        self.assertIn("orig_hash=_fingerprints(original,cols)",source)
        self.assertIn("cmp_hash=np.concatenate",source)
        self.assertIn("keep=~overlap & ~duplicate",source)
        self.assertIn('original_teacher(train,test)',source)
        self.assertIn('outer_valid_used_for_early_stopping',source)
        self.assertNotIn('m1.fit(xtr,ym,eval_set=[(xva,yv)]',source)
        self.assertNotIn('m2.fit(xtr,ym,eval_set=[(xva,yv)]',source)
        self.assertIn('full.fit_transform(tr_key, y[outer_train])',source)

    def test_exact_fingerprint_robust_float_format(self):
        cols=["Gender","Age","Flight Distance"]
        first=pd.DataFrame({"Gender":["Male"],"Age":[24],"Flight Distance":[501]})
        second=pd.DataFrame({"Gender":[" male "],"Age":[24.0],"Flight Distance":[501.000000]})
        self.assertEqual(model._fingerprints(first,cols).tolist(),model._fingerprints(second,cols).tolist())

    def test_route_categories_fit_training_only(self):
        tr=pd.DataFrame({"Flight Distance":[101.,101.,101.,202.],
                         "Class":[1.,1.,2.,1.],
                         "Type of Travel":[0.,1.,1.,0.],
                         "Customer Type":[0.,0.,0.,1.]})
        va=pd.DataFrame({"Flight Distance":[101.,404.],"Class":[1.,1.],
                         "Type of Travel":[0.,0.],"Customer Type":[0.,0.]})
        te=va.copy()
        a,b,c=model.route_category_profile(tr,va,te)
        np.testing.assert_allclose(b["v294_route_share_Class"],[2/3,0])
        self.assertEqual(list(a.columns),list(c.columns))

    def test_context_frequencies_do_not_include_holdout(self):
        tr=pd.DataFrame({"travel_segment":["a","a","b"],"exact_Flight Distance":["1","1","2"]})
        va=pd.DataFrame({"travel_segment":["unseen"],"exact_Flight Distance":["999"]})
        a,b,c=model.fold_frequencies(tr,va,va.copy())
        self.assertEqual(float(b["v294_freq_0"].iloc[0]),0.0)
        self.assertGreater(float(a["v294_freq_0"].iloc[0]),0.0)

    def test_controller_no_direct_submit_and_new_report(self):
        content=(ROOT/"kaggle_agent/v294_pipeline.py").read_text(encoding="utf-8")
        ast.parse(content)
        self.assertIn("candidate_ready_unsubmitted",content)
        self.assertIn("v2_9_4_status.json",content)
        self.assertIn("teejmahal20/airline-passenger-satisfaction",content)
        self.assertNotIn('"kaggle","competitions","submit"',content)

if __name__=="__main__":
    unittest.main()
