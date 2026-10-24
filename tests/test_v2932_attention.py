"""V2932 preregistered attention model, leakage, unique-owner and no-submit tests."""
import ast
import hashlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import v2932_attention_train as m
import v2932_pipeline as p
import v2932_actions_gate as gate


def toy_frames(n=16):
    rng=np.random.default_rng(20261101)
    cats=[c for c in p.CANONICAL if c in p.CATEGORICAL]
    base={}
    for col in p.CANONICAL:
        if col in cats:
            base[col]=pd.Categorical(["A","B"]*(n//2),categories=["A","B"])
        else:
            base[col]=rng.normal(size=n).astype("float32")
    train=pd.DataFrame(base)[p.CANONICAL]
    val=train.iloc[:4].copy()
    test=train.iloc[4:8].copy()
    for col in cats:
        val[col]=pd.Categorical(["UNSEEN"]*4,categories=["A","B"])
    return train,val,test


class V2932Safety(unittest.TestCase):
    def test_immutable_preregistration_and_champion(self):
        state=p.read(p.STATUS)
        self.assertTrue(p.verify_prereg(state))
        self.assertEqual(p.KERNEL,gate.KERNEL)
        self.assertEqual(m.FEATURE_FAMILY,p.FEATURE_FAMILY)
        self.assertEqual(state["architecture"],p.frozen_params()["architecture"])
        self.assertEqual(len(p.CANONICAL),37)
        self.assertEqual(p.CANONICAL[-3:],["te_0","te_1","te_2"])
        self.assertEqual(state["submission_policy"],"evaluation_only_no_submission")
        self.assertFalse(state["submitted"])

    def test_tokenizer_uses_train_only_and_preserves_order(self):
        train,val,test=toy_frames()
        nums,cats,order,types,cat_cols,num_cols,cards=m.feature_tokens(train,val,test)
        self.assertEqual(order,p.CANONICAL)
        self.assertEqual(types,["categorical" if c in p.CATEGORICAL else "numeric" for c in order])
        self.assertEqual(nums[0].shape,(16,len(num_cols)))
        self.assertEqual(cats[0].shape,(16,len(cat_cols)))
        self.assertTrue(np.all(cats[1]==0))
        self.assertTrue(np.allclose(nums[0].mean(axis=0),0,atol=1e-5))
        val2=val.copy()
        val2[num_cols[0]]=1e8
        nums2,cats2,*_=m.feature_tokens(train,val2,test)
        np.testing.assert_array_equal(nums[0],nums2[0])
        np.testing.assert_array_equal(cats[0],cats2[0])
        np.testing.assert_array_equal(nums[2],nums2[2])
        self.assertTrue(all(card==3 for card in cards))

    def test_attention_forward_backward_cpu_dry_run(self):
        train,val,test=toy_frames()
        nums,cats,order,types,cat_cols,num_cols,cards=m.feature_tokens(train,val,test)
        torch.manual_seed(20261101)
        model=m.FTTransformer(order,cat_cols,num_cols,cards)
        self.assertEqual(len(model.blocks),2)
        self.assertEqual(model.blocks[0].attn.num_heads,4)
        out=model(torch.from_numpy(nums[0]),torch.from_numpy(cats[0]))
        self.assertEqual(tuple(out.shape),(16,))
        self.assertTrue(torch.isfinite(out).all())
        loss=torch.nn.BCEWithLogitsLoss()(out,torch.tensor([0.,1.]*8))
        loss.backward()
        self.assertTrue(torch.isfinite(model.head.weight.grad).all())

    def test_wrong_schema_or_nonfinite_fails_closed(self):
        train,val,test=toy_frames()
        with self.assertRaises(ValueError):
            m.feature_tokens(train.drop(columns="te_2"),val,test)
        train.loc[0,"Age"]=float("nan")
        with self.assertRaises(ValueError):
            m.feature_tokens(train,val,test)

    def test_recover_only_cannot_create_kernel(self):
        s=p.read(p.STATUS)
        with patch.object(p,"preflight",return_value=s),patch.object(
                p,"checked_owner_exists",side_effect=AssertionError("should not query/push")):
            result=p.run(recover_only=True)
        self.assertEqual(result["stage"],"preregistered_not_started")
        self.assertFalse(result["notebook_created"])

    def test_fail_closed_without_cuda(self):
        with patch.object(torch.cuda,"is_available",return_value=False):
            with self.assertRaises(RuntimeError):
                m.prove_gpu()

    def test_no_submission_pickle_or_workflow_bypass(self):
        paths=[
            ROOT/"kaggle_agent/v2932_attention_train.py",
            ROOT/"kaggle_agent/v2932_pipeline.py",
            ROOT/"kaggle_agent/v2932_actions_gate.py",
        ]
        for path in paths:
            text=path.read_text(encoding="utf-8")
            ast.parse(text,filename=str(path))
            self.assertNotIn("allow_pickle=True",text)
            self.assertNotIn("kaggle competitions submit",text)
            self.assertNotIn("kaggle\",\"competitions\",\"submit",text)
        pipeline=paths[1].read_text()
        guard=paths[2].read_text()
        self.assertIn("allow_pickle=False",pipeline)
        self.assertIn("durably_reserve_unique_kernel(s)",pipeline)
        self.assertIn("if recover_only:",pipeline)
        self.assertIn("assert_no_other_actions()",guard)
        self.assertIn("assert_kaggle_idle_and_official()",guard)
        self.assertEqual(p.NEW_WEIGHT,0.15)
        self.assertEqual(p.MIN_GAIN,0.00005)
        self.assertEqual(p.BASELINE,0.9612850399332336)


if __name__=="__main__":
    unittest.main()
