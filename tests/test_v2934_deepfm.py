"""V2934 immutable DeepFM architecture, fold isolation, owner and no-submit safety tests."""
import ast
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"kaggle_agent"))
import v2934_deepfm_train as m
import v2934_pipeline as p
import v2934_actions_gate as gate


def toy_frames(n=16):
    rng=np.random.default_rng(20261103)
    cats=[c for c in p.CANONICAL if c in p.CATEGORICAL]
    data={}
    for col in p.CANONICAL:
        if col in cats:
            data[col]=pd.Categorical(["A","B"]*(n//2),categories=["A","B"])
        else:
            data[col]=rng.normal(size=n).astype("float32")
    train=pd.DataFrame(data)[p.CANONICAL]
    val=train.iloc[:4].copy()
    test=train.iloc[4:8].copy()
    for col in cats:
        val[col]=pd.Categorical(["UNSEEN"]*4,categories=["A","B"])
    return train,val,test


class V2934Safety(unittest.TestCase):
    def test_immutable_preregistration_and_champion(self):
        state=p.read(p.STATUS)
        self.assertEqual(p.verify_prereg(state),state["stage"] not in p.TERMINAL)
        self.assertEqual(p.KERNEL,gate.KERNEL)
        self.assertEqual(m.FEATURE_FAMILY,p.FEATURE_FAMILY)
        self.assertEqual(state["architecture"],p.frozen_params()["architecture"])
        self.assertEqual(state["loss"],"BCEWithLogitsLoss")
        self.assertEqual(len(p.CANONICAL),37)
        self.assertEqual(p.CANONICAL[-3:],["te_0","te_1","te_2"])
        self.assertEqual(state["submission_policy"],"evaluation_only_no_submission")
        self.assertFalse(state["submitted"])

    def test_tokenizer_uses_outer_train_only_and_preserves_order(self):
        train,val,test=toy_frames()
        nums,cats,order,types,cat_cols,num_cols,cards=m.feature_tokens(train,val,test)
        self.assertEqual(order,p.CANONICAL)
        self.assertEqual(types,["categorical" if c in p.CATEGORICAL else "numeric" for c in order])
        self.assertEqual(nums[0].shape,(16,23))
        self.assertEqual(cats[0].shape,(16,14))
        self.assertTrue(np.all(cats[1]==0))
        self.assertTrue(np.allclose(nums[0].mean(axis=0),0,atol=1e-5))
        val2=val.copy()
        val2[num_cols[0]]=1e8
        nums2,cats2,*_=m.feature_tokens(train,val2,test)
        np.testing.assert_array_equal(nums[0],nums2[0])
        np.testing.assert_array_equal(cats[0],cats2[0])
        np.testing.assert_array_equal(nums[2],nums2[2])
        self.assertTrue(all(card==3 for card in cards))

    def test_deepfm_forward_backward_cpu(self):
        train,val,test=toy_frames()
        nums,cats,order,types,cat_cols,num_cols,cards=m.feature_tokens(train,val,test)
        torch.manual_seed(20261103)
        model=m.DeepFM(order,cat_cols,num_cols,cards)
        self.assertEqual(len(model.cat_vectors),14)
        self.assertEqual(len(model.cat_linear),14)
        self.assertEqual(tuple(model.num_vectors.shape),(23,8))
        out=model(torch.from_numpy(nums[0]),torch.from_numpy(cats[0]))
        self.assertEqual(tuple(out.shape),(16,))
        self.assertTrue(torch.isfinite(out).all())
        loss=torch.nn.functional.binary_cross_entropy_with_logits(
            out,torch.tensor([0.,1.]*8))
        loss.backward()
        self.assertTrue(torch.isfinite(model.num_vectors.grad).all())
        self.assertTrue(torch.isfinite(model.deep[-1].weight.grad).all())

    def test_exact_explicit_fm_second_order_identity(self):
        train,val,test=toy_frames()
        nums,cats,order,types,cat_cols,num_cols,cards=m.feature_tokens(train,val,test)
        model=m.DeepFM(order,cat_cols,num_cols,cards).eval()
        with torch.no_grad():
            model.num_vectors.zero_()
            model.num_vectors[0].fill_(1.)
            model.num_vectors[1].fill_(1.)
            model.num_linear.zero_()
            model.bias.zero_()
            for emb in model.cat_vectors:
                emb.weight.zero_()
            for emb in model.cat_linear:
                emb.weight.zero_()
            for layer in model.deep:
                if isinstance(layer,torch.nn.Linear):
                    layer.weight.zero_()
                    layer.bias.zero_()
            n=torch.zeros((2,23))
            n[0,0],n[0,1]=2.,3.
            n[1,0],n[1,1]=4.,-1.
            c=torch.zeros((2,14),dtype=torch.long)
            output=model(n,c)
        torch.testing.assert_close(output,torch.tensor([48.,-32.]),atol=1e-5,rtol=0)

    def test_deepfm_rejects_incomplete_schema(self):
        train,val,test=toy_frames()
        nums,cats,order,types,cat_cols,num_cols,cards=m.feature_tokens(train,val,test)
        model=m.DeepFM(order,cat_cols,num_cols,cards)
        with self.assertRaises(ValueError):
            model(torch.zeros(8,22),torch.zeros(8,14,dtype=torch.long))
        with self.assertRaises(ValueError):
            model(torch.zeros(8,23),torch.zeros(8,13,dtype=torch.long))
        with self.assertRaises(ValueError):
            m.DeepFM(order[:-1],cat_cols,num_cols,cards)

    def test_wrong_schema_or_nonfinite_fails_closed(self):
        train,val,test=toy_frames()
        with self.assertRaises(ValueError):
            m.feature_tokens(train.drop(columns="te_2"),val,test)
        train.loc[0,"Age"]=float("nan")
        with self.assertRaises(ValueError):
            m.feature_tokens(train,val,test)

    def test_recover_only_cannot_create_kernel(self):
        pristine=dict(p.read(p.STATUS))
        pristine.update(stage="preregistered_not_started",kernel=None,
                        notebook_created=False,training_started=False)
        with patch.object(p,"read",return_value=pristine),patch.object(
                p,"preflight",return_value=pristine),patch.object(
                p,"checked_owner_exists",side_effect=AssertionError("must not query/push")),patch.object(
                p,"durably_reserve_unique_kernel",side_effect=AssertionError("must not reserve")):
            result=p.run(recover_only=True)
        self.assertEqual(result["stage"],"preregistered_not_started")
        self.assertFalse(result["notebook_created"])

    def test_persisted_owner_recovery_never_repushes(self):
        active=dict(p.read(p.STATUS))
        active.update(stage="kernel_running",kernel=p.KERNEL,
                      notebook_created=True,training_started=True)
        with patch.object(p,"read",return_value=active),patch.object(
                p,"preflight",return_value=active),patch.object(
                p,"checked_owner_exists",side_effect=AssertionError("must not query/push")),patch.object(
                p,"durably_reserve_unique_kernel",side_effect=AssertionError("must not reserve")),patch.object(
                p.v24,"poll_kernel_strict",return_value=("kernel_running","KernelWorkerStatus.RUNNING")),patch.object(
                p,"persist") as persist:
            result=p.run(recover_only=True)
        self.assertEqual(result["stage"],"kernel_running")
        persist.assert_called_once()

    def test_fail_closed_without_cuda(self):
        with patch.object(torch.cuda,"is_available",return_value=False):
            with self.assertRaises(RuntimeError):
                m.prove_gpu()

    def test_authenticated_missing_unrelated_kernel_is_not_active(self):
        def fake_kaggle(args,timeout=120):
            if args[:2]==["competitions","submissions"]:
                return "57446286,0.96099"
            if args[:2]==["kernels","list"]:
                return "ref,title\nmjw3177109/agent,deleted\n"
            if args[:2]==["kernels","status"]:
                if args[2]=="mjw3177109/agent":
                    raise LookupError("authenticated NOT_FOUND")
                return "KernelWorkerStatus.COMPLETE"
            raise AssertionError(args)
        with patch.dict(os.environ,{"KAGGLE_USERNAME":"mjw3177109",
                                     "KAGGLE_API_TOKEN":"test-only-placeholder"}):
            with patch.object(gate,"kaggle",side_effect=fake_kaggle):
                gate.assert_kaggle_idle_and_official()
        def fail_known(args,timeout=120):
            if args[:2]==["kernels","status"] and "v2933" in args[2]:
                raise LookupError("authenticated NOT_FOUND")
            return fake_kaggle(args,timeout)
        with patch.dict(os.environ,{"KAGGLE_USERNAME":"mjw3177109",
                                     "KAGGLE_API_TOKEN":"test-only-placeholder"}):
            with patch.object(gate,"kaggle",side_effect=fail_known):
                with self.assertRaises(RuntimeError):
                    gate.assert_kaggle_idle_and_official()

    def test_historical_cancellation_acknowledged_twice_is_not_training(self):
        seen=[]
        def fake_kaggle(args,timeout=120):
            if args[:2]==["competitions","submissions"]:
                return "57446286,0.96099"
            if args[:2]==["kernels","list"]:
                return "ref,title\nmjw3177109/best-cv-score-0-95625,canceled\n"
            if args[:2]==["kernels","status"]:
                seen.append(args[2])
                if args[2]=="mjw3177109/best-cv-score-0-95625":
                    return 'has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"'
                return "KernelWorkerStatus.COMPLETE"
            raise AssertionError(args)
        with patch.dict(os.environ,{"KAGGLE_USERNAME":"mjw3177109",
                                     "KAGGLE_API_TOKEN":"test-only-placeholder"}):
            with patch.object(gate,"kaggle",side_effect=fake_kaggle):
                gate.assert_kaggle_idle_and_official()
        self.assertEqual(seen.count("mjw3177109/best-cv-score-0-95625"),2)

    def test_cancel_requested_is_not_terminal(self):
        def fake_kaggle(args,timeout=120):
            if args[:2]==["competitions","submissions"]:
                return "57446286,0.96099"
            if args[:2]==["kernels","list"]:
                return "ref,title\nmjw3177109/best-cv-score-0-95625,canceling\n"
            if args[:2]==["kernels","status"]:
                if args[2]=="mjw3177109/best-cv-score-0-95625":
                    return 'has status "KernelWorkerStatus.CANCEL_REQUESTED"'
                return "KernelWorkerStatus.COMPLETE"
            raise AssertionError(args)
        with patch.dict(os.environ,{"KAGGLE_USERNAME":"mjw3177109",
                                     "KAGGLE_API_TOKEN":"test-only-placeholder"}):
            with patch.object(gate,"kaggle",side_effect=fake_kaggle):
                with self.assertRaises(RuntimeError):
                    gate.assert_kaggle_idle_and_official()

    def test_unknown_kaggle_kernel_status_blocks(self):
        def fake_kaggle(args,timeout=120):
            if args[:2]==["competitions","submissions"]:
                return "57446286,0.96099"
            if args[:2]==["kernels","list"]:
                return "ref,title\nmjw3177109/other,other\n"
            return "KernelWorkerStatus.UNKNOWN"
        with patch.dict(os.environ,{"KAGGLE_USERNAME":"mjw3177109",
                                     "KAGGLE_API_TOKEN":"test-only-placeholder"}):
            with patch.object(gate,"kaggle",side_effect=fake_kaggle):
                with self.assertRaises(RuntimeError):
                    gate.assert_kaggle_idle_and_official()

    def test_no_submission_pickle_or_workflow_bypass(self):
        paths=[ROOT/"kaggle_agent/v2934_deepfm_train.py",
               ROOT/"kaggle_agent/v2934_pipeline.py",
               ROOT/"kaggle_agent/v2934_actions_gate.py"]
        for path in paths:
            content=path.read_text(encoding="utf-8")
            ast.parse(content,filename=str(path))
            self.assertNotIn("allow_pickle=True",content)
            self.assertNotIn("kaggle competitions submit",content)
            self.assertNotIn('kaggle","competitions","submit',content)
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
