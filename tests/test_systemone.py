"""Typed requests, public checkpoint loading, and a raw-request demonstration."""
from argparse import Namespace
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from data import encode
from demo import run as demo
from model import Bazza, BazzaDecisionHead, Gemma4, gemma
from systemone import SystemOne
from utils import atomic_json, file_hash


def tokenizer():
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tok = Tokenizer(models.BPE(vocab={'[UNK]':0, **{c:i+1 for i,c in enumerate(alphabet)}},
                              merges=[], unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    return tok


def encoder():
    return gemma.GemmaTextModel(gemma.GemmaConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, num_global_key_value_heads=1, head_dim=8, global_head_dim=16,
        vocab_size=512, sliding_window=16, layer_types=('sliding_attention','full_attention')))


def api():
    torch.manual_seed(31)
    backbone = Gemma4(encoder=encoder())
    head = BazzaDecisionHead(torch.randn(26,16),30,encoder_width=16,width=8,
                             attention_heads=2,low_steps=1,high_cycles=1,gate_width=4,rank=2)
    return SystemOne(Bazza(backbone,head,head_precision='fp32'), tokenizer())


def request():
    return dict(state={'ticket':'A refund requested after 40 days; policy allows 30.'}, questions={
        'refund':{'type':'noul','instructions':'Is the refund allowed?',
                  'criteria':{'false':'Outside the refund window.','true':'Inside the refund window.'}},
        'next':{'type':'choice','instructions':'Select an action.',
                'criteria':{'explain_policy':'Explain the refund window.','refund_order':'Issue a refund.'}}})


class SystemOneTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.api = api()

    def test_unlabelled_inputs_match_training_and_candidates_round_trip(self):
        payload=request();original=deepcopy(payload)
        prepared=self.api._prepare(**payload)
        for name,kind,keys,row in prepared:
            raw=dict(id=name,group_id=name,source='systemone',kind=kind,state=payload['state'],
                     question=payload['questions'][name]['instructions'],
                     options=list(payload['questions'][name]['criteria'].values()),
                     target=[1.]+[0.]*(len(keys)-1))
            expected=encode(raw,self.api.tokenizer,4096)
            self.assertEqual(row,{k:v for k,v in expected.items() if k!='target'})
            self.assertNotIn('target',row)
        result=self.api(**payload)
        self.assertEqual(list(result['answers']),['refund','next'])
        for name,kind,keys,row in prepared:
            answer=result['answers'][name]
            self.assertEqual(list(answer['probabilities']),keys)
            batch=self.api.model.backbone.prepare_batch([row],'inference')
            with torch.no_grad():p=self.api.model(batch).logits[0,:len(keys)].softmax(-1).tolist()
            self.assertEqual(answer['probabilities'],dict(zip(keys,p)))
            self.assertEqual(answer['choice'],keys[max(range(len(p)),key=p.__getitem__)])
        refund=result['answers']['refund']
        self.assertEqual(refund['noul'],refund['probabilities']['true'])
        self.assertEqual(refund['value'],refund['noul']>.5)
        self.assertEqual(payload,original)
        self.assertFalse(self.api.model.training)
        self.assertFalse(self.api.model.backbone.training)

    def test_score_is_probability_weighted_numeric_value(self):
        result=self.api(state='Example',questions={'grade':{'type':'score','instructions':'Rate quality.',
                       'criteria':{'10':'Excellent.','0':'Poor.','5':'Average.'}}})['answers']['grade']
        expected=sum(float(k)*p for k,p in result['probabilities'].items())
        self.assertEqual(result['score'],expected)
        self.assertEqual(result['value'],expected)
        self.assertTrue(0<=expected<=10)

    def test_invalid_payload_and_overlong_input_fail_before_forward(self):
        bad=[]
        p=request();p['state']={'n':float('nan')};bad.append(p)
        p=request();p['questions']={};bad.append(p)
        p=request();p['questions']['refund']['type']='generate';bad.append(p)
        p=request();p['questions']['refund']['criteria']={'maybe':'Unknown','true':'Yes'};bad.append(p)
        p=request();p['questions']['next']['criteria']={str(i):str(i) for i in range(27)};bad.append(p)
        p=request();p['questions']['next']['instructions']='';bad.append(p)
        p=request();p['questions']['next']['criteria']={'a':'','b':'Text'};bad.append(p)
        p=request();p['questions']['next']['type']='score';bad.append(p)
        p=request();p['questions']['next']['type']='score';p['questions']['next']['criteria']={'NaN':'A','1':'B'};bad.append(p)
        with patch.object(self.api.model,'forward',side_effect=AssertionError('Must validate first')):
            for payload in bad:
                with self.assertRaises(ValueError):self.api(**payload)
            self.api.max_length=8
            with self.assertRaisesRegex(ValueError,'not truncated'):self.api(**request())

    def test_public_checkpoint_loader_needs_no_training_selection_or_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.api.tokenizer.save(str(root/'tokenizer.json'))
            atomic_json(root/'config.json',{})
            config=dict(backbone_checkpoint=gemma.CHECKPOINT,backbone_revision=gemma.REVISION,
                        encoder_width=16,hidden_width=8,attention_heads=2,low_steps=1,high_cycles=1,
                        gate_width=4,adapter_rank=2,softcap=30,quantization='nf4',precision='fp32')
            checkpoint=root/'head.pt'
            saved={'head':self.api.model.head.state_dict(),'config':config,'readout':{'softcap':30},
                   'identity':{'readout_files':{'tokenizer.json':file_hash(root/'tokenizer.json')}}}
            torch.save(saved,checkpoint)
            with patch('model.gemma.load_model',return_value=self.api.model.backbone.encoder) as load:
                loaded=SystemOne.from_checkpoint(checkpoint,backbone_path=root,device='cpu')
                self.assertEqual(loaded(**request()),self.api(**request()))
                load.assert_called_once()
            saved['config']['backbone_revision']='different'
            torch.save(saved,checkpoint)
            with self.assertRaisesRegex(ValueError,'identity differs'):
                SystemOne.from_checkpoint(checkpoint,backbone_path=root,device='cpu')

    def test_demo_runs_new_request_and_resumes_completed_response(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'head.pt').write_bytes(b'identity fixture')
            atomic_json(root/'request.json',request())
            args=Namespace(request=root/'request.json',checkpoint=root/'head.pt',backbone_path=root,
                           config=None,device='cpu',max_length=4096,output_dir=root/'demo',resume=False,plan_only=False)
            with patch('demo.SystemOne.from_checkpoint',return_value=self.api) as load:
                with contextlib.redirect_stdout(io.StringIO()):
                    first=demo(args);args.resume=True;second=demo(args)
                self.assertEqual(first,second);load.assert_called_once()
                self.assertEqual(json.loads((root/'demo/status.json').read_text())['phase'],'completed')
                self.assertTrue((root/'demo/complete.json').exists())
                changed=request();changed['state']='Different request';atomic_json(root/'request.json',changed)
                with self.assertRaisesRegex(ValueError,'different demo'):demo(args)


if __name__=='__main__':unittest.main()
