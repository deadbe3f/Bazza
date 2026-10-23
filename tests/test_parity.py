"""Exact CPU parity with the preserved pre-refactor research implementation."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock
import torch
from data import collate_records
from losses import DecisionLoss
from model import Bazza, BazzaDecisionHead
from fixtures import PrecomputedBackbone
from optimizers import make_optimizer
from trainer import Trainer

# Test-only reference imports. Production code never imports archived runners.
sys.path.insert(0, str(Path(__file__).resolve().parent / 'reference/legacy'))
from architecture.model import make_head as old_head
from architecture.readout_ablation import LetterReadout
from architecture.readout_mixed import MixedReadout
from architecture.train_v04 import OnlineTrainer


def rows():
    return [dict(id=str(i), group_id=str(i), ids=[1] * n, positions=[1, n-1], target=[float(i%2==0),float(i%2==1)],
                 category='tools', dataset_source='fixture', source='fixture', kind='nool', task_key='fixture/choice',
                 options=['yes','no'], semantic_class='yes', balance_schema='fixture') for i,n in enumerate([5,9])]


class ParityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.weight = torch.randn(26, 32)
        self.batch = collate_records(rows(), [torch.randn(5,32), torch.randn(9,32)])
        torch.manual_seed(42)
        self.old = MixedReadout(old_head(32,16,4,1,1,'bigru'), LetterReadout(self.weight,30),32,16,8,4)
        torch.manual_seed(42)
        self.new = BazzaDecisionHead(self.weight,30,encoder_width=32,width=16,attention_heads=4,
                                    low_steps=1,high_cycles=1,gate_width=8,rank=4)

    def test_state_keys_logits_losses_and_gradients_are_exact(self):
        self.assertEqual(list(self.old.state_dict()),list(self.new.state_dict()))
        for key,value in self.old.state_dict().items():
            self.assertTrue(torch.equal(value,self.new.state_dict()[key]),key)
        b=self.batch
        expected=self.old(b.hidden,b.attention_mask,b.option_positions,b.option_mask)
        actual=self.new(**b.inputs())
        self.assertTrue(torch.equal(expected,actual))
        old_loss=-(b.targets*expected.log_softmax(-1)).sum(-1).mean()
        new_loss,_=DecisionLoss()(actual,b)
        self.assertTrue(torch.equal(old_loss,new_loss))
        old_loss.backward();new_loss.backward()
        for (key,left),(_,right) in zip(self.old.named_parameters(),self.new.named_parameters()):
            self.assertEqual(left.grad is None,right.grad is None,key)
            if left.grad is not None:self.assertTrue(torch.equal(left.grad,right.grad),key)

    def test_muon_adamw_update_and_sample_schedule_match_reference(self):
        b=self.batch
        old_optimizer=make_optimizer(self.old,'muon',3e-5,.01,.999,.003)
        new_optimizer=make_optimizer(self.new,'muon',3e-5,.01,.999,.003)
        old=OnlineTrainer(self.old,old_optimizer,None,None,'cpu',online=Mock())
        old.online.collate.return_value=(b.hidden,b.attention_mask,b.option_positions,b.option_mask,b.targets)
        new=Trainer(Bazza(PrecomputedBackbone(),self.new),new_optimizer,checkpoint_model=self.new,device='cpu',progress=False)
        old.scheduler=torch.optim.lr_scheduler.LambdaLR(old_optimizer,lambda _: (old.state['examples_seen']+32)/640)
        new.scheduler=torch.optim.lr_scheduler.LambdaLR(new_optimizer,lambda _: (new.state['examples_seen']+32)/640)
        expected=old.train_batch(b.rows)
        actual=new.train_batch(b)
        self.assertEqual(expected,actual)
        self.assertEqual(old.state['examples_seen'],new.state['examples_seen'])
        self.assertEqual(old.state['step'],new.state['step'])
        self.assertEqual([g['lr'] for g in old_optimizer.param_groups],[g['lr'] for g in new_optimizer.param_groups])
        for key,value in self.old.state_dict().items():self.assertTrue(torch.equal(value,self.new.state_dict()[key]),key)


if __name__=='__main__':unittest.main()
