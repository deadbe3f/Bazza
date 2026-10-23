"""Public commands, cached data identity and resumable prediction journals."""
from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
from train import run as train
from evaluate import run as evaluate
from utils import atomic_json,file_hash


class CommandTests(unittest.TestCase):
    def fixture(self,root):
        torch.manual_seed(4)
        records=[];files={}
        for i in range(16):
            row=dict(id=str(i),group_id=str(i),ids=[1]*5,positions=[1,4],target=[float(i%2==0),float(i%2==1)],
                category='tools',dataset_source='fixture',source='fixture',kind='nool',task_key='fixture/choice',
                options=['yes','no'],semantic_class='yes',balance_schema='fixture',cache=f'{i}.pt')
            # Historical test caches omit training-only diagnostic annotations.
            if i >= 12:
                row.pop('category'); row.pop('semantic_class')
                row['domain'] = 'external'
            records.append(row)
            torch.save({'hidden':torch.randn(5,16)},root/row['cache'])
            files[row['id']]={'file':row['cache'],'sha256':file_hash(root/row['cache'])}
        atomic_json(root/'index.json',{'files':files})
        selection={'identity':{'hidden_width':16},'splits':{'train':records[:8],'validation':records[8:12],'test':records[12:]}}
        atomic_json(root/'selection.json',selection)
        torch.save({'weight':torch.randn(26,16),'softcap':30},root/'letters.pt')
        (root/'config.py').write_text("""from pathlib import Path
from settings import Config
config=Config(
    data_selection=Path('selection.json'),
    backbone_mode='cache',feature_format='pt',feature_cache_path=Path('.'),
    letter_readout_path=Path('letters.pt'),prefetch_batches=2,
    encoder_width=16,hidden_width=8,attention_heads=2,low_steps=1,high_cycles=1,
    gate_width=4,adapter_rank=2,
    batch_size=2,token_budget=32,cpu_threads=2,log_every_steps=1,save_every_examples=2,
    optimizer='adamw',learning_rate=.001,warmup_examples=2,decay_start_examples=4,
    validation_batch_size=2,validation_token_budget=32,validation_every_examples=4,
)
""")
        return root/'config.py'

    def test_train_evaluate_demo_and_partial_journal_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=self.fixture(root)
            args=Namespace(config=config,output_dir=root/'train',device='cpu',resume=False,initialize_from=None,plan_only=False,no_progress=True)
            train(args)
            summary=json.loads((root/'train/summary.json').read_text())
            self.assertEqual(summary['examples_seen'],8)
            self.assertEqual(summary['completed_epochs'],1)
            log=[json.loads(line) for line in (root/'train/logging/metrics.jsonl').read_text().splitlines()]
            self.assertTrue(any('train/loss' in row['values'] for row in log))
            self.assertEqual(len(json.loads((root/'train/checkpoints.json').read_text())['checkpoints']),3)
            args.resume=True;train(args)
            with self.assertRaises(ValueError):
                args.resume=False;train(args)
            checkpoints=json.loads((root/'train/checkpoints.json').read_text())['checkpoints']
            selected=root/'train'/checkpoints[0]['file']
            ev=Namespace(config=config,checkpoint=selected,output_dir=root/'test',split='test',device='cpu',resume=False,plan_only=False,no_progress=True)
            report=evaluate(ev)
            self.assertEqual(report['evaluated_rows'],4)
            lines=(root/'test/predictions.jsonl').read_bytes().splitlines(keepends=True)
            self.assertEqual(len(lines),2)
            (root/'test/predictions.jsonl').write_bytes(lines[0]+b'{"unfinished":')
            (root/'test/report.json').unlink();(root/'test/complete.json').unlink()
            ev.resume=True;resumed=evaluate(ev)
            self.assertEqual(report['by_dataset_source'],resumed['by_dataset_source'])
            ids=[p['id'] for line in (root/'test/predictions.jsonl').read_text().splitlines()
                 for p in json.loads(line)['predictions']]
            self.assertEqual(len(ids),len(set(ids)))
            result=subprocess.run([sys.executable,'demo.py','--config',str(config),'--checkpoint',str(selected),
                '--output-dir',str(root/'demo'),'--split','test','--example-id','12','--device','cpu','--no-progress'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('"letter":',result.stdout)
            prior=(root/'test/predictions.jsonl').read_bytes()
            index=json.loads((root/'index.json').read_text());index['changed']=True;atomic_json(root/'index.json',index)
            with self.assertRaisesRegex(ValueError,'different evaluation'):evaluate(ev)
            self.assertEqual(prior,(root/'test/predictions.jsonl').read_bytes())


    def test_automatic_test_selects_validation_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config_path=self.fixture(root)
            with config_path.open('a') as stream:stream.write('\nconfig.test_config_path = Path(__file__)\n')
            train(Namespace(config=config_path,output_dir=root/'train',device='cpu',resume=False,
                            initialize_from=None,plan_only=False,no_progress=True))
            result=json.loads((root/'train/workflow-complete.json').read_text())
            retained=json.loads((root/'train/checkpoints.json').read_text())['checkpoints']
            self.assertEqual(result['selected_checkpoint'],retained[0])
            self.assertEqual(result['test_rows'],4)


if __name__=='__main__':unittest.main()
