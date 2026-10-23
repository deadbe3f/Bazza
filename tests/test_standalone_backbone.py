"""Run the bundled backbone without an external model checkout or weights."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from data import collate_records
from model import Gemma4, Gemma4FeatureCache
from model import gemma
from settings import Config
from utils import atomic_json, file_hash


class StandaloneBackboneTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(73)
        self.config = gemma.GemmaConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1,
            num_global_key_value_heads=1, head_dim=8, global_head_dim=16,
            vocab_size=32, sliding_window=4,
            layer_types=('sliding_attention', 'full_attention'))

    def test_bundled_source_matches_feature_export_implementation(self):
        self.assertEqual(Path(gemma.__file__).resolve(),
                         Path(__file__).resolve().parents[1]/'model/gemma.py')
        self.assertEqual(file_hash(Path(gemma.__file__)),
            'e6eef82665a55340ca62e248d7aa3502d196e45b5689ab2e754166c8b2cef2d0')

    def test_live_loader_and_cached_states_use_local_backbone(self):
        encoder = gemma.GemmaTextModel(self.config).eval()
        rows = [dict(id=str(i), group_id=str(i), cache=f'{i}.pt',
                     ids=list(range(1, n+1)), positions=[2, n-1], target=[1., 0.])
                for i,n in enumerate([5, 9])]
        identity = dict(checkpoint=gemma.CHECKPOINT, revision=gemma.REVISION,
                        weight_sha256=gemma.WEIGHT_SHA256, quantization='nf4',
                        implementation_sha256=file_hash(Path(gemma.__file__)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['config.json', 'tokenizer.json']:
                atomic_json(root/name, {})
            settings = Config(data_selection=root/'selection.json', backbone_mode='live',
                              backbone_path=root, quantization='nf4').resolved(root)
            self.assertNotIn('kiln_model_path', settings.to_dict())
            live = Gemma4(root, quantization='nf4', selection={'identity':identity})
            batch = collate_records(rows)
            with patch('model.gemma.load_model', return_value=encoder) as load:
                with torch.no_grad():
                    result = live(batch).hidden
                load.assert_called_once_with(root, device=torch.device('cpu'), quantization='nf4')
            self.assertFalse(any(p.requires_grad for p in live.encoder.parameters()))
            self.assertIn(str(Path(gemma.__file__)), live.identity())
            for i,row in enumerate(rows):
                n = len(row['ids'])
                with torch.no_grad():
                    expected = encoder(batch.input_ids[i:i+1,:n], batch.attention_mask[i:i+1,:n])
                self.assertTrue(torch.equal(result[i,:n], expected[0]))
                torch.save({'hidden':result[i,:n]}, root/row['cache'])
            atomic_json(root/'index.json', {'files':{r['id']:{'file':r['cache'],
                'sha256':file_hash(root/r['cache'])} for r in rows}})
            cached = Gemma4FeatureCache(root, format='pt',
                selection={'identity':identity,'splits':{'train':rows}})
            self.assertTrue(torch.equal(result, cached(cached.prepare_batch(rows,'train')).hidden))

    def test_changed_implementation_is_rejected_before_loading_weights(self):
        identity = dict(checkpoint=gemma.CHECKPOINT, revision=gemma.REVISION,
                        weight_sha256=gemma.WEIGHT_SHA256, quantization='nf4',
                        implementation_sha256='wrong-source')
        live = Gemma4('/unused', quantization='nf4', selection={'identity':identity})
        with patch('model.gemma.load_model') as load:
            with self.assertRaisesRegex(ValueError, 'implementation_sha256'):
                live._load()
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
