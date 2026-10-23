"""Serialization, labels, split ownership and raw-feature normalization contracts."""
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch
from data import encode, load_selection
from features.bank import normalize_final
from model import Gemma4FeatureCache
from data import collate_records
from utils import atomic_json, digest, file_hash

sys.path.insert(0,str(Path(__file__).resolve().parent/'reference/legacy'))
from gemma_decisions.pilot import encode as old_encode
from architecture.data import identity_keys as old_keys
from architecture.patch_when2call import added_candidate as old_candidate
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/data'))
from common import identity_keys
from when2call_candidates import added_candidate


class DataTests(unittest.TestCase):
    def test_rounded_teacher_probabilities_keep_original_values(self):
        probabilities = [.0707, .2828, .2727, .2828, .0808, .0101]
        row = {'id':'rounded-teacher', 'ids':[1]*6,
               'positions':list(range(6)), 'target':probabilities}
        batch = collate_records([row])
        self.assertTrue(torch.equal(batch.targets[0], torch.tensor(probabilities)))
        # This is the real rounding boundary that the FP32 sum rejected.
        self.assertGreater(abs(float(batch.targets.sum()) - 1), 1e-4)
        for invalid in ([.2]*6, [-.1, .3, .2, .2, .2, .2],
                        [float('nan'), .2, .2, .2, .2, .2]):
            with self.assertRaisesRegex(ValueError, 'Invalid target distribution'):
                collate_records([{**row, 'target':invalid}])

    def test_serializer_source_and_dataset_repairs_are_unchanged(self):
        self.assertEqual(inspect.getsource(encode),inspect.getsource(old_encode))
        row={'id':'fixture-é','group_id':'g','state':{'question':'café','tools':[]},
             'question':'décision?','options':['oui','non'],'source':'fixture'}
        self.assertEqual(old_keys(row),identity_keys(row))
        for policy in ['tool_call','request_for_info','cannot_answer','direct']:
            self.assertEqual(old_candidate(policy,row),added_candidate(policy,row))

    def test_train_and_heldout_groups_must_not_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'selection.json'
            atomic_json(path,{'splits':{'train':[{'id':'1','group_id':'g'}],
                                       'test':[{'id':'2','group_id':'g'}]}})
            with self.assertRaisesRegex(ValueError,'membership overlap'):load_selection(path)

    def test_raw_features_are_normalized_once_and_mismatches_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);split=root/'train';split.mkdir()
            row={'id':'one','ids':[1]*5,'positions':[1,4],'order':[0,1],'target':[1.,0.]}
            atomic_json(split/'records.json',[row])
            hidden=torch.randn(5,16).to(torch.bfloat16)
            norm={'weight':torch.randn(16),'eps':1e-6}
            torch.save(norm,root/'final-rmsnorm.pt')
            atomic_json(root/'manifest.json',{'backbone':{'checkpoint':'fixture'},'serialization_sha256':'abc'})
            file='layers/layer-48/shard-00000.pt';(split/file).parent.mkdir(parents=True)
            torch.save({'hidden':hidden},split/file)
            entry={'shard':0,'records_sha256':digest([row]),'rows':[{'id':'one','offset':0,'tokens':5}],
                   'files':{'layer-48':{'file':file,'sha256':file_hash(split/file),'bytes':(split/file).stat().st_size}}}
            (split/'layer-additions.jsonl').write_text(json.dumps(entry)+'\n')
            atomic_json(split/'index.json',{'rows_by_layer':{'layer-48':1}})
            selection={'identity':{'checkpoint':'fixture','serialization_sha256':'abc'},'splits':{'train':[row]}}
            provider=Gemma4FeatureCache(root,selection=selection)
            batch=provider(collate_records([row]))
            self.assertTrue(torch.equal(batch.hidden[0],normalize_final(hidden,norm)))
            changed={**row,'positions':[2,4]}
            with self.assertRaisesRegex(ValueError,'input identity'):provider(collate_records([changed]))
            selection['identity']['checkpoint']='other'
            with self.assertRaisesRegex(ValueError,'backbone differs'):
                Gemma4FeatureCache(root,selection=selection)


if __name__=='__main__':unittest.main()
