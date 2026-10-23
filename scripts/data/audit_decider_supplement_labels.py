#!/usr/bin/env python3
"""Check exported targets directly against native records and teacher annotations.

This does not call the supplement's adapters or selection functions. Rule rows
are reproduced from their individually recorded seeds and pinned generator.
"""
import argparse
from collections import Counter
import csv
import importlib
import json
from pathlib import Path
import sys

from common import REPO, canonical, digest, file_hash, jsonl, key, parquet
from decider_supplement import input_texts
from download_decider_supplement import DECIDER_REVISION


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',type=Path,default=REPO/'build/groundbeef/decider-supplement-v0.1')
    a=ap.parse_args();rows={};aliases={};groups={};counts=Counter()
    manifest=json.loads((a.dataset/'manifest.json').read_text())
    for f in manifest['files']:
        assert file_hash(a.dataset/f['file'])==f['sha256'],f['file']
    for split in ['train','validation','calibration']:
        for r in jsonl(a.dataset/(split+'.jsonl.gz')):
            assert r['id'] not in rows
            rows[r['id']]=r;g=r['group_id']
            assert g not in groups or groups[g]==split
            groups[g]=split
            for t in input_texts(r):
                k=key(t)
                assert k not in aliases or aliases[k]==split,('cross_split_alias',r['id'])
                aliases[k]=split
            counts['exported:'+split]+=1
    lock=json.loads((a.dataset/'sources.json').read_text())
    for s in lock['sources']:
        for f in s['files']:
            p=REPO/'datasets/sources'/('decider-'+s['name'])/s['revision']/'raw'/f['path']
            assert file_hash(p)==f['sha256']
            if p.name=='README.md':continue
            original=parquet(p) if p.suffix=='.parquet' else csv.DictReader(p.open()) if p.suffix=='.csv' else jsonl(p)
            for i,x in enumerate(original):
                ident=s['name']+':'+f['path']+':'+str(i)
                for suffix in ['']+([':jailbreak'] if s['name']=='toxic_chat' else []):
                    r=rows.get(ident+suffix)
                    if not r:continue
                    assert 'train' in p.name
                    assert r['metadata']['source_row_sha256']==digest(x)
                    y=round(float(x['label'])) if s['name']=='stsb' else int(x['jailbreaking' if suffix else 'toxicity']) if s['name']=='toxic_chat' else int(x['label'])
                    assert r['target'].index(1.)==y,(r['id'],'native_target')
                    if s['name']=='toxic_chat':
                        assert x['human_annotation'].lower()=='true'
                        assert r['state']==x['user_input']
                        assert r['options']==['no','yes']
                    elif s['name']=='ag_news':
                        assert r['options']==['World','Sports','Business','Science and technology']
                        assert r['state']==x['text']
                    elif s['name']=='sst2':
                        assert r['options']==['negative','positive']
                        assert r['state']==x['sentence']
                    elif s['name']=='paws':
                        assert r['options']==['not paraphrases','paraphrases']
                        assert json.loads(r['state'])=={'text1':x['sentence1'],'text2':x['sentence2']}
                    elif s['name']=='wiki_qa':
                        assert r['options']==['no','yes']
                        assert json.loads(r['state'])=={'question':x['question'],'document_title':x['document_title'],'candidate_sentence':x['answer']}
                    elif s['name']=='stsb':
                        assert json.loads(r['state'])=={'text1':x['text1'],'text2':x['text2']}
                        assert r['metadata']['native_score']==float(x['label'])
                    counts['native:'+s['name']]+=1
    upstream=REPO/'datasets/sources/mapika-decider'/DECIDER_REVISION/'raw'
    sys.path.insert(0,str(upstream))
    render=importlib.import_module('decider.systemone');rule=importlib.import_module('decider.data.rules')
    for filename,name in [('custom_questions.jsonl','decider_custom'),('routing_messages.jsonl','decider_routing'),
                          ('routing_terse.jsonl','decider_routing'),('commands.jsonl','decider_commands')]:
        for i,x in enumerate(jsonl(upstream/'teacher_data'/filename)):
            for j,q in enumerate(x['questions']):
                r=rows.get(f'{name}:{filename}:{i}:{j}')
                if not r:continue
                expected=render.render_question(q)
                gold=list(q['criteria']).index(q['answer']) if q['type']=='choice' else int(q['answer'])
                if 'teacher_ok' in q and 'teacher_pred' in q:
                    assert q['teacher_ok'] is True
                    assert q['teacher_pred']==(q['answer'] if q['type']=='choice' else gold)
                else:
                    assert .5001<q['teacher_p']<=1
                assert r['target'].index(1.)==gold
                assert r['options']==expected['options'] and r['question']==expected['question']
                assert r['state']==render.render_state(x['state'],index_arrays=False)
                assert r['metadata']['source_row_sha256']==digest(x)
                counts['teacher:'+name]+=1
    generated={}
    for r in rows.values():
        if r['source']!='decider_rules':continue
        m=r['metadata'];seed=m['generator_seed']
        if seed not in generated:
            families=[f for f in rule.FAMILIES if f not in rule.HELD_FAMILIES and f!='form']
            generated[seed]=rule.build(3,seed=seed,domains=rule.HELD_DOMAINS if m['heldout_domain'] else None,families=families)
        e=generated[seed][m['generator_row']];q=e.qs[0]
        assert r['state']==e.context and r['question']==q.text
        assert r['options']==q.options and r['target'].index(1.)==q.gold
        counts['computed_rules']+=1
    checked=sum(v for k,v in counts.items() if not k.startswith('exported:'))
    assert checked==len(rows),(checked,len(rows))
    result={'verified':True,'checked_every_exported_target':checked,'counts':dict(counts),
            'cross_split_context_aliases':0,'cross_split_parents':0,
            'label_checks':'Native values without adapters; exact verifier annotations or stored gold probability >0.5001; seeded upstream rule reproduction.',
            'audit_script_sha256':file_hash(__file__)}
    (a.dataset/'label_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
