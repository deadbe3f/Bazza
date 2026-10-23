#!/usr/bin/env python3
"""Build an additive, source-locked Decider-inspired decision dataset.

Never loads model weights or modifies a prior dataset. Exact input/alias and
approximate trigram checks apply against all prior Groundbeef releases, the
standalone source ledger, protected benchmarks and native new-source holdouts.
"""
import argparse
from collections import Counter, defaultdict, deque
import csv
import importlib
import json
from pathlib import Path
import random
import re
import shutil
import sqlite3
import sys
import zlib
from functools import lru_cache

from common import REPO, canonical, digest, dumps, file_hash, jsonl, key, parquet, write_jsonl
from groundbeef import prompt, validate, visible_hash, state_object
from download_decider_supplement import DECIDER_REVISION

SEED = 20261012
VERSION = 'decider-supplement-v0.1'
MAX_TOKENS = 4096
SCALE = ['completely unrelated','same topic but different meaning','share some details',
         'roughly equivalent, important details differ','mostly equivalent, minor details differ',
         'completely equivalent in meaning']
QUOTAS = {'ag_news':6000,'sst2':5000,'paws':6000,'toxic_chat':3000,'stsb':4000,'wiki_qa':4000,
          'decider_custom':7000,'decider_routing':7000,'decider_commands':3000,'decider_rules':18000}


def teacher_agrees(q,answer):
    if 'teacher_ok' in q and 'teacher_pred' in q:
        return bool(q['teacher_ok']) and q['teacher_pred']==(q['answer'] if q['type']=='choice' else answer)
    # Older custom files retained only the probability assigned to the gold.
    # P(gold)>0.5 mathematically certifies strict argmax agreement. Leave a
    # margin for the recorded four-decimal rounding. Never infer from a low P.
    return isinstance(q.get('teacher_p'),(int,float)) and .5001<q['teacher_p']<=1


def save(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def input_texts(r):
    """Aliases cover raw contexts, structured request fields and question pairs.

    Both members of a sentence pair are protected. Short generic field values
    are not independently blacklisted; the whole input still always is.
    """
    value = state_object(r.get('state',''))
    texts = [dumps(value), value] if isinstance(value,str) else [dumps(value)]
    def walk(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k in {'request','current_user_request','question','prompt','dialogue_prefix',
                         'conversation_prompt','text1','text2','sentence1','sentence2','user_input'} and isinstance(v,str):
                    texts.append(v)
                elif isinstance(v,(dict,list)): walk(v)
            if 'premise' in x and 'hypothesis' in x: texts.append(str(x['premise'])+'\n'+str(x['hypothesis']))
        elif isinstance(x,list):
            for v in x:
                if isinstance(v,dict) and v.get('role')=='user' and isinstance(v.get('content'),str): texts.append(v['content'])
                walk(v)
    walk(value)
    if not value and r.get('question'): texts.append(r['question'])
    return sorted({canonical(t) for t in texts if isinstance(t,str) and t.strip()})


@lru_cache(maxsize=1024)
def shingles(text):
    words = re.findall(r'\w+', canonical(text))
    return {' '.join(words[i:i+3]) for i in range(len(words)-2)}


def sketch(text):
    ss = shingles(text)
    if len(ss)<8: return None
    return sorted({zlib.crc32(s.encode()) for s in ss})[:4]


class Protection:
    def __init__(self, path, readonly=False):
        self.cache={}
        self.db = sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True) if readonly else sqlite3.connect(path)
        if not readonly:
            self.db.executescript('''PRAGMA journal_mode=WAL;
                CREATE TABLE exact (key TEXT PRIMARY KEY, origin TEXT);
                CREATE TABLE visible (hash TEXT PRIMARY KEY);
                CREATE TABLE near (key TEXT PRIMARY KEY, text TEXT, s0 INTEGER,s1 INTEGER,s2 INTEGER,s3 INTEGER);
                CREATE TABLE exclusions (id TEXT, source TEXT, reason TEXT, origin TEXT);
            ''')
    def add(self,r,origin,near=True):
        if 'options' in r and 'question' in r:
            self.db.execute('INSERT OR IGNORE INTO visible VALUES (?)',(visible_hash(r),))
        for t in input_texts(r):
            self.db.execute('INSERT OR IGNORE INTO exact VALUES (?,?)',(key(t),origin))
            sk=sketch(t) if near else None
            if sk and len(sk)==4:
                self.db.execute('INSERT OR IGNORE INTO near VALUES (?,?,?,?,?,?)',(key(t),t,*sk))
    def finish(self):
        for i in range(4): self.db.execute(f'CREATE INDEX IF NOT EXISTS near_s{i} ON near(s{i})')
        self.db.commit()
    def match(self,r,near=True):
        if self.db.execute('SELECT 1 FROM visible WHERE hash=?',(visible_hash(r),)).fetchone():
            return 'existing_complete_input','visible input fingerprint'
        for t in input_texts(r):
            ck=(t,near)
            if ck in self.cache:
                if self.cache[ck]:return self.cache[ck]
                continue
            found=self.db.execute('SELECT origin FROM exact WHERE key=?',(key(t),)).fetchone()
            if found: return 'existing_or_protected_context',found[0]
            sk=sketch(t) if near else None
            if not sk or len(sk)!=4:
                self.cache[ck]=None;continue
            placeholders=','.join('?' for _ in sk)
            # Approximate candidate retrieval requires two shared bottom-four
            # signatures, followed by the full Jaccard check. Exact protection
            # is unconditional. This avoids scanning common single trigrams.
            queries=[f'SELECT key FROM near WHERE s{i} IN ({placeholders})' for i in range(4)]
            sql='WITH hits AS ('+' UNION ALL '.join(queries)+') SELECT n.key,n.text FROM near n JOIN (SELECT key FROM hits GROUP BY key HAVING count(*)>=2) h ON n.key=h.key'
            ss=shingles(t)
            for identity,other in self.db.execute(sql,sk*4):
                os=shingles(other)
                if len(ss&os)/len(ss|os)>=.85:
                    found=('near_existing_or_protected_context',identity)
                    self.cache[ck]=found;return found
            self.cache[ck]=None
        return None


def record(source, ident, group, state, question, options, answer, family, kind='choice', **meta):
    name=source['name']
    return {'id':name+':'+str(ident),'group_id':name+':'+str(group),'source':name,
            'domain':meta.pop('category','language'),'family':family,'kind':kind,
            'state':state if isinstance(state,str) else dumps(state),'question':question,
            'options':options,'target':[float(i==answer) for i in range(len(options))],
            'split':'candidate','metadata':{'dataset_source':name,'source_repo':source['repo'],
              'source_revision':source['revision'],'native_id':str(ident),'task_type':family,
              'subcategory':family,'prior_training_exposure':False,'prior_validation_monitoring':False,
              'retained_from_v02':False,'loss_weight':1.,**meta}}


def public_records(s, rawroot):
    for file in s['files']:
        path=rawroot/('decider-'+s['name'])/s['revision']/'raw'/file['path']
        if path.name=='README.md': continue
        native='train' if 'train' in path.name else 'validation' if 'validation' in path.name else 'test'
        rows=parquet(path) if path.suffix=='.parquet' else csv.DictReader(path.open()) if path.suffix=='.csv' else jsonl(path)
        for i,x in enumerate(rows):
            name=s['name']; ident=file['path']+':'+str(i)
            meta={'source_file':file['path'],'source_row_sha256':digest(x),'native_split':native,
                  'label_origin':'native_label','label_basis':'native_label'}
            if name=='ag_news':
                state=x['text'];opts=['World','Sports','Business','Science and technology'];y=int(x['label'])
                question='Which topic best describes this news article?';family='language.news_topic'; group=key(state)
            elif name=='sst2':
                state=x['sentence'];opts=['negative','positive'];y=int(x['label'])
                question='What is the sentiment of this movie-review passage?';family='language.sentiment';group=key(state)
            elif name in {'paws','stsb'}:
                a,b=(x['sentence1'],x['sentence2']) if name=='paws' else (x['text1'],x['text2'])
                state={'text1':a,'text2':b};group=digest(sorted([canonical(a),canonical(b)]))
                opts=['not paraphrases','paraphrases'] if name=='paws' else [f'{j}: {v}' for j,v in enumerate(SCALE)]
                y=int(x['label']) if name=='paws' else round(float(x['label']))
                question='Do these two sentences express the same meaning?' if name=='paws' else 'How similar in meaning are the two texts?'
                family='language.paraphrase' if name=='paws' else 'ordinal.semantic_similarity'
                if name=='stsb': meta.update(native_score=float(x['label']),score_transform='round nearest integer (Python ties-to-even), 0..5')
            elif name=='toxic_chat':
                state=x['user_input']; group=x['conv_id'];opts=['no','yes'];y=int(x['toxicity'])
                question='Is this user request toxic?';family='moderation.toxicity'
                # Never silently call model-labelled negatives human judgments.
                meta.update(label_origin='human_annotation' if x['human_annotation'].lower()=='true' else 'upstream_unannotated',
                            label_basis='human_annotation' if x['human_annotation'].lower()=='true' else 'upstream_unannotated')
            elif name=='wiki_qa':
                state={'question':x['question'],'document_title':x['document_title'],'candidate_sentence':x['answer']}
                group=x['question_id'];opts=['no','yes'];y=int(x['label'])
                question='Does the candidate sentence answer the question?';family='grounding.answer_relevance'
            else: raise ValueError(name)
            r=record(s,ident,group,state,question,opts,y,family,kind='score' if name=='stsb' else 'choice',
                     semantic_class=str(y),category='ordinal' if name=='stsb' else 'grounding' if name=='wiki_qa' else 'moderation' if name=='toxic_chat' else 'language',**meta)
            r['_native']=native
            r['_invalid']=y<0 or y>=len(opts) or meta['label_origin']=='upstream_unannotated'
            yield r
            if name=='toxic_chat':
                r=record(s,ident+':jailbreak',group,state,'Is this user request a jailbreak attempt?',opts,int(x['jailbreaking']),
                         'moderation.jailbreak',kind='noul',semantic_class=str(x['jailbreaking']),category='moderation',**meta)
                r['_native']=native;r['_invalid']=meta['label_origin']=='upstream_unannotated';yield r


def teacher_records(upstream, renderer, counters):
    configs=[('custom_questions.jsonl','decider_custom'),('routing_messages.jsonl','decider_routing'),
             ('routing_terse.jsonl','decider_routing'),('commands.jsonl','decider_commands')]
    for filename,name in configs:
        source={'name':name,'repo':'Mapika/decider','revision':DECIDER_REVISION}
        for i,x in enumerate(jsonl(upstream/'teacher_data'/filename)):
            state=renderer.render_state(x['state'],index_arrays=False)
            for j,q in enumerate(x['questions']):
                counters[name+':raw_questions']+=1
                rendered=renderer.render_question(q)
                answer=rendered['names'].index(q['answer'])
                pred=q.get('teacher_pred')
                # Strict equality, including Score. No generic/catch-all exception,
                # adjacent Score/risk tolerance or unverified command outside flag.
                agreed=teacher_agrees(q,answer)
                if not agreed:
                    counters[name+':teacher_disagreement_or_missing_check']+=1;continue
                yield record(source,filename+':'+str(i)+':'+str(j),key(state),state,rendered['question'],rendered['options'],answer,
                             'typed.'+('command_risk' if name=='decider_commands' else x.get('recipe','custom')),
                             kind=q['type'],category='typed_decisions',semantic_class=str(q['answer']),
                             label_origin='Qwen3.5-27B_generated_same_model_rechecked',
                             label_basis='teacher_consistency_not_independent_truth',source_file='teacher_data/'+filename,
                             source_row_sha256=digest(x),teacher_question_index=j,teacher_probability=q.get('teacher_p'),
                             native_domain=x['domain'],native_recipe=x['recipe'],strict_teacher_agreement=True)


def rule_records(rules):
    source={'name':'decider_rules','repo':'Mapika/decider','revision':DECIDER_REVISION}
    # One generator call per parent, retained together. A call can emit base,
    # state twin, rule twin, or another record; conservative grouping is safe.
    for split,count in [('train',6000),('validation',600),('calibration',300)]:
        for i in range(count):
            seed=int(digest([SEED,split,i])[:12],16)
            domain=rules.HELD_DOMAINS if split=='validation' else None
            families=[f for f in rules.FAMILIES if f not in rules.HELD_FAMILIES and f!='form']
            exs=rules.build(3,seed=seed,domains=domain,families=families)
            for j,e in enumerate(exs):
                q=e.qs[0]
                r=record(source,f'{split}:{i}:{j}',f'{split}:{i}',e.context,q.text,q.options,q.gold,e.task,
                         category='reasoning',semantic_class=str(q.options[q.gold]),
                         label_origin='programmatic_exact',label_basis='computed_by_pinned_rule_generator',
                         generator_seed=seed,generator_parent=f'{split}:{i}',
                         generator_row=j,heldout_domain=split=='validation')
                r['_forced_split']=split;yield r


def add_existing(protection,history,current,panel,ledger):
    paths=set()
    for root in [history,current]:
        if root==history:
            for version in ['v0.1','v0.2']:
                paths.update((root/version).glob('*.jsonl.gz'))
                paths.update((root/version/'heldout_pools').glob('*.jsonl.gz'))
        else: paths.update(root.glob('*.jsonl.gz'))
    # Membership is provenance, not a model-visible dataset.
    paths={p for p in paths if p.name not in {'membership.jsonl.gz','new_source_ownership.jsonl.gz','protected_membership.jsonl.gz'}}
    if not paths or not current.is_dir(): raise ValueError('Prior Groundbeef releases unavailable')
    manifest=[]
    for path in sorted(paths):
        count=0
        for r in jsonl(path):
            protection.add(r,str(path));count+=1
        manifest.append({'path':str(path),'rows':count,'sha256':file_hash(path)})
        protection.db.commit()
        print('Protected prior dataset',path.name,count,flush=True)
    for path in sorted(panel.glob('test.jsonl')):
        count=0
        for r in jsonl(path): protection.add(r,str(path));count+=1
        manifest.append({'path':str(path),'rows':count,'sha256':file_hash(path)})
    db=sqlite3.connect('file:'+str(ledger)+'?mode=ro',uri=True)
    for table,query in [('native_source_pool','SELECT DISTINCT key FROM examples'),('protected_native','SELECT key FROM protected')]:
        for (k,) in db.execute(query): protection.db.execute('INSERT OR IGNORE INTO exact VALUES (?,?)',(k,table))
    manifest.append({'path':str(ledger),'sha256':file_hash(ledger),'protection':'all source-pool question keys and complete protected native keys'})
    db.close();protection.finish();return manifest


class Groups:
    def __init__(self): self.parent={}
    def find(self,x):
        self.parent.setdefault(x,x)
        if self.parent[x]!=x: self.parent[x]=self.find(self.parent[x])
        return self.parent[x]
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)


def grouped_split(r, groups):
    if '_forced_split' in r: return r['_forced_split']
    n=int(digest([SEED,'group-split',groups.find(r['group_id'])])[:8],16)%100
    return 'validation' if n<8 else 'calibration' if n<12 else 'train'


def choose(rows,quota,tokenizer,counters,protection):
    buckets=defaultdict(list)
    for r in rows:buckets[(r['family'],r['metadata']['semantic_class'])].append(r)
    pools={k:deque(sorted(v,key=lambda r:digest([SEED,r['id']]))) for k,v in buckets.items()}
    selected=[];parents=Counter()
    while len(selected)<quota:
        batch=[]
        for k in sorted(pools):
            if pools[k]:batch.append(pools[k].popleft())
        if not batch:break
        encs=tokenizer.encode_batch([prompt(r) for r in batch],add_special_tokens=False)
        for r,enc in zip(batch,encs):
            if len(selected)==quota:break
            overlap=protection.match(r)
            if overlap:
                counters[r['source']+':'+overlap[0]]+=1
                protection.db.execute('INSERT INTO exclusions VALUES (?,?,?,?)',(r['id'],r['source'],*overlap))
                continue
            if len(enc.ids)>MAX_TOKENS:counters[r['source']+':over_token_budget']+=1;continue
            if parents[r['group_id']]>=3:counters[r['source']+':parent_cap']+=1;continue
            r['metadata']['input_tokens']=len(enc.ids);selected.append(r);parents[r['group_id']]+=1
    return selected


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--history',type=Path,required=True)
    ap.add_argument('--tokenizer',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=REPO/'build/groundbeef'/VERSION)
    ap.add_argument('--audit-only',action='store_true')
    ap.add_argument('--reuse-protection',type=Path,help='An audited draft with identical source lock and preserved historical files')
    a=ap.parse_args()
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(a.tokenizer))
    if a.audit_only:
        audit(a.output,a.tokenizer,tokenizer);return
    if a.output.exists():raise ValueError('Immutable output already exists; use --audit-only or another output directory')
    stage=a.output.with_name('.'+a.output.name+'-staging')
    if stage.exists():raise ValueError('Staging exists; inspect it before retrying')
    stage.mkdir(parents=True)
    lockpath=Path(__file__).with_name('decider_supplement_sources.json')
    lock=json.loads(lockpath.read_text());rawroot=REPO/'datasets/sources'
    upstream=rawroot/'mapika-decider'/DECIDER_REVISION/'raw'
    for s in lock['sources']:
        for f in s['files']:
            p=rawroot/('decider-'+s['name'])/s['revision']/'raw'/f['path']
            if file_hash(p)!=f['sha256']:raise ValueError('Raw source changed: '+str(p))
    for f in lock['decider']['files']:
        if file_hash(upstream/f['path'])!=f['sha256']:raise ValueError('Upstream file changed: '+f['path'])
    sys.path.insert(0,str(upstream))
    renderer=importlib.import_module('decider.systemone');rules=importlib.import_module('decider.data.rules')
    if a.reuse_protection:
        previous=json.loads((a.reuse_protection/'manifest.json').read_text())
        if not json.loads((a.reuse_protection/'audit.json').read_text())['verified']:raise ValueError('Reuse draft not audited')
        if previous['source_lock_sha256']!=file_hash(lockpath):raise ValueError('Reuse protection has different sources')
        expected=next(f['sha256'] for f in previous['files'] if f['file']=='protection.sqlite')
        if file_hash(a.reuse_protection/'protection.sqlite')!=expected:raise ValueError('Reuse protection changed')
        for p in previous['protected_files']:
            if file_hash(p['path'])!=p['sha256']:raise ValueError('Reuse protected file changed')
        shutil.copyfile(a.reuse_protection/'protection.sqlite',stage/'protection.sqlite')
        protect=Protection(stage/'protection.sqlite',readonly=True)
        protect.db.close()
        # Reopen existing verified ledger for this build's exclusion trace.
        protect.db=sqlite3.connect(stage/'protection.sqlite')
        protect.db.execute('DELETE FROM exclusions');protect.db.commit()
        files=previous['protected_files']
        print('Reused hash-verified complete protection ledger',flush=True)
    else:
        protect=Protection(stage/'protection.sqlite')
        files=add_existing(protect,a.history,REPO/'build/groundbeef/v0.3',REPO/'datasets/evaluation-panel',
                           REPO/'datasets/decisions/source-pool-v0.1/membership.sqlite')
    used_sources=set()
    used_repos=set()
    def collect_repos(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k in {'repo','source_repo'} and isinstance(v,str):used_repos.add(v.casefold())
                elif isinstance(v,(dict,list)):collect_repos(v)
        elif isinstance(x,list):
            for v in x:collect_repos(v)
    for p in [REPO/'build/groundbeef/v0.3/sources.json',a.history/'v0.2/sources.json']:
        # Human-reviewable source inventory, not heuristics over names of domains.
        used_sources.add(file_hash(p))
        collect_repos(json.loads(p.read_text()))
    for s in lock['sources']:
        if s['repo'].casefold() in used_repos:raise ValueError('Source already used by Groundbeef: '+s['repo'])
    excluded={'banking77','clinc_oos','boolq','helpsteer2','toolace','hermes_tools','mmlu','arc','openbookqa','sciq',
              'commonsense_qa','winogrande','hellaswag','gsm8k','agenttraj','mind2web','games','mario','synth'}
    # Only six explicitly novel native sources are fetched; nothing from this
    # excluded list is selected, and no raw benchmark file is exported.
    if set(QUOTAS)&excluded:raise ValueError('Previously used/excluded source selected')
    counters=Counter();candidates=[]
    for s in lock['sources']:
        for r in public_records(s,rawroot):
            if r['_native']!='train':
                protect.add(r,'native_holdout:'+s['name']);counters[s['name']+':protected_native_rows']+=1
            elif r['_invalid']:counters[s['name']+':unlabelled_or_unannotated']+=1
            else:candidates.append(r)
    protect.finish()
    candidates.extend(teacher_records(upstream,renderer,counters))
    candidates.extend(rule_records(rules))
    print('Candidates before novelty filtering:',len(candidates),flush=True)
    # Connect equal contexts/pair sentences before split assignment. Native
    # source group IDs and generated twin IDs always remain together.
    groups=Groups();owners={};visible={};conflicts=set()
    for r in candidates:
        gid=r['group_id'];groups.find(gid)
        for t in input_texts(r):
            k=key(t)
            if k in owners:groups.union(gid,owners[k])
            else:owners[k]=gid
        h=visible_hash(r)
        label=sorted((canonical(str(o)),v) for o,v in zip(r['options'],r['target']))
        if h in visible and visible[h]!=label:conflicts.add(h)
        visible[h]=label
    # Full label conflict groups are quarantined, not just one contradictory row.
    badgroups={groups.find(r['group_id']) for r in candidates if visible_hash(r) in conflicts}
    forced_owners={}
    priority={'train':0,'calibration':1,'validation':2}
    for r in candidates:
        if '_forced_split' in r:
            root=groups.find(r['group_id']);sp=r['_forced_split']
            if root not in forced_owners or priority[sp]>priority[forced_owners[root]]:
                forced_owners[root]=sp
    pools=defaultdict(list);seen=set()
    for i,r in enumerate(candidates):
        source=r['source'];h=visible_hash(r)
        reason=('conflicting_labels',h) if groups.find(r['group_id']) in badgroups else ('duplicate_candidate',h) if h in seen else protect.match(r,near=False)
        if reason:
            counters[source+':'+reason[0]]+=1
            protect.db.execute('INSERT INTO exclusions VALUES (?,?,?,?)',(r['id'],source,*reason));continue
        seen.add(h)
        root=groups.find(r['group_id'])
        r['group_id']='supplement:'+digest(root)
        # Split uses original group root, before replacing it with output ID.
        if root in forced_owners:split=forced_owners[root]
        else:
            n=int(digest([SEED,'group-split',r['group_id']])[:8],16)%100
            split='validation' if n<8 else 'calibration' if n<12 else 'train'
        r['split']=split;pools[(source,split)].append(r)
        if i%10000==0:print('Novelty scan',i,'/',len(candidates),flush=True)
    protect.db.commit()
    outputs={s:[] for s in ['train','validation','calibration']}
    source_counts={}
    for source,trainquota in QUOTAS.items():
        source_counts[source]={}
        for split in outputs:
            quota=trainquota if split=='train' else max(100,trainquota//12) if split=='validation' else max(50,trainquota//30)
            selected=choose(pools[source,split],quota,tokenizer,counters,protect)
            outputs[split].extend(selected)
            source_counts[source][split]={'available':len(pools[source,split]),'quota':quota,'selected':len(selected)}
        print('Selected',source,source_counts[source],flush=True)
    statistics={}
    for split,rows in outputs.items():
        random.Random(str(SEED)+split).shuffle(rows)
        statistics[split]=write_jsonl(stage/(split+'.jsonl.gz'),rows)
    # Fixed nested quick view, equal fractions per source to rounding.
    quick=[]
    for source in QUOTAS:
        rr=[r for r in outputs['validation'] if r['source']==source]
        quick.extend(sorted(rr,key=lambda r:digest([SEED,'quick',r['id']]))[:max(1,len(rr)//4)])
    statistics['validation-quick']=write_jsonl(stage/'validation-quick.jsonl.gz',quick)
    save(stage/'sources.json',lock)
    save(stage/'statistics.json',{'splits':statistics,'by_source':source_counts,'exclusions':dict(counters)})
    save(stage/'recipe.json',{'dataset':'groundbeef-decider-supplement','version':'0.1','seed':SEED,
        'train_quotas':QUOTAS,'max_input_tokens':MAX_TOKENS,'max_rows_per_parent':3,
        'novel_sources_only':list(QUOTAS),'excluded_existing_or_benchmark_sources':sorted(excluded-{'agenttraj','mind2web','games','mario','synth'}),
        'deferred_sources':['agenttraj','mind2web','games','mario','synth'],
        'split_recipe':'native training only, alias-connected parent groups hash-split 88/8/4; generated twin groups have independent seeds by split; rule validation uses upstream held-out domains',
        'augmentation':'no replay, random option shuffling, JSON wrapping or label renaming of existing Groundbeef rows',
        'novelty_policy':'full visible input and canonical context/request aliases; full existing native pool keys; bottom-four trigram sketch (at least two shared signatures) followed by Jaccard >=0.85 approximate near screening',
        'teacher_policy':'teacher_ok AND exact teacher_pred agreement, or stored P(gold)>0.5001 where only probabilities survived; no adjacent-score/risk or generic/catch-all exception',
        'limitations':['Exact and approximate textual checks do not establish semantic/paraphrase disjointness.',
            'Teacher labels are self-consistency checked, not independently rejudged.',
            'The new source cards retain their own license terms; this is a local research dataset, not a redistributable Apache-only corpus.',
            'SST2 supplies phrase examples without original review IDs; split grouping protects supplied text, not unprovided review ancestry.',
            'No native public test rows are exported; there is no new blind external benchmark in this release.']})
    snapshot=stage/'recipe_source';snapshot.mkdir()
    protect.db.commit()
    codefiles=['decider_supplement.py','download_decider_supplement.py','decider_supplement_sources.json','common.py','groundbeef.py','test_decider_supplement.py','audit_decider_supplement_labels.py']
    for name in codefiles:shutil.copyfile(Path(__file__).with_name(name),snapshot/name)
    protect.db.execute('PRAGMA wal_checkpoint(TRUNCATE)');protect.db.close()
    frozen=[p for p in sorted(stage.iterdir()) if p.is_file()]
    save(stage/'manifest.json',{'dataset':'groundbeef-decider-supplement','version':'0.1',
        'tokenizer_sha256':file_hash(a.tokenizer),'source_lock_sha256':file_hash(lockpath),
        'protected_files':files,'files':[{ 'file':p.name,'sha256':file_hash(p),'bytes':p.stat().st_size} for p in frozen],
        'converter_sha256':{n:file_hash(snapshot/n) for n in codefiles},'prior_source_inventory_sha256':sorted(used_sources)})
    report(stage,statistics,source_counts,counters)
    audit(stage,a.tokenizer,tokenizer)
    stage.rename(a.output)
    print('Created immutable supplement:',a.output,flush=True)


def report(out,statistics,counts,counters):
    lines=['# Groundbeef: Decider supplement v0.1','',
           'An additive dataset of new source records, separate from Groundbeef v0.3. No training was launched.',
           '', '| Source | Train | Validation | Calibration |','|---|---:|---:|---:|']
    for s in QUOTAS:lines.append('| '+s+' | '+' | '.join(str(counts[s][sp]['selected']) for sp in ['train','validation','calibration'])+' |')
    lines+=['','## Novelty and ownership','',
      'All prior Groundbeef v0.1/v0.2/v0.3 train and held-out inputs, their archived held-out pools, the external evaluation panel, the existing standalone native source ledger, and complete native held-outs of the new sources are protected. The protection inventory is in manifest.json. Exact visible input fingerprints ignore option order. Canonical context and structured-request aliases additionally prevent relabelled/reformatted old records from entering. Approximate trigram Jaccard screening uses a threshold of 0.85.',
      '', 'Related source parents, paired sentences, shared contexts and generated twins are assigned as groups. Duplicate candidates and contradictory presentations are quarantined. No old rows are replayed into this supplement.',
      '', '## Labels and use','',
      'Native public labels, strict same-teacher-consistency labels, and computed exact rule labels retain separate provenance. Custom teacher records with no explicit verifier flag require stored P(gold)>0.5001, certifying argmax agreement even with four-decimal rounding. STS-B continuous scores are rounded to six described levels using the pinned Decider convention. ToxicChat exports only human_annotation=true requests; neither model replies nor moderation API scores enter model inputs. Shell outside-project flags lacking an exact teacher check are excluded.',
      '', 'Decider teacher state strings and named options/rubrics are preserved. There is no fresh teacher inference. Generated rules use the pinned Apache-2.0 upstream generator, with counterfactual/state and rule twins retained in one parent split. No wide-candidate padding or transformations of Groundbeef rows are counted as novel data.',
      '', 'Inputs fit the current 4,096-token Bazza layout without truncation, with at most 26 candidates. Full schema/provenance details are in recipe.json and sources.json. The fixed quick validation file is a subset of full validation.',
      '', '## Limits','',
      'These checks establish textual novelty under the stated aliases and approximate matcher, not semantic disjointness or absence from pretrained-model corpora. Teacher agreement is not an independent correctness check. Source licenses vary (including ToxicChat CC-BY-NC-4.0 and unstated/other terms); retain source cards and do not describe the entire collection as Apache-2.0. SST2 lacks full-review parent identifiers. No new blind benchmark is claimed.',
      '', '## Files','',
      'train.jsonl.gz, validation.jsonl.gz, calibration.jsonl.gz, validation-quick.jsonl.gz, statistics.json, sources.json, recipe.json, manifest.json, audit.json, membership.jsonl.gz, protection.sqlite, recipe_source/.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def audit(out,tokenizer_path,tokenizer):
    manifest=json.loads((out/'manifest.json').read_text())
    if file_hash(tokenizer_path)!=manifest['tokenizer_sha256']:raise ValueError('Tokenizer changed')
    for f in manifest['files']:
        if file_hash(out/f['file'])!=f['sha256']:raise ValueError('Frozen file changed: '+f['file'])
    protect=Protection(out/'protection.sqlite',readonly=True)
    groups={};inputs=set();ids=set();counts={};membership=[];full_validation={}
    for split in ['train','validation','calibration']:
        n=0
        for r in jsonl(out/(split+'.jsonl.gz')):
            validate(r)
            if r['split']!=split:raise ValueError('Split label differs')
            if r['id'] in ids or visible_hash(r) in inputs:raise ValueError('Duplicate exported row')
            ids.add(r['id']);inputs.add(visible_hash(r))
            if r['group_id'] in groups and groups[r['group_id']]!=split:raise ValueError('Cross-split parent')
            groups[r['group_id']]=split
            if protect.match(r):raise ValueError('Existing/protected overlap: '+r['id'])
            length=len(tokenizer.encode(prompt(r),add_special_tokens=False).ids)
            if length>MAX_TOKENS or length!=r['metadata']['input_tokens']:raise ValueError('Token budget mismatch')
            membership.append({'id':r['id'],'group_id':r['group_id'],'split':split,'input_sha256':visible_hash(r),
                'source':r['source'],'source_revision':r['metadata']['source_revision'],
                'source_row_sha256':r['metadata'].get('source_row_sha256'),'label_basis':r['metadata']['label_basis'],
                'generator_seed':r['metadata'].get('generator_seed')})
            if split=='validation':full_validation[r['id']]=digest(r)
            n+=1
        counts[split]=n
    for r in jsonl(out/'validation-quick.jsonl.gz'):
        if full_validation.get(r['id'])!=digest(r):raise ValueError('Quick validation is not an exact nested subset')
    # Reading each public source again validates targets and supplied menu order,
    # independently of the serialization/selection pass.
    lock=json.loads((out/'sources.json').read_text());lookup={}
    for s in lock['sources']:
        for r in public_records(s,REPO/'datasets/sources'):
            if r['id'] in ids:lookup[r['id']]=r
    for split in ['train','validation','calibration']:
        for r in jsonl(out/(split+'.jsonl.gz')):
            if r['source'] in {s['name'] for s in lock['sources']}:
                native=lookup[r['id']]
                if native['_native']!='train' or native['_invalid']:raise ValueError('Native holdout/unlabelled selected')
                for field in ['state','question','options','target']:
                    if r[field]!=native[field]:raise ValueError('Native conversion differs: '+r['id'])
    # Hash-only verification confirms old immutable releases and evaluation
    # files remained unchanged throughout assembly.
    for p in manifest['protected_files']:
        if file_hash(p['path'])!=p['sha256']:raise ValueError('Protected file changed: '+p['path'])
    protect.db.close()
    import gzip
    memberpath=out/'membership.jsonl.gz'
    with memberpath.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw,filename='',mode='wb',mtime=0) as f:
            for r in membership:f.write((dumps(r)+'\n').encode())
    save(out/'audit.json',{'verified':True,'rows':counts,'distinct_groups':len(groups),
         'duplicate_exported_inputs':0,'existing_or_protected_overlap':0,'cross_split_parents':0,
         'token_budget':MAX_TOKENS,'native_targets_and_candidate_order_verified':len(lookup),
         'prior_files_preserved':True,'quick_is_exact_nested_subset':True,
         'membership_sha256':file_hash(memberpath),
         'scope':'Exact aliases plus approximate trigram screening; no semantic-disjointness claim.'})
    print('Independent saved-file audit passed:',counts,flush=True)


if __name__=='__main__': main()
