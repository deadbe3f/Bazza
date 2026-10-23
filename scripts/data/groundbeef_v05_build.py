"""Resumable offline construction of the 32K addition and combined 96K release."""
import argparse
import ast
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import copy
import gzip
import json
import os
from pathlib import Path
import re
import sqlite3
import sys

from common import canonical, code_key, digest, dumps, file_hash, jsonl, key, parquet
from groundbeef import cap_group, decorate, prompt, state_object, stats, validate, visible_hash
from groundbeef_v04 import original_aliases
from common import identity_keys
from prepare import NearIndex
from python_execution_worker import check as check_program
from v05_code import EXECUTION_PROTOCOL, code_row, execute_menu, features, invocation, normalized_program
from v05_job import Job, run, save

ROOT = Path('/work')
SEED = 20261017
QUOTAS = {'openscience-10':7000, 'openscience-4':1000, 'sciq':1000, 'mmlu':1000,
          'codeio-python-repaired':3000, 'codeio-short':4500, 'mbpp-execution':500,
          'gsm8k-4choice':250, 'gsm8k-10choice':750, 'aqua':1500, 'mathqa':1500,
          'winogrande':2000, 'hellaswag':500, 'commonsenseqa':500, 'anli':2000,
          'ragtruth':500, 'paws':500, 'when2call-fourway':1000, 'api-bank':500,
          'toolret':500, 'wiki_qa':500, 'stsb':500, 'decider_custom':500, 'decider_routing':500}
AUX = {'paws','wiki_qa','stsb','decider_custom','decider_routing'}


def clean(r): return {k:v for k,v in r.items() if not k.startswith('_')}


def validate_example(r):
    if r.get('metadata',{}).get('dataset_source') not in {'codeio-short','codeio-python-repaired','mbpp-execution'}:
        return validate(r)
    # Whitespace and case inside Python string literals are semantic content.
    # Preserve the generic structural checks without its prose canonicalizer.
    proxy={**r,'options':[str(i)+': '+o if isinstance(o,str) and o.strip() else '' for i,o in enumerate(r['options'])]}
    validate(proxy)
    values=[]
    for option in r['options']:
        value=ast.literal_eval(option)
        if any(type(value) is type(v) and value==v for v in values):raise ValueError('duplicate Python values')
        values.append(value)


def write(path, rows):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        with gzip.GzipFile(filename='', fileobj=f, mode='wb', mtime=0) as out:
            for r in rows: out.write((dumps(clean(r))+'\n').encode())
    tmp.replace(path)
    return {'file':path.name,'rows':len(rows),'sha256':file_hash(path),'bytes':path.stat().st_size}


def code_from(r):
    state = state_object(r.get('state', ''))
    return state.get('code') if isinstance(state, dict) else None


def question_text(r):
    state = state_object(r['state'])
    if isinstance(state, str): return state
    if isinstance(state, dict) and isinstance(state.get('question'), str): return state['question']
    return r['question'] if state == {} else dumps(state)


class Guards:
    def __init__(self, old, job):
        self.ids=set(); self.groups=set(); self.alias=set(); self.inputs=set(); self.programs=set(); self.native_codes=set()
        self.exposure_ids=set(); self.exposure_groups=set(); self.exposure_programs=set()
        self.near=NearIndex(); self.excluded=Counter(); self.selected_ids=set(); self.selected_groups=Counter()
        self.dev_programs=set(); self.selected_inputs=set(); self.selected_alias_groups=defaultdict(set)
        self.source_db=sqlite3.connect('file:/work/datasets/decisions/source-pool-v0.1/membership.sqlite?mode=ro',uri=True)
        self.alias.update(k for k, in self.source_db.execute('SELECT key FROM protected'))
        self.groups.update(g for g, in self.source_db.execute('SELECT key FROM protected_group'))
        self.native_codes.update(k for k, in self.source_db.execute('SELECT key FROM protected_code'))
        arch=json.loads((ROOT/'build/groundbeef/architecture-v0.1/exclusions.json').read_text())
        self.arch_ids=set(arch['blocked_training_ids']); self.arch_keys=set(arch['keys'])
        self.groups.update(k[len('group:'):] for k in self.arch_keys if k.startswith('group:'))
        # Raw source duplicates can carry new IDs and parent keys. Protect the
        # reserved pilot by normalized program content as well as its ledger.
        for r in jsonl(ROOT/'build/groundbeef/architecture-v0.1/train.jsonl.gz'):
            code=code_from(r)
            if code:
                try:self.programs.add(normalized_program(code))
                except (SyntaxError,ValueError):pass
        self.files=[]
        for r in old:
            self.add_block(r)
        roots=[Path('/history/v0.1'),Path('/history/v0.2'),ROOT/'build/groundbeef/v0.3',ROOT/'build/groundbeef/v0.4',ROOT/'build/groundbeef/decider-supplement-v0.1']
        for root in roots:
            for split in ['train','validation','calibration','test','ood']:
                path=root/(split+'.jsonl.gz')
                if not path.exists(): continue
                self.files.append({'file':str(path),'sha256':file_hash(path),'role':split})
                if split=='train':
                    if root.name=='decider-supplement-v0.1': continue # Prepared is not necessarily trained.
                    for r in jsonl(path):
                        self.exposure_ids.add(r['id']);self.exposure_groups.add(cap_group(r)[0])
                        code=code_from(r)
                        if code:
                            try:self.exposure_programs.add(normalized_program(code))
                            except (SyntaxError,ValueError):pass
                else:
                    for r in jsonl(path): self.add_block(r, near=True)
                job.check()
            for path in sorted((root/'heldout_pools').glob('*.jsonl.gz')):
                self.files.append({'file':str(path),'sha256':file_hash(path),'role':'historical_complete_heldout'})
                for r in jsonl(path): self.add_block(r, near=True)
        # Complete original code held-outs, not only selected test rows.
        crux=ROOT/'datasets/evaluation-panel/sources/work/artifacts/benchmark-suite/raw/repos/cruxeval'
        seen=0
        for path in crux.rglob('*.jsonl'):
            for r in jsonl(path):
                if 'code' in r:
                    self.native_codes.add(code_key(r['code']));self.programs.add(normalized_program(r['code'],'f'));seen+=1
        if seen<800: raise ValueError('Full CRUX program protection missing')
        for body, in self.source_db.execute("SELECT body FROM examples WHERE source='codeio' AND split!='train'"):
            r=json.loads(body);r['metadata'].setdefault('dataset_source','codeio');self.add_block(r)
        for variant in ['full','sanitized']:
            for split in ['test','validation','prompt']:
                for r in parquet(job.root/'raw/mbpp'/variant/f'{split}-00000-of-00001.parquet'):
                    if r.get('code'):
                        try:self.programs.add(normalized_program(r['code']))
                        except (SyntaxError,ValueError):pass
        # Native MMLU/MMLU-Pro question text provides approximate as well as exact protection.
        for source in ['mmlu-pro-protected','mmlu']:
            for path in (ROOT/'datasets/sources'/source/'v0.1/raw').rglob('*.parquet'):
                if source=='mmlu' and 'auxiliary_train' in path.name: continue
                for i,r in enumerate(parquet(path)):
                    text=r.get('question')
                    if text:self.near.add(text,f'native:{source}:{path}:{i}')
        for r in jsonl(ROOT/'datasets/evaluation-panel/test.jsonl'):
            state=r['state']; texts=[state] if isinstance(state,str) else [state.get('question','')] if isinstance(state,dict) else []
            for text in texts:
                if text:self.near.add(text,r['_evaluation']['run_id'])
        print('Protected IDs, parents, aliases, programs:',len(self.ids),len(self.groups),len(self.alias),len(self.programs),flush=True)

    def add_block(self,r,near=False):
        self.ids.add(r['id']);self.groups.add(cap_group(r)[0]);self.alias.update(original_aliases(r));self.inputs.add(visible_hash(r))
        code=code_from(r)
        if code:
            try:self.programs.add(normalized_program(code))
            except (SyntaxError,ValueError):pass
        if near and r.get('domain') in {'math','science','knowledge','commonsense'}:
            self.near.add(question_text(r),r['id'])

    def reason(self,r,program=None,near=False):
        if r['id'] in self.ids or r['id'] in self.arch_ids:return 'prior_control_or_heldout_id'
        if identity_keys(r)&self.arch_keys:return 'architecture_exclusion'
        if cap_group(r)[0] in self.groups:return 'prior_control_or_heldout_parent'
        if original_aliases(r)&self.alias or visible_hash(r) in self.inputs:return 'prior_or_native_protected_alias'
        if program and program in self.programs:return 'prior_or_native_protected_program'
        if near and self.near.match(question_text(r)):return 'near_heldout_question'
        return None

    def accept(self,r,cap=1,program=None):
        if r['id'] in self.selected_ids or visible_hash(r) in self.selected_inputs:return False
        if self.selected_groups[r['group_id']]>=cap:return False
        if any(self.selected_alias_groups[a]-{r['group_id']} for a in original_aliases(r)):return False
        if program and program in self.dev_programs and r['split']=='train':return False
        self.selected_ids.add(r['id']);self.selected_inputs.add(visible_hash(r));self.selected_groups[r['group_id']]+=1
        for a in original_aliases(r):self.selected_alias_groups[a].add(r['group_id'])
        if program and r['split']=='validation':self.dev_programs.add(program)
        return True


def parse_openscience(raw, origin, index):
    text=raw['input']; output=raw['output']
    if re.search(r'[\u3400-\u9fff]',text):raise ValueError('mixed_language_input')
    matches=list(re.finditer(r'(?m)^\s*([A-J])\s*[:.)]\s+',text))
    if [m.group(1) for m in matches]!=list('ABCDEFGHIJ'[:len(matches)]) or len(matches) not in {4,10}:
        raise ValueError('malformed_or_repeated_menu')
    stem=text[:matches[0].start()].strip()
    options=[text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)].strip() for i,m in enumerate(matches)]
    if not stem or any(re.search(r'\n\s*(Which|What|A:|B:)',o) for o in options):raise ValueError('multiple_questions_in_input')
    if '</think>' not in output:raise ValueError('missing_reasoning_boundary')
    answer=output.rsplit('</think>',1)[1]
    if re.search(r'(?i)\b(?:ambiguous|multiple correct|more than one correct|forced conclusion|formatting (?:issue|error)|question (?:is|may be) flawed)\b',answer):
        raise ValueError('solution_explicitly_flags_ambiguity')
    if re.search(r'(?i)(?:answer|answers)\s+(?:is|are)\s*[:=]?\s*\(?\s*[A-J](?:\s*[,/&]\s*|\s*(?:and|or)\s*|\s*)[A-J]\b',answer):
        raise ValueError('solution_declares_multiple_answer_letters')
    boxed=re.findall(r'\\boxed\s*\{\s*([A-J])\s*\}',answer)
    declarations=re.findall(r'(?:the\s+)?(?:correct\s+)?answer\s+is\s*[:=]?\s*\(?\s*([A-J])\b',answer,re.I)
    letters={x.upper() for x in boxed+declarations}
    if not boxed or len(letters)!=1:raise ValueError('missing_or_conflicting_final_answer')
    gold=ord(next(iter(letters)))-65
    if gold>=len(options):raise ValueError('gold_outside_menu')
    n=len(options); ident='openscience:'+digest(stem)
    r={'id':ident,'group_id':'openscience:'+key(stem),'source':'openscience','kind':'choice','domain':'knowledge',
       'family':'knowledge.synthetic_mcq','state':stem,'question':'Which option is the correct answer?',
       'options':options,'target':[float(i==gold) for i in range(n)],'split':'candidate',
       'metadata':{'dataset_source':f'openscience-{n}','category':'knowledge','task_type':'knowledge.synthetic_mcq',
           'subcategory':'knowledge.synthetic_mcq','source_repo':'nvidia/OpenScience',
           'source_revision':'7bd0437e4756f761768fe7e5cebeaa75480a4fd6','source_file':origin,'native_id':str(index),
           'original_row_sha256':digest(raw),'label_origin':'upstream_synthetic_R1_consensus',
           'label_basis':'publisher_consensus_not_independent_truth','native_subject':'not_provided',
           'semantic_class':'answer_position:'+str(gold),'loss_weight':1.,'input_transform':'groundbeef-v0.5/native_menu_no_solution',
           'source_menu_order_preserved':True}}
    validate(r);return r


class Build:
    def __init__(self,job):
        from tokenizers import Tokenizer
        self.job=job;self.root=job.root;self.parts=self.root/'parts';self.parts.mkdir(exist_ok=True)
        self.db=sqlite3.connect(self.root/'progress.sqlite')
        self.db.executescript('PRAGMA journal_mode=WAL; CREATE TABLE IF NOT EXISTS executions (key TEXT PRIMARY KEY, body TEXT); CREATE TABLE IF NOT EXISTS raw_candidates (id TEXT PRIMARY KEY, rank TEXT, body TEXT); CREATE TABLE IF NOT EXISTS cursor (name TEXT PRIMARY KEY, value TEXT);')
        self.tok=Tokenizer.from_file('/base/tokenizer.json');self.old=list(jsonl(ROOT/'build/groundbeef/v0.4/train.jsonl.gz'))
        self.base_repairs=[]
        self.repair_base_inputs()
        self.g=Guards(self.old,job);self.addon=[];self.newdev=[];self.counts=Counter();self.rejected=Counter();self.sources=[]

    def repair_base_inputs(self):
        """Repair confirmed post-call mutations in copies, never the v0.4 files."""
        lookup={int(r['metadata']['native_id']):(i,r) for i,r in enumerate(self.old)
                if r['metadata']['dataset_source']=='codeio-python'}
        path=ROOT/'datasets/sources/codeio/v0.1/raw/0_368500_filtered_v2_ds25.sced.jsonl'
        with path.open() as stream:
            for native_index,line in enumerate(stream):
                if native_index not in lookup:continue
                index,original=lookup[native_index];raw=json.loads(line)
                io_index=original['metadata']['io_index'];pair=raw['ios'][io_index];supplied=pair['input']
                try:
                    call=ast.parse('f('+original['state']['input']+')',mode='eval').body
                    displayed={k.arg:ast.literal_eval(k.value) for k in call.keywords} if isinstance(supplied,dict) else \
                              [ast.literal_eval(v) for v in call.args] if isinstance(supplied,list) else ast.literal_eval(call.args[0])
                except (ValueError,SyntaxError,IndexError):continue
                if displayed==supplied:continue # Keyword order alone is not an input change.
                task={'id':'repair:'+original['id'],'code':raw['refcode'],'function':raw['funcname'],
                      'input':supplied,'expected_json':dumps(pair['output']),
                      'other_inputs':[p for j,p in enumerate(raw['ios']) if j!=io_index][:3]}
                result=execute_menu(task)
                if 'rejection' in result:raise ValueError('Cannot verify original-control input repair '+original['id']+': '+result['rejection'])
                gold=original['target'].index(max(original['target']))
                if ast.literal_eval(original['options'][gold])!=ast.literal_eval(result['execution']['literal']):
                    raise ValueError('Original-control gold disagrees with repaired-input execution')
                repaired=copy.deepcopy(original)
                repaired['state']['input']=result['execution']['invocation'][2:-1]
                repaired['metadata'].update(previous_input=original['state']['input'],
                    original_control_row_sha256=digest(original),input_transform='groundbeef-v0.5/precall_input_repair',
                    feature_compatible_with_v04=False,input_repair_protocol=EXECUTION_PROTOCOL)
                repaired['metadata']['execution']['input']=result['execution']['invocation']
                repaired['metadata']['input_tokens']=len(self.tok.encode(prompt(repaired),add_special_tokens=False).ids)
                self.old[index]=repaired;self.base_repairs.append(repaired)
                print('Verified original-control input repair:',original['id'],flush=True)

    def note(self,name,**kw):
        self.job.check();self.job.update(stage=name,selected_train=len(self.addon),selected_new_validation=len(self.newdev),**kw)
        print(name,kw,flush=True)

    def ready(self,name):
        p=self.parts/(name+'.jsonl.gz');m=p.with_name(p.name+'.receipt.json')
        if not m.exists():return None
        receipt=json.loads(m.read_text())
        if file_hash(p)!=receipt['sha256']:raise ValueError('Corrupt committed selection '+name)
        rows=list(jsonl(p))
        for r in rows:
            if not self.g.accept(r,cap=2 if r['metadata']['dataset_source']=='mbpp-execution' else 3):
                raise ValueError('Committed selection overlap on resume '+r['id'])
            if r['split']=='validation':self.g.add_block(r,near=True)
        self.register(rows);return rows

    def register(self,rows):
        for r in rows:
            (self.addon if r['split']=='train' else self.newdev).append(r)
            self.counts[r['split']+'/'+r['metadata']['dataset_source']]+=1

    def commit(self,name,rows):
        p=self.parts/(name+'.jsonl.gz');save(p.with_name(p.name+'.receipt.json'),write(p,rows));self.register(rows)
        self.note('selected '+name,rows=len(rows))

    def qualify(self,r,split,near=False,program=None):
        why=self.g.reason(r,program,near)
        if why:self.rejected[r['metadata']['dataset_source']+':'+why]+=1;return False
        if split=='validation' and (r['id'] in self.g.exposure_ids or r['group_id'] in self.g.exposure_groups or program in self.g.exposure_programs):
            self.rejected['development_prior_training_membership']+=1;return False
        r['split']=split;m=r['metadata'];m.setdefault('category',r['domain'])
        m['prior_training_exposure']=r['id'] in self.g.exposure_ids or r['group_id'] in self.g.exposure_groups
        m['exposure_basis']='conservative historical training-dataset membership, not confirmed execution cursor'
        m['prior_validation_monitoring']=False;m.setdefault('label_basis',m.get('label_origin','native_label'));m.setdefault('loss_weight',1.)
        try:validate_example(r)
        except (ValueError,SyntaxError) as e:
            self.rejected['invalid_example:'+str(e)]+=1;return False
        length=len(self.tok.encode(prompt(r),add_special_tokens=False).ids)
        if length>4096:self.rejected['over_token_budget']+=1;return False
        m['input_tokens']=length
        return True

    def science(self):
        if self.ready('openscience') is not None:return
        self.note('filtering OpenScience')
        bins=defaultdict(list)
        for folder in sorted((self.root/'raw/openscience-broad').iterdir()):
            for path in sorted(folder.glob('*.jsonl.gz')):
                receipt=json.loads(path.with_name(path.name+'.json').read_text())
                if file_hash(path)!=receipt['sha256']:raise ValueError('Corrupt acquired OpenScience window')
                for i,raw in enumerate(jsonl(path)):
                    try:
                        r=parse_openscience(raw,folder.name+'/'+path.name,i)
                        split='validation' if int(key(r['group_id'])[:8],16)%20==0 else 'train'
                        if self.qualify(r,split,near=True):bins[(split,len(r['options']))].append(r)
                    except ValueError as e:self.rejected['openscience:'+str(e)]+=1
                self.job.check()
        rows=[]; quotas={('validation',10):450,('validation',4):50,('train',10):7000,('train',4):1000}
        self.note('OpenScience qualified pool',pools={str(k):len(v) for k,v in bins.items()})
        for bucket,n in quotas.items():
            pool=sorted(bins[bucket],key=lambda r:digest([SEED,r['id']]))
            # Near-identical stems within this source do not provide new breadth.
            near=NearIndex();got=[]
            for r in pool:
                if near.match(question_text(r)):continue
                if self.g.reason(r,near=True):continue
                if self.g.accept(r):
                    got.append(r);near.add(question_text(r),r['id'])
                    if len(got)==n:break
            if len(got)!=n:raise ValueError(f'OpenScience shortfall {bucket}: {len(got)}/{n}')
            rows+=got
            if bucket[0]=='validation':
                for r in got:self.g.add_block(r,near=True)
        self.commit('openscience',rows)

    def local(self):
        aux=defaultdict(list)
        for r in jsonl(ROOT/'build/groundbeef/decider-supplement-v0.1/train.jsonl.gz'):
            if r['metadata']['dataset_source'] in AUX:aux[r['metadata']['dataset_source']].append(r)
        for source,n in QUOTAS.items():
            if source.startswith(('openscience','codeio')) or source=='mbpp-execution':continue
            name='local-'+source
            if self.ready(name) is not None:continue
            self.note('selecting '+source)
            if source in AUX:
                rawrows=aux[source];formatted=True;path=ROOT/'build/groundbeef/decider-supplement-v0.1/train.jsonl.gz'
            elif source=='when2call-fourway':
                path=ROOT/'build/groundbeef/v0.4-sources/when2call-fourway-v0.2/train.jsonl.gz';rawrows=jsonl(path);formatted=True
            else:
                path=ROOT/'datasets/decisions'/source/'v0.1/train.jsonl.gz';rawrows=jsonl(path);formatted=False
            self.sources.append({'file':str(path),'sha256':file_hash(path)})
            bins=defaultdict(list)
            for raw in rawrows:
                if raw['id'] in self.g.ids or raw['group_id'] in self.g.groups:continue
                r=copy.deepcopy(raw) if formatted else decorate(raw,source,'train')
                if source=='mmlu':
                    text=question_text(r)
                    if len(text)>1400 or re.search(r'(?i)\b(passage|paragraph|author|best title|according to the text|underlined|in the story)\b',text):
                        self.rejected['mmlu:reading_or_long_context']+=1;continue
                    r['metadata']['native_subject']='unspecified'
                r=clean(r);m=r['metadata'];m['dataset_source']=source;m['parent_file']=str(path);m['original_row_sha256']=digest(raw)
                bins[(m.get('task_type',r['family']),m.get('semantic_class',str(r['target'].index(max(r['target'])))))].append(r)
            queues=deque(deque(sorted(v,key=lambda r:digest([SEED,r['id']]))) for _,v in sorted(bins.items()))
            rows=[]
            while queues and len(rows)<n:
                q=queues.popleft();r=q.popleft()
                if q:queues.append(q)
                if not self.qualify(r,'train',near=r['domain'] in {'math','science','knowledge','commonsense'}):continue
                cap=1 if source.startswith(('gsm8k','mmlu')) else 3
                if self.g.accept(r,cap=cap):rows.append(r)
                if len(rows)%100==0:self.job.check()
            if len(rows)!=n:
                # The agreed WikiQA fallback uses extra unused PAWS/STS-B only.
                if source=='wiki_qa':
                    needed=n-len(rows)
                    for alt in ['paws','stsb']:
                        target=(needed+1)//2 if alt=='paws' else n-len(rows)
                        used=0
                        for raw in sorted(aux[alt],key=lambda r:digest([SEED,'fallback',r['id']])):
                            r=copy.deepcopy(raw)
                            if self.qualify(r,'train') and self.g.accept(r,cap=3):
                                r['metadata']['quota_reallocation']='wiki_qa_shortfall';rows.append(r);used+=1
                                if used>=target:break
                if len(rows)!=n:raise ValueError(f'{source} eligible shortfall {len(rows)}/{n}')
            self.commit(name,rows)

    def scan_raw(self):
        path=ROOT/'datasets/sources/codeio/v0.1/raw/0_368500_filtered_v2_ds25.sced.jsonl'
        cursor=self.db.execute("SELECT value FROM cursor WHERE name='raw_scan_max20'").fetchone()
        state=json.loads(cursor[0]) if cursor else {'offset':0,'line':0,'complete':False}
        if state['complete']:return
        self.note('scanning raw CodeIO for short programs',cursor=state)
        with path.open('rb') as stream:
            stream.seek(state['offset'])
            while True:
                line=stream.readline()
                if not line:break
                index=state['line'];state['line']+=1
                raw=json.loads(line);code=raw.get('refcode','')
                try:
                    linecount=sum(bool(x.strip()) and not x.lstrip().startswith('#') for x in code.splitlines())
                    if linecount>20 or len(code)>2400 or not isinstance(raw.get('ios'),list) or not raw['ios']:continue
                    if not all(isinstance(pair,dict) and 'input' in pair and 'output' in pair for pair in raw['ios']):continue
                    for pair in raw['ios']:dumps(pair)
                    check_program(code);p=normalized_program(code,raw['funcname'])
                    if p in self.g.programs or code_key(code) in self.g.native_codes:continue
                    if 'codeio:'+str(index) in self.g.arch_ids or 'codeio:'+code_key(code) in self.g.groups:continue
                    split='validation' if int(p[:8],16)%10==0 else 'train'
                    if split=='validation' and p in self.g.exposure_programs:continue
                    item={'raw_index':index,'raw':raw,'program':p,'split':split}
                    self.db.execute('INSERT OR IGNORE INTO raw_candidates VALUES (?,?,?)',(p,digest([SEED,p]),dumps(item)))
                except (SyntaxError,ValueError,TypeError,KeyError):pass
                finally:
                    if state['line']%2000==0:
                        state['offset']=stream.tell()
                        self.db.execute("INSERT OR REPLACE INTO cursor VALUES ('raw_scan_max20',?)",(dumps(state),));self.db.commit()
                        if state['line']%20000==0:self.note('raw CodeIO scan',scanned=state['line'],candidates=self.db.execute('SELECT count(*) FROM raw_candidates').fetchone()[0])
                        self.job.check()
            state['offset']=stream.tell();state['complete']=True
            self.db.execute("INSERT OR REPLACE INTO cursor VALUES ('raw_scan_max20',?)",(dumps(state),));self.db.commit()
        self.note('raw CodeIO scan complete',scanned=state['line'],candidates=self.db.execute('SELECT count(*) FROM raw_candidates').fetchone()[0])

    def cached_execute(self,task):
        k=digest([EXECUTION_PROTOCOL,task]);found=self.db.execute('SELECT body FROM executions WHERE key=?',(k,)).fetchone()
        return k,json.loads(found[0]) if found else None

    def menus(self,tasks,quota,split,cap=1):
        chosen=[]
        with ThreadPoolExecutor(max_workers=8) as pool:
            for start in range(0,len(tasks),32):
                self.job.check();batch=tasks[start:start+32];cached=[self.cached_execute(t) for t in batch]
                missing=[t for t,(_,v) in zip(batch,cached) if v is None]
                fresh=iter(pool.map(execute_menu,missing))
                results=[]
                for task,(k,result) in zip(batch,cached):
                    if result is None:
                        result=next(fresh);self.db.execute('INSERT OR REPLACE INTO executions VALUES (?,?)',(k,dumps(result)))
                    results.append(result)
                self.db.commit()
                for task,result in zip(batch,results):
                    if 'rejection' in result:self.rejected[task['source']+':'+result['rejection']]+=1;continue
                    r=code_row(task,result,split);p=normalized_program(task['code'],task['function'])
                    if self.qualify(r,split,program=p) and self.g.accept(r,cap=cap,program=p):
                        chosen.append(r)
                        if len(chosen)==quota:return chosen
                if start%256==0:self.note('executing '+tasks[0]['source'] if tasks else 'empty',attempted=start+len(batch),accepted=len(chosen),target=quota)
        return chosen

    def raw_tasks(self,split):
        tasks=[]
        for body, in self.db.execute('SELECT body FROM raw_candidates ORDER BY rank'):
            item=json.loads(body)
            if item['split']!=split:continue
            raw=item['raw'];i=item['raw_index'];pairs=raw['ios'];j=int(digest([SEED,i])[:8],16)%len(pairs);pair=pairs[j]
            tasks.append({'id':'codeio-v05:'+str(i),'source':'codeio-short','code':raw['refcode'],'function':raw['funcname'],
                'input':pair['input'],'expected_json':dumps(pair['output']),'other_inputs':[x for k,x in enumerate(pairs) if k!=j][:3],
                'metadata':{'source_repo':'hkust-nlp/CodeIO-PyEdu-Reasoning-Raw','source_revision':'413a78caa01e634318ef8aa0e84f5ff242f25807',
                    'source_file':'0_368500_filtered_v2_ds25.sced.jsonl','native_id':str(i),'original_row_sha256':digest(raw),
                    'io_index':j,'original_category':raw.get('category'),'native_split':'train_derived_group_split'}})
        return sorted(tasks,key=lambda t:(features(t['code'])['lines']>12, not features(t['code'])['strings'],digest([SEED,t['id']])))

    def mbpp_tasks(self):
        path=self.root/'raw/mbpp/full/train-00000-of-00001.parquet';tasks=[]
        for raw in parquet(path):
            code=raw['code'];p=normalized_program(code)
            if p in self.g.programs:continue
            count=0
            for i,test in enumerate(raw['test_list']):
                try:
                    assertion=ast.parse(test).body[0]
                    compare=assertion.test
                    if not isinstance(assertion,ast.Assert) or not isinstance(compare,ast.Compare) or len(compare.ops)!=1 or not isinstance(compare.ops[0],ast.Eq):continue
                    call=compare.left
                    if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name):continue
                    supplied={k.arg:ast.literal_eval(k.value) for k in call.keywords} if call.keywords and not call.args else [ast.literal_eval(v) for v in call.args] if not call.keywords else None
                    if supplied is None:continue
                    dumps(supplied)
                    expected=ast.literal_eval(compare.comparators[0]);dumps(expected)
                    tasks.append({'id':f'mbpp-execution:{raw["task_id"]}:{i}','source':'mbpp-execution','code':code,'function':call.func.id,
                        'input':supplied,'expected_json':dumps(expected),'other_inputs':[],
                        'metadata':{'source_repo':'google-research-datasets/mbpp','source_revision':'4bb6404fdc6cacfda99d4ac4205087b89d32030c',
                            'source_file':'full/train-00000-of-00001.parquet','native_id':str(raw['task_id']),'test_index':i,'original_row_sha256':digest(raw),
                            'native_split':'train'}})
                    count+=1
                    if count==2:break
                except (ValueError,SyntaxError,AttributeError,TypeError):continue
        return sorted(tasks,key=lambda t:digest([SEED,t['id']]))

    def codes(self):
        self.scan_raw()
        if self.ready('code-development') is None:
            rows=self.menus(self.raw_tasks('validation'),500,'validation')
            if len(rows)!=500:raise ValueError(f'Only {len(rows)}/500 new code validation parents')
            for r in rows:self.g.add_block(r)
            self.commit('code-development',rows)
        if self.ready('code-training') is not None:return
        path=ROOT/'build/groundbeef/v0.4-sources/codeio-python-v0.2/train.jsonl.gz';tasks=[]
        raw_index={}
        wanted={r['id'] for r in jsonl(path) if r['id'] not in self.g.ids}
        rawpath=ROOT/'datasets/sources/codeio/v0.1/raw/0_368500_filtered_v2_ds25.sced.jsonl'
        for i,raw in enumerate(jsonl(rawpath)):
            if 'codeio:'+str(i) in wanted:raw_index['codeio:'+str(i)]=raw
        for base in jsonl(path):
            if base['id'] not in raw_index:continue
            raw=raw_index[base['id']];j=base['metadata']['io_index'];pairs=raw['ios'];p=normalized_program(raw['refcode'],raw['funcname'])
            if p in self.g.programs or cap_group(base)[0] in self.g.groups:continue
            try:raw_hash=digest(raw)
            except ValueError:
                self.rejected['codeio-python-repaired:nonfinite_source_record']+=1;continue
            tasks.append({'id':base['id'],'source':'codeio-python-repaired','code':raw['refcode'],'function':raw['funcname'],
                'input':pairs[j]['input'],'expected_json':dumps(pairs[j]['output']),
                'other_inputs':[x for i,x in enumerate(pairs) if i!=j][:3],
                'metadata':{**base['metadata'],'original_row_sha256':raw_hash,'prior_view_row_sha256':digest(base)}})
        tasks.sort(key=lambda t:(features(t['code'])['lines']>12,digest([SEED,t['id']])))
        repaired=self.menus(tasks,3000,'train');mbpp=self.menus(self.mbpp_tasks(),500,'train',cap=2)
        needed=8000-len(repaired)-len(mbpp)
        short=self.menus(self.raw_tasks('train'),needed,'train')
        rows=repaired+mbpp+short
        if len(rows)!=8000:raise ValueError(f'Code qualification shortfall {len(rows)}/8000')
        self.note('achieved code length distribution',at_most_12_lines=sum(r['metadata']['program_features']['lines']<=12 for r in rows),
                  at_most_20_lines=sum(r['metadata']['program_features']['lines']<=20 for r in rows),
                  original_12_line_target=5000,rows=len(rows))
        self.commit('code-training',rows)

    def publish(self):
        self.note('auditing and writing release')
        if len(self.old)!=64000 or len(self.addon)!=32000 or len(self.newdev)!=1000:raise ValueError('Wrong release sizes')
        anchors=list(jsonl(ROOT/'build/groundbeef/v0.4/validation.jsonl.gz'));selected=[]
        for domain,n in [('math',200),('science',200),('commonsense',200)]:
            rr=[r for r in anchors if r['metadata']['category']==domain]
            selected+=sorted(rr,key=lambda r:digest([SEED,'anchor',r['id']]))[:n]
        for source,n in [('anli',150),('ragtruth',150),('when2call-fourway',100)]:
            rr=[r for r in anchors if r['metadata']['dataset_source']==source]
            selected+=sorted(rr,key=lambda r:digest([SEED,'anchor',r['id']]))[:n]
        if len(selected)!=1000:raise ValueError('Development anchors unavailable')
        validation=self.newdev+selected
        train=self.old+self.addon
        train.sort(key=lambda r:digest([SEED,'mixed-order',r['id']]))
        ids=set();inputs=set();train_alias=set()
        for start in range(0,len(train),128):
            batch=train[start:start+128];lengths=self.tok.encode_batch([prompt(r) for r in batch],add_special_tokens=False)
            for r,t in zip(batch,lengths):
                validate_example(r)
                if len(t.ids)>4096 or len(t.ids)!=r['metadata']['input_tokens']:raise ValueError('Token metadata mismatch '+r['id'])
                if r['id'] in ids or visible_hash(r) in inputs:raise ValueError('Duplicate combined training row')
                ids.add(r['id']);inputs.add(visible_hash(r));train_alias.update(original_aliases(r))
            self.job.check()
        for r in validation:
            if r['id'] in ids or visible_hash(r) in inputs or original_aliases(r)&train_alias:raise ValueError('Training/development overlap '+r['id'])
        staging=self.root/'.release-staging';staging.mkdir(exist_ok=True)
        files=[]
        for name,rows in [('train',train),('train-addon',self.addon),('validation',validation),('validation-new',self.newdev)]:
            files.append(write(staging/(name+'.jsonl.gz'),rows))
        files.append(write(staging/'train-repaired-base.jsonl.gz',self.base_repairs))
        old_indices={r['id']:i for i,r in enumerate(self.old)};addon_indices={r['id']:i for i,r in enumerate(self.addon)}
        membership=[]
        for i,r in enumerate(train):
            m=r['metadata'];old=r['id'] in old_indices;repaired=old and m.get('feature_compatible_with_v04') is False
            membership.append({'id':r['id'],'group_id':r['group_id'],'split':'train','mixed_index':i,
                'component':'v0.4-repaired' if repaired else 'v0.4' if old else 'v0.5-addon','component_index':old_indices[r['id']] if old else addon_indices[r['id']],
                'source':m['dataset_source'],'source_repo':m.get('source_repo',m.get('repo')),'source_revision':m.get('source_revision',m.get('revision')),
                'native_id':m.get('native_id',m.get('upstream_index')),'source_file':m.get('source_file',m.get('parent_file')),
                'original_row_sha256':m.get('original_row_sha256'),'row_sha256':digest(r),'input_sha256':visible_hash(r),
                'alias_keys':sorted(original_aliases(r)),'label_basis':m.get('label_basis'),'input_tokens':m['input_tokens'],
                'prior_training_exposure':m.get('prior_training_exposure',False),'input_transform':m.get('input_transform'),
                'selection_reason':'retained_control' if old else 'targeted_gap_uplift'})
        files.append(write(staging/'membership.jsonl.gz',membership))
        # Legacy reporting requires a historical flag absent from new sources;
        # default it on copies, preserving all original/cache-visible row data.
        breakdown={name:stats([{**r,'metadata':{'retained_from_v02':False,**r['metadata']}} for r in rows])
                   for name,rows in [('train',train),('addon',self.addon),('validation',validation)]}
        save(staging/'statistics.json',breakdown)
        actual=dict(Counter(r['metadata']['dataset_source'] for r in self.addon))
        save(staging/'recipe.json',{'seed':SEED,'requested_addon':QUOTAS,'actual_addon':actual,'training_rows':96000,
             'original_rows_preserved':64000-len(self.base_repairs),'original_rows_input_repaired':len(self.base_repairs),
             'addon_rows':32000,'order':'seeded global hash mix; native choice order retained',
             'head_initialization':'fresh; no trained head checkpoint','feature_layers':['final'],'no_training_launched':True,
             'code_pool_adjustment':'Only 4525 unused <=12-line programs before execution and validation; prioritize these and extend raw qualification to <=20 lines. Report achieved coverage; do not invent program parents.'})
        save(staging/'sources.json',{'acquisition':json.loads((self.root/'source-lock.json').read_text()),
             'broad_acquisition':json.loads((self.root/'broad-source-lock.json').read_text()),'local':self.sources,'history':self.g.files,
             'all_pinned_inputs':json.loads((self.job.path/'config.json').read_text())['inputs']})
        panel=ROOT/'datasets/evaluation-panel/manifest.json';pm=json.loads(panel.read_text())
        for e in pm['entries']:
            if file_hash(panel.parent/e['file'])!=e['sha256']:raise ValueError('External test changed')
        manifest={'dataset':'groundbeef','version':'0.5','seed':SEED,'training_rows':96000,'validation_rows':2000,'files':files,
             'tokenizer_sha256':file_hash('/base/tokenizer.json'),'base_v04_manifest_sha256':file_hash(ROOT/'build/groundbeef/v0.4/manifest.json'),
             'source_lock_sha256':file_hash(staging/'sources.json'),'recipe_sha256':file_hash(staging/'recipe.json'),
             'external_panel_manifest_sha256':file_hash(panel),'code_sha256':self.job_config_code(),
             'exclusions':dict(self.rejected),'limits':['OpenScience subjects are not provided; no native subject balance claimed.',
               'Synthetic consensus and teacher consistency are not independently verified truth.',
               'Approximate overlap screening does not prove semantic independence.',
               'Original v0.4 files remain unchanged; confirmed post-call input mutations are repaired in the v0.5 copy and require fresh features.',
               'Validation anchors have been previously observed; no new blind test claimed.']}
        save(staging/'manifest.json',manifest)
        save(staging/'audit.json',{'verified':True,'training_rows':96000,'addon_rows':32000,'new_validation_rows':1000,
             'combined_validation_rows':2000,'duplicate_training_ids':0,'duplicate_training_inputs':0,'development_exact_input_or_alias_overlap':0,
             'all_training_inputs_retokenized_without_truncation':96000,'old_rows_preserved_exactly':64000-len(self.base_repairs),
             'old_rows_with_verified_input_repairs':[r['id'] for r in self.base_repairs],
             'complete_external_test_files_unchanged':True,'new_code_execution_menu_provenance_checked':True})
        text=f'# Groundbeef v0.5\n\n96,000 mixed training rows: original 64K slots plus 32K gap uplift. {len(self.base_repairs)} original inputs corrected in this copy after execution verification; original files remain unchanged.\n\n'
        text+='## Addition source counts\n\n| Source | Rows | Share of addition |\n|---|---:|---:|\n'
        for s,n in sorted(actual.items()):text+=f'| {s} | {n:,} | {n/320:.2f}% |\n'
        text+='\nValidation: 500 new knowledge + 500 new code + 1,000 previously monitored anchors. Old 4,500 validation remains a separate retention panel.\n\n'
        text+='Fresh decision-head training is planned; no feature extraction or training was launched. Reuse original training/test caches only by exact component/input identity. Rows in train-repaired-base.jsonl.gz require fresh features; membership marks them v0.4-repaired.\n\n'
        text+='Requested versus achieved quotas are in recipe.json. Membership records preserve original control indices for cache reuse and label/source provenance. OpenScience lacks subject labels; its targets are synthetic consensus. Code negatives are re-executed and type-consistent. See manifest.json for limits.\n'
        (staging/'README.md').write_text(text)
        release=self.root/'release'
        if release.exists():raise ValueError('Immutable release already exists')
        staging.replace(release)
        self.job.complete(release=str(release),training_rows=96000,addon_rows=32000,validation_rows=2000,actual_addon=actual)

    def job_config_code(self):return json.loads((self.job.path/'config.json').read_text())['code']


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--resume',action='store_true')
    a=ap.parse_args()
    scripts=[Path(__file__),*[Path(__file__).with_name(n) for n in ['v05_code.py','v05_code_worker.py','v05_job.py','python_execution_worker.py',
              'common.py','groundbeef.py','groundbeef_v04.py','prepare.py']],ROOT/'architecture/data.py']
    inputs=set()
    for source in QUOTAS:
        if source.startswith(('openscience','codeio')) or source in AUX or source in {'mbpp-execution','when2call-fourway'}:continue
        inputs.add(ROOT/'datasets/decisions'/source/'v0.1/train.jsonl.gz')
    for base in [Path('/history/v0.1'),Path('/history/v0.2'),ROOT/'build/groundbeef/v0.3',ROOT/'build/groundbeef/v0.4',ROOT/'build/groundbeef/decider-supplement-v0.1']:
        for split in ['train','validation','calibration','test','ood']:inputs.update(base.glob(split+'.jsonl.gz'))
        inputs.update((base/'heldout_pools').glob('*.jsonl.gz'))
    inputs.update([ROOT/'build/groundbeef/architecture-v0.1/exclusions.json',ROOT/'build/groundbeef/architecture-v0.1/train.jsonl.gz',ROOT/'datasets/decisions/source-pool-v0.1/membership.sqlite',
        ROOT/'build/groundbeef/v0.4-sources/codeio-python-v0.2/train.jsonl.gz',ROOT/'build/groundbeef/v0.4-sources/when2call-fourway-v0.2/train.jsonl.gz',
        ROOT/'datasets/sources/codeio/v0.1/raw/0_368500_filtered_v2_ds25.sced.jsonl',ROOT/'datasets/evaluation-panel/manifest.json'])
    config={'quotas':QUOTAS,'seed':SEED,'source_lock':file_hash(a.output/'source-lock.json'),
            'broad_source_lock':file_hash(a.output/'broad-source-lock.json'),'base_manifest':file_hash(ROOT/'build/groundbeef/v0.4/manifest.json'),
            'tokenizer':file_hash('/base/tokenizer.json'),'code':{p.name:file_hash(p) for p in scripts},
            'inputs':{str(p):file_hash(p) for p in sorted(inputs)}}
    import importlib.metadata
    config['runtime']={'python':sys.version,'image_id':os.environ.get('GROUND_IMAGE_ID'),
                       'packages':{n:importlib.metadata.version(n) for n in ['tokenizers','pyarrow','numpy']}}
    job=Job(a.output,'build',config,a.resume)
    def work():
        build=Build(job);build.science();build.local();build.codes();build.publish()
    return run(job,work)


if __name__=='__main__':raise SystemExit(main())
