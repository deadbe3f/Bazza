#!/usr/bin/env python3
"""Select immutable Groundbeef views from audited sources, without model loading.

Standalone sources stay unchanged. Only model-visible fields are reformatted;
source labels, candidate order and complete contexts are preserved. Historical
exposure, protected benchmarks and membership are recorded separately.
"""
import argparse
import ast
from collections import Counter, defaultdict, deque
import copy
import gzip
import json
import math
from pathlib import Path
import re
import sqlite3

from common import REPO, canonical, digest, dumps, file_hash, jsonl, key

SEED = 20261011


def state_object(value):
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, (dict, list)):
                return parsed
        except ValueError:
            pass
    return value


def visible_hash(r):
    return digest([canonical(dumps(state_object(r['state']))), canonical(r['question']),
                   sorted(canonical(str(o)) for o in r['options'])])


def aliases(r):
    """Conservative exact aliases independent of prompt placement and option order."""
    value = state_object(r['state'])
    texts = [dumps(value)]
    if isinstance(value, str):
        texts.append(value)
    elif isinstance(value, dict):
        for name in ['request', 'current_user_request', 'question', 'prompt', 'dialogue_prefix']:
            if isinstance(value.get(name), str):
                texts.append(value[name])
        if 'premise' in value and 'hypothesis' in value:
            texts.append(value['premise']+'\n'+value['hypothesis'])
        if 'conversation_prompt' in value:
            texts.append(value['conversation_prompt'])
        messages = value.get('messages', [])
        query = '\n'.join(m['content'] for m in messages if m.get('role') == 'user')
        if query:
            texts.append(query)
    return {key(t) for t in texts if t}


def prompt(r):
    # Byte-equivalent to gemma_decisions.pilot.encode / build_dataset_v01.prompt.
    state = r['state'] if isinstance(r['state'], str) else json.dumps(r['state'], ensure_ascii=False, sort_keys=True)
    text = '<bos><|turn>user\nChoose the best answer. Reply with its letter only.\n'
    text += 'Context:\n'+state+'\nQuestion:\n'+r['question']+'\nOptions:\n'
    text += ''.join(chr(65+i)+'. '+str(o)+'\n' for i, o in enumerate(r['options']))
    return text+'<turn|>\n<|turn>model\n<|channel>thought\n<channel|>'


def validate(r):
    if not 2 <= len(r['options']) <= 26 or len(r['options']) != len(r['target']):
        raise ValueError('invalid candidate count')
    if len({canonical(str(o)) for o in r['options']}) != len(r['options']):
        raise ValueError('duplicate candidates')
    if any(not str(o).strip() for o in r['options']):
        raise ValueError('empty candidate')
    if any(not math.isfinite(x) or x < 0 for x in r['target']) or abs(sum(r['target'])-1) > 1e-4:
        raise ValueError('invalid distribution')
    if not r['question'] or not r['state'] and r['state'] != {}:
        raise ValueError('empty input')


def family(source):
    if source in {'when2call', 'toolace', 'api-bank', 'toolret', 'hermes', 'banking77'}:
        return 'tools'
    if source in {'anli', 'ragtruth', 'contractnli', 'boolq'}:
        return 'grounding'
    if source in {'gsm8k-4choice', 'gsm8k-10choice', 'aqua', 'mathqa'}:
        return 'math'
    if source in {'arc-challenge', 'arc-easy', 'openbookqa', 'sciq'}:
        return 'science'
    if source == 'mmlu':
        return 'knowledge'
    if source in {'winogrande', 'hellaswag', 'commonsenseqa'}:
        return 'commonsense'
    if source in {'codeio', 'swe_qa'}:
        return 'code'
    if source == 'helpsteer2':
        return 'ordinal'
    return 'typed_decisions'


def eval_format(r, source):
    """Apply pinned benchmark layouts to other split questions, never test content."""
    state = state_object(r['state'])
    transform = 'original_source_layout'
    if source.startswith('gsm8k-'):
        r['state'] = {'question': state, 'task': f'GSM8K deterministic {len(r["options"])}-choice numeric selection'}
        r['question'] = 'Choose the numeric answer to the problem in state. Do not provide reasoning. This is a named multiple-choice adaptation of GSM8K.'
        transform = 'decision_index_gsm8k_answer_groups_v1_layout'
    elif source == 'anli':
        r['state'] = {}
        r['question'] = 'Classify the relationship between the premise and hypothesis.\nPremise: '+state['premise']+'\nHypothesis: '+state['hypothesis']
        r['options'] = [str(o).lower() for o in r['options']]
        transform = 'decision_index_nli_layout'
    elif source in {'arc-challenge', 'arc-easy', 'openbookqa', 'sciq', 'commonsenseqa', 'aqua', 'mathqa'}:
        r['state'] = {}
        r['question'] = state
        transform = 'decision_index_direct_mcq_layout'
    elif source == 'winogrande':
        r['state'] = {}
        r['question'] = 'Which option correctly fills the blank?\n'+state
        transform = 'decision_index_winogrande_layout'
    elif source == 'mmlu':
        r['state'] = state
        r['question'] = 'Which option is the correct answer?'
        transform = 'decision_index_mmlu_layout_native_four_choices'
    elif source == 'ragtruth':
        r['state'] = {'prompt': state['prompt'], 'response': state['response']}
        r['question'] = 'The response contains content that is not supported by the context in the prompt.'
        # Native span annotations supply no candidate menu. Use the benchmark's
        # fixed true/false schema; explicitly remap the standalone view's labels.
        r['options'] = ['The response contains content that is not supported by the context in the prompt.',
                        'All content of the response is supported by the context in the prompt.']
        r['target'] = list(reversed(r['target']))
        if 'target_native' in r:
            r['target_native'] = list(reversed(r['target_native']))
        r['metadata']['source_candidate_mapping'] = [1,0]
        r['kind'] = 'noul'
        transform = 'decision_index_ragtruth_true_false_layout'
    elif source == 'when2call':
        r['state'] = state
        r['question'] = "Which response should the assistant give to the user's question, given the available tools?"
        transform = 'when2call_native_preference_two_responses'
    r['metadata']['input_transform'] = transform
    return r


def semantic_class(r, source):
    meta = r['metadata']
    gold = max(range(len(r['target'])), key=r['target'].__getitem__)
    if source == 'when2call':
        return 'tool_call' if str(r['options'][gold]).startswith('<TOOLCALL>') else 'free_text'
    if source in {'toolace', 'hermes'}:
        return meta.get('action_type', 'tool_call' if meta.get('reference_call') else 'no_call')
    if source == 'ragtruth':
        return 'supported' if gold == 0 else 'unsupported'
    if source == 'anli':
        return ['entailment', 'neutral', 'contradiction'][gold]
    if source == 'groundbeef_rules' and meta['task_type'] == 'capability_routing':
        outcome = str(r['options'][gold])
        return outcome if outcome in {'ask_for_destination','no_compatible_tool'} else 'compatible_tool'
    if source in {'toolret', 'boolq', 'helpsteer2', 'groundbeef_rules'}:
        return str(r['options'][gold])
    if source == 'api-bank':
        return 'selected_tool'  # Native examples do not supervise the added absent-tool candidate.
    if source == 'distill_v3':
        return 'teacher_argmax:'+str(r['options'][gold])
    if source == 'open_jev':
        return '|'.join(str(o) for o, t in zip(r['options'], r['target']) if t == max(r['target']))
    return 'answer_position:'+str(gold)


def stratum(r):
    m = r['metadata']
    source = m['dataset_source']
    if source == 'ragtruth':
        return (m['subcategory'], m['semantic_class'])
    if source == 'anli':
        return (m['native_round'], m['semantic_class'])
    if source == 'open_jev':
        return (r['source'], m.get('upstream_metadata', {}).get('domain', ''), r['kind'])
    if source in {'helpsteer2', 'groundbeef_rules', 'distill_v3', 'swe_qa'}:
        return (m['task_type'], m['semantic_class'])
    if source in {'toolace', 'api-bank', 'hermes'}:
        position = max(range(len(r['target'])),key=r['target'].__getitem__)
        return (m['semantic_class'], min(len(r['options']), 10), position)
    if source in {'when2call', 'toolret'}:
        return (m['semantic_class'], min(len(r['options']), 10))
    return (m['subcategory'], m['semantic_class'])


def cap_group(r):
    source = r['metadata']['dataset_source']
    if source == 'swe_qa':
        return source+':question:'+str(r['metadata']['upstream_index']), 1
    return r['group_id'], 1 if source in {'codeio', 'distill_v3'} or source.startswith('gsm8k-') else 4


def parse_call(text, tools):
    """Replace supplied function names with identifiers, then parse literals only.

    ToolACE names can contain spaces/slashes. No reference code is executed.
    Multiple calls and non-literal arguments remain outside this release.
    """
    if not text.strip().startswith('['):
        return None
    translated = text.strip()
    names = [t['name'] for t in tools]
    for i in sorted(range(len(names)), key=lambda i: -len(names[i])):
        translated = re.sub(r'(?<=[\[,])\s*'+re.escape(names[i])+r'\s*(?=\()', 'tool_'+str(i), translated)
    tree = ast.parse(translated, mode='eval').body
    if not isinstance(tree, ast.List) or len(tree.elts) != 1 or not isinstance(tree.elts[0], ast.Call):
        raise ValueError('not_single_call')
    call = tree.elts[0]
    if not isinstance(call.func, ast.Name) or not re.fullmatch(r'tool_\d+', call.func.id) or call.args:
        raise ValueError('invalid_call_name_or_positional_arguments')
    index = int(call.func.id[5:])
    arguments = {}
    for item in call.keywords:
        if item.arg is None or item.arg in arguments:
            raise ValueError('invalid_argument_names')
        arguments[item.arg] = ast.literal_eval(item.value)
    schema = tools[index].get('parameters', {})
    required = schema.get('required', []) or tools[index].get('required', []) or []
    if not isinstance(required, list) or set(required)-set(arguments):
        raise ValueError('missing_required_arguments')
    if set(arguments)-set(schema.get('properties', {})):
        raise ValueError('unknown_arguments')
    return index, arguments


def toolace(raw_root, source, protected, output):
    path = raw_root/'toolace/v0.1/raw/data.json'
    if file_hash(path) != next(f['sha256'] for f in source['files'] if f['path'] == 'data.json'):
        raise ValueError('ToolACE raw checksum differs')
    records, exclusions = [], Counter()
    for i, raw in enumerate(json.loads(path.read_text())):
        try:
            start = re.search(r'\[\s*\{\s*"name"', raw['system']).start()
            tools, _ = json.JSONDecoder().raw_decode(raw['system'][start:])
            names = [t['name'] for t in tools]
            if not 1 <= len(names) <= 25 or len(set(names)) != len(names):
                raise ValueError('invalid_registry')
        except (AttributeError, ValueError, KeyError, TypeError):
            exclusions['invalid_registry'] += 1
            continue
        group = 'toolace:registry:'+digest(sorted(tools, key=lambda t: t['name']))
        cv = raw['conversations']
        for turn, message in enumerate(cv[:-1]):
            if message['from'] != 'user' or cv[turn+1]['from'] != 'assistant':
                continue
            if key(message['value']) in protected:
                exclusions['protected_query'] += 1
                continue
            try:
                call = parse_call(cv[turn+1]['value'], tools)
                answer = call[0] if call else len(names)
                if not call and not cv[turn+1]['value'].strip():
                    raise ValueError('empty_response')
            except (ValueError, SyntaxError, TypeError, IndexError):
                exclusions['unparseable_or_invalid_native_call'] += 1
                continue
            state = {'available_tools': tools,
                     'conversation_so_far': [{'role': m['from'], 'content': m['value']} for m in cv[:turn]],
                     'current_user_request': message['value']}
            records.append({'id':f'toolace:{i}:{turn}', 'group_id':group, 'split':'train', 'kind':'choice',
                'state':state, 'question':'Choose the next action. Select a supplied tool when a call is appropriate; otherwise respond without calling a tool (including asking for missing information).',
                'options':[t['name']+': '+t.get('description', '') for t in tools]+['respond_without_calling_a_tool'],
                'target':[float(j == answer) for j in range(len(names)+1)], 'domain':'tools', 'family':'tools.selection', 'source':'toolace',
                'metadata':{'native_id':f'{i}:{turn}', 'native_conversation_id':i, 'native_turn':turn,
                    'source_file':'data.json', 'source_repo':source['repo'], 'source_revision':source['revision'],
                    'category':'tools', 'subcategory':'tools.selection', 'label_origin':'native_synthetic_assistant_action',
                    'reference_call':{'name':names[call[0]], 'arguments':call[1]} if call else None,
                    'reference_response':cv[turn+1]['value'] if call is None else None,
                    'action_type':'tool_call' if call else 'no_call', 'arguments_not_supervised':True}})
    # Complete registry owns its split, before any mixture selection.
    groups = sorted({r['group_id'] for r in records}, key=lambda g: digest([SEED, 'toolace-dev', g]))
    holdout = set(groups[:700])
    for r in records:
        if r['group_id'] in holdout:
            r['split'] = 'validation'
    root = output/'toolace/v0.1'
    root.mkdir(parents=True)
    files = [write_rows(root/(s+'.jsonl.gz'), [r for r in records if r['split'] == s]) for s in ['train','validation']]
    manifest = {'dataset':'toolace-action-selection', 'version':'0.1', 'source':source, 'files':files,
        'exclusions':dict(exclusions), 'split_recipe':{'seed':SEED, 'validation_registries':700},
        'converter_sha256':file_hash(Path(__file__)),
        'scope':'Single literal native call name, or native no-call response; no inferred clarify/refusal labels, argument generation, or tool execution. Native output order preserved.'}
    save(root/'manifest.json', manifest)
    return records, manifest


def write_rows(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as stream:
        for r in records:
            stream.write((dumps({k:v for k,v in r.items() if not k.startswith('_')})+'\n').encode())
    return {'file':path.name, 'rows':len(records), 'sha256':file_hash(path), 'bytes':path.stat().st_size}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def decorate(raw, source, split, retained=False):
    r = copy.deepcopy(raw)
    m = r.setdefault('metadata', {})
    r['_original_sha256'] = digest(raw)
    r['_aliases'] = aliases(raw)
    r['_input_hash'] = visible_hash(raw)
    r['split'] = split
    m['dataset_source'] = source
    m.setdefault('native_domain', raw.get('domain'))
    m.setdefault('native_family', raw.get('family'))
    m.setdefault('subcategory', m.get('task_type', r['family']))
    m.setdefault('task_type', m['subcategory'])
    m['category'] = family(source)
    m['semantic_class'] = semantic_class(r, source)
    m['retained_from_v02'] = retained
    m['prior_training_exposure'] = retained and split == 'train'
    m['prior_validation_monitoring'] = retained and split == 'validation'
    m['label_basis'] = m.get('label_basis', m.get('label_origin', 'native_label'))
    m['loss_weight'] = 1.0
    if 'selection_weight_within_stratum' in m:
        m['parent_selection_weight_within_stratum'] = m.pop('selection_weight_within_stratum')
    r.setdefault('target_native', r['target'].copy() if source != 'distill_v3' else None)
    if r['target_native'] is None:
        del r['target_native']
    r['domain'] = family(source)
    r['family'] = m['subcategory']
    if not retained:
        eval_format(r, source)
    return r


def stats(rows):
    keys = ['category','dataset_source','task_type','semantic_class','label_basis']
    counts = {k:dict(Counter(r['metadata'][k] for r in rows)) for k in keys}
    by_source = {}
    for source in sorted(counts['dataset_source']):
        rr = [r for r in rows if r['metadata']['dataset_source'] == source]
        by_source[source] = {'rows':len(rr), 'groups':len({r['group_id'] for r in rr}),
            'cap_groups':len({cap_group(r)[0] for r in rr}),
            'tasks':dict(Counter(r['metadata']['task_type'] for r in rr)),
            'classes':dict(Counter(r['metadata']['semantic_class'] for r in rr)),
            'candidate_counts':dict(Counter(str(len(r['options'])) for r in rr)),
            'answer_positions':dict(Counter(str(max(range(len(r['target'])),key=r['target'].__getitem__)) for r in rr))}
    lengths = sorted(r['metadata']['input_tokens'] for r in rows)
    return {'rows':len(rows), 'distinct_groups':len({r['group_id'] for r in rows}),
        'retained_rows':sum(r['metadata']['retained_from_v02'] for r in rows),
        'counts':counts, 'by_source':by_source,
        'tokens':{'p50':lengths[len(lengths)//2], 'p95':lengths[int(.95*(len(lengths)-1))],
                  'maximum':lengths[-1], 'total':sum(lengths)}}


def select(rows, count, namespace, tokenizer, max_length, rejected, used, allow_gsm_pair=False):
    buckets = defaultdict(list)
    for r in rows:
        buckets[stratum(r)].append(r)
    for k in buckets:
        buckets[k] = deque(sorted(buckets[k], key=lambda r: digest([SEED, namespace, r['id']])))
    order = sorted(buckets)
    chosen = []
    capcounts = used.setdefault('capcounts', Counter())
    alias_groups = used.setdefault('alias_groups', defaultdict(set))
    # Tokenize candidates in batches, without tokenizing the giant source pool.
    while len(chosen) < count:
        candidates = []
        while len(candidates) < min(256, count-len(chosen)):
            found = False
            for k in order:
                if buckets[k] and len(candidates) < min(256, count-len(chosen)):
                    candidates.append(buckets[k].popleft()); found = True
            if not found:
                break
        if not candidates:
            raise ValueError(f'Insufficient qualified examples for {namespace}: {len(chosen)} / {count}; '+dumps(dict(rejected)))
        encodings = tokenizer.encode_batch([prompt(r) for r in candidates], add_special_tokens=False)
        for r, encoded in zip(candidates, encodings):
            if len(chosen) == count:
                break
            source = r['metadata']['dataset_source']
            group, cap = cap_group(r)
            split = r['split']
            identity = r['id']
            pair = allow_gsm_pair and source.startswith('gsm8k-') and r['group_id'] in used['gsm_pairs']
            if identity in used['ids'] or visible_hash(r) in used['inputs']:
                rejected[namespace+':duplicate'] += 1; continue
            if not pair and capcounts[group] >= cap:
                rejected[namespace+':group_cap'] += 1; continue
            if not pair and any(alias_groups[k]-{r['group_id']} for k in r['_aliases']):
                rejected[namespace+':input_alias'] += 1; continue
            if len(encoded.ids) > max_length:
                rejected[namespace+':over_token_budget'] += 1; continue
            validate(r)
            r['metadata']['input_tokens'] = len(encoded.ids)
            chosen.append(r)
            capcounts[group] += 1
            used['ids'].add(identity); used['inputs'].add(visible_hash(r)); used['aliases'].update(r['_aliases'])
            for k in r['_aliases']:
                alias_groups[k].add(r['group_id'])
            if source.startswith('gsm8k-'):
                used['gsm_pairs'].add(r['group_id'])
        print(f'{namespace}: {len(chosen)}/{count}', flush=True)
    return chosen


def quick_quotas(recipe):
    """Preserve full validation source proportions, rather than task multiplicity."""
    counts = recipe['validation']
    scale = recipe['quick_validation_rows']/sum(counts.values())
    quotas = {source:int(count*scale) for source,count in counts.items()}
    ranked = sorted(counts, key=lambda source: (-(counts[source]*scale-quotas[source]),source))
    for source in ranked[:recipe['quick_validation_rows']-sum(quotas.values())]:
        quotas[source] += 1
    return quotas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history-root', type=Path, required=True, help='Parent containing Groundbeef v0.1 and v0.2')
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--source-root', type=Path, default=REPO/'datasets/decisions')
    p.add_argument('--raw-root', type=Path, default=REPO/'datasets/sources')
    p.add_argument('--recipe', type=Path, default=Path(__file__).with_name('groundbeef_v03_recipe.json'))
    p.add_argument('--output', type=Path, default=REPO/'build/groundbeef/v0.3')
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output directory; releases are immutable')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    recipe = json.loads(args.recipe.read_text())
    assert sum(recipe['train'].values()) == 80000
    assert sum(recipe['validation'].values()) == 4000
    assert sum(recipe['calibration'].values()) == 500
    report = json.loads((args.source_root/'source-pool-v0.1/report.json').read_text())
    audit = json.loads((args.source_root/'source-pool-v0.1/audit.json').read_text())
    assert audit['verified']
    db = sqlite3.connect('file:'+str(args.source_root/'source-pool-v0.1/membership.sqlite')+'?mode=ro', uri=True)
    protected = {k for k, in db.execute('SELECT key FROM protected')}
    native_keys = {identity:k for identity,k in db.execute('SELECT id,key FROM examples WHERE eligible=1')}
    historical, historical_files = {}, []
    exposure = {'ids':set(), 'groups':set(), 'inputs':set(), 'aliases':set()}
    holdout = {'ids':set(), 'groups':set(), 'inputs':set(), 'aliases':set()}
    for version in ['v0.1','v0.2']:
        for split in ['train','validation','calibration','test','ood']:
            path = args.history_root/version/(split+'.jsonl.gz')
            if not path.is_file():
                continue
            rr = list(jsonl(path))
            historical[version,split] = rr
            historical_files.append({'version':version, 'split':split, 'rows':len(rr), 'sha256':file_hash(path)})
            target = exposure if split == 'train' else holdout
            for r in rr:
                target['ids'].add(r['id']); target['groups'].add(r['group_id'])
                target['inputs'].add(visible_hash(r)); target['aliases'].update(aliases(r))
    for path in sorted((args.history_root/'v0.1/heldout_pools').glob('*.jsonl.gz')):
        count = 0
        for r in jsonl(path):
            holdout['ids'].add(r['id']); holdout['groups'].add(r['group_id'])
            holdout['inputs'].add(visible_hash(r)); holdout['aliases'].update(aliases(r))
            count += 1
        historical_files.append({'version':'v0.1/heldout_pools', 'split':path.stem.split('.')[0],
                                 'rows':count, 'sha256':file_hash(path)})
    parent_sources = json.loads((args.history_root/'v0.2/sources.json').read_text())
    lock = json.loads(Path(__file__).with_name('sources.json').read_text())
    args.output.mkdir(parents=True)
    added, added_manifest = toolace(args.raw_root, next(s for s in lock['sources'] if s['name']=='toolace'),
                                    protected | holdout['aliases'], args.output/'standalone')
    pools = defaultdict(list)
    rejected = Counter()
    source_manifests = []
    for manifest in report['datasets']:
        name = manifest['name']
        if name not in recipe['train']:
            continue
        directory = args.source_root/name/'v0.1'
        source_manifests.append({'name':name, 'manifest_sha256':file_hash(directory/'manifest.json'), 'manifest':manifest})
        for split in ['train','validation']:
            path = directory/(split+'.jsonl.gz')
            expected = manifest['split_files'][split]['sha256']
            if file_hash(path) != expected:
                raise ValueError('Source changed: '+str(path))
            for raw in jsonl(path):
                r = decorate(raw,name,split)
                if name.startswith('gsm8k-'):
                    r['_aliases'].add(r['group_id'].split(':',1)[1])
                elif raw['id'] in native_keys:
                    r['_aliases'].add(native_keys[raw['id']])
                if split == 'validation' and (r['_aliases'] & exposure['aliases'] or r['group_id'] in exposure['groups'] or r['_input_hash'] in exposure['inputs']):
                    rejected[name+':old_training_exposure'] += 1; continue
                if split == 'train' and (r['_aliases'] & holdout['aliases'] or r['group_id'] in holdout['groups'] or r['_input_hash'] in holdout['inputs']):
                    rejected[name+':historical_heldout'] += 1; continue
                pools[name,split].append(r)
        print('Loaded '+name, flush=True)
    for raw in added:
        r = decorate(raw,'toolace',raw['split'])
        if raw['split'] == 'validation' and r['_aliases'] & exposure['aliases']:
            rejected['toolace:old_training_exposure'] += 1; continue
        pools['toolace',raw['split']].append(r)
    for split in ['train','validation']:
        for raw in historical['v0.2',split]:
            source = raw['metadata']['dataset_source']
            name = 'retained:'+source
            if name not in recipe[split]:
                continue
            r = decorate(raw,source,split,True)
            r['metadata'].update(parent_dataset='groundbeef-v0.2', **parent_sources['sources'][source])
            if r['_aliases'] & protected:
                rejected[name+':protected_benchmark_alias'] += 1; continue
            if split == 'train' and (r['_aliases'] & holdout['aliases'] or r['group_id'] in holdout['groups'] or r['_input_hash'] in holdout['inputs']):
                rejected[name+':historical_heldout'] += 1; continue
            if split == 'validation' and (r['_aliases'] & exposure['aliases'] or r['group_id'] in exposure['groups'] or r['_input_hash'] in exposure['inputs']):
                rejected[name+':old_training_exposure'] += 1; continue
            pools[name,split].append(r)
    # Every development parent owns its split, including unselected siblings.
    all_dev_aliases = {k for (source, split), rows in pools.items() if split == 'validation'
                       for r in rows for k in r['_aliases']}
    all_dev_groups = {r['group_id'] for (source, split), rows in pools.items()
                      if split == 'validation' for r in rows}
    for (source, split), rows in list(pools.items()):
        if split == 'train':
            clean = [r for r in rows if r['group_id'] not in all_dev_groups and not r['_aliases'] & all_dev_aliases]
            rejected[source+':complete_development_ownership'] += len(rows)-len(clean)
            pools[source,split] = clean
    # Validation and calibration take source-owned development groups. Protect
    # their complete parent groups, including siblings not selected in the view.
    selected = {}
    dev_groups = set()
    dev_aliases = set()
    used_dev = {'ids':set(), 'inputs':set(), 'aliases':set(), 'gsm_pairs':set()}
    gsm_validation_groups = set()
    for split in ['calibration','validation']:
        selected[split] = []
        for source, count in recipe[split].items():
            rows = pools[source,'validation']
            if not (source == 'gsm8k-10choice' and split == 'validation'):
                rows = [r for r in rows if r['group_id'] not in dev_groups and not r['_aliases'] & dev_aliases]
            if source == 'gsm8k-10choice' and split == 'validation':
                rows = [r for r in rows if r['group_id'] in gsm_validation_groups]
            rows = [copy.deepcopy(r) for r in rows]
            for r in rows:
                r['split'] = split
            rr = select(rows,count,split+'/'+source,tokenizer,recipe['max_input_tokens'],rejected,used_dev,
                        allow_gsm_pair=source=='gsm8k-10choice' and split=='validation')
            selected[split].extend(rr)
            if source == 'gsm8k-4choice' and split=='validation':
                gsm_validation_groups.update(r['group_id'] for r in rr)
            dev_groups.update(r['group_id'] for r in rr)
            dev_aliases.update(k for r in rr for k in r['_aliases'])
    used_train = {'ids':set(), 'inputs':set(), 'aliases':set(), 'gsm_pairs':set()}
    selected['train'] = []
    for source, count in recipe['train'].items():
        rows = [r for r in pools[source,'train'] if r['group_id'] not in dev_groups and not r['_aliases'] & dev_aliases]
        if source in {'retained:open_jev','retained:groundbeef_rules'}:
            require_counts = recipe['retained_open_sources' if source == 'retained:open_jev' else 'retained_rule_tasks']
            assert sum(require_counts.values()) == count
            rr = []
            for generator, quota in require_counts.items():
                subset = [r for r in rows if (r['source'] if source == 'retained:open_jev' else r['metadata']['task_type']) == generator]
                rr.extend(select(subset,quota,'train/'+source+'/'+generator,
                                 tokenizer,recipe['max_input_tokens'],rejected,used_train))
        else:
            rr = select(rows,count,'train/'+source,tokenizer,recipe['max_input_tokens'],rejected,used_train)
        selected['train'].extend(rr)
    files, statistics = [], {}
    for split, rows in selected.items():
        rows.sort(key=lambda r: digest([SEED,'global-row-order',split,r['id']]))
        files.append(write_rows(args.output/(split+'.jsonl.gz'), rows))
        statistics[split] = stats(rows)
    # The quick set is a fixed, task/label-balanced nested subset, not resampled
    # every validation. It includes legacy anchors, separately marked in metadata.
    quick, quick_used = [], {'ids':set(),'inputs':set(),'aliases':set(),'gsm_pairs':set()}
    for source,count in quick_quotas(recipe).items():
        original_source = source.removeprefix('retained:')
        rows = [copy.deepcopy(r) for r in selected['validation']
                if r['metadata']['dataset_source'] == original_source]
        quick.extend(select(rows,count,'validation-quick/'+source,tokenizer,recipe['max_input_tokens'],
                            rejected,quick_used,True))
    quick.sort(key=lambda r: digest([SEED,'quick-order',r['id']]))
    files.append(write_rows(args.output/'validation-quick.jsonl.gz', quick))
    statistics['validation-quick'] = stats(quick)
    # Retain historical test and OOD bytes, without treating them as a new blind
    # test. New independent comparisons remain the unchanged external panel.
    import shutil
    for split in ['test','ood']:
        src = args.history_root/'v0.2'/(split+'.jsonl.gz')
        shutil.copyfile(src,args.output/src.name)
        files.append({'file':src.name,'rows':len(historical['v0.2',split]),'sha256':file_hash(src),'bytes':src.stat().st_size,
                      'role':'unchanged historical development reference; already evaluated'})
    members = []
    for split, rows in selected.items():
        for r in rows:
            m = r['metadata']
            members.append({'id':r['id'],'split':split,'group_id':r['group_id'],
                'source':m['dataset_source'],'native_id':m.get('native_id',m.get('upstream_index',r['id'])),
                'source_revision':m.get('source_revision',m.get('revision')), 'source_repo':m.get('source_repo',m.get('repo')),
                'source_file':m.get('source_file',m.get('upstream_file')), 'category':m['category'],
                'task':m['task_type'],'semantic_class':m['semantic_class'],'label_basis':m['label_basis'],
                'native_parent_group':r['group_id'],'cap_group':cap_group(r)[0],
                'original_row_sha256':r['_original_sha256'],'row_sha256':digest({k:v for k,v in r.items() if not k.startswith('_')}),
                'original_input_sha256':r['_input_hash'],'input_sha256':visible_hash(r),
                'alias_keys':sorted(r['_aliases']),'prior_training_exposure':m['prior_training_exposure'],
                'prior_validation_monitoring':m['prior_validation_monitoring'],
                'input_transform':m.get('input_transform','retained_original'),
                'input_tokens':m['input_tokens']})
    files.append(write_rows(args.output/'membership.jsonl.gz',members))
    save(args.output/'statistics.json',statistics)
    save(args.output/'training_weights.json',{'class_power':0.0,'class_cap':2.0,'sources':{},'tasks':{}})
    save(args.output/'sources.json',{'standalone_sources':source_manifests,'toolace':added_manifest,
                                   'retained_sources':parent_sources, 'historical_files':historical_files})
    save(args.output/'recipe.json',recipe)
    save(args.output/'manifest.json',{'dataset':'groundbeef','version':'0.3','stage':'training_release',
        'training_rows':80000,'files':files,'seed':SEED,'max_input_tokens':recipe['max_input_tokens'],
        'tokenizer_sha256':file_hash(args.tokenizer),'recipe_sha256':file_hash(args.output/'recipe.json'),
        'source_recipe_sha256':file_hash(args.recipe),
        'converter_sha256':file_hash(Path(__file__)), 'sources_sha256':file_hash(args.output/'sources.json'),
        'statistics_sha256':file_hash(args.output/'statistics.json'),
        'source_pool_audit_sha256':file_hash(args.source_root/'source-pool-v0.1/audit.json'),
        'source_pool_manifest_sha256':file_hash(args.source_root/'source-pool-v0.1/report.json'),
        'excluded_during_selection':dict(rejected),
        'validation':{'rows':4000,'quick_rows':1000,'new_source_rows':3700,'previously_monitored_legacy_rows':300,
            'fixed':True,'choice_augmentation':False,'training_exposed_rows':0,
            'selection_metric':'macro category cross entropy, unweighted; report new-source and legacy slices separately',
            'class_policy':'balanced semantic classes/tasks for diagnostic development; benchmark tests retain natural frequencies',
            'mismatches':['When2Call training provides 2-way preferences; official test has 4-way response policy. No clarification/refusal labels inferred from English.',
                          'MMLU auxiliary train has four candidates and no subjects; MMLU-Pro external test has ten candidates and named subjects.',
                          'CodeIO inputs/JSON outputs differ from CRUXEval Python literal outputs. Keep both source identities explicit.']},
        'label_policy':'Native hard labels / Open-Jev oracle distributions / explicit teacher probabilities stay separate. target is the active objective; never rename teacher argmax ground truth.',
        'sampling_policy':'Hash rank without replacement, balanced source strata, source group caps. Row order mixed; candidate order preserved. Training GSM has one presentation per problem.',
        'excluded_sources':{'PrimeVul':'deferred', 'Prometheus GPT-4':'user exclusion', 'vision_audio':'deferred',
            'CLINC':'entire training set overlaps protected ToolRet benchmark queries', 'Banking77':'omit to avoid protected ToolRet bank queries',
            'HoVer':'supporting evidence not available', 'ContractNLI':'whole-contract budget; deferred rather than extract answer spans'},
        'external_evaluation':{'panel':'datasets/evaluation-panel','manifest_sha256':file_hash(REPO/'datasets/evaluation-panel/manifest.json'),
                               'role':'unchanged previously measured panel; not validation or training'},
        'quality_limits':['No semantic/paraphrase contamination guarantee.', 'Backbone and competitors may have seen public corpora.',
                          'When2Call/ToolACE/Hermes native examples are synthetic; source labels are not independently verified.',
                          'CodeIO execution outputs are source-provided, not reexecuted locally.',
                          'Balanced development scores are diagnostic; use natural benchmark test metrics for comparison.']})
    print(dumps({s:{'rows':v['rows'],'counts':v['counts']['category']} for s,v in statistics.items()}), flush=True)


if __name__ == '__main__':
    main()
