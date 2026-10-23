"""Type-consistent code menus; execute only in the offline preparation container."""
import ast
import copy
import json
from pathlib import Path
import random
import subprocess
import sys

from common import code_key, digest, dumps
from python_execution_worker import check

INSTRUCTION = 'Choose the correct output of f called with the supplied input arguments. Candidates are Python literal values.'
EXECUTION_PROTOCOL = 'v3-precall-literal-input-request-tags'


def invocation(supplied):
    """Describe the actual call before source code can mutate its arguments."""
    if isinstance(supplied, dict):
        return 'f(' + ', '.join(k + '=' + repr(v) for k, v in supplied.items()) + ')'
    if isinstance(supplied, list):
        return 'f(' + ', '.join(repr(v) for v in supplied) + ')'
    return 'f(' + repr(supplied) + ')'


def normalized_program(code, function=None):
    tree = ast.parse(code)
    # Ignore the wrapper alias used by the older Python view.
    tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign) and
        any(isinstance(t, ast.Name) and t.id == 'f' for t in n.targets) and isinstance(n.value, ast.Name))]
    functions = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    function = function or ('main_solution' if 'main_solution' in functions else functions[-1] if functions else None)
    class Rename(ast.NodeTransformer):
        def visit_FunctionDef(self, n):
            if n.name == function: n.name = 'ENTRYPOINT'
            self.generic_visit(n); return n
        def visit_Name(self, n):
            if n.id == function: n.id = 'ENTRYPOINT'
            return n
    tree = Rename().visit(tree)
    for n in ast.walk(tree):
        body = getattr(n, 'body', None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            body.pop(0)
    return digest(ast.dump(tree, include_attributes=False))


def features(code):
    tree = ast.parse(code); ns = list(ast.walk(tree))
    attrs = {n.attr for n in ns if isinstance(n, ast.Attribute)}
    return {'lines': sum(bool(x.strip()) and not x.lstrip().startswith('#') for x in code.splitlines()),
            'imports': any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ns),
            'strings': bool(attrs & {'split', 'join', 'replace', 'strip', 'find', 'count', 'lower', 'upper'}),
            'slicing': any(isinstance(n, ast.Slice) for n in ns),
            'mutation': bool(attrs & {'append', 'extend', 'pop', 'remove', 'insert', 'sort', 'reverse'}),
            'loops': any(isinstance(n, (ast.For, ast.While, ast.comprehension)) for n in ns)}


def mutations(code, maximum=3):
    tree = ast.parse(code); result = []
    for index, original in enumerate(ast.walk(tree)):
        modes = []
        if isinstance(original, ast.Call) and isinstance(original.func, ast.Attribute):
            if original.func.attr == 'replace' and len(original.args) == 2: modes.append('replace_first_only')
            if original.func.attr == 'join' and isinstance(original.func.value, ast.Constant) and isinstance(original.func.value.value, str): modes.append('join_separator')
            if original.func.attr in {'lower', 'upper'}: modes.append('swap_case')
        if isinstance(original, ast.Call) and isinstance(original.func, ast.Name) and original.func.id == 'sorted': modes.append('sort_direction')
        if isinstance(original, ast.Slice) and original.step is not None: modes.append('slice_direction')
        if isinstance(original, ast.Subscript) and isinstance(original.slice, ast.Constant) and type(original.slice.value) is int: modes.append('index_off_by_one')
        if isinstance(original, ast.BinOp) and isinstance(original.op, (ast.Add, ast.Sub)): modes.append('arithmetic_sign')
        if isinstance(original, ast.Compare) and len(original.ops) == 1 and isinstance(original.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE)): modes.append('comparison_boundary')
        for mode in modes:
            other = copy.deepcopy(tree); node = list(ast.walk(other))[index]
            if mode == 'replace_first_only': node.args.append(ast.Constant(1))
            elif mode == 'join_separator': node.func.value.value = '' if node.func.value.value else ' '
            elif mode == 'swap_case': node.func.attr = 'upper' if node.func.attr == 'lower' else 'lower'
            elif mode == 'sort_direction':
                kw = next((k for k in node.keywords if k.arg == 'reverse'), None)
                if kw: kw.value = ast.UnaryOp(op=ast.Not(), operand=kw.value)
                else: node.keywords.append(ast.keyword(arg='reverse', value=ast.Constant(True)))
            elif mode == 'slice_direction': node.step = ast.UnaryOp(op=ast.USub(), operand=node.step)
            elif mode == 'index_off_by_one': node.slice.value += 1
            elif mode == 'arithmetic_sign': node.op = ast.Sub() if isinstance(node.op, ast.Add) else ast.Add()
            elif mode == 'comparison_boundary':
                op = node.ops[0]; node.ops[0] = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt}[type(op)]()
            changed = ast.unparse(ast.fix_missing_locations(other))
            if changed != ast.unparse(tree): result.append((mode, changed))
    # Prefer syntax-sensitive mistakes rather than whichever node happened first.
    order = {'replace_first_only': 0, 'join_separator': 1, 'slice_direction': 2, 'index_off_by_one': 3, 'sort_direction': 4}
    result.sort(key=lambda x: (order.get(x[0], 5), digest(x)))
    return result[:maximum]


def execute_menu(task):
    """Return a verified menu or explicit rejection; child processes carry limits."""
    code, function, supplied = task['code'], task['function'], task['input']
    try:
        check(code)
        cases = [{'tag': 'gold', 'code': code, 'function': function, 'input': supplied},
                 {'tag': 'repeat', 'code': code, 'function': function, 'input': supplied}]
        for i, (mode, changed) in enumerate(mutations(code)):
            cases.append({'tag': 'mutant:' + mode + ':' + str(i), 'code': changed, 'function': function, 'input': supplied,
                          'mutated_code_sha256': digest(changed)})
        for i, pair in enumerate(task.get('other_inputs', [])[:3]):
            cases.append({'tag': 'other_observed_input:' + str(i), 'code': code, 'function': function, 'input': pair['input'],
                          'expected_json': dumps(pair['output'])})
        # JSON alone turns tuple arguments into lists. Literal-only transport
        # preserves input types while never evaluating arbitrary expressions.
        for case in cases:case['input_literal']=repr(case['input'])
        cases_by_tag={case['tag']:case for case in cases}
        if len(cases_by_tag)!=len(cases):raise ValueError('duplicate execution request tags')
        proc = subprocess.Popen([sys.executable, str(Path(__file__).with_name('v05_code_worker.py'))],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                env={'PATH': '/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONHASHSEED': '0'})
        try: stdout, stderr = proc.communicate(json.dumps(cases), timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill(); stdout, stderr = proc.communicate()
        results = [json.loads(line) for line in stdout.splitlines() if line.strip()]
        by_tag = {r['tag']: r for r in results if r.get('ok')}
        if not all(x in by_tag for x in ['gold', 'repeat']): return {'rejection': 'execution_failed_or_timeout'}
        gold = by_tag['gold']
        if gold['literal'] != by_tag['repeat']['literal']: return {'rejection': 'nondeterministic_gold'}
        if task.get('expected_json') is not None and gold['json'] != task['expected_json']:
            return {'rejection': 'executed_gold_disagrees_with_native_expected'}
        # Cross-process verification also catches list(set(...)) hash-order targets.
        probe=subprocess.run([sys.executable,str(Path(__file__).with_name('v05_code_worker.py'))],
            input=json.dumps([cases[0]]),capture_output=True,text=True,timeout=5,
            env={'PATH':'/usr/bin:/bin','PYTHONDONTWRITEBYTECODE':'1','PYTHONHASHSEED':'1'})
        check_result=json.loads(probe.stdout.splitlines()[0]) if probe.stdout.strip() else {}
        if not check_result.get('ok') or check_result['json']!=gold['json']:
            return {'rejection':'cross_process_hash_seed_dependent_output'}
        gold_value = ast.literal_eval(gold['literal']); options = [gold['literal']]
        origins = [{'origin': 'executed_gold', 'invocation': gold['invocation']}]
        for item in results:
            if not item.get('ok') or item['tag'] in {'gold', 'repeat'}: continue
            value = ast.literal_eval(item['literal'])
            if type(value) is not type(gold_value) or any(value == ast.literal_eval(o) for o in options): continue
            source = cases_by_tag.get(item['tag'])
            if source is None:raise ValueError('unknown execution result tag')
            if 'expected_json' in source and item['json'] != source['expected_json']: continue
            if len(options) >= 6: break
            options.append(item['literal'])
            origins.append({'origin': item['tag'], 'invocation': item['invocation'],
                            'mutated_code_sha256': source.get('mutated_code_sha256'),
                            'source_output_independently_reexecuted': item['tag'].startswith('other_observed_input:')})
        if len(options) < 2: return {'rejection': 'no_distinct_same_type_executed_negative'}
        order = list(range(len(options))); random.Random('20261017:' + task['id']).shuffle(order)
        return {'options': [options[i] for i in order], 'gold': order.index(0),
                'origins': [origins[i] for i in order], 'execution': gold,
                'features': features(code), 'original_code_sha256': digest(code)}
    except (ValueError, SyntaxError, TypeError, KeyError, IndexError, json.JSONDecodeError, subprocess.TimeoutExpired) as e:
        return {'rejection': type(e).__name__ + ': ' + str(e)[:100]}


def code_row(task, result, split):
    code = task['code']
    if task['function'] != 'f': code += '\n\nf = ' + task['function'] + '\n'
    r = {'id': task['id'], 'group_id': 'python:' + normalized_program(task['code'], task['function']),
         'source': task['source'], 'domain': 'code', 'family': 'code.execution', 'kind': 'choice',
         'state': {'code': code, 'input': result['execution']['invocation'][2:-1], 'task': 'Python function output selection'},
         'question': INSTRUCTION, 'options': result['options'],
         'target': [float(i == result['gold']) for i in range(len(result['options']))], 'split': split,
         'metadata': {**task['metadata'], 'dataset_source': task['source'], 'category': 'code',
             'task_type': 'code.execution', 'subcategory': 'code.execution', 'loss_weight': 1.,
             'label_basis': 'repeated_restricted_execution_matches_native_expected',
             'execution': result['execution'], 'candidate_origins': result['origins'],
             'prior_view_distractor_method':task['metadata'].get('distractor_method'),
             'distractor_method':'executed semantic mutations and independently reexecuted same-program inputs; consistent Python types',
             'execution_protocol':EXECUTION_PROTOCOL,
             'runtime_verification':'restricted CPython 3.11; primary hash seed 0, gold checked in separate process at seed 1',
             'program_features': result['features'], 'original_code_sha256': result['original_code_sha256'],
             'input_transform': 'groundbeef-v0.5/type_consistent_python_execution',
             'semantic_class': 'answer_position:' + str(result['gold'])}}
    return r
