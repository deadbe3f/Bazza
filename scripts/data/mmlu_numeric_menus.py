"""Conservative larger MMLU menus with numerically distinct scalar negatives.

Free-text retrieved alternatives are kept as an experimental source view; they
are not certified false for a new question. This release instead expands only
scalar-answer questions, with rational-value inequality checked explicitly.
"""
from collections import Counter
import copy
from fractions import Fraction
import json
from pathlib import Path
import random
import re

from common import canonical, digest, file_hash, jsonl
from groundbeef import decorate, save, write_rows
from groundbeef_v04_views import topic

NUMBER=re.compile(r'^\s*([$]?)([-+]?\d+(?:\.\d+)?(?:/\d+)?)(\s*[%a-zA-Z]*)\s*$')
NEGATIVE=re.compile(r'\b(false|incorrect|except|least|not true|not correct|not equal)\b',re.I)


def convert(raw):
    if NEGATIVE.search(str(raw['state'])):raise ValueError('negative predicate')
    matches=[NUMBER.fullmatch(o) for o in raw['options']]
    if not all(matches):raise ValueError('not a scalar menu')
    prefix={m[1] for m in matches};suffix={m[3].strip().casefold() for m in matches}
    if len(prefix)!=1 or len(suffix)!=1:raise ValueError('inconsistent units')
    values=[Fraction(m[2]) for m in matches]
    if len(set(values))!=4:raise ValueError('numerically equivalent native options')
    gold_index=raw['target'].index(1.);gold=values[gold_index]
    step=min(abs(v-gold) for v in values if v!=gold)
    original=list(raw['options']);extra=[];seen=set(values)
    for i in range(1,30):
        for sign in [-1,1]:
            value=gold+sign*i*step
            if value in seen:continue
            seen.add(value)
            number=str(value.numerator) if value.denominator==1 else str(value.numerator)+'/'+str(value.denominator)
            extra.append(matches[gold_index][1]+number+matches[gold_index][3])
            if len(extra)==6:break
        if len(extra)==6:break
    if len(extra)!=6:raise ValueError('insufficient distinct scalar negatives')
    options=original+extra;order=list(range(10));random.Random('20261013:'+raw['id']+':numeric-menu').shuffle(order)
    r=decorate(raw,'mmlu',raw['split']);r['options']=[options[i] for i in order]
    r['target']=[float(i==gold_index) for i in order];r['target_native']=list(r['target'])
    r['metadata'].update(dataset_source='mmlu-tenchoice',selection_topic=topic(raw),
        selection_topic_basis='keyword heuristic, not a native subject',
        input_transform='groundbeef-v0.4/mmlu-numeric-tenchoice',original_options=raw['options'],
        original_target=raw['target'],candidate_order=order,added_numeric_values=[str(Fraction(NUMBER.fullmatch(o)[2])) for o in extra],
        native_numeric_gold=str(gold),distractor_method='six rationally distinct scalar values; spacing from native wrong answers',
        added_candidate_sources=[],original_row_sha256=digest(raw),
        distractor_limit='Native scalar gold assumed unique; original human label preserved. No free-text negatives are invented.')
    return r


def main():
    root=Path('build/groundbeef/v0.4-sources/mmlu-numeric-menus-v0.1')
    if (root/'manifest.json').exists():raise ValueError('Numeric source view already exists; preserve it and choose a new version')
    root.mkdir(parents=True,exist_ok=True)
    files={}
    for split in ['train','validation']:
        rows=[];rejected=Counter()
        for raw in jsonl(Path('datasets/decisions/mmlu/v0.1')/(split+'.jsonl.gz')):
            try:rows.append(convert(raw))
            except (ValueError,ZeroDivisionError) as e:rejected[str(e)]+=1
        files[split]={'file':write_rows(root/(split+'-tenchoice.jsonl.gz'),rows),'rejected':dict(rejected)}
    # The native development pool has only eight scalar menus. Reserve 100
    # additional training-owned parents before combined selection, excluding
    # every previously selected v0.3 training ID. All views of these parents
    # become development-owned in the combined builder.
    train=list(jsonl(root/'train-tenchoice.jsonl.gz'));dev=list(jsonl(root/'validation-tenchoice.jsonl.gz'))
    seen={r['id'] for r in jsonl(Path('build/groundbeef/v0.3/train.jsonl.gz'))}
    eligible=sorted((r for r in train if r['id'] not in seen),key=lambda r:digest(['numeric-development',r['id']]))
    reserved=eligible[:100]
    if len(reserved)!=100:raise ValueError('Not enough previously unselected scalar parents')
    owned={r['group_id'] for r in reserved}
    for r in reserved:
        r['split']='validation';r['metadata']['derived_development_from']='native auxiliary_train; hash reserve, no previous v0.3 training ID'
    files['train']['file']=write_rows(root/'train-tenchoice.jsonl.gz',[r for r in train if r['group_id'] not in owned])
    files['validation']['file']=write_rows(root/'validation-tenchoice.jsonl.gz',dev+reserved)
    save(root/'manifest.json',{'files':files,'builder_sha256':file_hash(__file__),
         'source_manifest_sha256':file_hash('datasets/decisions/mmlu/v0.1/manifest.json'),
         'limits':'A numeric subset of auxiliary MMLU, not a MMLU-Pro or subject-annotated training set.'})
    print(json.dumps({k:v['file']['rows'] for k,v in files.items()}))


if __name__=='__main__':main()
