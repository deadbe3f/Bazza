#!/usr/bin/env python3
"""Independently verify frozen standalone files and their native GSM/ARC targets."""
import argparse
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import sqlite3

from common import REPO, file_hash, jsonl, parquet, validate
from common import key
from adapters import native_split


def check_complete_native_holdouts(raw_root, db):
    """Include unlabelled and rejected held-out questions, not only exports."""
    train_keys = {k for k, in db.execute("SELECT DISTINCT key FROM examples WHERE eligible=1 AND split='train'")}
    train_groups = {k for k, in db.execute("SELECT DISTINCT groupkey FROM examples WHERE eligible=1 AND split='train'")}
    checked, overlaps = Counter(), []
    fields = {'gsm8k':'question', 'arc-challenge':'question', 'arc-easy':'question',
              'openbookqa':'question_stem', 'sciq':'question', 'commonsenseqa':'question',
              'winogrande':'sentence', 'hellaswag':'ctx', 'mmlu':'question', 'aqua':'question',
              'clinc':'text'}
    for name, field in fields.items():
        for file in (raw_root/name/'v0.1/raw').rglob('*.parquet'):
            if native_split(name, file) == 'train':
                continue
            for i, raw in enumerate(parquet(file)):
                checked[name] += 1
                if key(raw[field]) in train_keys:
                    overlaps.append((name, file.name, i, 'question'))
                if name == 'hellaswag' and 'hellaswag:'+key(raw['source_id']) in train_groups:
                    overlaps.append((name, file.name, i, 'video_group'))
    # API-Bank test prefixes can have no usable target or different schema
    # formatting; their known input text is still an exclusion candidate.
    for file in (raw_root/'api-bank/v0.1/raw/test-data').glob('*.json'):
        for i, raw in enumerate(json.loads(file.read_text())):
            if 'input' not in raw:
                continue
            text = raw['input'].lstrip()
            try:
                while text.startswith('{'):
                    _, offset = json.JSONDecoder().raw_decode(text)
                    text = text[offset:].lstrip()
            except ValueError:
                continue
            checked['api-bank'] += 1
            if key(text) in train_keys:
                overlaps.append(('api-bank', file.name, i, 'dialogue_prefix'))
    assert not overlaps, f'Raw native held-out overlap requires exclusion: {overlaps[:20]}'
    return dict(checked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO/'datasets/decisions')
    parser.add_argument('--raw-root', type=Path, default=REPO/'datasets/sources')
    args = parser.parse_args()
    report = json.loads((args.root/'source-pool-v0.1/report.json').read_text())
    totals = Counter()
    ids, gsm_views = set(), {}
    for manifest in report['datasets']:
        name = manifest['name']
        folder = args.root/name/'v0.1'
        for split, expected in manifest['split_files'].items():
            path = folder/(split+'.jsonl.gz')
            assert file_hash(path) == expected['sha256'], f'Frozen file changed: {path}'
            count = 0
            for record in jsonl(path):
                validate(record)
                assert record['split'] == split
                assert record['domain'] == manifest['category']
                assert record['id'] not in ids, record['id']
                ids.add(record['id'])
                assert not any(k.startswith('_') for k in record)
                if name.startswith('gsm8k-'):
                    gsm_views[(name, record['metadata']['native_id'])] = record
                count += 1
            assert count == expected['rows'], (name, split, count)
            totals[(manifest['category'], split)] += count
        print('Verified', name, {s:f['rows'] for s,f in manifest['split_files'].items()}, flush=True)
    raw = list(parquet(args.raw_root/'gsm8k/v0.1/raw/main/train-00000-of-00001.parquet'))
    split_pools = {}
    split_ids = {}
    for (name, identity), record in gsm_views.items():
        if name != 'gsm8k-10choice':
            continue
        split = record['split']
        split_pools.setdefault(split, set()).add(record['metadata']['gold_numeric'])
        split_ids.setdefault(split, set()).add(int(identity))
    assert not split_ids['train'] & split_ids['validation']
    for (name, identity), record in gsm_views.items():
        original = raw[int(identity)]
        assert record['state'] == original['question'], 'GSM question was changed'
        gold = Decimal(original['answer'].split('####')[-1].strip().replace(',', ''))
        predicted_gold = Decimal(record['options'][record['target'].index(1.)])
        assert gold == predicted_gold, (identity, gold, predicted_gold)
        assert set(record['options']) <= split_pools[record['split']], 'Foreign split distractor'
        assert set(record['metadata']['distractor_parent_ids']) <= split_ids[record['split']]
        assert record['metadata']['candidate_pool_split'] == record['split']
        if name.endswith('4choice'):
            ten = gsm_views[('gsm8k-10choice', identity)]
            assert record['split'] == ten['split']
            assert set(record['options']) <= set(ten['options'])
            assert record['group_id'] == ten['group_id']
    # ARC has both letter and numeric choice labels. Check EVERY exported row
    # directly against its raw native answer key, including kept native order.
    for name in ['arc-challenge', 'arc-easy']:
        originals = {}
        for path in (args.raw_root/name/'v0.1/raw').rglob('*.parquet'):
            relative = str(path.relative_to(args.raw_root/name/'v0.1/raw'))
            for original in parquet(path):
                originals[f'{name}:{relative}:{original["id"]}'] = original
        for path in (args.root/name/'v0.1').glob('*.jsonl.gz'):
            for record in jsonl(path):
                original = originals[record['id']]
                assert record['options'] == original['choices']['text']
                native = original['choices']['label'].index(original['answerKey'])
                assert record['target'].index(1.) == native
    db = sqlite3.connect('file:'+str(args.root/'source-pool-v0.1/membership.sqlite')+'?mode=ro', uri=True)
    overlap = db.execute("SELECT count(*) FROM examples a WHERE a.eligible=1 AND a.split='train' AND EXISTS (SELECT 1 FROM examples b WHERE b.eligible=1 AND b.split!='train' AND (a.key=b.key OR a.groupkey=b.groupkey))").fetchone()[0]
    assert overlap == 0
    development_test_overlap = db.execute("SELECT count(*) FROM examples a WHERE a.eligible=1 AND a.split='validation' AND EXISTS (SELECT 1 FROM examples b WHERE b.eligible=1 AND b.split='test' AND (a.key=b.key OR a.groupkey=b.groupkey))").fetchone()[0]
    assert development_test_overlap == 0
    complete_holdouts = check_complete_native_holdouts(args.raw_root, db)
    for path, expected in report['preserved_existing_test_sha256'].items():
        assert file_hash(path) == expected, f'Existing benchmark changed: {path}'
    result = {'verified':True, 'rows':len(ids), 'train_heldout_exact_or_group_overlap':overlap,
              'all_gsm_native_answers_and_split_local_candidates_verified':True,
              'all_arc_native_labels_and_order_verified':True,
              'validation_test_exact_or_group_overlap':development_test_overlap,
              'category_splits':{f'{c}/{s}':n for (c,s),n in totals.items()},
              'unique_gsm_training_problems':len(split_ids['train']),
              'gsm_alternative_presentations_are_not_extra_problems':True,
              'raw_native_heldouts_checked_including_unlabelled':complete_holdouts,
              'limitations':['Approximate near-duplicate audit does not prove paraphrase disjointness.',
                             'CodeIO source outputs were not reexecuted.',
                             'Archived Groundbeef protection uses exact input fingerprints.']}
    (args.root/'source-pool-v0.1/audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
