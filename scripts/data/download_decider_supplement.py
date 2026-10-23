#!/usr/bin/env python3
"""Download pinned supplement sources; freeze file hashes on the first download."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

from common import REPO, file_hash

DECIDER_REVISION = 'e50e549b47e2da69223734fee4efa1ddd4528e93'
SOURCES = [
    ('ag_news', 'fancyzhx/ag_news', 'eb185aade064a813bc0b7f42de02595523103ca4', 'unknown',
     ['data/train-00000-of-00001.parquet', 'data/test-00000-of-00001.parquet']),
    ('sst2', 'stanfordnlp/sst2', '8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb', 'unknown',
     ['data/'+s+'-00000-of-00001.parquet' for s in ['train','validation','test']]),
    ('paws', 'google-research-datasets/paws', '161ece9501cf0a11f3e48bd356eaa82de46d6a09', 'other; see source card',
     ['labeled_final/'+s+'-00000-of-00001.parquet' for s in ['train','validation','test']]),
    ('toxic_chat', 'lmsys/toxic-chat', '29df8e4dba60e1f4af4b4075c0705c5b313548a8', 'CC-BY-NC-4.0',
     ['data/0124/toxic-chat_annotation_'+s+'.csv' for s in ['train','test']]),
    ('stsb', 'SetFit/stsb', '77bd85cf73a244d4503e477ce6de81196187839f', 'not stated in card; preserve upstream terms',
     [s+'.jsonl' for s in ['train','validation','test']]),
    ('wiki_qa', 'microsoft/wiki_qa', '3f104672b5de699878fe7907afc486f0de325eb5', 'other; see source card',
     ['data/'+s+'-00000-of-00001.parquet' for s in ['train','validation','test']]),
]


def download(item):
    url, destination, expected = item
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        part = destination.with_suffix(destination.suffix+'.part')
        with urllib.request.urlopen(url, timeout=180) as src, part.open('wb') as dst:
            shutil.copyfileobj(src, dst)
        part.rename(destination)
    sha = file_hash(destination)
    if expected and sha != expected:
        raise ValueError('Pinned source file changed: '+str(destination))
    return {'path': str(destination), 'url': url, 'sha256': sha, 'bytes': destination.stat().st_size}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--decider-checkout', type=Path, default=Path('/tmp/decider-dataset-review'))
    ap.add_argument('--lock', type=Path, default=Path(__file__).with_name('decider_supplement_sources.json'))
    a = ap.parse_args()
    old = json.loads(a.lock.read_text()) if a.lock.exists() else None
    upstream = REPO/'datasets/sources/mapika-decider'/DECIDER_REVISION/'raw'
    if not upstream.exists():
        actual = subprocess.check_output(['git','-C',str(a.decider_checkout),'rev-parse','HEAD'],text=True).strip()
        if actual != DECIDER_REVISION:
            raise ValueError('Wrong Decider checkout revision')
        shutil.copytree(a.decider_checkout, upstream, ignore=shutil.ignore_patterns('.git','__pycache__'))
    upstream_files = []
    for p in sorted(upstream.rglob('*')):
        if p.is_file():
            upstream_files.append({'path':str(p.relative_to(upstream)), 'sha256':file_hash(p), 'bytes':p.stat().st_size})
    if old and upstream_files != old['decider']['files']:
        raise ValueError('Frozen Decider source copy changed')
    result = {'dataset':'groundbeef-decider-supplement', 'version':'0.1',
              'decider':{'repo':'https://github.com/Mapika/decider','revision':DECIDER_REVISION,
                         'license':'Apache-2.0; preserve LICENSE', 'files':upstream_files}, 'sources':[]}
    for name, repo, rev, license_, paths in SOURCES:
        entries = []
        prior = next((s for s in old['sources'] if s['name']==name),None) if old else None
        for path in ['README.md']+paths:
            expected = next((f['sha256'] for f in prior['files'] if f['path']==path),None) if prior else None
            destination = REPO/'datasets/sources'/('decider-'+name)/rev/'raw'/path
            entries.append(('https://huggingface.co/datasets/'+repo+'/resolve/'+rev+'/'+path, destination, expected))
        with ThreadPoolExecutor(max_workers=4) as pool:
            files = list(pool.map(download, entries))
        for file, path in zip(files, ['README.md']+paths): file['path']=path
        result['sources'].append({'name':name,'repo':repo,'revision':rev,'license':license_,'files':files})
        print('Verified',name,sum(f['bytes'] for f in files),'bytes',flush=True)
    if old and result != old:
        raise ValueError('Source lock differs from frozen inventory')
    if not old:
        a.lock.write_text(json.dumps(result,indent=2)+'\n')
    print('Frozen source lock:',a.lock,flush=True)


if __name__ == '__main__': main()
