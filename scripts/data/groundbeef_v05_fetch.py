"""Acquire bounded pinned source windows; no model inference or code execution."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
import urllib.request

from v05_job import Job, run, save, sha

SOURCES = {
    'openscience': {'repo': 'nvidia/OpenScience',
                    'revision': '7bd0437e4756f761768fe7e5cebeaa75480a4fd6'},
    'mbpp': {'repo': 'google-research-datasets/mbpp',
             'revision': '4bb6404fdc6cacfda99d4ac4205087b89d32030c'},
}
WINDOWS = {'OS-Q2.5-32B-10.jsonl': 28000, 'OS-Q3-235B-4.jsonl': 7000}


def url(s, filename):
    return f'https://huggingface.co/datasets/{s["repo"]}/resolve/{s["revision"]}/{filename}'


def request(address, offset=None):
    headers = {'User-Agent': 'KneeJerk-Groundbeef-v0.5'}
    if offset is not None:
        headers['Range'] = f'bytes={offset}-'
        address += ('&' if '?' in address else '?') + f'offset={offset}'
    return urllib.request.urlopen(urllib.request.Request(address, headers=headers), timeout=90)


def small_file(s, filename, root):
    path = root / filename; marker = path.with_name(path.name + '.receipt.json')
    if marker.exists():
        m = json.loads(marker.read_text())
        if sha(path) != m['sha256']:
            raise ValueError(f'Corrupt committed source: {path}')
        return m
    path.parent.mkdir(parents=True, exist_ok=True)
    with request(url(s, filename)) as response:
        data = response.read(16 << 20)
        if response.read(1):
            raise ValueError('Unexpected oversized small source')
    tmp = path.with_name(path.name + '.tmp'); tmp.write_bytes(data); tmp.replace(path)
    m = {'file': filename, 'sha256': sha(path), 'bytes': len(data), **s}
    save(marker, m)
    return m


def fetch_window(job, s, filename, limit, root):
    folder = root / filename.removesuffix('.jsonl'); folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        **s, 'source_file': filename, 'limit': limit, 'shards': [], 'rows': 0, 'next_offset': 0}
    if manifest['revision'] != s['revision'] or manifest['limit'] != limit:
        raise ValueError('Source window configuration changed')
    for shard in manifest['shards']:
        if sha(folder / shard['file']) != shard['sha256']:
            raise ValueError('Corrupt committed source shard')
    while manifest['rows'] < limit:
        job.check(); offset = manifest['next_offset']; batch = []; total = 0
        try:
            with request(url(s, filename), offset) as stream:
                if offset and (stream.status != 206 or not stream.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                    raise ValueError('Server refused exact byte-range resume')
                while manifest['rows'] < limit:
                    job.check()
                    line = stream.readline(2 << 20)
                    if not line or not line.endswith(b'\n'):
                        raise ValueError('Incomplete source record or exhausted source window')
                    record = json.loads(line)
                    batch.append(record); total += len(line)
                    if len(batch) == min(256, limit - manifest['rows']):
                        start = manifest['rows']; name = f'{start:07d}.jsonl.gz'
                        tmp = folder / (name + '.tmp')
                        with gzip.open(tmp, 'wt') as f:
                            for record in batch:
                                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                        tmp.replace(folder / name)
                        manifest['shards'].append({'file': name, 'rows': len(batch), 'first_row': start,
                            'start_offset': offset, 'end_offset': offset + total,
                            'sha256': sha(folder / name)})
                        manifest['rows'] += len(batch); offset += total
                        manifest['next_offset'] = offset
                        save(manifest_path, manifest)
                        job.update(source=filename, rows=manifest['rows'], target_rows=limit)
                        print(f'{filename}: {manifest["rows"]:,}/{limit:,} rows committed', flush=True)
                        batch = []; total = 0
        except (OSError, TimeoutError) as e:
            print(f'Source stream interrupted; resume from committed offset: {e}', flush=True)
            job.check(); time.sleep(2)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True); ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    config = {'sources': SOURCES, 'windows': WINDOWS,
              'code': {p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('v05_job.py')]}}
    job = Job(a.output, 'acquire', config, a.resume)
    def work():
        raw = a.output / 'raw'; receipts = []
        for name, s in SOURCES.items():
            receipts.append(small_file(s, 'README.md', raw / name)); job.check()
        for split in ['train', 'validation', 'test', 'prompt']:
            for variant in ['full', 'sanitized']:
                receipts.append(small_file(SOURCES['mbpp'], f'{variant}/{split}-00000-of-00001.parquet', raw / 'mbpp'))
                job.check()
        for filename, count in WINDOWS.items():
            fetch_window(job, SOURCES['openscience'], filename, count, raw / 'openscience')
        save(a.output / 'source-lock.json', {**config, 'small_files': receipts,
             'source_window_manifests': {name: sha(raw / 'openscience' / name.removesuffix('.jsonl') / 'manifest.json') for name in WINDOWS}})
        job.complete(rows=sum(WINDOWS.values()), source_lock=str(a.output / 'source-lock.json'))
    return run(job, work)


if __name__ == '__main__':
    raise SystemExit(main())
