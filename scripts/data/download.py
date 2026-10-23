#!/usr/bin/env python3
"""Download pinned public source files into separate, resumable dataset folders."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

REPO = Path(__file__).resolve().parents[2]


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(temp, path)


def download(source, item, root):
    path = root / source['name'] / 'v0.1' / 'raw' / item['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = item.get('sha256')
    if path.exists():
        actual = sha256(path)
        if expected and actual != expected:
            raise ValueError(f'Existing source file differs from its lock: {path}')
        return {'path': item['path'], 'bytes': path.stat().st_size, 'sha256': actual,
                'url': item['url'], 'reused': True}
    part = path.with_suffix(path.suffix + '.part')
    for attempt in range(3):
        try:
            # A failed transfer can be resumed if the server honours Range.
            offset = part.stat().st_size if part.exists() else 0
            headers = {'User-Agent': 'KneeJerk-training-data/0.1'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            request = urllib.request.Request(item['url'], headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                append = offset and response.status == 206
                expected_bytes = int(response.headers.get('Content-Length', '0'))
                received = 0
                with part.open('ab' if append else 'wb') as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
                        received += len(block)
                if expected_bytes and received != expected_bytes:
                    raise ValueError('Incomplete HTTP response')
            actual = sha256(part)
            if expected and actual != expected:
                raise ValueError(f'Download differs from its lock: {path}')
            os.replace(part, path)
            return {'path': item['path'], 'bytes': path.stat().st_size, 'sha256': actual,
                    'url': item['url'], 'reused': False}
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, default=Path(__file__).with_name('sources.json'))
    parser.add_argument('--root', type=Path, default=REPO / 'datasets/sources')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--freeze-hashes', action='store_true')
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    jobs = [(source, item) for source in lock['sources'] for item in source['files']]
    done, failures = {}, []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(download, s, f, args.root): (s, f) for s, f in jobs}
        for future in as_completed(pending):
            source, item = pending[future]
            try:
                value = future.result()
                done[(source['name'], item['path'])] = value
                if args.freeze_hashes:
                    item['sha256'] = value['sha256']
                print(f"[{len(done)}/{len(jobs)}] {source['name']}/{item['path']} "
                      f"{value['bytes'] / 1024**2:.1f} MiB", flush=True)
            except Exception as exc:
                failures.append({'source': source['name'], 'path': item['path'],
                                 'error': str(exc)})
                print(f"FAILED {source['name']}/{item['path']}: {exc}", flush=True)
    if args.freeze_hashes:
        save(args.lock, lock)
    for source in lock['sources']:
        files = [done[(source['name'], f['path'])] for f in source['files']
                 if (source['name'], f['path']) in done]
        errors = [f for f in failures if f['source'] == source['name']]
        save(args.root / source['name'] / 'v0.1/manifest.json', {
            'source': source['name'], 'version': '0.1', 'repo': source['repo'],
            'revision': source['revision'], 'category': source['category'],
            'adapter': source['adapter'], 'license': source['license'],
            'files': files, 'failures': errors, 'complete': not errors,
            'registry_sha256': sha256(args.lock),
        })
    save(REPO / 'build/training-data-uplift/downloads.json', {
        'files': len(done), 'bytes': sum(f['bytes'] for f in done.values()),
        'failures': failures, 'elapsed_seconds': time.monotonic() - started,
        'registry_sha256': sha256(args.lock),
    })
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
