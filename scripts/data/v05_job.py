"""Durable lifecycle helpers for the Groundbeef v0.5 CPU preparation jobs."""
import hashlib
import json
import os
from pathlib import Path
import signal
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


class Stopped(Exception):
    pass


class Job:
    def __init__(self, root, phase, config, resume):
        self.root = Path(root); self.phase = phase
        self.path = self.root / 'jobs' / phase
        self.path.mkdir(parents=True, exist_ok=True)
        self.stop = False; self.started = time.time(); self.status = {}
        cfg = self.path / 'config.json'
        if cfg.exists():
            if not resume:
                raise ValueError('Job exists; use --resume or a new output directory')
            if json.loads(cfg.read_text()) != config:
                raise ValueError('Resume rejected: source/configuration/code fingerprint changed')
            if (self.path / 'complete.json').exists():
                raise ValueError('Job already complete; use --status')
        else:
            save(cfg, config)
        for sig in [signal.SIGTERM, signal.SIGINT]:
            signal.signal(sig, lambda *_: setattr(self, 'stop', True))
        self.update('starting')

    def update(self, state='running', **values):
        self.status.update(values)
        self.status.update(phase=self.phase, state=state, updated_at=time.time(),
                           session_seconds=time.time() - self.started)
        save(self.path / 'status.json', self.status)

    def check(self):
        if self.stop:
            self.update('stopped', reason='signal; committed shards/rows retained')
            raise Stopped()

    def complete(self, **values):
        self.update('complete', **values)
        save(self.path / 'complete.json', self.status)


def run(job, function):
    try:
        function()
    except Stopped:
        print('Stopped cleanly; use --resume to continue.', flush=True)
        return 130
    except BaseException as e:
        job.update('failed', error=type(e).__name__ + ': ' + str(e))
        raise
    return 0
