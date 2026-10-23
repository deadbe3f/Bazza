"""Durable local metrics with optional, retryable Overfit & Chill publishing.

The experiment never needs the dashboard to be available. The JSONL is the
source of truth; Overfit's run/key/step upserts make replay safe.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from utils import atomic_json


class RunLog:
    def __init__(
        self, directory, name, config, url="", project="KneeJerk Gemma decisions"
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.url = url.rstrip("/")
        self.project = project
        self.next_retry = 0.0
        self.warned = False
        self.status = "running"
        self.path = self.directory / "metrics.jsonl"
        self.meta_path = self.directory / "logging.json"
        self.remote_path = self.directory / "overfit.json"
        metadata = {"name": name, "project": project, "config": config}
        if self.meta_path.exists():
            if json.loads(self.meta_path.read_text()) != metadata:
                raise ValueError(
                    "Logging identity changed; choose a fresh output directory"
                )
        else:
            atomic_json(self.meta_path, metadata)
        # Repair an interrupted final append; never discard a complete event.
        if self.path.exists():
            with self.path.open("rb+") as stream:
                data = stream.read()
                if data and not data.endswith(b"\n"):
                    stream.truncate(data.rfind(b"\n") + 1)
        self.offset = self.path.stat().st_size if self.path.exists() else 0
        self.flush()

    def request(self, method, path, body=None):
        encoded = None if body is None else json.dumps(body, allow_nan=False).encode()
        request = urllib.request.Request(
            self.url + "/api" + path,
            data=encoded,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            return json.load(response)

    def log(self, values, step):
        if step < 0 or any(
            not key.strip() or not math.isfinite(float(v)) for key, v in values.items()
        ):
            raise ValueError("Metrics require nonnegative steps and finite values")
        event = {
            "step": int(step),
            "values": {k: float(v) for k, v in values.items()},
            "wall_time": time.time(),
        }
        with self.path.open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.offset = self.path.stat().st_size
        self.flush()

    def flush(self, force=False):
        if not self.url or (not force and time.monotonic() < self.next_retry):
            return not self.url
        try:
            remote = (
                json.loads(self.remote_path.read_text())
                if self.remote_path.exists()
                else {}
            )
            if remote.get("url") != self.url:
                remote = {}
            if not remote:
                projects = self.request("GET", "/projects")
                project = next((p for p in projects if p["name"] == self.project), None)
                if project is None:
                    try:
                        project = self.request(
                            "POST", "/projects", {"name": self.project}
                        )
                    except urllib.error.HTTPError as error:
                        if error.code != 409:
                            raise
                        project = next(
                            p
                            for p in self.request("GET", "/projects")
                            if p["name"] == self.project
                        )
                meta = json.loads(self.meta_path.read_text())
                # Reuse a run if creation succeeded but the response was lost.
                run = next(
                    (
                        r
                        for r in self.request("GET", f"/projects/{project['id']}/runs")
                        if r["name"] == meta["name"]
                        and (
                            json.loads(r["config"])
                            if isinstance(r["config"], str)
                            else r["config"]
                        )
                        == meta["config"]
                    ),
                    None,
                )
                if run is None:
                    run = self.request(
                        "POST",
                        f"/projects/{project['id']}/runs",
                        {
                            "name": meta["name"],
                            "config": meta["config"],
                            "tags": ["gemma4", "frozen-backbone", "sweep"],
                        },
                    )
                remote = {"url": self.url, "run_id": run["id"], "offset": 0}
                atomic_json(self.remote_path, remote)
            endpoint = f"/runs/{remote['run_id']}"
            if self.path.exists():
                with self.path.open("rb") as stream:
                    stream.seek(remote["offset"])
                    for line in stream:
                        event = json.loads(line)
                        self.request(
                            "POST",
                            endpoint + "/metrics",
                            {"step": event["step"], "values": event["values"]},
                        )
                        remote["offset"] += len(line)
                        atomic_json(self.remote_path, remote)
            self.request("PATCH", endpoint, {"status": self.status})
            self.next_retry = 0.0
            self.warned = False
            return True
        except (OSError, ValueError, KeyError) as error:
            self.next_retry = time.monotonic() + 30
            if not self.warned:
                print(
                    f"Overfit unavailable ({error}); metrics remain in {self.path}",
                    flush=True,
                )
                self.warned = True
            return False

    def finish(self, status="completed"):
        self.status = status
        atomic_json(self.directory / "logging_status.json", {"status": status})
        self.flush(force=True)


def sync_tree(root, url):
    failures = 0
    for metadata in sorted(Path(root).rglob("logging.json")):
        value = json.loads(metadata.read_text())
        log = RunLog(
            metadata.parent, value["name"], value["config"], url, value["project"]
        )
        status_path = metadata.parent / "logging_status.json"
        if status_path.exists():
            log.status = json.loads(status_path.read_text())["status"]
        failures += not log.flush(force=True)
    if failures:
        raise RuntimeError(f"{failures} run logs could not be synchronized")
