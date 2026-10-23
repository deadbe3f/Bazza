"""Frozen token features for validation only; training stays online."""
from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from . import sweep


def validation_due(examples, batch_rows, quick_every, full_every, final=False):
    """Crossing an example boundary also works for non-divisible batch sizes."""
    previous = examples - batch_rows
    if final or examples // full_every > previous // full_every:
        return "full"
    if examples // quick_every > previous // quick_every:
        return "quick"
    return None


def length_batches(records, max_rows, token_budget):
    """Bucket by length, bounding padded tokens without dropping long rows."""
    batch = []
    for row in sorted(records, key=lambda r: (len(r["ids"]), r["id"])):
        if batch and (len(batch) >= max_rows or
                      (len(batch) + 1) * len(row["ids"]) > token_budget):
            yield batch
            batch = []
        batch.append(row)
    if batch:
        yield batch


class ValidationCache:
    def __init__(self, directory, identity, records, *, reserve_gib=10):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.allowed = {r["id"]: sweep.digest({k: r[k] for k in
                        ("id", "ids", "positions")}) for r in records}
        self.reserve_gib = reserve_gib
        self.path = self.root / "index.json"
        if self.path.exists():
            self.index = json.loads(self.path.read_text())
            if self.index["identity"] != identity or self.index["allowed"] != self.allowed:
                raise ValueError("Validation cache identity changed; use a fresh output")
        else:
            self.index = {"identity": identity, "allowed": self.allowed, "files": {}}
            sweep.atomic_json(self.path, self.index)
        self.checked = set()

    def ensure(self, records, online, logger, step, status, progress=False):
        for row in records:
            if self.allowed.get(row["id"]) != sweep.digest({k: row[k] for k in
                                                          ("id", "ids", "positions")}):
                raise ValueError("Only the fixed validation inputs may use this cache")
        missing = []
        for row in records:
            entry = self.index["files"].get(row["id"])
            if entry:
                if row["cache"] != entry["file"] or not (self.root / entry["file"]).exists():
                    raise ValueError("Committed validation feature is missing")
                if row["id"] not in self.checked:
                    if sweep.file_hash(self.root / entry["file"]) != entry["sha256"]:
                        raise ValueError("Committed validation feature checksum changed")
                    self.checked.add(row["id"])
            else:
                missing.append(row)
        if not missing:
            return
        required = sum(len(r["ids"]) * self.identity["hidden_width"] * 2 + 8192
                       for r in missing)
        if shutil.disk_usage(self.root).free < required + self.reserve_gib * 2**30:
            raise OSError("Not enough disk for validation features and the reserve")
        started = time.perf_counter()
        print(f"Caching {len(missing)} validation inputs only; training is uncached.", flush=True)
        for i, row in enumerate(tqdm(missing, desc="Cache validation", unit="row",
                                     disable=not progress), 1):
            if sweep.STOP_REQUESTED:
                raise sweep.SweepStopped()
            hidden = online.collate([row])[0][0].detach().to("cpu", dtype=torch.bfloat16).contiguous()
            if hidden.shape != (len(row["ids"]), self.identity["hidden_width"]):
                raise ValueError("Unexpected backbone feature shape")
            path = self.root / row["cache"]
            path.parent.mkdir(parents=True, exist_ok=True)
            sweep.atomic_save({"hidden": hidden}, path)
            self.index["files"][row["id"]] = {"file": row["cache"],
                "sha256": sweep.file_hash(path), "tokens": len(row["ids"])}
            self.checked.add(row["id"])
            # The committed index makes interrupted exports safely resumable.
            sweep.atomic_json(self.path, self.index)
            if i % 32 == 0 or i == len(missing):
                elapsed = time.perf_counter() - started
                values = {"validation_cache/completed": len(self.index["files"]),
                          "validation_cache/total": len(self.allowed),
                          "validation_cache/rows_per_second": i / max(elapsed, 1e-6)}
                logger.log(values, step)
                status("caching_validation", **values)

    @torch.no_grad()
    def predict(self, head, records, *, max_rows, token_budget, device, progress=False):
        head.eval()
        result = {}
        batches = list(length_batches(records, max_rows, token_budget))
        for rows in tqdm(batches, desc="Validate cached head", unit="batch",
                         leave=False, disable=not progress):
            if sweep.STOP_REQUESTED:
                raise sweep.SweepStopped()
            tensors = sweep.collate(rows, self.root, device)
            with sweep.autocast(device):
                logits = head(*tensors[:4]).float().cpu()
            for row, logit in zip(rows, logits):
                result[row["id"]] = logit[:len(row["target"])].clone()
        return [result[r["id"]] for r in records]
