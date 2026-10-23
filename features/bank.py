"""Read only committed raw feature shards, including added layers."""
from collections import OrderedDict
import json
from pathlib import Path
import torch
from utils import digest, file_hash

def normalize_final(hidden, artifact):
    """Kiln RMSNorm: stored weight is the effective multiplier, already converted."""
    x = hidden.float()
    x = x * (x.square().mean(-1, keepdim=True) + artifact["eps"]).pow(-0.5)
    return (x * artifact["weight"].to(x.device).float()).to(hidden.dtype)


def read_journal_readonly(path):
    # Readers never repair or mutate an active export's journal.
    with path.open("rb") as stream:
        return [json.loads(line) for line in stream if line.endswith(b"\n")]

def commits(directory):
    read = read_journal_readonly
    merged = {}
    for name in ("shards.jsonl", "layer-additions.jsonl"):
        p = directory / name
        if not p.exists():
            continue
        for entry in read(p):
            n = entry["shard"]
            if n not in merged:
                merged[n] = {**entry, "files": dict(entry["files"])}
                continue
            previous = merged[n]
            if previous["records_sha256"] != entry["records_sha256"] or previous["rows"] != entry["rows"]:
                raise ValueError("Layer journals disagree about inputs/offsets")
            for layer, item in entry["files"].items():
                if layer in previous["files"] and previous["files"][layer] != item:
                    raise ValueError("Conflicting committed layer file")
                previous["files"][layer] = item
    return merged

class LayerBank:
    """Read only the requested layers and examples; mmap shards, bounded handles."""
    def __init__(self, split_dir, layers, max_open_shards=8):
        self.directory = Path(split_dir)
        self.layers = list(layers)
        self.max_open = max_open_shards
        self.cache = OrderedDict()
        self.records = {r["id"]: r for r in json.loads((self.directory / "records.json").read_text())}
        self.locations = {}
        for entry in read_journal_readonly(self.directory / "shards.jsonl"):
            for row in entry["rows"]:
                self.locations[row["id"]] = (entry, row)

    def load(self, example_id):
        entry, row = self.locations[example_id]
        result = {}
        for layer in self.layers:
            key = entry["files"][layer]["file"]
            if key not in self.cache:
                self.cache[key] = torch.load(self.directory / key, map_location="cpu",
                                             weights_only=True, mmap=True)
                while len(self.cache) > self.max_open:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(key)
            result[layer] = self.cache[key]["hidden"][row["offset"]:row["offset"] + row["tokens"]]
        return result

class AddedLayerBank(LayerBank):
    """Mmap only requested layers; works with original and added-layer commits."""
    def __init__(self, split_dir, layers, max_open_shards=8):
        self.directory = Path(split_dir)
        self.layers = list(layers)
        self.max_open = max_open_shards
        self.cache = OrderedDict()
        self.records = {r["id"]: r for r in json.loads((self.directory / "records.json").read_text())}
        self.locations = {}
        for entry in commits(self.directory).values():
            if set(self.layers).issubset(entry["files"]):
                for row in entry["rows"]:
                    self.locations[row["id"]] = (entry, row)
