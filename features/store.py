"""CPU IO for immutable per-example or raw-layer feature banks."""
import json
from pathlib import Path
import torch
from data import collate_records
from utils import file_hash
from .bank import AddedLayerBank


class FeatureStore:
    def __init__(self, config, selection, device="cuda"):
        self.config, self.selection, self.device = config, selection, device
        self.kind = config["kind"]
        self.resident = {}
        self.checked = set()
        self.banks = {}
        self.normalization = None
        if self.kind in ("layer", "routed"):
            if not config.get("layer"):
                raise ValueError("Raw cache requires an explicit final layer")
            roots = config.get("banks", {"default": config.get("root")})
            self.roots = {k: Path(v) for k, v in roots.items()}
            manifests = [json.loads((root / "manifest.json").read_text()) for root in self.roots.values()]
            for manifest in manifests:
                if manifest.get("final_layer", config["layer"]) != config["layer"]:
                    raise ValueError("Final normalization requires the bank's final layer")
                for name, value in manifest["backbone"].items():
                    if selection["identity"].get(name) != value:
                        raise ValueError("Feature backbone differs from selection: " + name)
                if manifest.get("serialization_sha256") != selection["identity"].get("serialization_sha256"):
                    raise ValueError("Feature serialization differs from selection")
            norms = [torch.load(root / "final-rmsnorm.pt", map_location="cpu", weights_only=True)
                     for root in self.roots.values()]
            self.normalization = norms[0]
            for norm in norms[1:]:
                if norm["eps"] != norms[0]["eps"] or not torch.equal(norm["weight"], norms[0]["weight"]):
                    raise ValueError("Feature components have different normalization")
        elif self.kind == "pt":
            self.root = Path(config["root"])
            self.index = json.loads((self.root / "index.json").read_text()) if (self.root / "index.json").exists() else {}
        else:
            raise ValueError("Unknown feature mode: " + self.kind)

    def identity(self):
        """Hash manifests/indices, not hundreds of GB of tensors on every run."""
        paths = []
        if self.kind in ("layer", "routed"):
            for root in self.roots.values():
                paths.extend([root / "manifest.json", root / "final-rmsnorm.pt"])
                for child in root.iterdir():
                    if child.is_dir() and (child / "records.json").exists():
                        paths.extend([child / "records.json", child / "index.json"])
                        paths.extend(p for p in [child / "shards.jsonl", child / "layer-additions.jsonl"] if p.exists())
        elif self.kind == "pt":
            paths.extend(p for p in [self.root / "index.json", self.root / "records.json"] if p.exists())
        return {str(p): file_hash(p) for p in sorted(set(paths))}

    def _load(self, row, split):
        if row["id"] in self.resident:
            return self.resident[row["id"]]
        if self.kind == "pt":
            entry = self.index.get("files", {}).get(row["id"])
            if entry and row["id"] not in self.checked:
                if entry["file"] != row["cache"] or file_hash(self.root / row["cache"]) != entry["sha256"]:
                    raise ValueError("Cached feature checksum differs: " + row["id"])
                self.checked.add(row["id"])
            return torch.load(self.root / row["cache"], map_location="cpu", weights_only=True)["hidden"]
        if self.kind == "routed":
            route_split = self.config.get("split_map", {}).get(split, split)
            component, bank_split = self.selection["routes"][route_split][row["id"]]
        else:
            component = "default"
            bank_split = self.config.get("split_map", {}).get(split, split)
        key = component, bank_split
        if key not in self.banks:
            self.banks[key] = AddedLayerBank(self.roots[component] / bank_split,
                [self.config["layer"]], self.config.get("max_open_shards", 128))
        bank = self.banks[key]
        expected = bank.records[row["id"]]
        for name in ("ids", "positions", "order", "target"):
            actual = row.get("_cache_target", row.get("target")) if name == "target" else row.get(name)
            if actual != expected.get(name):
                raise ValueError("Cached input identity differs: " + row["id"] + "/" + name)
        return bank.load(row["id"])[self.config["layer"]]

    def prepare(self, rows, split):
        return collate_records(rows, [self._load(row, split) for row in rows])

    def preload(self, rows, split):
        for row in rows:
            self.resident[row["id"]] = self._load(row, split).clone()
