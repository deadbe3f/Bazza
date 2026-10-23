"""Build a diverse training slice and a reusable, group-aware exclusion ledger.

Selection uses training metadata only, never model errors on held-out tests.
This module has no torch/tokenizer dependency and can run on the host.
"""

import argparse
from collections import Counter, defaultdict, deque
import gzip
import hashlib
import json
from pathlib import Path
import unicodedata

from gemma_decisions.tracking import atomic_json

QUOTAS = {"tools": 1400, "grounding": 1200, "math": 1200, "code": 1200,
          "knowledge": 800, "science": 800, "commonsense": 600,
          "typed_decisions": 500, "ordinal": 300}
SEED = 20261013


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def read_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_rows(path, rows):
    """Deterministic gzip, preserving every record field and target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
            for row in rows:
                stream.write((json.dumps(row, ensure_ascii=False, sort_keys=True,
                                         separators=(",", ":"), allow_nan=False) + "\n").encode())
                count += 1
    temporary.replace(path)
    return {"rows": count, "sha256": file_hash(path), "bytes": path.stat().st_size}


def canonical(value):
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def visible_key(row):
    """Conservative exclusion key; choice order and labels are not identity."""
    state = row["state"]
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False, sort_keys=True)
    return digest([canonical(state), canonical(row["question"]),
                   sorted(canonical(o) for o in row["options"])])


def identity_keys(row):
    """Support both original data rows and Groundbeef membership entries."""
    meta = row.get("metadata", {})
    keys = {"id:" + str(row["id"])}
    for name in ("group_id", "native_parent_group", "cap_group"):
        value = row.get(name, meta.get(name))
        # SWE-QA uses entire repositories as train/validation split units.
        # Its native question/cap group is the appropriate exclusion unit.
        if str(value).startswith("swe_qa:repo:"):
            continue
        if value:
            keys.add("group:" + str(value))
    for value in row.get("alias_keys", meta.get("alias_keys", [])):
        keys.add("alias:" + str(value))
    for name in ("input_sha256", "original_input_sha256"):
        value = row.get(name, meta.get(name))
        if value:
            keys.add("upstream-input:" + str(value))
    native = row.get("native_id", meta.get("native_id", meta.get("upstream_index")))
    if native is not None:
        repo = (row.get("source_repo") or meta.get("source_repo") or meta.get("repo")
                or meta.get("dataset_source") or row.get("dataset_source") or row.get("source"))
        revision = row.get("source_revision") or meta.get("source_revision") or meta.get("revision") or ""
        keys.add("native:" + digest([repo, revision, str(native)]))
    if all(name in row for name in ("state", "question", "options")):
        keys.add("visible:" + visible_key(row))
    return keys


class Components:
    """Union all known aliases, so filtering includes transitive relatives."""

    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != key:
            next_key = self.parent[key]
            self.parent[key] = root
            key = next_key
        return root

    def add(self, keys):
        keys = sorted(keys)
        root = self.find(keys[0])
        for key in keys[1:]:
            other = self.find(key)
            if other != root:
                self.parent[other] = root


def length_bin(length):
    return next((str(limit) for limit in (128, 256, 512, 1024, 2048)
                 if length <= limit), "4096")


def select(records, components, quotas=QUOTAS, seed=SEED):
    """Round-robin sources/tasks and class/choice/length strata within them."""
    buckets = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for row in records:
        source = row.get("dataset_source", row["source"])
        inner = (len(row["target"]), row["semantic_class"], length_bin(len(row["ids"])),
                 "hard" if max(row["target"]) >= 1 - 1e-6 else "teacher")
        buckets[row["category"]][source][row["task_type"]][inner].append(row)
    selected, used = [], set()
    for category, quota in quotas.items():
        pools = []
        for source, tasks in sorted(buckets[category].items()):
            task_queues = []
            for task, strata in sorted(tasks.items()):
                queues = []
                for key, rows in sorted(strata.items()):
                    rows.sort(key=lambda r: digest([seed, r["id"]]))
                    queues.append(deque(rows))
                task_queues.append(deque(queues))
            pools.append(deque(task_queues))
        active = deque(pools)
        taken = 0
        while active and taken < quota:
            source = active.popleft()
            task = source.popleft()
            queue = task.popleft()
            row = queue.popleft()
            if queue:
                task.append(queue)
            if task:
                source.append(task)
            if source:
                active.append(source)
            group = components.find("id:" + row["id"])
            if group in used:
                continue
            used.add(group)
            selected.append(row)
            taken += 1
        if taken != quota:
            raise ValueError("Not enough distinct groups for %s: %s/%s" % (category, taken, quota))
    selected.sort(key=lambda r: digest([seed, "training-order", r["id"]]))
    return selected, used


def statistics(records):
    result = {}
    for category in sorted({r["category"] for r in records}):
        rows = [r for r in records if r["category"] == category]
        result[category] = {"rows": len(rows), "tokens": sum(len(r["ids"]) for r in rows),
            "sources": dict(sorted(Counter(r.get("dataset_source", r["source"]) for r in rows).items())),
            "tasks": dict(sorted(Counter(r["task_type"] for r in rows).items())),
            "semantic_classes": dict(sorted(Counter(r["semantic_class"] for r in rows).items())),
            "choice_counts": dict(sorted(Counter(str(len(r["target"])) for r in rows).items())),
            "length_bins": dict(sorted(Counter(length_bin(len(r["ids"])) for r in rows).items())),
            "teacher_targets": sum(max(r["target"]) < 1 - 1e-6 for r in rows)}
    return result


def build(data_dir, selection_dir, output_dir):
    release = json.loads((data_dir / "release.json").read_text())
    for name in ("manifest.json", "sources.json"):
        if file_hash(data_dir / name) != release["sha256"][name]:
            raise ValueError("Original dataset identity changed: " + name)
    for name in ("train.jsonl.gz", "membership.jsonl.gz", "validation-quick.jsonl.gz"):
        if file_hash(data_dir / name) != release["data_files_sha256"][name]:
            raise ValueError("Original data changed: " + name)
    encoded = json.loads((selection_dir / "manifest.json").read_text())
    if encoded["identity"]["dataset_manifest_sha256"] != file_hash(data_dir / "manifest.json"):
        raise ValueError("Tokenized manifest belongs to a different Groundbeef release")
    records = encoded["splits"]["train"]
    if len(records) != 80000:
        raise ValueError("Expected the original 80K training pool")
    identities, components = {}, Components()
    for row in read_rows(data_dir / "membership.jsonl.gz"):
        if row["id"] in identities:
            raise ValueError("Duplicate membership id")
        identities[row["id"]] = row
        components.add(identity_keys(row))
    for row in read_rows(data_dir / "train.jsonl.gz"):
        keys = identity_keys(row)
        keys.update(identity_keys(identities[row["id"]]))
        components.add(keys)
    protected_roots = {components.find("id:" + row["id"]) for row in identities.values()
                       if row["split"] != "train"}
    protected_overlap = {r["id"] for r in records
                         if components.find("id:" + r["id"]) in protected_roots}
    selected, used = select([r for r in records if r["id"] not in protected_overlap], components)
    selected_ids = {r["id"] for r in selected}
    selected_related = {r["id"] for r in records if components.find("id:" + r["id"]) in used}
    blocked_ids = selected_related | protected_overlap
    protected = {r["id"] for name, rr in encoded["splits"].items() if name != "train" for r in rr}
    if any(components.find("id:" + identity) in used for identity in protected):
        raise ValueError("Selected group overlaps validation or calibration")
    for entry in identities.values():
        if entry["split"] != "train" and components.find("id:" + entry["id"]) in used:
            raise ValueError("Selected alias overlaps a protected split")
    excluded_roots = used | {components.find("id:" + identity) for identity in protected_overlap}
    exclusion_keys = sorted(key for key in components.parent if components.find(key) in excluded_roots)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "manifest.json").exists():
        old = json.loads((output_dir / "manifest.json").read_text())
        if old["recipe_sha256"] != digest({"seed": SEED, "quotas": QUOTAS}) or old["builder_sha256"] != file_hash(__file__):
            raise ValueError("Existing slice identity differs; choose a new version/output")
        for name, entry in old["files"].items():
            if file_hash(output_dir / name) != entry["sha256"]:
                raise ValueError("Existing slice file changed: " + name)
        print(json.dumps({"reused": True, **old["counts"]}), flush=True)
        return
    selected_raw = []
    files = {}
    def remaining():
        for row in read_rows(data_dir / "train.jsonl.gz"):
            if row["id"] in selected_ids:
                selected_raw.append(row)
            if row["id"] not in blocked_ids:
                yield row
    files["remaining-train.jsonl.gz"] = write_rows(output_dir / "remaining-train.jsonl.gz", remaining())
    files["train.jsonl.gz"] = write_rows(output_dir / "train.jsonl.gz", selected_raw)
    selected_membership = [dict(identities[r["id"]], role="architecture_training",
                               original_split="train", exclusion_keys=sorted(identity_keys(identities[r["id"]])))
                           for r in selected]
    files["membership.jsonl.gz"] = write_rows(output_dir / "membership.jsonl.gz", selected_membership)
    files["validation-quick.jsonl.gz"] = write_rows(output_dir / "validation-quick.jsonl.gz",
                                                   read_rows(data_dir / "validation-quick.jsonl.gz"))
    ledger = {"dataset": "groundbeef-architecture", "version": "0.1", "selected_ids": sorted(selected_ids),
              "blocked_training_ids": sorted(blocked_ids), "keys": exclusion_keys,
              "protected_native_overlap_ids": sorted(protected_overlap),
              "selected_related_training_ids": sorted(selected_related),
              "swe_qa_policy": "Native question and alias exclusion; repository-level split groups are recorded in membership but not treated as duplicate questions.",
              "policy": "Exclude exact selected rows, linked parent groups, aliases, native identities and normalized inputs.",
              "limitations": "Does not detect every paraphrase or undo exposure in existing checkpoints."}
    atomic_json(output_dir / "exclusions.json", ledger)
    mini = {"identity": encoded["identity"], "splits": {"train": selected,
            "validation-quick": encoded["splits"]["validation-quick"],
            "validation-full": encoded["splits"]["validation"]}}
    atomic_json(output_dir / "selection.json", mini)
    stats = statistics(selected)
    atomic_json(output_dir / "statistics.json", stats)
    counts = {"train": len(selected), "validation_quick": 1000, "validation_full": 4000,
              "excluded_from_final_training": len(blocked_ids),
              "additional_linked_rows_excluded": len(selected_related) - len(selected),
              "protected_native_overlap_rows_excluded": len(protected_overlap),
              "remaining_training": 80000 - len(blocked_ids)}
    tokens = sum(len(r["ids"]) for r in selected)
    report = ["# Bazza architecture development slice", "", "8K selected from Groundbeef v0.3 training only.",
              "Native inputs, option order and hard/teacher targets are unchanged.", "",
              "| Category | Rows | Share | Sources |", "|---|---:|---:|---|"]
    for category, entry in stats.items():
        report.append("| %s | %s | %.2f%% | %s |" % (category, entry["rows"], entry["rows"] / 80,
                       ", ".join("%s: %s" % item for item in entry["sources"].items())))
    report += ["", "Training feature estimate: **%.2f GiB** (full 3840-wide BF16 token states)." % (tokens * 3840 * 2 / 2**30),
               "", "Selected IDs, source revisions, native identities and parent groups: `membership.jsonl.gz`.",
               "Machine-readable exclusion closure: `exclusions.json`.",
               "Original training minus selected/linked rows: `remaining-train.jsonl.gz` (%s rows)." % counts["remaining_training"],
               "Additionally excluded %s training rows sharing native source identities with protected splits." % len(protected_overlap),
               "", "The original Groundbeef release and resumable training run are unchanged.",
               "Exclusion cannot erase prior exposure in an existing trained head; use fresh heads for clean comparisons.",
               "Selection balances sources/tasks, semantic classes, choice counts and token-length bands.",
               "Known group aliases and normalized duplicate inputs are deduplicated; paraphrase deduplication is not guaranteed.",
               "SWE-QA uses native question identities for exclusion; repository-level groups remain split provenance.",
               "The existing quick/full validation sets remain held out and reusable; test/OOD are not selected or cached here.", ""]
    (output_dir / "REPORT.md").write_text("\n".join(report))
    for name in ("exclusions.json", "selection.json", "statistics.json", "REPORT.md"):
        files[name] = {"sha256": file_hash(output_dir / name), "bytes": (output_dir / name).stat().st_size}
    manifest = {"dataset": "groundbeef-architecture", "version": "0.1", "seed": SEED,
                "quotas": QUOTAS, "counts": counts, "recipe_sha256": digest({"seed": SEED, "quotas": QUOTAS}),
                "builder_sha256": file_hash(__file__), "parent_dataset": {"name": "groundbeef", "version": "0.3",
                  "manifest_sha256": file_hash(data_dir / "manifest.json"),
                  "train_sha256": file_hash(data_dir / "train.jsonl.gz"),
                  "membership_sha256": file_hash(data_dir / "membership.jsonl.gz"),
                  "sources_sha256": file_hash(data_dir / "sources.json"),
                  "tokenized_manifest_sha256": file_hash(selection_dir / "manifest.json")},
                "feature_estimate_gib": tokens * 3840 * 2 / 2**30,
                "files": files, "selection": "Training-only stratified round-robin, one row per known linked group; no model/test error mining."}
    atomic_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"output": str(output_dir), **counts, "feature_estimate_gib": manifest["feature_estimate_gib"]}), flush=True)


def filter_rows(input_path, output_path, ledger_path, membership_path=None):
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError("Filtering must write a new file, not replace the original")
    ledger = json.loads(Path(ledger_path).read_text())
    blocked = set(ledger["keys"])
    removed, kept = [], []
    for row in read_rows(input_path):
        (removed if identity_keys(row) & blocked else kept).append(row)
    # Optional membership entries connect future formats to previously recorded aliases.
    if membership_path:
        graph = Components()
        for row in read_rows(membership_path):
            graph.add(identity_keys(row))
        for row in kept + removed:
            graph.add(identity_keys(row))
        roots = {graph.find(k) for k in blocked if k in graph.parent}
        kept, removed = [], []
        for row in read_rows(input_path):
            (removed if graph.find("id:" + row["id"]) in roots else kept).append(row)
    info = write_rows(output_path, kept)
    audit = {"ledger_sha256": file_hash(ledger_path), "input_sha256": file_hash(input_path),
             "kept": len(kept), "removed": len(removed), "removed_ids": sorted(r["id"] for r in removed),
             "output": info}
    atomic_json(Path(str(output_path) + ".exclusion-audit.json"), audit)
    print(json.dumps({"kept": len(kept), "removed": len(removed), "output": str(output_path)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("build")
    prepare.add_argument("--data-dir", type=Path, required=True)
    prepare.add_argument("--selection-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    exclude = sub.add_parser("exclude")
    exclude.add_argument("--input", type=Path, required=True)
    exclude.add_argument("--output", type=Path, required=True)
    exclude.add_argument("--ledger", type=Path, required=True)
    exclude.add_argument("--membership", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        build(args.data_dir, args.selection_dir, args.output_dir)
    else:
        filter_rows(args.input, args.output, args.ledger, args.membership)


if __name__ == "__main__":
    main()
