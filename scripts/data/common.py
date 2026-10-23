"""Shared data-only helpers. No model loading, generation, or code execution."""

import ast
import gzip
import hashlib
import json
import math
import random
import unicodedata
from collections import Counter
from pathlib import Path

SEED = 20261011
REPO = Path(__file__).resolve().parents[2]


def dumps(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value):
    return hashlib.sha256(
        (value if isinstance(value, str) else dumps(value)).encode()
    ).hexdigest()


def canonical(text):
    # Deliberately conservative for formulas/code: preserve punctuation/operators.
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def key(text):
    return digest(canonical(text))


def file_hash(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def parquet(path):
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
        yield from batch.to_pylist()


def placed(options, correct, identity):
    indices = list(range(len(options)))
    random.Random(f"{SEED}:{identity}:presentation").shuffle(indices)
    return [options[i] for i in indices], indices.index(correct)


def code_key(code):
    tree = ast.parse(code)
    # Remove comments (AST parser) and docstrings; preserve program semantics.
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                body.pop(0)
    return digest(ast.dump(tree, include_attributes=False))


def row(
    source,
    identity,
    split,
    stem,
    options,
    correct,
    subcategory,
    state=None,
    group=None,
    origin="native_hard_label",
    **metadata,
):
    state = stem if state is None else state
    out = {
        "id": f"{source['name']}:{identity}",
        "kind": "choice",
        "state": state if isinstance(state, str) else dumps(state),
        "question": "Choose the correct answer." if state == stem else stem,
        "options": options,
        "target": [float(i == correct) for i in range(len(options))],
        "domain": source["category"],
        "family": subcategory,
        "source": source["name"],
        "split": split,
        "group_id": f"{source['name']}:{group or key(stem)}",
        "metadata": {
            "category": source["category"],
            "subcategory": subcategory,
            "native_id": str(identity),
            "label_origin": origin,
            "source_repo": source["repo"],
            "source_revision": source["revision"],
            **metadata,
        },
        "_key": key(metadata.pop("protection_text", stem)),
        "_group": f"{source['name']}:{group or key(stem)}",
    }
    # protection_text is audit plumbing, never a duplicated input or target.
    out["metadata"].pop("protection_text", None)
    return out


def validate(record):
    options, target = record["options"], record["target"]
    if not 2 <= len(options) <= 26 or len(options) != len(target):
        raise ValueError("option_count")
    if any(not isinstance(x, str) or not x.strip() for x in options):
        raise ValueError("empty_option")
    if len(set(canonical(x) for x in options)) != len(options):
        raise ValueError("duplicate_options")
    if any(not math.isfinite(x) or x < 0 for x in target) or sum(target) != 1:
        raise ValueError("invalid_target")
    if target.count(1.0) != 1:
        raise ValueError("not_one_hot")
    if not record["state"].strip() or not record["question"].strip():
        raise ValueError("empty_input")


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count, classes, families, groups = 0, Counter(), Counter(), set()
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
            for record in records:
                validate(record)
                record = {k: v for k, v in record.items() if not k.startswith("_")}
                stream.write((dumps(record) + "\n").encode())
                count += 1
                correct = record["target"].index(1.0)
                classes[str(correct)] += 1
                families[record["family"]] += 1
                groups.add(record["group_id"])
    return {
        "rows": count,
        "groups": len(groups),
        "sha256": file_hash(path),
        "bytes": path.stat().st_size,
        "answer_positions": dict(classes),
        "subcategories": dict(families),
    }


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
        repo = (
            row.get("source_repo")
            or meta.get("source_repo")
            or meta.get("repo")
            or meta.get("dataset_source")
            or row.get("dataset_source")
            or row.get("source")
        )
        revision = (
            row.get("source_revision")
            or meta.get("source_revision")
            or meta.get("revision")
            or ""
        )
        keys.add("native:" + _membership_digest([repo, revision, str(native)]))
    if all(name in row for name in ("state", "question", "options")):
        keys.add("visible:" + visible_key(row))
    return keys


def _membership_digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def visible_key(row):
    state = row["state"]
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False, sort_keys=True)
    return _membership_digest(
        [
            canonical(state),
            canonical(row["question"]),
            sorted(canonical(o) for o in row["options"]),
        ]
    )
