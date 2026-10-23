#!/usr/bin/env python3
"""Build immutable standalone training sources, with shared split protection."""

import argparse
import importlib
import json
import os
import re
import shutil
import sqlite3
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from adapters import DERIVED_VALIDATION, native_split, records
from common import (
    REPO,
    SEED,
    canonical,
    code_key,
    digest,
    dumps,
    file_hash,
    jsonl,
    key,
    parquet,
    validate,
    write_jsonl,
)

READY = {
    "gsm",
    "arc",
    "openbook",
    "csqa",
    "aqua",
    "mathqa",
    "sciq",
    "wino",
    "hella",
    "mmlu",
    "anli",
    "clinc",
    "when_pref",
    "toolret",
    "api_bank",
    "rag",
    "codeio",
}
PRIORITY = {"train": 1, "validation": 2, "test": 3}
MCQ_CATEGORIES = {"math", "science", "knowledge", "commonsense"}


class NearIndex:
    """Candidate audit, not a proof against paraphrases: four bottom trigram hashes."""

    def __init__(self):
        self.index = defaultdict(list)
        self.texts = {}

    @staticmethod
    def shingles(text):
        words = re.findall(r"\w+", canonical(text))
        return {" ".join(words[i : i + 3]) for i in range(len(words) - 2)}

    def add(self, text, identity):
        shingles = self.shingles(text)
        if len(shingles) < 5:
            return
        self.texts[identity] = shingles
        for value in sorted(shingles, key=digest)[:4]:
            self.index[value].append(identity)

    def match(self, text):
        shingles = self.shingles(text)
        if len(shingles) < 5:
            return None
        candidates = set()
        for value in sorted(shingles, key=digest)[:4]:
            candidates.update(self.index.get(value, []))
        for identity in sorted(candidates):
            other = self.texts[identity]
            if len(shingles & other) / len(shingles | other) >= 0.85:
                return identity
        return None


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def protected_external(root, db, near):
    counts = Counter()
    # Register complete native held-outs even if choices/labels are unusable.
    fields = {
        "gsm8k": "question",
        "arc-challenge": "question",
        "arc-easy": "question",
        "openbookqa": "question_stem",
        "sciq": "question",
        "commonsenseqa": "question",
        "winogrande": "sentence",
        "hellaswag": "ctx",
        "mmlu": "question",
        "aqua": "question",
        "clinc": "text",
    }
    for name, field in fields.items():
        for file in (root / name / "v0.1/raw").rglob("*.parquet"):
            if native_split(name, file) == "train":
                continue
            for i, raw in enumerate(parquet(file)):
                text = raw[field]
                db.execute(
                    "INSERT OR REPLACE INTO protected VALUES (?, ?)",
                    (key(text), f"{name}:{file.name}:{i}"),
                )
                if native_split(name, file) == "test":
                    db.execute(
                        "INSERT OR REPLACE INTO protected_test VALUES (?)", (key(text),)
                    )
                if name != "clinc":
                    near.add(text, f"{name}:{file.name}:{i}")
                if name == "hellaswag":
                    db.execute(
                        "INSERT OR REPLACE INTO protected_group VALUES (?)",
                        ("hellaswag:" + key(raw["source_id"]),),
                    )
                counts["native_full:" + name] += 1
    with zipfile.ZipFile(root / "mathqa/v0.1/raw/MathQA.zip") as archive:
        for filename in ["dev.json", "test.json", "challenge_test.json"]:
            for i, raw in enumerate(json.loads(archive.read(filename))):
                text = raw["Problem"]
                db.execute(
                    "INSERT OR REPLACE INTO protected VALUES (?, ?)",
                    (key(text), f"mathqa:{filename}:{i}"),
                )
                if filename != "dev.json":
                    db.execute(
                        "INSERT OR REPLACE INTO protected_test VALUES (?)", (key(text),)
                    )
                near.add(text, f"mathqa:{filename}:{i}")
                counts["native_full:mathqa"] += 1
    for file in (root / "api-bank/v0.1/raw/test-data").glob("*.json"):
        for i, raw in enumerate(json.loads(file.read_text())):
            if "input" not in raw:
                continue
            text = raw["input"].lstrip()
            try:
                while text.startswith("{"):
                    _, offset = json.JSONDecoder().raw_decode(text)
                    text = text[offset:].lstrip()
            except ValueError:
                continue
            db.execute(
                "INSERT OR REPLACE INTO protected VALUES (?, ?)",
                (key(text), f"api-bank:{file.name}:{i}"),
            )
            db.execute("INSERT OR REPLACE INTO protected_test VALUES (?)", (key(text),))
            counts["native_full:api-bank"] += 1
    for source in ["mmlu-pro-protected", "toolret-protected"]:
        for file in sorted((root / source / "v0.1/raw").rglob("*.parquet")):
            for i, raw in enumerate(parquet(file)):
                text = raw["question"] if source.startswith("mmlu") else raw["query"]
                db.execute(
                    "INSERT OR REPLACE INTO protected VALUES (?, ?)",
                    (key(text), f"{source}:{file.name}:{i}"),
                )
                db.execute(
                    "INSERT OR REPLACE INTO protected_test VALUES (?)", (key(text),)
                )
                if source.startswith("mmlu"):
                    near.add(text, f"{source}:{file.name}:{i}")
                counts[source] += 1
    file = root / "when2call/v0.1/raw/test/when2call_test_mcq.jsonl"
    for raw in jsonl(file):
        db.execute(
            "INSERT OR REPLACE INTO protected VALUES (?, ?)",
            (key(raw["question"]), "when2call:" + raw["uuid"]),
        )
        db.execute(
            "INSERT OR REPLACE INTO protected_test VALUES (?)", (key(raw["question"]),)
        )
        counts["when2call_full_test"] += 1
    # Protect all 800 original CRUXEval programs, not just the 570 converted cases.
    cruxroot = (
        REPO
        / "datasets/evaluation-panel/sources/work/artifacts/benchmark-suite/raw/repos/cruxeval"
    )
    files = list(cruxroot.rglob("*.jsonl"))
    for file in files:
        for raw in jsonl(file):
            if "code" not in raw:
                continue
            db.execute(
                "INSERT OR REPLACE INTO protected_code VALUES (?)",
                (code_key(raw["code"]),),
            )
            counts["cruxeval_programs"] += 1
    if counts["cruxeval_programs"] < 800:
        raise ValueError("Full CRUXEval source protection is missing")
    return dict(counts)


def historical_protection(path, db):
    protected_files = []
    if path is None:
        return protected_files
    if not path.is_dir():
        raise ValueError(f"Historical held-out root unavailable: {path}")
    for split in ["validation", "calibration", "test", "ood"]:
        for file in path.glob(split + ".jsonl*"):
            count = 0
            for raw in jsonl(file):
                db.execute(
                    "INSERT OR IGNORE INTO protected_input VALUES (?)",
                    (digest([raw["state"], raw["question"], sorted(raw["options"])]),),
                )
                count += 1
            protected_files.append(
                {"path": str(file), "rows": count, "sha256": file_hash(file)}
            )
    return protected_files


def mark_excluded(db, query, params, source_stats, reason):
    counts = db.execute(
        "SELECT source,count(*) FROM examples WHERE eligible=1 AND ("
        + query
        + ") GROUP BY source",
        params,
    ).fetchall()
    for source, count in counts:
        source_stats[source][reason] += count
    db.execute(
        "UPDATE examples SET eligible=0 WHERE eligible=1 AND (" + query + ")", params
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock", type=Path, default=Path(__file__).with_name("sources.json")
    )
    parser.add_argument("--raw-root", type=Path, default=REPO / "datasets/sources")
    parser.add_argument("--output-root", type=Path, default=REPO / "datasets/decisions")
    parser.add_argument("--historical-root", type=Path)
    parser.add_argument("--code-limit", type=int, default=20000)
    parser.add_argument(
        "--reuse-ingestion",
        type=Path,
        help="Reuse a prior audited source ledger only when raw registry and adapter hashes match",
    )
    args = parser.parse_args()
    started = time.monotonic()
    lock = json.loads(args.lock.read_text())
    sources = [
        s
        for s in lock["sources"]
        if s["adapter"] in READY or s["name"] == "contractnli"
    ]
    output_names = [s["name"] for s in sources if s["name"] != "gsm8k"] + [
        "gsm8k-4choice",
        "gsm8k-10choice",
    ]
    for name in output_names:
        if (args.output_root / name / "v0.1").exists():
            raise ValueError(
                f"Immutable output exists: {name}/v0.1; use another output root"
            )
    if (args.output_root / "source-pool-v0.1").exists():
        raise ValueError("Immutable pool ledger exists; use another output root")
    preserved = {
        str(p): file_hash(p) for p in sorted((REPO / "datasets").glob("*/test.jsonl"))
    }
    stage = REPO / "build/training-data-uplift/preparing"
    if stage.exists():
        shutil.rmtree(
            stage
        )  # Only disposable conversion staging, never frozen datasets.
    stage.mkdir(parents=True)
    reuse_report = None
    if args.reuse_ingestion:
        reuse_report = json.loads((args.reuse_ingestion / "report.json").read_text())
        for manifest in reuse_report["datasets"]:
            if manifest["registry_sha256"] != file_hash(args.lock):
                raise ValueError("Cannot reuse a ledger from different source files")
            for name in ["adapters.py", "common.py"]:
                if manifest["converter_sha256"][name] != file_hash(
                    Path(__file__).with_name(name)
                ):
                    raise ValueError(
                        "Cannot reuse a ledger after changing source adapters: " + name
                    )
        shutil.copyfile(
            args.reuse_ingestion / "membership.sqlite", stage / "membership.sqlite"
        )
    db = sqlite3.connect(stage / "membership.sqlite")
    if reuse_report:
        tables = [
            name
            for (name,) in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            if name != "examples"
        ]
        for name in tables:
            db.execute('DROP TABLE "' + name + '"')
        db.execute("UPDATE examples SET eligible=1, split=json_extract(body,'$.split')")
    db.executescript("""
        PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS examples (seq INTEGER PRIMARY KEY, id TEXT UNIQUE, source TEXT,
          split TEXT, groupkey TEXT, key TEXT, inputhash TEXT, signature TEXT, body TEXT, eligible INTEGER);
        CREATE TABLE protected (key TEXT PRIMARY KEY, origin TEXT);
        CREATE TABLE protected_input (key TEXT PRIMARY KEY);
        CREATE TABLE protected_code (key TEXT PRIMARY KEY);
        CREATE TABLE protected_group (key TEXT PRIMARY KEY);
        CREATE TABLE protected_test (key TEXT PRIMARY KEY);
    """)
    near = NearIndex()
    protection = protected_external(args.raw_root, db, near)
    historical = historical_protection(args.historical_root, db)
    stats = defaultdict(Counter)
    if reuse_report:
        for manifest in reuse_report["datasets"]:
            source = manifest["source"]["name"]
            stats[source] = Counter(
                {
                    k: v
                    for k, v in manifest["source_statistics"].items()
                    if k == "raw_decisions_seen" or k.startswith("rejected:")
                }
            )
    for source in sources:
        for file in source["files"]:
            path = args.raw_root / source["name"] / "v0.1/raw" / file["path"]
            if file_hash(path) != file["sha256"]:
                raise ValueError(f"Source lock mismatch: {path}")
        for record in () if reuse_report else records(source, args.raw_root):
            stats[source["name"]]["raw_decisions_seen"] += 1
            if "_reject" in record:
                stats[source["name"]]["rejected:" + record["_reject"]] += 1
                continue
            try:
                if not record.get("_gsm"):
                    validate(record)
            except ValueError as error:
                stats[source["name"]]["rejected:" + str(error)] += 1
                continue
            signature = digest(
                [record["state"], record["question"], sorted(record["options"])]
            )
            # Same question with genuinely different candidates remains a parent
            # alias; identical presentations are removed separately below.
            try:
                db.execute(
                    "INSERT INTO examples VALUES (NULL,?,?,?,?,?,?,?,?,1)",
                    (
                        record["id"],
                        source["name"],
                        record["split"],
                        record["_group"],
                        record["_key"],
                        signature,
                        signature,
                        dumps(record),
                    ),
                )
            except sqlite3.IntegrityError:
                raise ValueError(f"Duplicate source ID: {record['id']}")
        db.commit()
        rejected = sum(
            v for k, v in stats[source["name"]].items() if k.startswith("rejected:")
        )
        print(
            f"Ingested {source['name']}: {stats[source['name']]['raw_decisions_seen']} decisions; {rejected} rejected",
            flush=True,
        )
    db.executescript("""
        CREATE INDEX IF NOT EXISTS examples_key ON examples(key);
        CREATE INDEX IF NOT EXISTS examples_group ON examples(groupkey);
        CREATE INDEX IF NOT EXISTS examples_signature ON examples(signature);
        CREATE INDEX IF NOT EXISTS examples_source_split ON examples(source,split,eligible);
    """)
    # Any native held-out instance owns all exact question/context aliases and
    # all sibling decisions from its premise/document/video/program group.
    mark_excluded(
        db,
        "split='train' AND (key IN (SELECT key FROM examples WHERE split!='train') OR groupkey IN (SELECT groupkey FROM examples WHERE split!='train') OR groupkey IN (SELECT key FROM protected_group) OR key IN (SELECT key FROM protected) OR inputhash IN (SELECT key FROM protected_input))",
        (),
        stats,
        "excluded:heldout_overlap",
    )
    mark_excluded(
        db,
        "source='codeio' AND groupkey IN (SELECT 'codeio:'||key FROM protected_code)",
        (),
        stats,
        "excluded:protected_code",
    )
    mark_excluded(
        db,
        "split='validation' AND (key IN (SELECT key FROM examples WHERE split='test') OR groupkey IN (SELECT groupkey FROM examples WHERE split='test') OR key IN (SELECT key FROM protected_test) OR groupkey IN (SELECT key FROM protected_group))",
        (),
        stats,
        "excluded:validation_test_alias",
    )
    # Prefer source-specific native datasets to the heterogeneous MMLU auxiliary
    # pool; all surviving decisions for a query/context belong to one source.
    db.execute("CREATE TABLE source_order (source TEXT PRIMARY KEY, rank INTEGER)")
    source_priority = {"when2call": 0, "toolret": 2, "mathqa": 3, "mmlu": 4}
    order = sorted(
        sources, key=lambda s: (source_priority.get(s["name"], 1), s["name"])
    )
    db.executemany(
        "INSERT INTO source_order VALUES (?,?)",
        [(s["name"], i) for i, s in enumerate(order)],
    )
    db.execute("""CREATE TABLE owner AS SELECT key,min(rank) AS rank FROM examples
                  JOIN source_order USING(source) WHERE eligible=1 AND split='train' GROUP BY key""")
    db.execute("CREATE UNIQUE INDEX owner_key ON owner(key)")
    db.execute("CREATE UNIQUE INDEX source_rank ON source_order(rank)")
    mark_excluded(
        db,
        "split='train' AND EXISTS (SELECT 1 FROM owner JOIN source_order USING(rank) WHERE owner.key=examples.key AND source_order.source!=examples.source)",
        (),
        stats,
        "excluded:cross_source_alias",
    )
    # Identical inputs/candidate sets with contradictory native answers are
    # quarantined together, rather than arbitrarily choosing the first target.
    conflicts = []
    duplicates = db.execute(
        "SELECT signature FROM examples WHERE eligible=1 GROUP BY signature HAVING count(*)>1"
    ).fetchall()
    for (signature,) in duplicates:
        members = db.execute(
            "SELECT seq,source,body FROM examples WHERE eligible=1 AND signature=?",
            (signature,),
        ).fetchall()
        answers = set()
        for seq, source, body in members:
            r = json.loads(body)
            answers.add(
                r["metadata"]["gold_numeric"]
                if r.get("_gsm")
                else canonical(r["options"][r["target"].index(1.0)])
            )
        if len(answers) > 1:
            conflicts.append(
                {
                    "signature": signature,
                    "members": [(i, s) for i, s, _ in members],
                    "answers": sorted(answers),
                }
            )
            for seq, source, _ in members:
                db.execute("UPDATE examples SET eligible=0 WHERE seq=?", (seq,))
                stats[source]["excluded:contradictory_native_answers"] += 1
    save(stage / "label-conflicts.json", conflicts)
    # Same program/question can occur repeatedly in its source. Keep distinct
    # annotated responses/relevance pairs, remove identical presentations.
    mark_excluded(
        db,
        "seq NOT IN (SELECT min(seq) FROM examples WHERE eligible=1 GROUP BY source,split,signature)",
        (),
        stats,
        "excluded:duplicate_presentation",
    )
    # Audit close MCQ wording against the FULL native held-outs and MMLU-Pro.
    for seq, source, split, body in db.execute(
        "SELECT seq,source,split,body FROM examples WHERE eligible=1"
    ):
        record = json.loads(body)
        if record["domain"] in MCQ_CATEGORIES and split != "train":
            near.add(record["state"], f"{source}:{seq}")
    near_exclusions = []
    for seq, source, body in db.execute(
        "SELECT seq,source,body FROM examples WHERE eligible=1 AND split='train'"
    ):
        record = json.loads(body)
        if record["domain"] in MCQ_CATEGORIES:
            match = near.match(record["state"])
            if match:
                near_exclusions.append((seq, source, match))
    for seq, source, match in near_exclusions:
        db.execute("UPDATE examples SET eligible=0 WHERE seq=?", (seq,))
        stats[source]["excluded:near_heldout"] += 1
    save(
        stage / "near-overlap.json",
        {
            "method": "bottom-four word-trigram candidate sketch; Jaccard >= 0.85",
            "limitation": "Does not establish semantic/paraphrase disjointness.",
            "matches": near_exclusions,
        },
    )
    # Bound the first code view, while keeping the entire original source on disk.
    if args.code_limit:
        groups = [
            g
            for (g,) in db.execute(
                "SELECT DISTINCT groupkey FROM examples WHERE source='codeio' AND eligible=1"
            )
        ]
        selected = sorted(groups, key=lambda g: digest(f"{SEED}:{g}"))[
            : args.code_limit
        ]
        db.execute("CREATE TABLE code_selected (groupkey TEXT PRIMARY KEY)")
        db.executemany("INSERT INTO code_selected VALUES (?)", [(g,) for g in selected])
        mark_excluded(
            db,
            "source='codeio' AND groupkey NOT IN (SELECT groupkey FROM code_selected)",
            (),
            stats,
            "excluded:initial_code_view_budget",
        )
    # Derive dev only where native dev is absent or benchmark-reserved. Whole
    # groups first; target row counts can be exceeded by a group's siblings.
    split_recipes = {}
    for source, target in DERIVED_VALIDATION.items():
        groups = db.execute(
            "SELECT groupkey,count(*) FROM examples WHERE source=? AND split='train' AND eligible=1 GROUP BY groupkey",
            (source,),
        ).fetchall()
        count, chosen = 0, []
        for group, size in sorted(
            groups, key=lambda g: digest(f"{SEED}:validation:{g[0]}")
        ):
            if count >= target:
                break
            chosen.append(group)
            count += size
        for group in chosen:
            db.execute(
                "UPDATE examples SET split='validation' WHERE groupkey=? AND eligible=1",
                (group,),
            )
        split_recipes[source] = {
            "origin": "held out from native training before presentations",
            "target_rows": target,
            "actual_rows": count,
            "groups": len(chosen),
            "seed": SEED,
        }
    # Cross-source validation aliases now own their question as well.
    mark_excluded(
        db,
        "split='train' AND key IN (SELECT key FROM examples WHERE eligible=1 AND split='validation')",
        (),
        stats,
        "excluded:derived_validation_alias",
    )
    # Extend the same approximate MCQ overlap audit to freshly derived dev.
    for seq, body in db.execute(
        "SELECT seq,body FROM examples WHERE eligible=1 AND split='validation'"
    ):
        record = json.loads(body)
        if record["domain"] in MCQ_CATEGORIES:
            near.add(record["state"], f"derived-validation:{seq}")
    extra_near = []
    for seq, source, body in db.execute(
        "SELECT seq,source,body FROM examples WHERE eligible=1 AND split='train'"
    ):
        record = json.loads(body)
        if record["domain"] in MCQ_CATEGORIES:
            match = near.match(record["state"])
            if match:
                extra_near.append((seq, source, match))
    for seq, source, match in extra_near:
        db.execute("UPDATE examples SET eligible=0 WHERE seq=?", (seq,))
        stats[source]["excluded:near_derived_validation"] += 1
    near_exclusions.extend(extra_near)
    save(
        stage / "near-overlap.json",
        {
            "method": "bottom-four word-trigram candidate sketch; Jaccard >= 0.85",
            "limitation": "Does not establish semantic/paraphrase disjointness.",
            "matches": near_exclusions,
        },
    )
    db.commit()
    # GSM8K choices are generated AFTER ownership, within each split separately.
    protocol_root = REPO / "datasets/gsm8k/decision-index-source"
    protocol_manifest = json.loads((protocol_root / "manifest.json").read_text())
    for relative, expected in protocol_manifest["files_sha256"].items():
        if relative.endswith("test-00000-of-00001.parquet"):
            continue
        if file_hash(protocol_root / relative) != expected:
            raise ValueError("GSM protocol source changed: " + relative)
    sys.path.insert(0, str(protocol_root))
    gsm = importlib.import_module("decision_index.suite.build.gsm8k_v3")
    from decimal import Decimal

    manifest_rows = []
    for source in sources:
        name = source["name"]
        names = ["gsm8k-4choice", "gsm8k-10choice"] if name == "gsm8k" else [name]
        for output_name in names:
            output = stage / "decisions" / output_name / "v0.1"
            split_files = {}
            for split in ["train", "validation", "test"]:
                if name == "gsm8k" and split == "test":
                    continue  # Existing protocol test files stay untouched.
                query = "SELECT body FROM examples WHERE source=? AND split=? AND eligible=1 ORDER BY seq"
                if name == "gsm8k":
                    values = [
                        json.loads(b) for (b,) in db.execute(query, (name, split))
                    ]
                    golds = {
                        int(r["metadata"]["native_id"]): Decimal(
                            r["metadata"]["gold_numeric"]
                        )
                        for r in values
                    }
                    sets = gsm.option_sets(golds)
                    count = 4 if output_name.endswith("4choice") else 10
                    for record in values:
                        i = int(record["metadata"]["native_id"])
                        options, correct = gsm.options(
                            golds[i], sets[i][count], count, i
                        )
                        record["id"] = output_name + ":" + str(i)
                        record["options"], record["target"] = (
                            options,
                            [float(j == correct) for j in range(count)],
                        )
                        record["split"] = split
                        record["metadata"].update(
                            distractor_method=gsm.METHOD,
                            protocol_seed=gsm.SEED,
                            answer_group=sets[i]["group"],
                            parent_id="gsm8k:" + str(i),
                            distractor_parent_ids=sets[i]["members"],
                            candidate_pool_split=split,
                        )
                    iterator = iter(values)
                else:

                    def iterator_rows():
                        for (body,) in db.execute(query, (name, split)):
                            record = json.loads(body)
                            record["split"] = split
                            yield record

                    iterator = iterator_rows()
                split_files[split] = write_jsonl(
                    output / (split + ".jsonl.gz"), iterator
                )
            manifest = {
                "name": output_name,
                "version": "0.1",
                "category": source["category"],
                "source": source,
                "split_files": split_files,
                "source_statistics": dict(stats[name]),
                "validation_recipe": split_recipes.get(
                    name, {"origin": "native validation"}
                ),
                "registry_sha256": file_hash(args.lock),
                "converter_sha256": {
                    p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")
                },
                "schema": "Groundbeef v3-compatible decision keys; metadata is excluded from model input",
                "notes": {
                    "native_choice_order": "preserved when supplied",
                    "synthesized_choice_order": "one deterministic presentation, not repeated augmentation",
                    "labels": "native hard labels / explicit native preferences, never teacher probabilities",
                    "near_overlap_audit": "Approximate trigram audit; not proof against paraphrases",
                    "historical_protection": "exact complete input+option fingerprints only",
                    "gsm_views": "same parent problem; choose one view per training presentation",
                    "codeio": "source-executed I/O not independently reexecuted; variable-output subset",
                    "validation_targets": "do not select model architecture using benchmark test scores",
                },
            }
            if name == "gsm8k":
                manifest["gsm_protocol"] = protocol_manifest
                manifest["existing_test"] = preserved[
                    str(REPO / "datasets" / output_name / "test.jsonl")
                ]
            save(output / "manifest.json", manifest)
            manifest_rows.append(manifest)
            print(
                "Prepared",
                output_name,
                {k: v["rows"] for k, v in split_files.items()},
                flush=True,
            )
    # Mechanical ownership audit across the complete standalone collection.
    overlap = db.execute(
        "SELECT count(*) FROM (SELECT key FROM examples WHERE eligible=1 GROUP BY key HAVING count(DISTINCT split)>1)"
    ).fetchone()[0]
    group_overlap = db.execute(
        "SELECT count(*) FROM (SELECT groupkey FROM examples WHERE eligible=1 GROUP BY groupkey HAVING count(DISTINCT split)>1)"
    ).fetchone()[0]
    # Native val/test aliases are protected together, but must not appear in train.
    train_overlap = db.execute(
        "SELECT count(*) FROM examples a WHERE a.eligible=1 AND a.split='train' AND EXISTS (SELECT 1 FROM examples b WHERE b.eligible=1 AND b.split!='train' AND (a.key=b.key OR a.groupkey=b.groupkey))"
    ).fetchone()[0]
    if train_overlap:
        raise ValueError("Train/held-out ownership audit failed")
    for path, expected in preserved.items():
        if file_hash(path) != expected:
            raise ValueError("Existing test was modified: " + path)
    raw_only = [
        s["name"]
        for s in lock["sources"]
        if s not in sources and s["adapter"] != "protected"
    ]
    report = {
        "version": "0.1",
        "seed": SEED,
        "datasets": manifest_rows,
        "protected_external_counts": protection,
        "historical_heldouts": historical,
        "preserved_existing_test_sha256": preserved,
        "raw_only_pending_conversion": raw_only,
        "train_heldout_overlaps": train_overlap,
        "native_val_test_alias_keys": overlap,
        "native_val_test_alias_groups": group_overlap,
        "near_matches": len(near_exclusions),
        "elapsed_seconds": time.monotonic() - started,
        "contradictory_presentation_groups": len(conflicts),
        "combined_groundbeef_created": False,
    }
    if args.reuse_ingestion:
        report["reused_ingestion"] = {
            "path": str(args.reuse_ingestion),
            "ledger_sha256": file_hash(args.reuse_ingestion / "membership.sqlite"),
            "adapters_and_source_lock_unchanged": True,
        }
    save(stage / "report.json", report)
    # Preserve the source/split/parent ledger for future selection and growth.
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()
    for name in output_names:
        destination = args.output_root / name / "v0.1"
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.rename(stage / "decisions" / name / "v0.1", destination)
    audit_root = args.output_root / "source-pool-v0.1"
    if audit_root.exists():
        raise ValueError("Audit root exists")
    audit_root.mkdir(parents=True)
    for filename in [
        "membership.sqlite",
        "near-overlap.json",
        "label-conflicts.json",
        "report.json",
    ]:
        shutil.move(str(stage / filename), audit_root / filename)
    print("Complete:", audit_root / "report.json", flush=True)


if __name__ == "__main__":
    main()
