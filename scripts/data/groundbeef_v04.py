"""Assemble Groundbeef v0.4 with provenance, protected parents and fixed panels."""

import argparse
import copy
import heapq
import json
import sqlite3
from collections import Counter, defaultdict, deque
from pathlib import Path

from common import digest, file_hash, identity_keys, jsonl, key
from groundbeef import (
    aliases,
    cap_group,
    decorate,
    prompt,
    save,
    state_object,
    stats,
    validate,
    visible_hash,
    write_rows,
)

SEED = 20261013


def recipe_key(r):
    return r["metadata"]["v04_source_view"]


def original_aliases(r):
    result = aliases(r)
    state = state_object(r["state"])
    # Empty state is a prompt layout, not an example identity. Native MCQs
    # move their actual question into `question`; protect that text instead.
    result.difference_update(key(x) for x in ["{}", "[]", '""', ""])
    if state in ({}, [], "") and r.get("question"):
        result.add(key(r["question"]))
    if isinstance(state, dict) and isinstance(state.get("code"), str):
        result.add(key(state["code"]))
    return result


def prepare(raw, view, path, formatted=False):
    base = (
        view.removesuffix("-fourchoice").removesuffix("-tenchoice")
        if view.startswith("mmlu-")
        else view
    )
    if formatted:
        r = copy.deepcopy(raw)
        r["_original_sha256"] = digest(raw)
        r["_aliases"] = original_aliases(raw)
        r["_input_hash"] = visible_hash(raw)
    else:
        r = decorate(raw, base, raw["split"])
    m = r["metadata"]
    m["v04_source_view"] = view
    m["v04_parent_file"] = str(path)
    m.setdefault("category", r["domain"])
    m["original_row_sha256"] = digest(raw)
    m.setdefault("loss_weight", 1.0)
    m.setdefault("label_basis", m.get("label_origin", "native_label"))
    m.setdefault(
        "semantic_class",
        "answer_position:"
        + str(max(range(len(r["target"])), key=r["target"].__getitem__)),
    )
    m.setdefault("task_type", m.get("subcategory", r["family"]))
    m.setdefault("retained_from_v02", False)
    m.setdefault("prior_training_exposure", False)
    m.setdefault("prior_validation_monitoring", False)
    if view == "mmlu-fourchoice":
        m["dataset_source"] = view
    if view == "codeio-python":
        s = state_object(r["state"])
        call = m["execution"]["input"]
        if not call.startswith("f(") or not call.endswith(")"):
            raise ValueError("Python invocation")
        r["state"] = {
            "code": s["code"],
            "input": call[2:-1],
            "task": "Python function output selection",
        }
        r["question"] = (
            "Choose the correct output of f called with the supplied input arguments. Candidates are Python literal values."
        )
        m["input_transform"] = "groundbeef-v0.4/crux-style-python-arguments"
    r["_aliases"].update(original_aliases(r))
    r["_native_parent"] = r["group_id"]
    r["_architecture_keys"] = identity_keys(raw) | identity_keys(r)
    return r


def stratum(r):
    m = r["metadata"]
    gold = max(range(len(r["target"])), key=r["target"].__getitem__)
    return (
        m["task_type"],
        m["semantic_class"],
        len(r["options"]),
        m.get("selection_topic", ""),
        gold,
    )


def select(rows, count, name, tokenizer, used, excluded, max_tokens=4096, paired=None):
    buckets = defaultdict(list)
    for r in rows:
        if paired is not None and r["group_id"] not in paired:
            continue
        buckets[stratum(r)].append(r)
    queues = deque(
        deque(sorted(rr, key=lambda r: digest([SEED, name, r["id"]])))
        for _, rr in sorted(buckets.items())
    )
    chosen = []
    while queues and len(chosen) < count:
        batch = []
        while queues and len(batch) < min(128, count - len(chosen)):
            q = queues.popleft()
            batch.append(q.popleft())
            if q:
                queues.append(q)
        encodings = tokenizer.encode_batch(
            [prompt(r) for r in batch], add_special_tokens=False
        )
        for r, enc in zip(batch, encodings):
            if len(chosen) >= count:
                break
            group = cap_group(r)[0]  # SWE questions, not whole repository, own caps.
            pair = paired is not None and r["group_id"] in paired
            if r["id"] in used["ids"] or visible_hash(r) in used["inputs"]:
                excluded[name + ":duplicate"] += 1
                continue
            owner = name.split("/")[0]
            source = r["metadata"]["dataset_source"]
            cap = (
                1
                if source.startswith(("gsm8k", "mmlu", "codeio")) or source == "swe_qa"
                else 3
                if source.startswith("decider_")
                else 4
            )
            wrong_owner = group in used["owners"] and used["owners"][group] != owner
            alias_conflict = any(
                used["alias_groups"][a] - {group} for a in r["_aliases"]
            )
            if (
                wrong_owner
                or not pair
                and (used["capcounts"][group] >= cap or alias_conflict)
            ):
                excluded[name + ":parent_or_alias_cap"] += 1
                continue
            if len(enc.ids) > max_tokens:
                excluded[name + ":over_token_budget"] += 1
                continue
            validate(r)
            m = r["metadata"]
            m["input_tokens"] = len(enc.ids)
            chosen.append(r)
            used["ids"].add(r["id"])
            used["inputs"].add(visible_hash(r))
            used["groups"].add(group)
            used["aliases"].update(r["_aliases"])
            used["owners"][group] = owner
            used["capcounts"][group] += 1
            for a in r["_aliases"]:
                used["alias_groups"][a].add(group)
    if len(chosen) != count:
        raise ValueError(f"{name}: only {len(chosen)}/{count}; " + str(dict(excluded)))
    print(f"{name}: {len(chosen)}", flush=True)
    return chosen


def empty_used():
    return {
        "ids": set(),
        "groups": set(),
        "inputs": set(),
        "aliases": set(),
        "owners": {},
        "capcounts": Counter(),
        "alias_groups": defaultdict(set),
    }


def build(args):
    from tokenizers import Tokenizer

    if args.output.exists():
        raise ValueError(
            "Use a new output directory; release creation never overwrites a version"
        )
    recipe = json.loads(args.recipe.read_text())
    assert sum(recipe["train"].values()) == 64000
    tok = Tokenizer.from_file(str(args.tokenizer))
    excluded = Counter()
    sources = []
    pools = defaultdict(list)
    ledger = json.loads((args.architecture / "exclusions.json").read_text())
    arch_ids = set(ledger["blocked_training_ids"])
    arch_keys = set(ledger["keys"])
    db = sqlite3.connect(
        "file:"
        + str(args.source_root / "source-pool-v0.1/membership.sqlite")
        + "?mode=ro",
        uri=True,
    )
    protected = {k for (k,) in db.execute("SELECT key FROM protected")}
    native_keys = {
        i: k for i, k in db.execute("SELECT id,key FROM examples WHERE eligible=1")
    }
    held = empty_used()
    exposure = empty_used()
    historical = []
    for root in [args.history / "v0.1", args.history / "v0.2", args.parent, args.aux]:
        for p in sorted(root.glob("*.jsonl.gz")):
            split = p.name.split(".")[0]
            if split not in {"train", "validation", "calibration", "test", "ood"}:
                continue
            target = exposure if split == "train" else held
            for r in jsonl(p):
                target["ids"].add(r["id"])
                target["groups"].add(cap_group(r)[0])
                target["aliases"].update(original_aliases(r))
                target["inputs"].add(visible_hash(r))
            historical.append({"file": str(p), "sha256": file_hash(p), "role": split})
    for p in sorted((args.history / "v0.1/heldout_pools").glob("*.jsonl.gz")):
        for r in jsonl(p):
            held["ids"].add(r["id"])
            held["groups"].add(cap_group(r)[0])
            held["aliases"].update(original_aliases(r))
            held["inputs"].add(visible_hash(r))
        historical.append(
            {
                "file": str(p),
                "sha256": file_hash(p),
                "role": "complete historical heldout pool",
            }
        )
    legacy = {
        s
        for s in recipe["train"]
        if s in {"open_jev", "distill_v3", "swe_qa", "helpsteer2"}
    }
    auxiliary = set(json.loads((args.aux / "recipe.json").read_text())["train_quotas"])
    entries = []
    for view in recipe["train"]:
        if view in legacy or view in auxiliary:
            continue
        for split in ["train", "validation"]:
            formatted = False
            if view == "when2call-fourway":
                path = args.views / "when2call-fourway-v0.2" / (split + ".jsonl.gz")
                formatted = True
            elif view == "codeio-python":
                path = args.views / "codeio-python-v0.2" / (split + ".jsonl.gz")
                formatted = True
            elif view.startswith("mmlu-"):
                folder = (
                    "mmlu-numeric-menus-v0.1"
                    if view == "mmlu-tenchoice"
                    else "mmlu-menus-v0.1"
                )
                path = (
                    args.views
                    / folder
                    / (split + "-" + view.removeprefix("mmlu-") + ".jsonl.gz")
                )
                formatted = True
            elif view == "toolace":
                path = args.parent / "standalone/toolace/v0.1" / (split + ".jsonl.gz")
            else:
                path = args.source_root / view / "v0.1" / (split + ".jsonl.gz")
            entries.append((view, split, path, formatted))
    for root, names in [(args.parent, legacy), (args.aux, auxiliary)]:
        for split in ["train", "validation"]:
            path = root / (split + ".jsonl.gz")
            for raw in jsonl(path):
                view = raw["metadata"]["dataset_source"]
                if view in names:
                    r = prepare(raw, view, path, True)
                    pools[view, split].append(r)
            sources.append(
                {
                    "file": str(path),
                    "sha256": file_hash(path),
                    "already_audited_parent": True,
                }
            )
    for view, split, path, formatted in entries:
        if not path.exists():
            raise FileNotFoundError(path)
        reservoir = defaultdict(list)
        limit = (
            3000 if view == "when2call-fourway" else 2000 if view == "anli" else 1000
        )
        for raw in jsonl(path):
            r = prepare(raw, view, path, formatted)
            if raw["id"] in native_keys:
                r["_aliases"].add(native_keys[raw["id"]])
            if split == "train":
                bucket = reservoir[stratum(r)]
                entry = (
                    -int(digest([SEED, "candidate-pool", r["id"]]), 16),
                    r["id"],
                    r,
                )
                if len(bucket) < limit:
                    heapq.heappush(bucket, entry)
                elif entry[:2] > bucket[0][:2]:
                    heapq.heapreplace(bucket, entry)
            else:
                pools[view, split].append(r)
        if split == "train":
            pools[view, split] = [
                r for bucket in reservoir.values() for _, _, r in bucket
            ]
        sources.append(
            {"view": view, "split": split, "file": str(path), "sha256": file_hash(path)}
        )
        print(f"Loaded {view}/{split}: {len(pools[view, split])}", flush=True)
    # Filter before transformations can hide a source request/native identity.
    for (view, split), rows in list(pools.items()):
        clean = []
        for r in rows:
            group = cap_group(r)[0]
            if split == "train":
                if r["id"] in arch_ids or r["_architecture_keys"] & arch_keys:
                    excluded[view + ":architecture_experiment_exclusion"] += 1
                    continue
                if r["_aliases"] & protected:
                    excluded[view + ":native_benchmark_protection"] += 1
                    continue
                if (
                    r["id"] in held["ids"]
                    or group in held["groups"]
                    or r["_aliases"] & held["aliases"]
                    or r["_input_hash"] in held["inputs"]
                ):
                    excluded[view + ":historical_or_parent_holdout"] += 1
                    continue
                r["metadata"]["prior_parent_train_membership"] = (
                    r["id"] in exposure["ids"]
                )
            else:
                if (
                    r["id"] in exposure["ids"]
                    or group in exposure["groups"]
                    or r["_aliases"] & exposure["aliases"]
                    or r["_input_hash"] in exposure["inputs"]
                ):
                    excluded[view + ":previous_training_exposure_in_development"] += 1
                    continue
                r["metadata"]["prior_validation_monitoring"] = r["id"] in held["ids"]
            clean.append(r)
        pools[view, split] = clean
    # Every source-owned development parent stays out of training, even if its
    # row does not enter this small validation/calibration panel.
    dev_groups = {
        cap_group(r)[0] for (v, s), rr in pools.items() if s == "validation" for r in rr
    }
    dev_aliases = {
        a
        for (v, s), rr in pools.items()
        if s == "validation"
        for r in rr
        for a in r["_aliases"]
    }
    for (view, split), rows in list(pools.items()):
        if split == "train":
            clean = [
                r
                for r in rows
                if cap_group(r)[0] not in dev_groups and not r["_aliases"] & dev_aliases
            ]
            excluded[view + ":complete_development_ownership"] += len(rows) - len(clean)
            pools[view, split] = clean
    selected = {}
    used_dev = empty_used()
    # Calibration comes from separate development parents; legacy panels are
    # retained as validation anchors, never squeezed to supply calibration.
    cal_views = [
        v for v in recipe["validation"] if v not in legacy and v != "gsm8k-10choice"
    ]
    scale = recipe["calibration_rows"] / sum(recipe["validation"][v] for v in cal_views)
    cq = {v: int(recipe["validation"][v] * scale) for v in cal_views}
    for v in sorted(cq, key=lambda v: (-(recipe["validation"][v] * scale - cq[v]), v))[
        : recipe["calibration_rows"] - sum(cq.values())
    ]:
        cq[v] += 1
    selected["calibration"] = []
    for view, count in cq.items():
        rr = select(
            pools[view, "validation"],
            count,
            "calibration/" + view,
            tok,
            used_dev,
            excluded,
        )
        for r in rr:
            r["split"] = "calibration"
        selected["calibration"] += rr
    selected["validation"] = []
    paired = set()
    for view, count in recipe["validation"].items():
        rr = select(
            pools[view, "validation"],
            count,
            "validation/" + view,
            tok,
            used_dev,
            excluded,
            paired=paired if view == "gsm8k-10choice" else None,
        )
        for r in rr:
            r["split"] = "validation"
        if view == "gsm8k-4choice":
            paired.update(r["group_id"] for r in rr)
        selected["validation"] += rr
    selected["train"] = []
    used_train = empty_used()
    # MMLU auxiliary contains variants of the same question under different
    # IDs. Protect text similarity as well as exact/native parent ownership.
    from prepare import NearIndex

    near = NearIndex()

    def mcq_text(r):
        state = state_object(r["state"])
        text = state.get("question") if isinstance(state, dict) else state
        return text or r["question"]

    mcq_categories = {"math", "science", "knowledge", "commonsense"}
    for split in ["validation", "calibration"]:
        for r in selected[split]:
            if r["metadata"]["category"] in mcq_categories:
                near.add(mcq_text(r), r["id"])
    for view, count in recipe["train"].items():
        candidates = []
        for r in pools[view, "train"]:
            if r["metadata"]["category"] in mcq_categories and near.match(mcq_text(r)):
                excluded[view + ":near_development_question"] += 1
                continue
            candidates.append(r)
        rr = select(candidates, count, "train/" + view, tok, used_train, excluded)
        for r in rr:
            r["split"] = "train"
        selected["train"] += rr
    full_count = sum(recipe["validation"].values())
    scale = recipe["quick_validation_rows"] / full_count
    qq = {v: int(n * scale) for v, n in recipe["validation"].items()}
    for v in sorted(qq, key=lambda v: (-(recipe["validation"][v] * scale - qq[v]), v))[
        : recipe["quick_validation_rows"] - sum(qq.values())
    ]:
        qq[v] += 1
    quick = []
    quick_used = empty_used()
    for view, count in qq.items():
        # Quick validation is a subset of full; paired GSM presentations may
        # legitimately share one problem in this diagnostic subset too.
        rows = [r for r in selected["validation"] if recipe_key(r) == view]
        quick += select(
            rows,
            count,
            "quick/" + view,
            tok,
            quick_used,
            excluded,
            paired={r["group_id"] for r in rows} if view == "gsm8k-10choice" else None,
        )
    selected["validation-quick"] = quick
    args.output.mkdir(parents=True)
    files = []
    statistics = {}
    members = []
    for split, rows in selected.items():
        rows.sort(key=lambda r: digest([SEED, "final-order", split, r["id"]]))
        files.append(write_rows(args.output / (split + ".jsonl.gz"), rows))
        statistics[split] = stats(rows)
        if split == "validation-quick":
            continue
        for r in rows:
            clean = {k: v for k, v in r.items() if not k.startswith("_")}
            m = r["metadata"]
            members.append(
                {
                    "id": r["id"],
                    "split": split,
                    "group_id": r["group_id"],
                    "cap_group": cap_group(r)[0],
                    "source": m["dataset_source"],
                    "source_view": recipe_key(r),
                    "category": m["category"],
                    "task": m["task_type"],
                    "semantic_class": m["semantic_class"],
                    "source_repo": m.get("source_repo", m.get("repo")),
                    "source_revision": m.get("source_revision", m.get("revision")),
                    "native_id": m.get("native_id", m.get("upstream_index", r["id"])),
                    "parent_file": m["v04_parent_file"],
                    "original_row_sha256": r["_original_sha256"],
                    "row_sha256": digest(clean),
                    "original_input_sha256": r["_input_hash"],
                    "input_sha256": visible_hash(r),
                    "alias_keys": sorted(r["_aliases"]),
                    "architecture_identity_keys": sorted(r["_architecture_keys"]),
                    "label_basis": m["label_basis"],
                    "input_transform": m.get("input_transform", "preserved_parent"),
                    "input_tokens": m["input_tokens"],
                    "prior_parent_train_membership": m.get(
                        "prior_parent_train_membership", False
                    ),
                    "prior_validation_monitoring": m.get(
                        "prior_validation_monitoring", False
                    ),
                }
            )
    files.append(write_rows(args.output / "membership.jsonl.gz", members))
    save(args.output / "recipe.json", recipe)
    save(args.output / "statistics.json", statistics)
    save(
        args.output / "sources.json",
        {
            "source_files": sources,
            "historical_files": historical,
            "source_pool_db_sha256": file_hash(
                args.source_root / "source-pool-v0.1/membership.sqlite"
            ),
            "architecture_exclusions_sha256": file_hash(
                args.architecture / "exclusions.json"
            ),
            "source_views": {
                p.parent.name: json.loads(p.read_text())
                for p in args.views.glob("*/manifest.json")
            },
        },
    )
    save(
        args.output / "training_weights.json",
        {"class_power": 0.0, "class_cap": 2.0, "sources": {}, "tasks": {}},
    )
    panel = json.loads((args.panel / "manifest.json").read_text())
    for e in panel["entries"]:
        if file_hash(args.panel / e["file"]) != e["sha256"]:
            raise ValueError("Protected test changed")
    save(
        args.output / "manifest.json",
        {
            "dataset": "groundbeef",
            "version": "0.4",
            "training_rows": 64000,
            "seed": SEED,
            "files": files,
            "max_input_tokens": 4096,
            "tokenizer_sha256": file_hash(args.tokenizer),
            "recipe_sha256": file_hash(args.output / "recipe.json"),
            "sources_sha256": file_hash(args.output / "sources.json"),
            "builder_sha256": file_hash(__file__),
            "excluded": dict(excluded),
            "calibration_quotas": cq,
            "quick_quotas": qq,
            "external_evaluation": {
                "manifest_sha256": file_hash(args.panel / "manifest.json"),
                "tracks": len(panel["entries"]),
            },
            "quality_limits": [
                "Synthetic When2Call negatives are templates with retained native preference labels.",
                "MMLU larger menus are restricted to scalar questions with rationally distinct negatives; not general MMLU-Pro training.",
                "Knowledge topic tags are keyword heuristics, not native subject labels.",
                "Code gold independently matches source JSON; negative output types are derived from JSON.",
                "Auxiliary teacher consistency is not independent human truth.",
                "Legacy development anchors have already been monitored; this is not a blind test.",
            ],
        },
    )
    print("Created Groundbeef v0.4:", args.output, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output", type=Path, default=Path("build/groundbeef/.v0.4-staging")
    )
    p.add_argument(
        "--recipe",
        type=Path,
        default=Path(__file__).with_name("groundbeef_v04_recipe.json"),
    )
    p.add_argument("--source-root", type=Path, default=Path("datasets/decisions"))
    p.add_argument("--parent", type=Path, default=Path("build/groundbeef/v0.3"))
    p.add_argument(
        "--aux", type=Path, default=Path("build/groundbeef/decider-supplement-v0.1")
    )
    p.add_argument(
        "--architecture", type=Path, default=Path("build/groundbeef/architecture-v0.1")
    )
    p.add_argument("--views", type=Path, default=Path("build/groundbeef/v0.4-sources"))
    p.add_argument("--panel", type=Path, default=Path("datasets/evaluation-panel"))
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    build(p.parse_args())


if __name__ == "__main__":
    main()
