#!/usr/bin/env python3
"""Verify the mixture against its frozen parents, native labels and holdouts."""

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from common import REPO, digest, file_hash, jsonl
from groundbeef import aliases, prompt, quick_quotas, validate, visible_hash
from prepare import NearIndex


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--history-root", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--source-root", type=Path, default=REPO / "datasets/decisions")
    args = p.parse_args()
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    root = args.dataset
    manifest = json.loads((root / "manifest.json").read_text())
    recipe = json.loads((root / "recipe.json").read_text())
    require(
        file_hash(args.tokenizer) == manifest["tokenizer_sha256"], "Tokenizer identity"
    )
    require(
        file_hash(root / "recipe.json") == manifest["recipe_sha256"], "Recipe identity"
    )
    require(
        file_hash(root / "sources.json") == manifest["sources_sha256"],
        "Source ledger identity",
    )
    sources = json.loads((root / "sources.json").read_text())
    splits = {}
    for entry in manifest["files"]:
        path = root / entry["file"]
        require(file_hash(path) == entry["sha256"], "Changed file: " + entry["file"])
        rows = list(jsonl(path))
        require(len(rows) == entry["rows"], "Wrong file count: " + entry["file"])
        splits[entry["file"].split(".")[0]] = rows
    for split in ["train", "validation", "calibration"]:
        require(len(splits[split]) == sum(recipe[split].values()), split + " total")
        counts = Counter(
            ("retained:" if r["metadata"]["retained_from_v02"] else "")
            + r["metadata"]["dataset_source"]
            for r in splits[split]
        )
        require(dict(counts) == recipe[split], split + " mixture quotas")
    members = {(r["split"], r["id"]): r for r in splits["membership"]}
    require(len(members) == 84500, "Unique membership identities")
    groups, hashes, identities, ledger_aliases = {}, {}, {}, {}
    selected_originals = {}
    for split in ["train", "validation", "calibration"]:
        groups[split], hashes[split], identities[split], ledger_aliases[split] = (
            set(),
            set(),
            set(),
            set(),
        )
        rows = splits[split]
        for start in range(0, len(rows), 256):
            batch = rows[start : start + 256]
            encodings = tokenizer.encode_batch(
                [prompt(r) for r in batch], add_special_tokens=False
            )
            for r, encoded in zip(batch, encodings):
                validate(r)
                require(
                    len(encoded.ids)
                    == r["metadata"]["input_tokens"]
                    <= manifest["max_input_tokens"],
                    "Full token length: " + r["id"],
                )
                member = members[split, r["id"]]
                require(
                    member["row_sha256"] == digest(r), "Row membership hash: " + r["id"]
                )
                require(
                    member["input_sha256"] == visible_hash(r),
                    "Input membership hash: " + r["id"],
                )
                require(r["id"] not in identities[split], "Duplicate ID")
                identities[split].add(r["id"])
                groups[split].add(r["group_id"])
                hashes[split].add(member["input_sha256"])
                ledger_aliases[split].update(member["alias_keys"])
                source = r["metadata"]["dataset_source"]
                parent_key = (
                    "retained:" + source
                    if r["metadata"]["retained_from_v02"]
                    else source,
                    r["id"],
                )
                require(
                    parent_key not in selected_originals,
                    "Same source example in multiple splits",
                )
                selected_originals[parent_key] = (r, member)
                if source == "distill_v3":
                    require(
                        r["source"] == "yuri_v3"
                        and r["target"] == r["target_teacher"]
                        and "target_native" not in r,
                        "Teacher/native separation",
                    )
                else:
                    require(r["target_native"] == r["target"], "Native labels changed")
        print("Verified tokens and membership: " + split, flush=True)
    overlap = {}
    for a, b in [
        ("train", "validation"),
        ("train", "calibration"),
        ("validation", "calibration"),
    ]:
        overlap[a + "/" + b] = {
            "ids": len(identities[a] & identities[b]),
            "groups": len(groups[a] & groups[b]),
            "inputs": len(hashes[a] & hashes[b]),
            "original_aliases": len(ledger_aliases[a] & ledger_aliases[b]),
        }
        require(
            not any(overlap[a + "/" + b].values()),
            "Cross-split overlap: " + a + "/" + b,
        )
    # Read parent releases afresh. Their labels must be copied, not reconstructed
    # from an argmax or inferred from the source's English response.
    checked, parent_hashes = Counter(), {}
    near_dev = NearIndex()
    original_mcq_train = []
    parent_paths = []
    for entry in sources["standalone_sources"]:
        name = entry["name"]
        path = args.source_root / name / "v0.1"
        require(
            file_hash(path / "manifest.json") == entry["manifest_sha256"],
            "Standalone manifest changed",
        )
        for split in ["train", "validation"]:
            parent_paths.append((name, path / (split + ".jsonl.gz")))
    for split in ["train", "validation"]:
        parent_paths.append(
            ("toolace", root / "standalone/toolace/v0.1" / (split + ".jsonl.gz"))
        )
        parent_paths.append(
            ("retained", args.history_root / "v0.2" / (split + ".jsonl.gz"))
        )
    for name, path in parent_paths:
        for original in jsonl(path):
            source = (
                original["metadata"]["dataset_source"] if name == "retained" else name
            )
            lookup = (
                "retained:" + source if name == "retained" else name,
                original["id"],
            )
            if lookup not in selected_originals:
                continue
            r, member = selected_originals[lookup]
            require(
                digest(original) == member["original_row_sha256"],
                "Original source row changed: " + r["id"],
            )
            expected_target = (
                list(reversed(original["target"]))
                if source == "ragtruth"
                else original["target"]
            )
            require(
                expected_target == r["target"],
                "Source answer label changed: " + r["id"],
            )
            if source == "ragtruth":
                require(
                    r["metadata"]["source_candidate_mapping"] == [1, 0]
                    and r["options"][0]
                    == "The response contains content that is not supported by the context in the prompt."
                    and r["options"][1]
                    == "All content of the response is supported by the context in the prompt.",
                    "RAG benchmark true/false schema",
                )
            if source == "anli":
                require(
                    [str(o).lower() for o in original["options"]] == r["options"],
                    "NLI native order",
                )
            elif source != "ragtruth":
                require(
                    original["options"] == r["options"],
                    "Candidate order/contents changed: " + r["id"],
                )
            if r["metadata"]["category"] in {
                "math",
                "science",
                "knowledge",
                "commonsense",
            }:
                if r["split"] == "train":
                    original_mcq_train.append(original)
                else:
                    near_dev.add(original["state"], r["id"])
            checked[source] += 1
        parent_hashes[name + "/" + path.name] = file_hash(path)
    require(sum(checked.values()) == 84500, "Missing parent row identities")
    near_matches = [
        (r["id"], near_dev.match(r["state"]))
        for r in original_mcq_train
        if near_dev.match(r["state"])
    ]
    require(not near_matches, "Near development overlap: " + str(near_matches[:10]))
    # Validate every historical split/group, including unselected held-out pools.
    old_train = {"groups": set(), "aliases": set(), "inputs": set()}
    old_heldout = {"groups": set(), "aliases": set(), "inputs": set()}
    old_alias_members = defaultdict(set)
    for entry in sources["historical_files"]:
        path = args.history_root / entry["version"] / (entry["split"] + ".jsonl.gz")
        require(file_hash(path) == entry["sha256"], "Historical file changed")
        target = (
            old_train
            if entry["split"] == "train" and "heldout_pools" not in entry["version"]
            else old_heldout
        )
        for r in jsonl(path):
            target["groups"].add(r["group_id"])
            target["aliases"].update(aliases(r))
            target["inputs"].add(visible_hash(r))
            if target is old_train:
                for alias in aliases(r):
                    old_alias_members[alias].add((entry["version"], r["id"]))
    require(
        not groups["train"] & old_heldout["groups"],
        "Historical held-out parent trained",
    )
    require(
        not ledger_aliases["train"] & old_heldout["aliases"],
        "Historical held-out input trained",
    )
    for split in ["validation", "calibration"]:
        require(
            not groups[split] & old_train["groups"],
            "Historical training parent in development",
        )
        require(
            not ledger_aliases[split] & old_train["aliases"],
            "Historical training input in development",
        )
    new_source_alias_matches = []
    for r in splits["train"]:
        if r["metadata"]["retained_from_v02"]:
            continue
        member = members["train", r["id"]]
        require(
            member["original_input_sha256"] not in old_train["inputs"]
            and member["input_sha256"] not in old_train["inputs"],
            "New-source row has an unmarked identical historical input",
        )
        matches = set(member["alias_keys"]) & old_train["aliases"]
        if matches:
            new_source_alias_matches.append(
                {
                    "id": r["id"],
                    "source": member["source"],
                    "alias_keys": sorted(matches),
                    "historical_training_rows": sorted(
                        {v for k in matches for v in old_alias_members[k]}
                    ),
                    "scope": "Shared request text; the complete tool/context input differs. Not a new blind question.",
                }
            )
    db = sqlite3.connect(
        "file:"
        + str(args.source_root / "source-pool-v0.1/membership.sqlite")
        + "?mode=ro",
        uri=True,
    )
    protected = {k for (k,) in db.execute("SELECT key FROM protected")}
    require(
        not ledger_aliases["train"] & protected,
        "Protected raw benchmark alias in train",
    )
    gsm = {
        s: [
            r for r in splits[s] if r["metadata"]["dataset_source"].startswith("gsm8k-")
        ]
        for s in ["train", "validation"]
    }
    require(
        len({r["group_id"] for r in gsm["train"]}) == 6000,
        "GSM train has repeated presentations",
    )
    paired = defaultdict(set)
    for r in gsm["validation"]:
        paired[r["group_id"]].add(r["metadata"]["dataset_source"])
    require(
        len(paired) == 200 and all(len(v) == 2 for v in paired.values()),
        "GSM development pairs",
    )
    quick = splits["validation-quick"]
    require(
        len(quick) == 1000 and {r["id"] for r in quick} <= identities["validation"],
        "Fixed nested quick subset",
    )
    quick_counts = Counter(
        ("retained:" if r["metadata"]["retained_from_v02"] else "")
        + r["metadata"]["dataset_source"]
        for r in quick
    )
    require(dict(quick_counts) == quick_quotas(recipe), "Quick source proportions")
    require(
        all(
            r == next(x for x in splits["validation"] if x["id"] == r["id"])
            for r in quick
        ),
        "Quick subset altered a row",
    )
    panel = REPO / "datasets/evaluation-panel"
    require(
        file_hash(panel / "manifest.json")
        == manifest["external_evaluation"]["manifest_sha256"],
        "External evaluation manifest changed",
    )
    panel_manifest = json.loads((panel / "manifest.json").read_text())
    for entry in panel_manifest["entries"]:
        require(
            file_hash(panel / entry["file"]) == entry["sha256"],
            "External test track changed",
        )
    result = {
        "verified": True,
        "rows": {
            s: len(splits[s])
            for s in [
                "train",
                "validation",
                "validation-quick",
                "calibration",
                "test",
                "ood",
            ]
        },
        "split_overlap": overlap,
        "selected_parent_labels_and_order_verified": dict(checked),
        "all_inputs_retokenized_without_truncation": 84500,
        "prior_training_exposure_in_new_validation": 0,
        "previously_monitored_legacy_validation_rows": sum(
            r["metadata"]["prior_validation_monitoring"] for r in splits["validation"]
        ),
        "training_aliases_in_full_protected_raw_benchmark_ledger": 0,
        "old_heldout_group_or_input_overlap_with_training": 0,
        "near_mcq_development_matches": near_matches,
        "near_method": "Bottom-four word-trigram candidates, Jaccard >= .85; approximate screening only.",
        "gsm_train_unique_problems": 6000,
        "gsm_validation_unique_problems": 200,
        "historical_request_aliases_in_new_source_train": new_source_alias_matches,
        "external_test_tracks_unchanged": len(panel_manifest["entries"]),
        "parent_file_hashes": parent_hashes,
        "audit_script_sha256": file_hash(Path(__file__)),
        "limits": manifest["quality_limits"],
    }
    (root / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "parent_file_hashes"}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
