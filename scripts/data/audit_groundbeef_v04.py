"""Saved-file audit of Groundbeef v0.4 membership, labels and fixed panels."""

import argparse
import ast
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from common import digest, file_hash, jsonl
from groundbeef import prompt, state_object, validate, visible_hash
from when2call_candidates import plain_call


def audit(root, tokenizer_path):
    from tokenizers import Tokenizer

    m = json.loads((root / "manifest.json").read_text())
    recipe = json.loads((root / "recipe.json").read_text())
    tok = Tokenizer.from_file(str(tokenizer_path))
    assert file_hash(tokenizer_path) == m["tokenizer_sha256"]
    assert (
        file_hash(root / "recipe.json") == m["recipe_sha256"]
        and file_hash(root / "sources.json") == m["sources_sha256"]
    )
    splits = {}
    for e in m["files"]:
        path = root / e["file"]
        assert file_hash(path) == e["sha256"], e["file"]
        rr = list(jsonl(path))
        assert len(rr) == e["rows"]
        splits[e["file"].split(".")[0]] = rr
    members = {(r["split"], r["id"]): r for r in splits["membership"]}
    assert len(members) == 64000 + 4500 + 500
    forbidden = json.loads(
        Path("build/groundbeef/architecture-v0.1/exclusions.json").read_text()
    )
    blocked_ids = set(forbidden["blocked_training_ids"])
    blocked_keys = set(forbidden["keys"])
    ids = {}
    groups = {}
    inputs = {}
    aliases = {}
    checked = Counter()
    needed = defaultdict(dict)
    for split in ["train", "validation", "calibration"]:
        rr = splits[split]
        ids[split] = set()
        groups[split] = set()
        inputs[split] = set()
        aliases[split] = set()
        counts = Counter(r["metadata"]["v04_source_view"] for r in rr)
        expected = recipe[split] if split != "calibration" else m["calibration_quotas"]
        assert dict(counts) == expected, (split, counts, expected)
        for start in range(0, len(rr), 128):
            batch = rr[start : start + 128]
            enc = tok.encode_batch([prompt(r) for r in batch], add_special_tokens=False)
            for row, tokens in zip(batch, enc):
                validate(row)
                member = members[split, row["id"]]
                assert len(tokens.ids) == row["metadata"]["input_tokens"] <= 4096
                assert row["split"] == split and row["id"] not in ids[split]
                assert member["row_sha256"] == digest(row) and member[
                    "input_sha256"
                ] == visible_hash(row)
                ids[split].add(row["id"])
                groups[split].add(member["cap_group"])
                inputs[split].add(member["input_sha256"])
                aliases[split].update(member["alias_keys"])
                if split == "train":
                    assert (
                        row["id"] not in blocked_ids
                        and not set(member["architecture_identity_keys"]) & blocked_keys
                    )
                needed[member["parent_file"]][row["id"]] = (row, member)
                view = row["metadata"]["v04_source_view"]
                meta = row["metadata"]
                gold = max(range(len(row["target"])), key=row["target"].__getitem__)
                if view == "when2call-fourway":
                    assert len(row["options"]) == 4 and set(row["state"]) == {
                        "tools",
                        "question",
                    }
                    orig = meta["original_options"][meta["original_target"].index(1.0)]
                    assert row["options"][gold] == plain_call(orig)
                    assert meta["action_order"] == [
                        "direct",
                        "tool_call",
                        "request_for_info",
                        "cannot_answer",
                    ]
                if view == "mmlu-tenchoice":
                    assert len(row["options"]) == 10
                    native_gold = meta["original_options"][
                        meta["original_target"].index(1.0)
                    ]
                    assert (
                        row["options"][gold] == native_gold
                        and len(meta["added_numeric_values"]) == 6
                    )
                    values = [Fraction(v) for v in meta["added_numeric_values"]]
                    assert (
                        len(set(values)) == 6
                        and Fraction(meta["native_numeric_gold"]) not in values
                    )
                if view == "codeio-python":
                    result = ast.literal_eval(row["options"][gold])
                    w = meta["execution"]
                    assert (
                        json.dumps(
                            result,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        == w["json"]
                    )
                    assert row["state"]["input"] == w["input"][2:-1]
                checked[view] += 1
        print(
            "Verified token lengths, memberships and label witnesses:",
            split,
            flush=True,
        )
    overlap = {}
    for a, b in [
        ("train", "validation"),
        ("train", "calibration"),
        ("validation", "calibration"),
    ]:
        entry = {
            "ids": len(ids[a] & ids[b]),
            "parents": len(groups[a] & groups[b]),
            "inputs": len(inputs[a] & inputs[b]),
            "aliases": len(aliases[a] & aliases[b]),
        }
        assert not any(entry.values()), (a, b, entry)
        overlap[a + "/" + b] = entry
    source = json.loads((root / "sources.json").read_text())
    verified_parents = 0
    for e in source["source_files"]:
        path = Path(e["file"])
        assert file_hash(path) == e["sha256"], str(path)
        lookup = needed.get(str(path), {})
        if not lookup:
            continue
        for raw in jsonl(path):
            if raw["id"] not in lookup:
                continue
            row, member = lookup[raw["id"]]
            assert digest(raw) == member["original_row_sha256"]
            view = row["metadata"]["v04_source_view"]
            target = (
                list(reversed(raw["target"])) if view == "ragtruth" else raw["target"]
            )
            assert row["target"] == target, (view, row["id"])
            if view == "anli":
                assert row["options"] == [str(o).lower() for o in raw["options"]], row[
                    "id"
                ]
            elif view not in {"ragtruth", "codeio-python"}:
                assert row["options"] == raw["options"], row["id"]
            verified_parents += 1
        # The same underlying MMLU IDs can appear in native/expanded files, but
        # selected parents are never shared across view files in training.
    assert verified_parents == 69000, verified_parents
    quick = splits["validation-quick"]
    full = {r["id"]: r for r in splits["validation"]}
    assert len(quick) == 1000 and all(r == full[r["id"]] for r in quick)
    assert (
        dict(Counter(r["metadata"]["v04_source_view"] for r in quick))
        == m["quick_quotas"]
    )
    # Check near MCQ development matches, beyond exact parent/input equality.
    from prepare import NearIndex

    near = NearIndex()
    train_mcq = []
    for split in ["validation", "calibration"]:
        for r in splits[split]:
            if r["metadata"]["category"] in {
                "math",
                "science",
                "knowledge",
                "commonsense",
            }:
                state = state_object(r["state"])
                text = state.get("question") if isinstance(state, dict) else state
                if not text:
                    text = r["question"]
                near.add(text, r["id"])
    for r in splits["train"]:
        if r["metadata"]["category"] in {"math", "science", "knowledge", "commonsense"}:
            state = state_object(r["state"])
            text = state.get("question") if isinstance(state, dict) else state
            if not text:
                text = r["question"]
            match = near.match(text)
            if match:
                train_mcq.append({"id": r["id"], "match": match})
    assert not train_mcq, train_mcq[:5]
    for e in source["historical_files"]:
        assert file_hash(e["file"]) == e["sha256"], e["file"]
    panel = Path("datasets/evaluation-panel")
    pm = json.loads((panel / "manifest.json").read_text())
    assert (
        file_hash(panel / "manifest.json")
        == m["external_evaluation"]["manifest_sha256"]
    )
    for e in pm["entries"]:
        assert file_hash(panel / e["file"]) == e["sha256"]
    result = {
        "verified": True,
        "rows": {k: len(v) for k, v in splits.items()},
        "source_parent_records_verified": verified_parents,
        "split_overlap": overlap,
        "architecture_experiment_training_overlap": 0,
        "native_gold_witnesses": dict(checked),
        "all_inputs_retokenized_without_truncation": 69000,
        "quick_panel_is_fixed_nested_subset": True,
        "near_mcq_development_matches": 0,
        "external_test_tracks_unchanged": len(pm["entries"]),
        "script_sha256": file_hash(__file__),
        "limits": m["quality_limits"],
    }
    save = root / "audit.json"
    save.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    a = p.parse_args()
    audit(a.dataset, a.tokenizer)


if __name__ == "__main__":
    main()
