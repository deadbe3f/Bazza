"""Versioned When2Call four-action view and copy-on-write token feature bank.

Original preference labels stay authoritative. Added alternatives are explicitly
template-authored distractors, not recovered NVIDIA responses. Ambiguous pairs
remain pairs. No external test source is read by this builder.
"""

import argparse
from collections import Counter
import copy
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import time
import traceback

from architecture.data import digest, file_hash, read_rows, write_rows
from gemma_decisions.tracking import atomic_json

POLICIES = ("direct", "tool_call", "request_for_info", "cannot_answer")
VERSION = "when2call-fourway-v0.1"
# Reviewed ambiguity: both native responses refuse the same task. A one-hot
# distinction between them is inherited from the preference release, not a
# reliable action label. Keep both and do not expand this row.
KEEP_PAIR = {"when2call:3954": "two semantically equivalent native refusals"}
STOP = False


def classify(text):
    if text.startswith("<TOOLCALL>"):
        return "tool_call"
    if re.search(r"\b(unable|can't|cannot|can not|don't have|do not have|beyond my capabilities)\b", text, re.I):
        return "cannot_answer"
    if re.search(r"\b(could you|can you|please provide|please specify|i need|i.?ll need|which specific|in which|for what specific|just to confirm|just to clarify|to clarify)\b", text, re.I):
        return "request_for_info"
    if re.match(r"(what|which|how many)\b", text, re.I) and text.rstrip().endswith("?"):
        return "request_for_info"
    return "direct"


def plain_call(text):
    if not text.startswith("<TOOLCALL>"):
        return text
    if not text.endswith("</TOOLCALL>"):
        raise ValueError("Unclosed native tool-call wrapper")
    calls = json.loads(text[len("<TOOLCALL>"):-len("</TOOLCALL>")])
    if not isinstance(calls, list) or len(calls) != 1:
        raise ValueError("Four-action view requires exactly one native tool call")
    return json.dumps(calls[0], ensure_ascii=False)


def words(value):
    return set(re.findall(r"[a-z0-9]+", str(value).lower())) - {
        "the", "a", "an", "to", "of", "for", "from", "with", "and", "in", "me", "my", "please"}


def relevant_tool(query, tools):
    # Deterministic retrieval for a negative candidate, without consulting gold.
    query_words = words(query)
    return max(tools, key=lambda t: (len(query_words & words(t["name"] + " " + t.get("description", ""))),
                                     digest(t)))


def parameter_value(schema):
    if "default" in schema:
        return copy.deepcopy(schema["default"])
    if schema.get("enum"):
        return copy.deepcopy(schema["enum"][0])
    kind = schema.get("type", "string")
    if kind in ("object", "dict"):
        props = schema.get("properties", {})
        return {k: parameter_value(props.get(k, {})) for k in schema.get("required", list(props))}
    if kind in ("array", "list"):
        return [parameter_value(schema.get("items", {}))]
    if kind in ("integer", "int"):
        return 1
    if kind in ("number", "float"):
        return 1.0
    if kind in ("boolean", "bool"):
        return True
    return "example"


def added_candidate(policy, row):
    query, tools = row["state"]["question"], row["state"]["tools"]
    tool = relevant_tool(query, tools) if tools else {
        "name": "execute_request", "parameters": {"properties": {
            "request": {"type": "string", "default": query}}, "required": ["request"]}}
    schema = tool.get("parameters", {})
    props = schema.get("properties", {})
    required = tool.get("required", schema.get("required", list(props)))
    selector = int(digest([VERSION, row["id"], policy])[:8], 16)
    if policy == "tool_call":
        # Defaults are deliberately not treated as supplied user information.
        # This candidate is only added when the native preferred response is
        # non-call; it may guess a missing argument or use an irrelevant tool.
        args = {k: parameter_value(props.get(k, {})) for k in required}
        text = json.dumps({"name": tool["name"], "arguments": args}, ensure_ascii=False)
        method = ("closest available tool; schema defaults or typed example values" if tools else
                  "invented unavailable execute_request tool; no tools exist in this menu")
    elif policy == "request_for_info":
        name = (required or list(props) or ["request details"])[selector % max(1, len(required or list(props)))] if tools else "preferred information source"
        detail = str(name).replace("_", " ")
        templates = (
            "Before I proceed, could you confirm the {detail} you want me to use?",
            "Could you please specify the {detail} for this request?",
            "To assist you with this request, please provide the {detail}.",
            "Which {detail} should I use to carry out your request?",
        )
        text = templates[selector % len(templates)].format(detail=detail)
        method = "clarification about a closest-tool argument; possibly unnecessary or irrelevant"
    elif policy == "cannot_answer":
        templates = (
            "I'm sorry, but I can't complete that request with the tools currently available.",
            "The available tools don't support that request, so I'm unable to carry it out.",
            "I don't have a tool that can perform the requested operation. I can't complete it here.",
            "Apologies, I'm unable to carry out this request using the capabilities available to me.",
        )
        text = templates[selector % len(templates)]
        method = "unavailable-capability response template"
    else:
        # Explicit fabricated completion, never a new gold label. Prefer the
        # original (usually much more specific) direct negative when present.
        quoted = query.rstrip(".?!")
        templates = (
            'Done. I have completed your request: "{query}".',
            'Your request has been completed successfully: "{query}".',
            'I have carried out the requested operation: "{query}".',
            'All set. The requested result is ready: "{query}".',
        )
        text = templates[selector % len(templates)].format(query=quoted)
        method = "unsupported completion claim referencing the request"
    return text, {"origin": "synthetic_template", "author": "Codex", "method": method,
                  "generator_version": VERSION, "template_selector": selector}


def patch_row(original):
    row = copy.deepcopy(original)
    messages = row["state"]["messages"]
    if len(messages) != 1 or messages[0]["role"] != "user":
        raise ValueError("Expected one user turn")
    row["state"] = {"tools": row["state"]["tools"], "question": messages[0]["content"]}
    policies = [classify(t) for t in row["options"]]
    target = row["target"]
    if sorted(target) != [0., 1.]:
        raise ValueError("Expected a native hard preference pair")
    chosen = target.index(1.)
    gold_policy = policies[chosen]
    native = [{"origin": "native_preference", "native_option_index": i,
               "text_sha256": digest(t), "policy": policies[i]} for i, t in enumerate(row["options"])]
    row["options"] = [plain_call(t) for t in row["options"]]
    if len(set(policies)) != 2:
        if row["id"] not in KEEP_PAIR:
            raise ValueError("Unreviewed same-policy pair: " + row["id"])
        action_order, origins = policies, native
        reason = KEEP_PAIR[row["id"]]
    else:
        candidates = dict(zip(policies, row["options"]))
        origins_by_policy = dict(zip(policies, native))
        for policy in POLICIES:
            if policy not in candidates:
                candidates[policy], origins_by_policy[policy] = added_candidate(policy, row)
        row["options"] = [candidates[p] for p in POLICIES]
        row["target"] = [float(p == gold_policy) for p in POLICIES]
        row["target_native"] = list(row["target"])
        action_order = list(POLICIES)
        origins = [origins_by_policy[p] for p in POLICIES]
        reason = None
    if len(set(row["options"])) != len(row["options"]):
        raise ValueError("Duplicate candidates")
    if row["options"][row["target"].index(1.)] != plain_call(original["options"][chosen]):
        raise ValueError("Native preferred response changed")
    patch = {"version": VERSION, "mode": "four-action view with native preference gold",
             "original_row_sha256": digest(original), "original_options": original["options"],
             "original_target": target, "action_order": action_order, "native_gold_policy": gold_policy,
             "candidate_origins": origins, "kept_pair_reason": reason,
             "label_policy": "native gold retained; no new positive labels",
             "limitations": "synthetic distractors are templates, not recovered or independently relabeled responses"}
    row["metadata"].update(input_transform=VERSION, semantic_class=gold_policy,
                           when2call_patch=patch)
    return row


def build(args):
    from tokenizers import Tokenizer
    from gemma_decisions.pilot import encode
    source_index = json.loads((args.parent_bank / "index.json").read_text())
    records = json.loads((args.parent_bank / "records.json").read_text())
    parent_data = json.loads((args.parent_data / "manifest.json").read_text())
    if not source_index["complete"] or file_hash(args.parent_data / "manifest.json") != source_index["identity"]["dataset_manifest_sha256"]:
        raise ValueError("Parent dataset/feature bank mismatch")
    if file_hash(args.parent_bank / "records.json") != parent_data["files"]["selection.json"]["sha256"]:
        raise ValueError("Parent selection changed")
    identity = source_index["identity"]
    if file_hash(args.base_dir / "tokenizer.json") != identity["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs")
    tok = Tokenizer.from_file(str(args.base_dir / "tokenizer.json"))
    train = list(read_rows(args.parent_data / "train.jsonl.gz"))
    needed = {r["id"] for r in records["splits"]["validation-full"]}
    full = [r for r in read_rows(args.parent_validation) if r["id"] in needed]
    raw = {r["id"]: r for r in train + full}
    if len(raw) != len(source_index["files"]):
        raise ValueError("Raw input coverage differs")
    patched, review = {}, []
    for row in raw.values():
        if row.get("source") != "when2call":
            continue
        new = patch_row(row)
        encoded = encode(new, tok, 4096)
        if encoded is None:
            raise ValueError("Expanded row exceeds max length: " + row["id"])
        old = next(r for s in records["splits"].values() for r in s if r["id"] == row["id"])
        prepared = {**old, **encoded, "semantic_class": new["metadata"]["semantic_class"],
                    "when2call_patch": new["metadata"]["when2call_patch"]}
        prepared["cache"] = "patched/" + digest([VERSION, prepared["id"], prepared["ids"]]) + ".pt"
        patched[row["id"]] = (new, prepared)
        review.append({"id": row["id"], "split": row["split"], "state": new["state"],
                       "options": new["options"], "target": new["target"],
                       "patch": new["metadata"]["when2call_patch"],
                       "old_tokens": len(old["ids"]), "new_tokens": len(prepared["ids"])})
    selection = copy.deepcopy(records)
    for split, rows in selection["splits"].items():
        selection["splits"][split] = [patched[r["id"]][1] if r["id"] in patched else r for r in rows]
    selection["identity"]["when2call_patch"] = VERSION
    # Neither raw records nor encoded inputs for any other source may change.
    for split, rows in records["splits"].items():
        for before, after in zip(rows, selection["splits"][split]):
            if before["id"] != after["id"] or before["group_id"] != after["group_id"]:
                raise ValueError("Membership or order changed")
            if before["id"] not in patched and before != after:
                raise ValueError("Non-When2Call record changed")
    manifest = {"dataset": "groundbeef-architecture", "version": VERSION,
                "parent_data_manifest_sha256": file_hash(args.parent_data / "manifest.json"),
                "parent_feature_identity_sha256": digest(identity),
                "parent_records_sha256": file_hash(args.parent_bank / "records.json"),
                "builder_sha256": file_hash(__file__), "source": "nvidia/When2Call native train preferences",
                "source_revision": review[0]["patch"].get("source_revision", raw[review[0]["id"]]["metadata"]["source_revision"]),
                "protected_test_read": False, "original_gold_preserved": True,
                "choice_order": "direct, tool_call, request_for_info, cannot_answer (benchmark action order)",
                "changed_unique": len(patched), "unchanged_unique": len(raw) - len(patched),
                "statistics": {split: {"rows": len(rows), "when2call_rows": sum(r["source"] == "when2call" for r in rows),
                    "when2call_choice_counts": dict(Counter(str(len(r["target"])) for r in rows if r["source"] == "when2call")),
                    "when2call_gold_policies": dict(Counter(r["semantic_class"] for r in rows if r["source"] == "when2call"))}
                    for split, rows in selection["splits"].items()},
                "retained_pair_ids": [r["id"] for r in review if r["patch"]["kept_pair_reason"]],
                "added_candidates": sum(o["origin"] == "synthetic_template" for r in review for o in r["patch"]["candidate_origins"]),
                "limitations": ["Synthetic negatives are template-authored and can be easier than native MCQ distractors.",
                    "Native preference labels may be noisy; this patch does not establish correctness against tool semantics.",
                    "This changes action coverage, prompt formatting and choice order together; it is not a format-only ablation.",
                    "No direct-answer positives exist in this selected native preference subset."], "files": {}}
    # A repeated build must reproduce the data rather than overwrite another recipe.
    if (args.data_dir / "manifest.json").exists():
        prior = json.loads((args.data_dir / "manifest.json").read_text())
        if prior["builder_sha256"] != manifest["builder_sha256"] or prior["parent_records_sha256"] != manifest["parent_records_sha256"]:
            raise ValueError("Patch recipe changed: use a fresh directory")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train", train), ("validation-full", full)]:
        manifest["files"][name + ".jsonl.gz"] = write_rows(args.data_dir / (name + ".jsonl.gz"),
            [patched[r["id"]][0] if r["id"] in patched else r for r in rows])
    quick_ids = {r["id"] for r in records["splits"]["validation-quick"]}
    quick = [r for r in full if r["id"] in quick_ids]
    manifest["files"]["validation-quick.jsonl.gz"] = write_rows(args.data_dir / "validation-quick.jsonl.gz",
        [patched[r["id"]][0] if r["id"] in patched else r for r in quick])
    manifest["files"]["review.jsonl.gz"] = write_rows(args.data_dir / "review.jsonl.gz", review)
    atomic_json(args.data_dir / "selection.json", selection)
    manifest["files"]["selection.json"] = {"sha256": file_hash(args.data_dir / "selection.json"), "rows": len(raw)}
    # Preserve the original exclusion ledger, including native group/alias identities.
    for name in ["exclusions.json", "membership.jsonl.gz"]:
        src = args.parent_data / name
        if src.exists():
            dest = args.data_dir / name
            if not dest.exists():
                os.link(src, dest)
            if not os.path.samefile(src, dest):
                raise ValueError("Conflicting exclusion ledger")
            manifest["files"][name] = {"sha256": file_hash(src), "reused_parent": True}
    atomic_json(args.data_dir / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ["changed_unique", "unchanged_unique", "added_candidates", "retained_pair_ids", "statistics"]}, indent=2), flush=True)


def export(args):
    import torch
    from tqdm.auto import tqdm
    from architecture.features import append_entry, forward, hashlib_serialization, input_key, load_entries, unique_records
    from gemma_decisions.probe import load_kiln
    from gemma_decisions.sweep import atomic_save
    from gemma_decisions.tracking import RunLog
    args.bank_dir.mkdir(parents=True, exist_ok=True)
    with (args.bank_dir / ".export.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dataset = json.loads((args.data_dir / "manifest.json").read_text())
        old = json.loads((args.parent_bank / "index.json").read_text())
        if not old["complete"] or dataset["parent_feature_identity_sha256"] != digest(old["identity"]):
            raise ValueError("Wrong parent bank")
        if dataset["parent_records_sha256"] != file_hash(args.parent_bank / "records.json"):
            raise ValueError("Parent records changed")
        for name, entry in dataset["files"].items():
            if file_hash(args.data_dir / name) != entry["sha256"]:
                raise ValueError("Patched data changed: " + name)
        records = json.loads((args.data_dir / "selection.json").read_text())
        unique = unique_records(records)
        implementation = load_kiln(args.kiln_model_dir)
        actual = {"checkpoint": implementation.CHECKPOINT, "revision": implementation.REVISION,
                  "weight_sha256": implementation.WEIGHT_SHA256,
                  "implementation_sha256": file_hash(args.kiln_model_dir / "model.py"),
                  "tokenizer_sha256": file_hash(args.base_dir / "tokenizer.json"),
                  "serialization_sha256": hashlib_serialization(), "torch": torch.__version__,
                  "cuda": torch.version.cuda, "image_id": args.image_id}
        for k, value in actual.items():
            if old["identity"][k] != value:
                raise ValueError("Frozen feature runtime differs: " + k)
        identity = {**old["identity"], "dataset_manifest_sha256": file_hash(args.data_dir / "manifest.json"),
                    "when2call_patch": VERSION, "parent_feature_identity_sha256": digest(old["identity"]),
                    "patch_exporter_sha256": file_hash(__file__), "patched_backbone_batch_size": 1,
                    "reuse_policy": "hardlinked original tensors; no recomputation for other sources"}
        if (args.bank_dir / "identity.json").exists() and json.loads((args.bank_dir / "identity.json").read_text()) != identity:
            raise ValueError("Existing patch feature identity changed")
        atomic_json(args.bank_dir / "identity.json", identity)
        shutil.copyfile(args.data_dir / "selection.json", args.bank_dir / "records.json")
        entries = load_entries(args.bank_dir)
        for key, entry in entries.items():
            if key not in unique or entry["input_sha256"] != input_key(unique[key]) or not (args.bank_dir / entry["file"]).is_file():
                raise ValueError("Invalid resumed entry: " + key)
        logger = RunLog(args.bank_dir / "logging", VERSION + " feature patch", identity,
                        args.overfit_url, "Bazza When2Call fourway")
        def status(phase, **extra):
            atomic_json(args.bank_dir / "status.json", {"phase": phase, "completed": len(entries),
                        "total_unique": len(unique), "updated_at": time.time(), **extra})
        try:
            status("reusing_original")
            for row in unique.values():
                if row["source"] == "when2call" or row["id"] in entries:
                    continue
                original = old["files"][row["id"]]
                if original["input_sha256"] != input_key(row) or original["file"] != row["cache"]:
                    raise ValueError("Non-When2Call input differs")
                src, dest = args.parent_bank / original["file"], args.bank_dir / row["cache"]
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    os.link(src, dest)
                if not os.path.samefile(src, dest):
                    raise ValueError("Reused tensor must be a hard link")
                entry = {**original, "origin": "unchanged-parent-hardlink",
                         "parent_feature_identity_sha256": digest(old["identity"])}
                append_entry(args.bank_dir, entry)
                entries[row["id"]] = entry
            changed = [r for r in unique.values() if r["source"] == "when2call"]
            pending = [r for r in changed if r["id"] not in entries]
            required = sum(len(r["ids"]) * 3840 * 2 + 8192 for r in pending)
            if shutil.disk_usage(args.bank_dir).free < required + 5 * 2**30:
                raise OSError("Insufficient disk for patch")
            if pending:
                status("loading_backbone", patched_remaining=len(pending), backbone_batch_size=1)
                backbone = implementation.load_model(args.base_dir, quantization="bnb-nf4").requires_grad_(False).eval()
                start = time.perf_counter()
                progress = tqdm(pending, desc="Patch When2Call features", unit="row")
                for n, row in enumerate(progress, 1):
                    if STOP:
                        break
                    tensor = forward(backbone, [row])[0].cpu().to(torch.bfloat16).clone()
                    if tensor.shape != (len(row["ids"]), 3840):
                        raise ValueError("Unexpected patched shape")
                    path = args.bank_dir / row["cache"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.exists() and path.stat().st_nlink > 1:
                        raise ValueError("Never overwrite a hardlinked tensor")
                    atomic_save({"hidden": tensor, "input_sha256": input_key(row),
                                 "feature_identity_sha256": digest(identity)}, path)
                    entry = {"id": row["id"], "file": row["cache"], "sha256": file_hash(path),
                             "input_sha256": input_key(row), "tokens": len(row["ids"]),
                             "origin": VERSION, "backbone_batch_size": 1,
                             "original_input_sha256": old["files"][row["id"]]["input_sha256"]}
                    append_entry(args.bank_dir, entry)
                    entries[row["id"]] = entry
                    if n % 16 == 0 or n == len(pending):
                        rate = n / max(time.perf_counter() - start, 1e-6)
                        values = {"cache/patched_completed": len(entries) - dataset["unchanged_unique"],
                                  "cache/patched_total": len(changed), "cache/rows_per_second": rate}
                        logger.log(values, int(values["cache/patched_completed"]))
                        status("exporting", **values)
                        progress.set_postfix(rows_per_second=f"{rate:.2f}")
                del backbone
                torch.cuda.empty_cache()
            complete = len(entries) == len(unique)
            atomic_json(args.bank_dir / "index.json", {"identity": identity, "complete": complete, "files": entries})
            phase = "completed" if complete else "stopped"
            status(phase, reused=dataset["unchanged_unique"], patched=len(entries) - dataset["unchanged_unique"])
            logger.finish(phase)
        except BaseException as error:
            status("failed", error=str(error), traceback=traceback.format_exc())
            logger.finish("failed")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    for name in ["parent-bank", "parent-data", "parent-validation", "data-dir", "base-dir"]:
        b.add_argument("--" + name, required=True, type=Path)
    e = sub.add_parser("export")
    for name in ["parent-bank", "data-dir", "bank-dir", "base-dir", "kiln-model-dir"]:
        e.add_argument("--" + name, required=True, type=Path)
    for name in ["image-tag", "image-id"]:
        e.add_argument("--" + name, required=True)
    e.add_argument("--overfit-url", default="http://127.0.0.1:6006")
    args = parser.parse_args()
    if args.command == "build":
        build(args)
    else:
        import torch
        torch.set_num_threads(8)
        def stop(signum, frame):
            global STOP
            STOP = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        export(args)


if __name__ == "__main__":
    main()
