"""Small Open-Jev SFT pilot: frozen Gemma, cached states, shared recurrent head.

Targets and generator metadata never enter model inputs. The public upstream
train/validation/calibration/test/OOD partitions retain their separate roles.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.nn import functional as F

from .model import GemmaDecisionHead
from .probe import load_kiln, memory
from .decision_metadata import fields


DATA_REVISION = "c67699e13d0ae25e35b77165a4b6b079bedc8aba"
SEED = 20261003


def encode(row, tokenizer, max_length, order=None):
    options = row["options"]
    if order is None:
        order = list(range(len(options)))
    if not 2 <= len(order) <= 26:
        return None
    state = row["state"]
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False, sort_keys=True)
    prompt = "<bos><|turn>user\nChoose the best answer. Reply with its letter only.\n"
    prompt += "Context:\n" + state + "\nQuestion:\n" + row["question"] + "\nOptions:\n"
    ends = []
    for index, original in enumerate(order):
        prompt += chr(65 + index) + ". " + str(options[original])
        ends.append(len(prompt))
        prompt += "\n"
    prompt += "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    encoded = tokenizer.encode(prompt, add_special_tokens=False)
    if len(encoded.ids) > max_length:
        return None  # No silently truncated states, questions or choices.
    positions = []
    for end in ends:
        eligible = [i for i, (a, b) in enumerate(encoded.offsets) if 0 < b <= end]
        if not eligible:
            raise ValueError("Cannot locate option's final token")
        positions.append(eligible[-1])
    target = [float(row["target"][i]) for i in order]
    if len(target) != len(order) or min(target) < 0 or abs(sum(target) - 1) > 1e-4:
        raise ValueError("Invalid target distribution")
    return {"id": row["id"], "group_id": row["group_id"], "source": row["source"],
            "kind": row["kind"], "ids": encoded.ids, "positions": positions,
            "target": target, "order": order,
            **{**fields(row), "options": [str(options[i]) for i in order]}}


def select_split(path, tokenizer, max_length, limit):
    buckets = defaultdict(list)
    skipped = Counter()
    for line in gzip.open(path, "rt"):
        row = json.loads(line)
        encoded = encode(row, tokenizer, max_length)
        if encoded is None:
            skipped["too_long_or_unsupported_options"] += 1
            continue
        digest = hashlib.sha256((str(SEED) + row["id"]).encode()).hexdigest()
        buckets[(row["source"], row["kind"])].append((digest, row, encoded))
    available = sum(map(len, buckets.values()))
    for values in buckets.values():
        values.sort(key=lambda item: item[0], reverse=True)
    selected = []
    while len(selected) < min(limit, available):
        for key in sorted(buckets):
            if buckets[key] and len(selected) < limit:
                _, raw, encoded = buckets[key].pop()
                selected.append((raw, encoded))
    return selected, {"available": available, "selected": len(selected), "skipped": dict(skipped)}


def batch(records, output_dir):
    hidden_rows = [torch.load(output_dir / r["cache"], weights_only=True) for r in records]
    length = max(h.shape[0] for h in hidden_rows)
    choices = max(len(r["positions"]) for r in records)
    hidden = torch.zeros(len(records), length, hidden_rows[0].shape[-1], dtype=torch.bfloat16)
    mask = torch.zeros(len(records), length, dtype=torch.long)
    positions = torch.zeros(len(records), choices, dtype=torch.long)
    option_mask = torch.zeros_like(positions, dtype=torch.bool)
    targets = torch.zeros(len(records), choices)
    for i, (record, h) in enumerate(zip(records, hidden_rows)):
        hidden[i, :len(h)] = h
        mask[i, :len(h)] = 1
        count = len(record["positions"])
        positions[i, :count] = torch.tensor(record["positions"])
        option_mask[i, :count] = True
        targets[i, :count] = torch.tensor(record["target"])
    return tuple(t.cuda() for t in (hidden, mask, positions, option_mask, targets))


@torch.no_grad()
def predict(head, records, output_dir, batch_size):
    head.eval()
    result = []
    for start in range(0, len(records), batch_size):
        tensors = batch(records[start:start + batch_size], output_dir)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = head(*tensors[:4]).float().cpu()
        for row, record in zip(logits, records[start:start + batch_size]):
            result.append(row[:len(record["target"])])
    return result


def metrics(logits, records, temperature=1.0):
    values = defaultdict(list)
    bins = [[] for _ in range(10)]
    for logit, record in zip(logits, records):
        prob = (logit.float() / temperature).softmax(-1)
        target = torch.tensor(record["target"])
        correct = float(target[prob.argmax()] >= target.max() - 1e-6)
        item = {"accuracy": correct, "cross_entropy": float(-(target * prob.clamp_min(1e-9).log()).sum()),
                "brier": float((prob - target).square().sum()), "confidence": float(prob.max()),
                "target_mass_at_prediction": float(target[prob.argmax()])}
        for key, value in item.items():
            values[key].append(value)
        values["chance"].append(float((target >= target.max() - 1e-6).float().mean()))
        values["source:" + record["source"]].append(correct)
        if target.max() >= 1 - 1e-6:
            values["hard_target_accuracy"].append(correct)
        bins[min(9, int(item["confidence"] * 10))].append(
            (item["confidence"], item["target_mass_at_prediction"]))
    means = {key: sum(items) / len(items) for key, items in values.items()}
    means["soft_target_ece"] = sum(abs(sum(c for c, _ in b) - sum(a for _, a in b)) for b in bins) / len(records)
    means["hard_target_rows"] = len(values["hard_target_accuracy"])
    means["rows"] = len(records)
    means["accuracy_definition"] = "argmax prediction belongs to maximum-target set; ties accepted"
    means["per_source"] = {key[7:]: {"accuracy": means.pop(key), "rows": len(values[key])}
                           for key in list(means) if key.startswith("source:")}
    return means


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=Path, required=True)
    p.add_argument("--kiln-model-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=Path("build/open_jev"))
    p.add_argument("--image-tag", required=True)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--train-rows", type=int, default=1024)
    p.add_argument("--validation-rows", type=int, default=256)
    p.add_argument("--calibration-rows", type=int, default=128)
    p.add_argument("--test-rows", type=int, default=512)
    p.add_argument("--ood-rows", type=int, default=256)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--augment-orders", type=int, default=2)
    args = p.parse_args()
    if (args.data_dir / "revision.txt").read_text().strip() != DATA_REVISION:
        raise ValueError("Open-Jev revision mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "features").mkdir(exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(8)
    tokenizer = Tokenizer.from_file(str(args.base_dir / "tokenizer.json"))
    splits = {}
    counts = {}
    group_sets = {}
    for split in ("train", "validation", "calibration", "test", "ood"):
        path = args.data_dir / "raw/release-v2-redistributable" / (split + ".jsonl.gz")
        selected, info = select_split(path, tokenizer, args.max_length, getattr(args, split + "_rows"))
        group_sets[split] = {row["group_id"] for row, _ in selected}
        records = []
        for row, encoded in selected:
            records.append(encoded)
            if split == "train":
                for variant in range(1, args.augment_orders):
                    rng = random.Random(str(SEED) + row["id"] + str(variant))
                    order = list(range(len(row["options"])))
                    rng.shuffle(order)
                    augmented = encode(row, tokenizer, args.max_length, order)
                    if augmented is not None:
                        records.append(augmented)
        splits[split] = records
        info["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        info["cached_rows"] = len(records)
        info["source_kind_counts"] = dict(Counter(r["source"] + "/" + r["kind"] for r in records))
        counts[split] = info
    for a, groups in group_sets.items():
        for b, others in group_sets.items():
            if a != b and groups & others:
                raise ValueError(f"Linked groups overlap between {a} and {b}")
    implementation = load_kiln(args.kiln_model_dir)
    identity = {"checkpoint": implementation.CHECKPOINT, "revision": implementation.REVISION,
                "weight_sha256": implementation.WEIGHT_SHA256,
                "quantization": "bnb-nf4", "implementation_sha256": hashlib.sha256(
                    (args.kiln_model_dir / "model.py").read_bytes()).hexdigest(),
                "tokenizer_sha256": hashlib.sha256((args.base_dir / "tokenizer.json").read_bytes()).hexdigest(),
                "dataset_revision": DATA_REVISION, "max_length": args.max_length,
                "seed": SEED, "counts": counts}
    previous = args.output_dir / "feature_identity.json"
    if previous.exists() and json.loads(previous.read_text()) != identity:
        raise ValueError("Existing feature cache has a different experiment identity; choose a fresh output directory")
    previous.write_text(json.dumps(identity, indent=2))
    print("Selected splits", json.dumps({s: len(r) for s, r in splits.items()}), flush=True)
    backbone = implementation.load_model(args.base_dir, quantization="bnb-nf4")
    letter_ids = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        ids = tokenizer.encode(letter, add_special_tokens=False).ids
        if len(ids) != 1:
            raise ValueError("Baseline requires single-token answer letters")
        letter_ids.append(ids[0])
    letter_ids = torch.tensor(letter_ids, device="cuda")
    feature_start = time.perf_counter()
    all_records = [(split, i, r) for split, rows in splits.items() for i, r in enumerate(rows)]
    with torch.no_grad():
        for count, (split, i, record) in enumerate(all_records, 1):
            name = f"features/{split}-{i:05d}.pt"
            record["cache"] = name
            baseline_path = args.output_dir / name.replace(".pt", "-baseline.pt")
            if (args.output_dir / name).exists() and baseline_path.exists():
                record["baseline_logits"] = torch.load(baseline_path, weights_only=True).tolist()
                continue
            ids = torch.tensor([record["ids"]], device="cuda")
            hidden = backbone(ids)
            baseline = backbone.logits(hidden[:, -1], letter_ids[:len(record["target"])])[0].cpu()
            record["baseline_logits"] = baseline.tolist()
            torch.save(hidden[0].cpu(), args.output_dir / name)
            torch.save(baseline, baseline_path)
            del hidden, ids
            if count % 100 == 0:
                print(json.dumps({"cached": count, "total": len(all_records),
                                  "elapsed_seconds": time.perf_counter() - feature_start}), flush=True)
    feature_seconds = time.perf_counter() - feature_start
    del backbone
    gc.collect()
    torch.cuda.empty_cache()
    (args.output_dir / "split_records.json").write_text(json.dumps(splits))
    head = GemmaDecisionHead().cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.01)
    history = []
    best_loss, best_epoch = float("inf"), 0
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        head.train()
        order = list(range(len(splits["train"])))
        random.shuffle(order)
        total = 0
        for start in range(0, len(order), args.batch_size):
            records = [splits["train"][i] for i in order[start:start + args.batch_size]]
            h, mask, pos, option_mask, target = batch(records, args.output_dir)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = head(h, mask, pos, option_mask)
            loss = -(target * logits.log_softmax(-1)).sum(-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(records)
        validation = predict(head, splits["validation"], args.output_dir, args.batch_size)
        score = metrics(validation, splits["validation"])
        history.append({"epoch": epoch, "train_cross_entropy": total / len(order),
                        "validation_cross_entropy": score["cross_entropy"],
                        "validation_accuracy": score["accuracy"]})
        print(json.dumps(history[-1]), flush=True)
        if score["cross_entropy"] < best_loss:
            best_loss, best_epoch = score["cross_entropy"], epoch
            torch.save(head.state_dict(), args.output_dir / "head.pt")
        elif epoch - best_epoch >= 3:
            break
    head.load_state_dict(torch.load(args.output_dir / "head.pt", weights_only=True))
    calibration = predict(head, splits["calibration"], args.output_dir, args.batch_size)
    temperatures = [0.5 + i * 0.1 for i in range(46)]
    temperature = min(temperatures, key=lambda t: metrics(calibration, splits["calibration"], t)["cross_entropy"])
    results = {}
    for split in ("test", "ood"):
        predictions = predict(head, splits[split], args.output_dir, args.batch_size)
        baseline = [torch.tensor(r["baseline_logits"]) for r in splits[split]]
        results[split] = {"recurrent": metrics(predictions, splits[split]),
                          "recurrent_calibrated": metrics(predictions, splits[split], temperature),
                          "stock_letter_logits": metrics(baseline, splits[split])}
    report = {**identity, "results": results, "training_history": history,
              "best_epoch": best_epoch, "calibration_temperature": temperature,
              "head_parameters": sum(p.numel() for p in head.parameters()),
              "feature_seconds": feature_seconds, "head_training_seconds": time.perf_counter() - train_start,
              "image": args.image_tag, "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0), "head_training_memory": memory(),
              "protocol": "balanced source/kind sampling; upstream grouped splits; train-only option permutations; frozen NF4 backbone",
              "scope": "Small text-only supervised pilot. No adapter training, RL, arguments, CUA execution, or Decision Index score."}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"best_epoch": best_epoch, "results": results}), flush=True)


if __name__ == "__main__":
    main()
