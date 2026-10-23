"""Unattended, resumable frozen-Gemma head training, online or with a feature bank.

Training and hyperparameter selection use train/validation only. Calibration,
test and OOD are evaluated once for the validation-selected winner.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import gzip
import hashlib
import inspect
import itertools
import json
import math
import os
import random
import shutil
import signal
import time
import traceback
from contextlib import nullcontext
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tqdm.auto import tqdm

from .model import GemmaDecisionHead
from .online import OnlineFeatures
from .pilot import DATA_REVISION, SEED, encode, metrics, select_split
from .probe import load_kiln
from .tracking import RunLog, atomic_json, sync_tree
from .dataset_paths import resolve_dataset
from .weighting import fit_weights, weighted_loss
from .evaluation import breakdown, log_values
from .decision_metadata import fields


SPLITS = ("train", "validation", "calibration", "test", "ood")
STOP_REQUESTED = False


class SweepStopped(Exception):
    pass


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("Stop requested; finishing the current operation and saving progress.", flush=True)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def check_groups(splits):
    for i, a in enumerate(SPLITS):
        ids = {r["group_id"] for r in splits[a]}
        for b in SPLITS[i + 1:]:
            if ids & {r["group_id"] for r in splits[b]}:
                raise ValueError(f"Linked groups overlap between {a} and {b}")


def prepare(args, implementation):
    source_paths, dataset_revision = resolve_dataset(args.data_dir)
    source_hashes = {s: file_hash(p) for s, p in source_paths.items()}
    identity = {
        "format_version": 1, "checkpoint": implementation.CHECKPOINT,
        "revision": implementation.REVISION, "weight_sha256": implementation.WEIGHT_SHA256,
        "quantization": "bnb-nf4", "implementation_sha256": file_hash(args.kiln_model_dir / "model.py"),
        "tokenizer_sha256": file_hash(args.base_dir / "tokenizer.json"),
        "serialization_sha256": hashlib.sha256(inspect.getsource(encode).encode()).hexdigest(),
        "selection_sha256": hashlib.sha256(inspect.getsource(select_split).encode()).hexdigest(),
        "dataset_revision": dataset_revision, "source_sha256": source_hashes,
        "max_length": args.max_length, "selection_seed": SEED,
        "limits": {"train": max(args.train_sizes), "validation": args.validation_rows,
                   "calibration": args.calibration_rows, "test": args.test_rows, "ood": args.ood_rows},
        "augment_orders": args.augment_orders,
        "groundbeef_panels": dataset_revision.startswith("groundbeef-v0.1:"),
    }
    selection_dir = "selection" if args.feature_mode == "online" else "cache"
    path = args.output_dir / selection_dir / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["identity"] != identity:
            raise ValueError("Feature-bank identity changed; choose a fresh --output-dir")
        check_groups(manifest["splits"])
        return manifest
    tokenizer = Tokenizer.from_file(str(args.base_dir / "tokenizer.json"))
    splits, counts = {}, {}
    for split in SPLITS:
        if identity["groundbeef_panels"] and split != "ood":
            selected, count = [], {"skipped": {}}
            with gzip.open(source_paths[split], "rt") as stream:
                for line in tqdm(stream, desc=f"Tokenize {split}", unit="row", disable=not getattr(args, "progress", False)):
                    raw = json.loads(line)
                    record = encode(raw, tokenizer, args.max_length)
                    if record is None:
                        raise ValueError(f"Fixed groundbeef {split} row {raw['id']} exceeds max length; no silent selection changes")
                    selected.append((raw, record))
            if len(selected) != identity["limits"][split]:
                raise ValueError(f"groundbeef {split} is frozen at {len(selected)} rows; requested {identity['limits'][split]}")
            count.update(available=len(selected), selected=len(selected))
        else:
            selected, count = select_split(source_paths[split], tokenizer, args.max_length, identity["limits"][split])
        if len(selected) != identity["limits"][split]:
            raise ValueError(f"{split}: requested {identity['limits'][split]} usable rows, found {len(selected)}")
        records = []
        for source_index, (raw, original) in enumerate(selected):
            variants = [(0, original)]
            if split == "train":
                for variant in range(1, args.augment_orders):
                    order = list(range(len(raw["options"])))
                    random.Random(str(SEED) + raw["id"] + str(variant)).shuffle(order)
                    augmented = encode(raw, tokenizer, args.max_length, order)
                    if augmented is None:
                        count["skipped_augmentations"] = count.get("skipped_augmentations", 0) + 1
                    else:
                        variants.append((variant, augmented))
            for variant, record in variants:
                record.update(source_index=source_index, variant=variant,
                              cache=f"features/{split}-{source_index:06d}-{variant:02d}.pt")
                records.append(record)
        splits[split] = records
        count["cached_rows"] = len(records)
        counts[split] = count
        print(f"Selected {split}: {len(selected)} unique rows, {len(records)} encoded examples", flush=True)
    check_groups(splits)
    manifest = {"identity": identity, "counts": counts, "splits": splits}
    atomic_json(path, manifest)
    return manifest


def import_pilot(args, manifest):
    """Reuse only records with matching provenance, inputs, positions and targets."""
    if args.reuse_pilot is None:
        return 0
    previous = args.reuse_pilot / "feature_identity.json"
    records_path = args.reuse_pilot / "split_records.json"
    if not previous.exists() or not records_path.exists():
        return 0
    old = json.loads(previous.read_text())
    keys = ("checkpoint", "revision", "quantization", "implementation_sha256", "tokenizer_sha256",
            "dataset_revision", "max_length", "seed")
    identity = {**manifest["identity"], "seed": SEED}
    if any(old.get(k) != identity.get(k) for k in keys):
        print("Pilot cache provenance differs; creating fresh features instead.", flush=True)
        return 0
    previous_splits = json.loads(records_path.read_text())
    comparable = ("id", "ids", "positions", "target", "order")
    count = 0
    for split, records in manifest["splits"].items():
        lookup = {digest({k: r[k] for k in comparable}): r for r in previous_splits[split]}
        for record in records:
            path = args.output_dir / "cache" / record["cache"]
            old_record = lookup.get(digest({k: record[k] for k in comparable}))
            if path.exists() or old_record is None:
                continue
            original = args.reuse_pilot / old_record["cache"]
            if not original.exists():
                continue
            hidden = torch.load(original, weights_only=True, map_location="cpu")
            if hidden.shape != (len(record["ids"]), 3840) or hidden.dtype != torch.bfloat16:
                raise ValueError("Pilot feature shape/dtype mismatch")
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_save({"hidden": hidden, "baseline": torch.tensor(old_record["baseline_logits"])}, path)
            count += 1
    print(f"Imported {count} audited pilot feature records", flush=True)
    return count


def cache_estimate(args, manifest):
    records = [r for rows in manifest["splits"].values() for r in rows]
    root = args.output_dir / "cache"
    estimate = sum(len(r["ids"]) * 3840 * 2 + 8192 for r in records)
    remaining = sum(len(r["ids"]) * 3840 * 2 + 8192 for r in records if not (root / r["cache"]).exists())
    result = {"encoded_examples": len(records), "estimated_cache_gib": estimate / 2**30,
              "estimated_remaining_gib": remaining / 2**30,
              "disk_free_gib": shutil.disk_usage(args.output_dir).free / 2**30}
    print(json.dumps({"feature_plan": result}), flush=True)
    if result["estimated_cache_gib"] > args.max_cache_gib:
        raise ValueError("Feature estimate exceeds --max-cache-gib; reduce data or raise that explicit limit")
    if result["disk_free_gib"] < result["estimated_remaining_gib"] + args.disk_reserve_gib:
        raise ValueError("Insufficient disk space for the feature bank and reserved free space")
    atomic_json(args.output_dir / "cache_estimate.json", result)
    return result


def cache_features(args, implementation, manifest, logger):
    root = args.output_dir / "cache"
    records = [r for rows in manifest["splits"].values() for r in rows]
    missing = [r for r in records if not (root / r["cache"]).exists()]
    if not missing:
        print("Feature bank complete; backbone inference skipped.", flush=True)
        logger.finish()
        return
    if args.device != "cuda":
        raise ValueError("New Gemma features require CUDA; CPU is supported only for cached-head verification")
    print(f"Caching {len(missing)} new requests using frozen Gemma; progress every 100 requests.", flush=True)
    backbone = implementation.load_model(args.base_dir, quantization="bnb-nf4")
    tokenizer = Tokenizer.from_file(str(args.base_dir / "tokenizer.json"))
    letters = [tokenizer.encode(letter, add_special_tokens=False).ids for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    if any(len(ids) != 1 for ids in letters):
        raise ValueError("Baseline requires single-token answer letters")
    letters = torch.tensor([ids[0] for ids in letters], device="cuda")
    started = time.perf_counter()
    completed = len(records) - len(missing)
    with torch.no_grad():
        for index, record in enumerate(missing, 1):
            if STOP_REQUESTED:
                logger.finish("stopped")
                raise SweepStopped()
            if index % 100 == 1 and shutil.disk_usage(root).free < args.disk_reserve_gib * 2**30:
                raise OSError("Disk reserve reached; feature preparation can be resumed")
            ids = torch.tensor([record["ids"]], device="cuda")
            hidden = backbone(ids)
            baseline = backbone.logits(hidden[:, -1], letters[:len(record["target"])])[0].cpu()
            path = root / record["cache"]
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_save({"hidden": hidden[0].cpu(), "baseline": baseline}, path)
            del hidden, ids, baseline
            if index % 100 == 0 or index == len(missing):
                elapsed = time.perf_counter() - started
                values = {"cache/completed": completed + index, "cache/total": len(records),
                          "cache/requests_per_second": index / elapsed,
                          "cache/remaining_seconds": (len(missing) - index) * elapsed / index}
                logger.log(values, completed + index)
                print(json.dumps(values), flush=True)
    del backbone
    gc.collect()
    torch.cuda.empty_cache()
    logger.finish()


def collate(records, root, device):
    if isinstance(root, OnlineFeatures):
        return root.collate(records)
    rows = [torch.load(root / r["cache"], weights_only=True, map_location="cpu")["hidden"] for r in records]
    length, width = max(len(h) for h in rows), rows[0].shape[-1]
    choices = max(len(r["target"]) for r in records)
    hidden = torch.zeros(len(rows), length, width, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    mask = torch.zeros(len(rows), length, dtype=torch.long)
    positions = torch.zeros(len(rows), choices, dtype=torch.long)
    option_mask = torch.zeros_like(positions, dtype=torch.bool)
    targets = torch.zeros(len(rows), choices)
    for i, (record, h) in enumerate(zip(records, rows)):
        hidden[i, :len(h)] = h
        mask[i, :len(h)] = 1
        count = len(record["target"])
        positions[i, :count] = torch.tensor(record["positions"])
        option_mask[i, :count] = True
        targets[i, :count] = torch.tensor(record["target"])
    return tuple(value.to(device) for value in (hidden, mask, positions, option_mask, targets))


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()


@torch.no_grad()
def predictions(head, records, root, batch_size, device, progress=False, description="Evaluate"):
    head.eval()
    result = []
    for start in tqdm(range(0, len(records), batch_size), total=math.ceil(len(records) / batch_size),
                      desc=description, unit="batch", leave=False, disable=not progress):
        if STOP_REQUESTED:
            raise SweepStopped()
        subset = records[start:start + batch_size]
        tensors = collate(subset, root, device)
        with autocast(device):
            logits = head(*tensors[:4]).float().cpu()
        result.extend(row[:len(record["target"])] for row, record in zip(logits, subset))
    return result


def measure(logits, records, temperature=1.0, source_shares=None):
    if any(not torch.isfinite(row).all() for row in logits):
        raise FloatingPointError("Non-finite predictions")
    result = metrics(logits, records, temperature)
    # Stable, unclipped soft-target CE, matching the training objective.
    result["cross_entropy"] = sum(float(-(torch.tensor(r["target"]) *
                                             (l.float() / temperature).log_softmax(-1)).sum())
                                   for l, r in zip(logits, records)) / len(records)
    result.update(breakdown(logits, records, temperature, source_shares))
    return result


def grid(args):
    return [{"train_rows": rows, "lr": lr, "weight_decay": wd, "batch_size": batch,
             "width": width, "low_steps": low, "high_cycles": high, "seed": seed,
             "attention_heads": args.attention_heads, "epochs": args.epochs,
             "eval_every": args.eval_every, "patience": args.patience,
             "keep_best_checkpoints": args.keep_best_checkpoints,
             "lr_schedule": args.lr_schedule, "warmup_steps": args.warmup_steps,
             "warmup_fraction": args.warmup_fraction, "min_lr_fraction": args.min_lr_fraction,
             "gradient_clip": args.gradient_clip,
             "decay_fraction": args.decay_fraction, "loss_weighting": args.loss_weighting,
             "weighting_policy": json.loads(args.weighting_config.read_text()) if args.weighting_config else {},
             "selection_metric": args.selection_metric, "optimizer": "adamw",
             "feature_mode": args.feature_mode, "backbone_batch_size": args.backbone_batch_size}
            for rows, lr, wd, batch, width, low, high, seed in itertools.product(
                args.train_sizes, args.learning_rates, args.weight_decays, args.batch_sizes,
                args.widths, args.low_steps, args.high_cycles, args.seeds)]


def learning_rate(config, step, total):
    warmup = config.get("warmup_steps")
    if warmup is None:
        warmup = round(total * config["warmup_fraction"])
    if step < warmup:
        return config["lr"] * (step + 1) / warmup
    if config.get("lr_schedule", "cosine") == "constant":
        return config["lr"]
    start_decay = warmup
    if config.get("lr_schedule") == "wsd":
        start_decay = max(warmup, math.floor(total * (1 - config.get("decay_fraction", .2))))
        if step <= start_decay:
            return config["lr"]
    progress = min(1.0, (step - start_decay) / max(1, total - start_decay - 1))
    return config["lr"] * (config["min_lr_fraction"] +
                           (1 - config["min_lr_fraction"]) * (1 + math.cos(math.pi * progress)) / 2)


def make_head(config, device):
    torch.manual_seed(config["seed"])
    if device == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    return GemmaDecisionHead(width=config["width"], attention_heads=config["attention_heads"],
                             low_steps=config["low_steps"], high_cycles=config["high_cycles"]).to(device)


def trial_name(config):
    return f"n{config['train_rows']}-lr{config['lr']:g}-{digest(config)[:10]}"


def save_progress(path, head, optimizer, state, config):
    atomic_save({"head": head.state_dict(), "optimizer": optimizer.state_dict(), "state": state,
                 "config": config, "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if next(head.parameters()).is_cuda else []}, path)


def publish_checkpoints(directory, state, config):
    """Rebuild the index from committed resume state before pruning old files."""
    retained = state["best_checkpoints"]
    if retained:
        best = torch.load(directory / retained[0]["file"], weights_only=True, map_location="cpu")
        atomic_save(best["head"], directory / "best.pt")
    entries = [{"rank": rank, **entry, "sha256": file_hash(directory / entry["file"])}
               for rank, entry in enumerate(retained, 1)]
    atomic_json(directory / "checkpoints.json", {
        "format_version": 1, "selection": config.get("selection_metric", "cross_entropy") + "; minimum fixed-validation metric; earlier step breaks ties",
        "keep_best_checkpoints": config["keep_best_checkpoints"], "checkpoints": entries,
        "best_head_sha256": file_hash(directory / "best.pt") if retained else None,
    })
    # last.pt is already committed. If publication was interrupted, resume
    # rebuilds the index/alias before deleting only these managed snapshots.
    keep = {entry["file"] for entry in retained}
    for path in directory.glob("checkpoint-step-*.pt"):
        if path.name not in keep:
            path.unlink()


def save_validation_checkpoint(directory, head, optimizer, state, config, row):
    """Keep the top K immutable validation snapshots plus rolling last.pt."""
    if not math.isfinite(row["validation_cross_entropy"]):
        raise FloatingPointError("Non-finite validation loss")
    candidate = {"step": row["step"], "epoch": row["epoch"],
                 "validation_cross_entropy": row["validation_cross_entropy"],
                 "validation_accuracy": row["validation_accuracy"],
                 "file": f"checkpoint-step-{row['step']:07d}.pt"}
    ranked = sorted([*state["best_checkpoints"], candidate],
                    key=lambda entry: (entry["validation_cross_entropy"], entry["step"]))
    state["best_checkpoints"] = ranked[:config["keep_best_checkpoints"]]
    if candidate in state["best_checkpoints"]:
        save_progress(directory / candidate["file"], head, optimizer, state, config)
    # Commit the new ranking with the exact optimizer/model position before
    # publishing its index or evicting any formerly retained checkpoint.
    save_progress(directory / "last.pt", head, optimizer, state, config)
    publish_checkpoints(directory, state, config)


def train_trial(args, manifest, config, run_identity, feature_source=None):
    name = trial_name(config)
    directory = args.output_dir / "runs" / name
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary["status"] == "completed":
            print(f"Skipping completed trial {name}", flush=True)
            return summary
    root = feature_source if feature_source is not None else args.output_dir / "cache"
    train = [r for r in manifest["splits"]["train"] if r["source_index"] < config["train_rows"]]
    validation = manifest["splits"]["validation"]
    logger = RunLog(directory, args.output_dir.name + "/" + name,
                    {**config, "experiment_identity": run_identity, "train_encoded_rows": len(train)},
                    args.overfit_url, args.overfit_project)
    head = make_head(config, args.device)
    # Biases and norm gains receive no decay. Keep the historical all-parameter
    # grouping for old checkpoints to preserve their exact resume behavior.
    if "optimizer" in config:
        optimizer = torch.optim.AdamW([
            {"params": [p for p in head.parameters() if p.ndim >= 2], "weight_decay": config["weight_decay"]},
            {"params": [p for p in head.parameters() if p.ndim < 2], "weight_decay": 0.}], lr=config["lr"])
    else:
        optimizer = torch.optim.AdamW(head.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    fitted = fit_weights(train, config.get("weighting_policy")) if config.get("loss_weighting", "none") == "semantic" else None
    if fitted:
        atomic_json(directory / "loss_weights.json", fitted)
    shares = fitted["source_shares"] if fitted else {k: n / len(train) for k,n in
             Counter(fields(r)["dataset_source"] for r in train).items()}
    batches = math.ceil(len(train) / config["batch_size"])
    total_steps = batches * config["epochs"]
    if config.get("warmup_steps", 0) is not None and config.get("warmup_steps", 0) >= total_steps:
        raise ValueError("Warmup must finish before the run ends")
    state = {"epoch": 1, "next_batch": 0, "step": 0, "examples_seen": 0,
             "epoch_loss_sum": 0.0, "epoch_examples": 0, "best_loss": None,
             "best_step": 0, "best_epoch": 0, "bad_checks": 0, "history": [],
             "best_checkpoints": [],
             "elapsed_seconds": 0.0, "early_stopped": False}
    progress_path = directory / "last.pt"
    if progress_path.exists():
        # This is a trusted, local checkpoint created by this experiment.
        saved = torch.load(progress_path, weights_only=True, map_location=args.device)
        if saved["config"] != config:
            raise ValueError("Resume checkpoint config mismatch")
        head.load_state_dict(saved["head"])
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
        publish_checkpoints(directory, state, config)
        torch.set_rng_state(saved["torch_rng"].cpu())
        if saved["cuda_rng"]:
            torch.cuda.set_rng_state_all([r.cpu() for r in saved["cuda_rng"]])
        print(f"Resuming {name} from step {state['step']}", flush=True)
    else:
        save_progress(progress_path, head, optimizer, state, config)
    started = time.perf_counter()
    prior_seconds = state["elapsed_seconds"]
    print(f"Training {name}: {len(train)} encoded examples, at most {total_steps} updates", flush=True)
    bar = tqdm(total=len(train) * config["epochs"], initial=state["examples_seen"],
               desc="Train " + name, unit="example", dynamic_ncols=True, disable=not getattr(args, "progress", False))
    try:
        while state["epoch"] <= config["epochs"] and not state["early_stopped"]:
            epoch = state["epoch"]
            order = list(range(len(train)))
            random.Random(config["seed"] + epoch).shuffle(order)
            for batch_index in range(state["next_batch"], batches):
                head.train()
                chosen = order[batch_index * config["batch_size"]:(batch_index + 1) * config["batch_size"]]
                records = [train[i] for i in chosen]
                tensors = collate(records, root, args.device)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate(config, state["step"], total_steps)
                with autocast(args.device):
                    logits = head(*tensors[:4])
                    loss = weighted_loss(logits, tensors[4], records, fitted)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                state["step"] += 1
                state["examples_seen"] += len(chosen)
                state["next_batch"] = batch_index + 1
                state["epoch_loss_sum"] += loss.item() * len(chosen)
                state["epoch_examples"] += len(chosen)
                state["elapsed_seconds"] = prior_seconds + time.perf_counter() - started
                bar.update(len(chosen))
                bar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{optimizer.param_groups[0]['lr']:.2g}", refresh=False)
                end_epoch = batch_index + 1 == batches
                if state["step"] % args.log_every == 0 or end_epoch:
                    train_metrics = {"train/batch_cross_entropy": loss.item(),
                                "train/epoch_running_cross_entropy": state["epoch_loss_sum"] / state["epoch_examples"],
                                "train/learning_rate": optimizer.param_groups[0]["lr"],
                                "train/gradient_norm": float(norm), "progress/epoch": epoch,
                                "progress/examples_seen": state["examples_seen"],
                                "progress/examples_per_second": state["examples_seen"] / max(state["elapsed_seconds"], 1e-6),
                                "progress/remaining_seconds": max(0, len(train) * config["epochs"] - state["examples_seen"]) *
                                    state["elapsed_seconds"] / state["examples_seen"]}
                    logger.log(train_metrics, state["step"])
                    atomic_json(args.output_dir / "status.json", {"phase": "training", "trial": name,
                        "step": state["step"], "total_steps": total_steps, "updated_at": time.time(), **train_metrics})
                    if feature_source is not None:
                        print(json.dumps({"trial": name, "step": state["step"], **train_metrics}), flush=True)
                if state["step"] % config["eval_every"] == 0 or end_epoch:
                    atomic_json(args.output_dir / "status.json", {"phase": "validation", "trial": name,
                        "step": state["step"], "examples_seen": state["examples_seen"], "updated_at": time.time()})
                    pred = predictions(head, validation, root, args.eval_batch_size, args.device,
                                       progress=getattr(args, "progress", False), description="Validation")
                    score = measure(pred, validation, source_shares=shares)
                    selection_loss = score[config.get("selection_metric", "cross_entropy")]
                    atomic_json(directory / f"validation-step-{state['step']:07d}.json", score)
                    row = {"step": state["step"], "epoch": epoch,
                           "train_cross_entropy": state["epoch_loss_sum"] / state["epoch_examples"],
                           "validation_cross_entropy": selection_loss, "validation_raw_cross_entropy": score["cross_entropy"],
                           "validation_accuracy": score["accuracy"]}
                    state["history"].append(row)
                    if state["best_loss"] is None or selection_loss < state["best_loss"]:
                        state.update(best_loss=selection_loss, best_step=state["step"], best_epoch=epoch, bad_checks=0)
                    else:
                        state["bad_checks"] += 1
                    state["early_stopped"] = config["patience"] > 0 and state["bad_checks"] >= config["patience"]
                    state["elapsed_seconds"] = prior_seconds + time.perf_counter() - started
                    save_validation_checkpoint(directory, head, optimizer, state, config, row)
                    logger.log({"validation/cross_entropy": score["cross_entropy"],
                                "validation/accuracy": score["accuracy"], "validation/brier": score["brier"],
                                "validation/best_cross_entropy": state["best_loss"],
                                "progress/epoch": epoch, **log_values(score)}, state["step"])
                    atomic_json(directory / "history.json", state["history"])
                    print(json.dumps({"trial": name, **row}), flush=True)
                    bar.set_postfix(loss=f"{loss.item():.3f}", val=f"{score['accuracy']:.1%}", best=f"{state['best_loss']:.3f}")
                    atomic_json(args.output_dir / "status.json", {"phase": "training", "trial": name,
                        "step": state["step"], "examples_seen": state["examples_seen"], "updated_at": time.time(), **row})
                if STOP_REQUESTED or state["step"] % args.checkpoint_every == 0:
                    save_progress(progress_path, head, optimizer, state, config)
                if STOP_REQUESTED:
                    raise SweepStopped()
                if state["early_stopped"]:
                    break
            if not state["early_stopped"]:
                state.update(epoch=epoch + 1, next_batch=0, epoch_loss_sum=0.0, epoch_examples=0)
            save_progress(progress_path, head, optimizer, state, config)
        summary = {"status": "completed", "name": name, "config": config,
                   "head_parameters": sum(p.numel() for p in head.parameters()),
                   "train_encoded_rows": len(train), "best_validation_cross_entropy": state["best_loss"],
                   "best_epoch": state["best_epoch"], "best_step": state["best_step"],
                   "steps": state["step"], "examples_seen": state["examples_seen"],
                   "early_stopped": state["early_stopped"], "elapsed_seconds": state["elapsed_seconds"],
                   "history": state["history"], "best_checkpoint_sha256": file_hash(directory / "best.pt"),
                   "retained_checkpoints": json.loads((directory / "checkpoints.json").read_text())["checkpoints"]}
        atomic_json(summary_path, summary)
        logger.finish()
        return summary
    except SweepStopped:
        save_progress(progress_path, head, optimizer, state, config)
        logger.finish("stopped")
        atomic_json(args.output_dir / "status.json", {"phase": "stopped", "trial": name,
                    "step": state["step"], "examples_seen": state["examples_seen"], "updated_at": time.time()})
        raise
    except Exception:
        logger.finish("failed")
        atomic_json(args.output_dir / "status.json", {"phase": "failed", "trial": name,
                    "step": state["step"], "examples_seen": state["examples_seen"], "updated_at": time.time()})
        raise
    finally:
        bar.close()
        del head, optimizer
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()


def write_leaderboard(root, summaries):
    completed = sorted((r for r in summaries if r["status"] == "completed"),
                       key=lambda r: (r["best_validation_cross_entropy"], r["name"]))
    atomic_json(root / "leaderboard.json", completed)
    fields = ["name", "train_rows", "lr", "weight_decay", "batch_size", "width", "low_steps", "high_cycles",
              "seed", "best_validation_cross_entropy", "best_epoch", "best_step", "steps", "elapsed_seconds"]
    temporary = root / "leaderboard.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**r["config"], **r} for r in completed)
    temporary.replace(root / "leaderboard.csv")
    return completed


def evaluate_winner(args, manifest, winner, run_identity, feature_source=None):
    path = args.output_dir / "winner_evaluation.json"
    identity = {"experiment_identity": run_identity, "trial": winner["name"],
                "checkpoint_sha256": winner["best_checkpoint_sha256"]}
    if path.exists() and json.loads(path.read_text())["identity"] == identity:
        print("Winner evaluation already complete.", flush=True)
        return
    root = feature_source if feature_source is not None else args.output_dir / "cache"
    atomic_json(args.output_dir / "status.json", {"phase": "final_evaluation", "trial": winner["name"], "updated_at": time.time()})
    if feature_source is not None:
        feature_source.capture_baseline = True
    directory = args.output_dir / "runs" / winner["name"]
    head = make_head(winner["config"], args.device)
    if file_hash(directory / "best.pt") != identity["checkpoint_sha256"]:
        raise ValueError("Winning checkpoint checksum mismatch")
    head.load_state_dict(torch.load(directory / "best.pt", weights_only=True, map_location=args.device))
    cal = manifest["splits"]["calibration"]
    predicted = predictions(head, cal, root, args.eval_batch_size, args.device,
                            progress=args.progress, description="Calibration")
    stock_cal = (feature_source.stock_predictions(cal) if feature_source is not None else
                 [torch.load(root / r["cache"], weights_only=True)["baseline"] for r in cal])
    temperatures = [0.5 + i * .1 for i in range(46)]
    def cal_loss(values, temperature):
        return sum(float(-(torch.tensor(r["target"]) * (l.float() / temperature).log_softmax(-1)).sum())
                   for l,r in zip(values, cal)) / len(cal)
    temperature = min(temperatures, key=lambda t: cal_loss(predicted, t))
    stock_temperature = min(temperatures, key=lambda t: cal_loss(stock_cal, t))
    train = [r for r in manifest["splits"]["train"] if r["source_index"] < winner["config"]["train_rows"]]
    shares = {k: n / len(train) for k,n in Counter(fields(r)["dataset_source"] for r in train).items()}
    results = {}
    for split in ("test", "ood"):
        records = manifest["splits"][split]
        pred = predictions(head, records, root, args.eval_batch_size, args.device,
                           progress=args.progress, description=split.title())
        stock = (feature_source.stock_predictions(records) if feature_source is not None else
                 [torch.load(root / r["cache"], weights_only=True)["baseline"] for r in records])
        results[split] = {"recurrent": measure(pred, records, source_shares=shares),
                          "recurrent_calibrated": measure(pred, records, temperature, shares),
                          "stock_letter_logits": measure(stock, records, source_shares=shares),
                          "stock_calibrated": measure(stock, records, stock_temperature, shares)}
        atomic_json(args.output_dir / (split + "_predictions.json"), {
            "identity": identity, "rows": [{"id": r["id"], "target": r["target"], "options": fields(r)["options"],
                    "recurrent_logits": p.tolist(), "stock_logits": s.tolist()} for r,p,s in zip(records,pred,stock)]})
    report = {"identity": identity, "selection": winner["config"].get("selection_metric", "cross_entropy") + "; test/OOD never rank trials",
              "temperature": temperature, "stock_temperature": stock_temperature, "results": results,
              "scope": "Frozen text backbone; supervised head only; no adapter training, RL, arguments or CUA execution"}
    atomic_json(path, report)
    logger = RunLog(args.output_dir / "evaluation_log", args.output_dir.name + "/winner-evaluation",
                    identity, args.overfit_url, args.overfit_project)
    logger.log({f"{s}/{model}/{key}": row[key]
                for s, models in results.items() for model, row in models.items()
                for key in ("accuracy", "cross_entropy", "brier")}, 0)
    logger.finish()
    print(json.dumps(report), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-dir", type=Path, required=True)
    p.add_argument("--kiln-model-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=Path("build/open_jev"))
    p.add_argument("--reuse-pilot", type=Path, default=Path("build/gemma4_pilot"))
    p.add_argument("--feature-mode", choices=("online", "cached"), default="cached",
                   help="online runs frozen Gemma during each update; cached exports features first")
    p.add_argument("--backbone-batch-size", type=int, default=1,
                   help="maximum online backbone microbatch; 1 matches audited unpadded feature forwards")
    p.add_argument("--image-tag", required=True)
    p.add_argument("--train-sizes", type=int, nargs="+", default=[1024, 4096, 16384])
    p.add_argument("--learning-rates", type=float, nargs="+", default=[3e-5, 1e-4, 3e-4])
    p.add_argument("--weight-decays", type=float, nargs="+", default=[.01])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[8])
    p.add_argument("--widths", type=int, nargs="+", default=[256])
    p.add_argument("--attention-heads", type=int, default=4)
    p.add_argument("--low-steps", type=int, nargs="+", default=[2])
    p.add_argument("--high-cycles", type=int, nargs="+", default=[2])
    p.add_argument("--seeds", type=int, nargs="+", default=[SEED])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--augment-orders", type=int, default=2)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--validation-rows", type=int, default=1024)
    p.add_argument("--calibration-rows", type=int, default=256)
    p.add_argument("--test-rows", type=int, default=1024)
    p.add_argument("--ood-rows", type=int, default=1024)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=256, help="full fixed validation set, in optimizer updates")
    p.add_argument("--keep-best-checkpoints", type=int, default=3,
                   help="number of full-state snapshots retained by validation CE, in addition to last.pt")
    p.add_argument("--patience", type=int, default=6, help="validation checks without improvement; 0 disables")
    p.add_argument("--lr-schedule", choices=("constant", "cosine", "wsd"), default="cosine")
    p.add_argument("--decay-fraction", type=float, default=.2, help="wsd: fraction of total updates used for cosine tail")
    p.add_argument("--loss-weighting", choices=("none", "semantic"), default="none")
    p.add_argument("--weighting-config", type=Path, help="JSON with class_power, class_cap, sources and tasks loss multipliers")
    p.add_argument("--selection-metric", choices=("cross_entropy", "selection_cross_entropy"), default="cross_entropy")
    p.add_argument("--progress", action=argparse.BooleanOptionalAction, default=False,
                   help="show tqdm training progress even inside Docker")
    p.add_argument("--warmup-steps", type=int, default=None,
                   help="fixed warmup updates; overrides warmup fraction")
    p.add_argument("--warmup-fraction", type=float, default=.05)
    p.add_argument("--min-lr-fraction", type=float, default=.1)
    p.add_argument("--gradient-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--max-cache-gib", type=float, default=128)
    p.add_argument("--disk-reserve-gib", type=float, default=10)
    p.add_argument("--overfit-url", default=os.environ.get("OVERFIT_URL", "http://127.0.0.1:6006"))
    p.add_argument("--overfit-project", default="KneeJerk Gemma decisions")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--plan-only", action="store_true", help="select data and estimate disk; no inference/training")
    p.add_argument("--smoke-only", action="store_true", help="online: longest full batch forward/backward VRAM check; no optimizer update")
    p.add_argument("--prepare-only", action="store_true", help="build feature bank, skip training")
    p.add_argument("--sync-logs-only", action="store_true", help="replay local metrics to Overfit")
    return p


def run(args):
    if args.sync_logs_only:
        sync_tree(args.output_dir, args.overfit_url)
        return 0
    positive = [args.epochs, args.augment_orders, args.max_length, args.validation_rows, args.calibration_rows,
                args.test_rows, args.ood_rows, args.eval_batch_size, args.eval_every, args.log_every,
                args.checkpoint_every, args.keep_best_checkpoints, args.attention_heads, args.backbone_batch_size, *args.train_sizes, *args.batch_sizes,
                *args.widths, *args.low_steps, *args.high_cycles]
    if min(positive) < 1 or args.patience < 0 or any(s < 0 for s in args.seeds):
        raise ValueError("Counts must be positive; patience and seeds nonnegative")
    if any(lr <= 0 or not math.isfinite(lr) for lr in args.learning_rates) or any(wd < 0 or not math.isfinite(wd) for wd in args.weight_decays):
        raise ValueError("Learning rates must be positive and weight decays nonnegative")
    if not 0 <= args.warmup_fraction < 1 or not 0 < args.min_lr_fraction <= 1 or args.gradient_clip <= 0:
        raise ValueError("Invalid learning-rate schedule or gradient clipping")
    if args.warmup_steps is not None and args.warmup_steps < 0:
        raise ValueError("Warmup steps must be nonnegative")
    if not 0 < args.decay_fraction <= 1:
        raise ValueError("Decay fraction must be in (0,1]")
    if any(w % args.attention_heads for w in args.widths):
        raise ValueError("Head widths must be divisible by attention heads")
    if args.feature_mode == "online" and (args.device != "cuda" or args.prepare_only):
        raise ValueError("Online mode requires CUDA and has no prepare-only phase")
    if args.smoke_only and (args.feature_mode != "online" or args.plan_only):
        raise ValueError("Smoke-only requires online mode and cannot be combined with plan-only")
    torch.set_num_threads(8)
    implementation = load_kiln(args.kiln_model_dir)
    manifest = prepare(args, implementation)
    estimate = (cache_estimate(args, manifest) if args.feature_mode == "cached" else
                {"feature_mode": "online", "encoded_examples": sum(len(r) for r in manifest["splits"].values()),
                 "estimated_cache_gib": 0, "estimated_remaining_gib": 0})
    trials = grid(args)
    if len({digest(c) for c in trials}) != len(trials):
        raise ValueError("Repeated grid values create duplicate trials")
    atomic_json(args.output_dir / "plan.json", {"trials": trials, "feature_plan": estimate,
                                               "protocol": "nested balanced train subsets; fixed upstream held-out splits; validation selects winner"})
    print(f"Planned {len(trials)} trials; up to {args.epochs} passes per trial.", flush=True)
    if args.plan_only:
        if args.loss_weighting == "semantic":
            policy = json.loads(args.weighting_config.read_text()) if args.weighting_config else {}
            fitted = fit_weights(manifest["splits"]["train"], policy)
            atomic_json(args.output_dir / "loss_weight_plan.json", fitted)
            print(f"Training-only weight plan: {len(fitted['schemas'])} semantic schemas; source shares {fitted['source_shares']}", flush=True)
        atomic_json(args.output_dir / "status.json", {"phase": "plan_ready", "updated_at": time.time(),
                    "training_rows": max(args.train_sizes), "trials": len(trials), "no_model_loaded": True})
        return 0
    experiment = {"feature_identity": digest(manifest["identity"]), "trials": trials,
                  "training_source_sha256": file_hash(Path(__file__)),
                  "online_source_sha256": file_hash(Path(__file__).with_name("online.py")),
                  "head_source_sha256": file_hash(Path(__file__).with_name("model.py")),
                  "reasoner_source_sha256": file_hash(Path(__file__).parents[1] / "reasoning/model.py"),
                  "weighting_source_sha256": file_hash(Path(__file__).with_name("weighting.py")),
                  "evaluation_source_sha256": file_hash(Path(__file__).with_name("evaluation.py")),
                  "metadata_source_sha256": file_hash(Path(__file__).with_name("decision_metadata.py")),
                  "device": args.device, "torch": torch.__version__, "cuda": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU",
                  "image": args.image_tag, "eval_batch_size": args.eval_batch_size}
    experiment_path = args.output_dir / "experiment.json"
    if experiment_path.exists() and json.loads(experiment_path.read_text()) != experiment:
        raise ValueError("Training identity changed; use a fresh --output-dir to keep results comparable")
    atomic_json(experiment_path, experiment)
    run_identity = digest(experiment)
    feature_source = None
    if args.feature_mode == "online":
        atomic_json(args.output_dir / "status.json", {"phase": "loading_backbone", "updated_at": time.time()})
        print("Loading frozen Gemma for online training; no feature export or cache reads.", flush=True)
        backbone = implementation.load_model(args.base_dir, quantization="bnb-nf4")
        tokenizer = Tokenizer.from_file(str(args.base_dir / "tokenizer.json"))
        letters = [tokenizer.encode(letter, add_special_tokens=False).ids for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
        if any(len(ids) != 1 for ids in letters):
            raise ValueError("Baseline requires single-token answer letters")
        letter_ids = torch.tensor([ids[0] for ids in letters], device=args.device)
        feature_source = OnlineFeatures(backbone, args.device, args.backbone_batch_size, letter_ids)
    else:
        prepare_feature_bank(args, manifest, implementation, estimate)
    if args.prepare_only:
        return 0
    if args.smoke_only:
        config = trials[0]
        head = make_head(config, args.device)
        records = sorted(manifest["splits"]["train"], key=lambda r: len(r["ids"]), reverse=True)[:config["batch_size"]]
        fitted = fit_weights(manifest["splits"]["train"], config["weighting_policy"]) if args.loss_weighting == "semantic" else None
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        tensors = collate(records, feature_source, args.device)
        with autocast(args.device):
            loss = weighted_loss(head(*tensors[:4]), tensors[4], records, fitted)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite smoke loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), config["gradient_clip"], error_if_nonfinite=True)
        torch.cuda.synchronize()
        result = {"experiment_identity": run_identity, "optimizer_updates": 0,
                  "longest_batch_tokens": [len(r["ids"]) for r in records], "batch_rows": len(records),
                  "forward_backward_seconds": time.perf_counter() - started,
                  "loss": float(loss), "gradient_norm": float(norm),
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                  "device_total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
                  "gpu": experiment["gpu"], "image": args.image_tag,
                  "frozen_backbone_gradients": sum(p.grad is not None for p in backbone.parameters())}
        atomic_json(args.output_dir / "smoke_test.json", result)
        atomic_json(args.output_dir / "status.json", {"phase": "smoke_passed", "updated_at": time.time(), **result})
        print(json.dumps(result), flush=True)
        return 0
    summaries = []
    failures = []
    for config in trials:
        if STOP_REQUESTED:
            raise SweepStopped()
        try:
            summaries.append(train_trial(args, manifest, config, run_identity, feature_source))
        except SweepStopped:
            raise
        except Exception:
            failure = {"trial": trial_name(config), "config": config, "traceback": traceback.format_exc()}
            failures.append(failure)
            atomic_json(args.output_dir / "failures.json", failures)
            metadata = args.output_dir / "runs" / trial_name(config) / "logging.json"
            if metadata.exists():
                value = json.loads(metadata.read_text())
                RunLog(metadata.parent, value["name"], value["config"],
                       args.overfit_url, value["project"]).finish("failed")
            print(failure["traceback"], flush=True)
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
        write_leaderboard(args.output_dir, summaries)
    completed = write_leaderboard(args.output_dir, summaries)
    if not completed:
        raise RuntimeError("All trials failed; inspect failures.json")
    evaluate_winner(args, manifest, completed[0], run_identity, feature_source)
    atomic_json(args.output_dir / "completed.json", {"completed_trials": len(completed), "failed_trials": len(failures),
                                                    "winner": completed[0]["name"], "experiment_identity": run_identity})
    atomic_json(args.output_dir / "status.json", {"phase": "completed", "winner": completed[0]["name"], "updated_at": time.time()})
    print(f"Sweep finished. Results: {args.output_dir}/leaderboard.csv", flush=True)
    return 1 if failures else 0


def prepare_feature_bank(args, manifest, implementation, estimate):
    cache_logger = RunLog(args.output_dir / "cache_log", args.output_dir.name + "/feature-preparation",
                          {"feature_identity": digest(manifest["identity"]),
                           "encoded_examples": estimate["encoded_examples"],
                           "estimated_cache_gib": estimate["estimated_cache_gib"]},
                          args.overfit_url, args.overfit_project)
    try:
        import_pilot(args, manifest)
        cache_features(args, implementation, manifest, cache_logger)
    except SweepStopped:
        cache_logger.finish("stopped")
        raise
    except Exception:
        cache_logger.finish("failed")
        raise


def main():
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".sweep.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another runner owns this output directory")
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        try:
            raise SystemExit(run(args))
        except SweepStopped:
            print("Stopped safely. Repeat the same command to resume.", flush=True)
            raise SystemExit(130)


if __name__ == "__main__":
    main()
