"""Export unprojected BF16 token states once; train heads without the backbone.

Feature identity and a checksummed append-only journal make exports resumable.
The original run and its validation cache are opened read-only.
"""

import argparse
import fcntl
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import time
import traceback

import torch
from tokenizers import Tokenizer
from tqdm.auto import tqdm

from architecture.data import digest, file_hash, read_rows
from gemma_decisions.pilot import encode
from gemma_decisions.probe import load_kiln
from gemma_decisions.sweep import atomic_save, digest as legacy_digest
from gemma_decisions.tracking import RunLog, atomic_json
from gemma_decisions.validation_cache import length_batches

STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def input_key(row):
    return digest({k: row[k] for k in ("id", "ids", "positions")})


def unique_records(manifest):
    result = {}
    for rows in manifest["splits"].values():
        for row in rows:
            if row["id"] in result and input_key(row) != input_key(result[row["id"]]):
                raise ValueError("One record id has different inputs")
            result[row["id"]] = row
    return result


def load_entries(bank):
    journal = Path(bank) / "features.jsonl"
    result = {}
    if journal.exists():
        # Discard only a torn final append, keeping all committed rows.
        with journal.open("rb+") as stream:
            data = stream.read()
            if data and not data.endswith(b"\n"):
                stream.truncate(data.rfind(b"\n") + 1)
                data = data[:data.rfind(b"\n") + 1]
        for line in data.splitlines():
            row = json.loads(line)
            if row["id"] in result and result[row["id"]] != row:
                raise ValueError("Conflicting committed feature identities")
            result[row["id"]] = row
    return result


def append_entry(bank, entry):
    with (Path(bank) / "features.jsonl").open("a") as stream:
        stream.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@torch.no_grad()
def forward(backbone, rows):
    length = max(len(r["ids"]) for r in rows)
    ids = torch.zeros(len(rows), length, dtype=torch.long, device="cuda")
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        ids[i, :len(row["ids"])] = torch.tensor(row["ids"], device="cuda")
        mask[i, :len(row["ids"])] = 1
    hidden = backbone(ids, mask)
    if not torch.isfinite(hidden).all():
        raise FloatingPointError("Non-finite frozen backbone features")
    return hidden


@torch.no_grad()
def audit_batching(backbone, records, max_rows, token_budget):
    ordered = sorted(records, key=lambda r: (len(r["ids"]), r["id"]))
    audit_rows = [ordered[int(i * (len(ordered) - 1) / 11)] for i in range(12)]
    references = {}
    torch.cuda.synchronize()
    started = time.perf_counter()
    for row in audit_rows:
        references[row["id"]] = forward(backbone, [row])[0].cpu()
    torch.cuda.synchronize()
    single_seconds = time.perf_counter() - started
    samples = []
    started = time.perf_counter()
    for batch in length_batches(audit_rows, max_rows, token_budget):
        hidden = forward(backbone, batch).cpu()
        for row, value in zip(batch, hidden):
            ref = references[row["id"]].float()
            value = value[:len(row["ids"])].float()
            relative = float((value - ref).square().mean().sqrt() / ref.square().mean().sqrt().clamp_min(1e-8))
            cosine = float(torch.nn.functional.cosine_similarity(value, ref, dim=-1).mean())
            samples.append({"id": row["id"], "tokens": len(row["ids"]),
                            "relative_rms_error": relative, "mean_token_cosine": cosine})
    torch.cuda.synchronize()
    report = {"rows": samples, "single_seconds": single_seconds,
              "batched_seconds": time.perf_counter() - started,
              "max_relative_rms_error": max(r["relative_rms_error"] for r in samples),
              "min_mean_token_cosine": min(r["mean_token_cosine"] for r in samples),
              "requested_batch_size": max_rows, "token_budget": token_budget,
              "bitwise_identical_required": False,
              "thresholds": {"relative_rms_error": .02, "mean_token_cosine": .999}}
    report["passed"] = report["max_relative_rms_error"] <= .02 and report["min_mean_token_cosine"] >= .999
    return report


def run(args):
    args.bank_dir.mkdir(parents=True, exist_ok=True)
    with (args.bank_dir / ".export.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        export(args)


def export(args):
    dataset = json.loads((args.data_dir / "manifest.json").read_text())
    for name, entry in dataset["files"].items():
        if file_hash(args.data_dir / name) != entry["sha256"]:
            raise ValueError("Development data changed: " + name)
    manifest = json.loads((args.data_dir / "selection.json").read_text())
    records = unique_records(manifest)
    implementation = load_kiln(args.kiln_model_dir)
    model_identity = {"checkpoint": implementation.CHECKPOINT, "revision": implementation.REVISION,
                      "weight_sha256": implementation.WEIGHT_SHA256, "quantization": "bnb-nf4",
                      "implementation_sha256": file_hash(args.kiln_model_dir / "model.py"),
                      "tokenizer_sha256": file_hash(args.base_dir / "tokenizer.json"),
                      "serialization_sha256": hashlib_serialization()}
    for key, value in model_identity.items():
        if manifest["identity"][key] != value:
            raise ValueError("Frozen backbone/input identity changed: " + key)
    identity = {**model_identity, "dataset_manifest_sha256": file_hash(args.data_dir / "manifest.json"),
                "hidden_width": 3840, "dtype": "bfloat16", "tensor": "final-normalized-token-states-before-trainable-projection",
                "exporter_sha256": file_hash(__file__), "torch": torch.__version__, "cuda": torch.version.cuda,
                "data_helpers_sha256": file_hash(Path(__file__).with_name("data.py")),
                "gpu": torch.cuda.get_device_name(), "image_tag": args.image_tag, "image_id": args.image_id,
                "requested_batch_size": args.batch_size, "token_budget": args.token_budget}
    identity_path = args.bank_dir / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Feature bank identity changed; choose a new bank directory")
    atomic_json(identity_path, identity)
    # Re-encode selected raw requests to verify the imported tokenized manifest.
    tokenizer = Tokenizer.from_file(str(args.base_dir / "tokenizer.json"))
    for row in read_rows(args.data_dir / "train.jsonl.gz"):
        actual = encode(row, tokenizer, 4096)
        expected = records[row["id"]]
        if actual is None or any(actual[k] != expected[k] for k in actual):
            raise ValueError("Selected raw request differs from tokenized input: " + row["id"])
    atomic_json(args.bank_dir / "records.json", manifest)
    entries = load_entries(args.bank_dir)
    for identity_id, entry in tqdm(list(entries.items()), desc="Verify existing bank", unit="row"):
        if identity_id not in records or entry["input_sha256"] != input_key(records[identity_id]):
            raise ValueError("Unexpected cached input")
        if file_hash(args.bank_dir / entry["file"]) != entry["sha256"]:
            raise ValueError("Cached tensor checksum changed: " + identity_id)
    logger = RunLog(args.bank_dir / "logging", "Gemma 12B architecture feature export",
                    identity, args.overfit_url, "Bazza architecture")
    def status(phase, **extra):
        atomic_json(args.bank_dir / "status.json", {"phase": phase, "completed": len(entries),
                    "total_unique": len(records), "updated_at": time.time(), **extra})
    try:
        # Hardlinks keep validation available independently, without another 12 GiB copy.
        old = json.loads((args.validation_cache / "index.json").read_text())
        for key, value in model_identity.items():
            if old["identity"][key] != value:
                raise ValueError("Validation feature identity differs: " + key)
        status("reusing_validation")
        for row in tqdm(manifest["splits"]["validation-full"], desc="Reuse cached validation", unit="row"):
            if row["id"] in entries:
                continue
            item = old["files"][row["id"]]
            if old["allowed"][row["id"]] != input_key(row):
                raise ValueError("Validation cache input differs")
            src, dest = args.validation_cache / item["file"], args.bank_dir / row["cache"]
            if file_hash(src) != item["sha256"]:
                raise ValueError("Original validation feature checksum changed")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                os.link(src, dest)
            if file_hash(dest) != item["sha256"]:
                raise ValueError("Conflicting validation feature destination")
            entry = {"id": row["id"], "file": row["cache"], "sha256": item["sha256"],
                     "input_sha256": input_key(row), "tokens": len(row["ids"]),
                     "origin": "existing-unpadded-validation", "backbone_batch_size": 1}
            append_entry(args.bank_dir, entry)
            entries[row["id"]] = entry
        missing = [r for r in manifest["splits"]["train"] if r["id"] not in entries]
        required = sum(len(r["ids"]) * 3840 * 2 + 8192 for r in missing)
        if shutil.disk_usage(args.bank_dir).free < required + args.disk_reserve_gib * 2**30:
            raise OSError("Insufficient disk for the selected feature bank and reserve")
        if missing:
            status("loading_backbone", training_remaining=len(missing))
            backbone = implementation.load_model(args.base_dir, quantization="bnb-nf4").requires_grad_(False).eval()
            audit_path = args.bank_dir / "batching-audit.json"
            if audit_path.exists():
                audit = json.loads(audit_path.read_text())
            else:
                status("auditing_batching")
                try:
                    audit = audit_batching(backbone, manifest["splits"]["train"], args.batch_size, args.token_budget)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    audit = {"passed": False, "reason": "Requested batch exceeded VRAM in the audit"}
                audit["actual_batch_size"] = args.batch_size if audit["passed"] else 1
                atomic_json(audit_path, audit)
            batch_size = audit["actual_batch_size"]
            print("Batching audit: " + json.dumps({k: v for k, v in audit.items() if k != "rows"}), flush=True)
            status("exporting", backbone_batch_size=batch_size)
            batches = list(length_batches(missing, batch_size, args.token_budget))
            started, exported = time.perf_counter(), 0
            progress = tqdm(total=len(missing), desc="Cache architecture training", unit="row")
            for batch in batches:
                if STOP:
                    break
                try:
                    hidden = forward(backbone, batch).cpu().to(torch.bfloat16)
                except torch.cuda.OutOfMemoryError:
                    # The already-audited unpadded path is a safe fallback for long inputs.
                    torch.cuda.empty_cache()
                    hidden = None
                for i, row in enumerate(batch):
                    tensor = (forward(backbone, [row])[0].cpu().to(torch.bfloat16)
                              if hidden is None else hidden[i, :len(row["ids"])].clone())
                    if tensor.shape != (len(row["ids"]), 3840):
                        raise ValueError("Unexpected token feature shape")
                    path = args.bank_dir / row["cache"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    atomic_save({"hidden": tensor, "input_sha256": input_key(row),
                                 "feature_identity_sha256": digest(identity)}, path)
                    entry = {"id": row["id"], "file": row["cache"], "sha256": file_hash(path),
                             "input_sha256": input_key(row), "tokens": len(row["ids"]),
                             "origin": "architecture-training-export",
                             "backbone_batch_size": len(batch) if hidden is not None else 1}
                    append_entry(args.bank_dir, entry)
                    entries[row["id"]] = entry
                    exported += 1
                    progress.update(1)
                if exported % 64 < len(batch) or exported == len(missing):
                    elapsed = time.perf_counter() - started
                    values = {"cache/training_completed": len(entries) - 4000, "cache/training_total": 8000,
                              "cache/rows_per_second": exported / max(elapsed, 1e-6)}
                    logger.log(values, len(entries) - 4000)
                    status("exporting", **values, backbone_batch_size=batch_size)
                    progress.set_postfix(rows_per_sec="%.2f" % values["cache/rows_per_second"])
            progress.close()
            del backbone
            torch.cuda.empty_cache()
        complete = len(entries) == len(records)
        atomic_json(args.bank_dir / "index.json", {"identity": identity, "complete": complete, "files": entries})
        status("completed" if complete else "stopped", reusable_validation=4000, exported_training=len(entries) - 4000)
        logger.finish("completed" if complete else "stopped")
    except BaseException as error:
        status("failed", error=str(error), traceback=traceback.format_exc())
        logger.finish("failed")
        raise


def hashlib_serialization():
    # v0.3 stores a hash of the JSON-serialized source string, not raw bytes.
    return legacy_digest(inspect.getsource(encode))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("data-dir", "bank-dir", "base-dir", "kiln-model-dir", "validation-cache"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--disk-reserve-gib", type=float, default=20)
    parser.add_argument("--overfit-url", default="http://127.0.0.1:6006")
    args = parser.parse_args()
    if min(args.batch_size, args.token_budget) < 1:
        parser.error("Batch size and token budget must be positive")
    torch.set_num_threads(8)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    run(args)


if __name__ == "__main__":
    main()
