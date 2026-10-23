"""Evaluate a checkpoint on a fixed serialized panel, with resumable batch commits."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import torch
from tqdm.auto import tqdm
from configuration import load_config, make_model, prepare, record_settings
from data import BatchLoader, length_batches
from metrics import score, selection_metrics
from utils import atomic_json, autocast, digest, file_hash, job_lock


def input_identity(row):
    return digest({k: row.get(k) for k in ("id", "ids", "positions", "options", "order", "target")})


def main(args):
    config = load_config(args.config)
    selection, backbone, identity = prepare(config, args.device, settings_path=args.config)
    rows = selection["splits"][args.split]
    if getattr(args, "example_id", None):
        rows = [row for row in rows if row["id"] == args.example_id]
        if len(rows) != 1:
            raise ValueError("Example ID is not unique/present in the selected split")
    if not rows:
        raise ValueError("Evaluation split is empty")
    batches = list(length_batches(rows, config.validation_batch_size, config.validation_token_budget))
    ordered = [row for batch in batches for row in batch]
    identity = {**identity, "checkpoint_sha256": file_hash(args.checkpoint), "split": args.split,
                "input_sha256": digest([input_identity(row) for row in ordered])}
    if args.plan_only:
        print(json.dumps({"rows": len(rows), "batches": len(batches), "identity": identity}, indent=2))
        return
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    with job_lock(directory):
        path = directory / "plan.json"
        if path.exists():
            if not args.resume or json.loads(path.read_text()) != identity:
                raise ValueError("Existing/different evaluation; exact --resume or a fresh output directory required")
        elif args.resume:
            raise ValueError("No evaluation to resume")
        else:
            atomic_json(path, identity)
        record_settings(config, args.config, directory)
        journal = directory / "predictions.jsonl"
        predictions, boundaries = [], 0
        if journal.exists():
            with journal.open("rb+") as stream:
                for line in stream:
                    if not line.endswith(b"\n"):
                        stream.truncate(stream.tell() - len(line))
                        break
                    entry = json.loads(line)
                    expected = batches[boundaries]
                    if [p["input_sha256"] for p in entry["predictions"]] != [input_identity(row) for row in expected]:
                        raise ValueError("Prediction journal differs from the fixed batch order")
                    predictions.extend(entry["predictions"])
                    boundaries += 1
        stop = [False]
        old_handlers = {sig: signal.signal(sig, lambda signum, frame: stop.__setitem__(0, True))
                        for sig in (signal.SIGTERM, signal.SIGINT)}
        started = time.perf_counter()
        def status(phase, **extra):
            atomic_json(directory / "status.json", {"phase": phase, "evaluated_rows": len(predictions),
                "total_rows": len(rows), "updated_at": time.time(), **extra})
        try:
            status("loading_head")
            saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            model, _ = make_model(config, selection, saved, backbone=backbone)
            model.to(args.device).eval()
            loader = BatchLoader(backbone.prepare_batch, args.device, config.prefetch_batches)
            with torch.no_grad():
                for batch_rows in tqdm(batches[boundaries:], desc="Bazza test", unit="batch", disable=args.no_progress):
                    if stop[0]:
                        status("stopped")
                        return None
                    batch = loader(batch_rows, args.split)
                    with autocast(args.device, config.precision):
                        output = model(batch).logits.float().cpu()
                    if not torch.isfinite(output).all():
                        raise FloatingPointError("Non-finite test logits")
                    chunk = [{"id": row["id"], "input_sha256": input_identity(row),
                              "logits": value[:len(row["target"])].tolist()}
                             for row, value in zip(batch_rows, output)]
                    with journal.open("a") as stream:
                        stream.write(json.dumps({"predictions": chunk}, allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    predictions.extend(chunk)
                    status("evaluating")
            if len(predictions) != len(rows):
                raise ValueError("Incomplete evaluation")
            logits = [torch.tensor(entry["logits"]) for entry in predictions]
            report = {**score(logits, ordered), **selection_metrics(logits, ordered), "complete": True,
                      "evaluated_rows": len(rows), "plan_sha256": file_hash(path),
                      "checkpoint_sha256": identity["checkpoint_sha256"],
                      "session_seconds": time.perf_counter() - started}
            atomic_json(directory / "report.json", report)
            atomic_json(directory / "complete.json", {"rows": len(rows), "report_sha256": file_hash(directory / "report.json")})
            status("completed")
            print(f'{len(rows):,} rows; accuracy {report["accuracy"]:.3%}; equal-track macro {report["macro_track_accuracy"]:.3%}', flush=True)
            return report
        except BaseException as error:
            status("failed", error=str(error))
            raise
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.py"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    main(args)
