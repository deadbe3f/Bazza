"""Train or continue a Bazza head on the fixed feature bank; no backbone loading."""

import argparse
import math
from pathlib import Path
import random
import time
import json

import torch
from tqdm.auto import tqdm

from architecture.data import digest, file_hash
from architecture.trainer import Trainer
from architecture.model import make_head
from architecture.metrics import TrainingMetrics
from architecture.optimizers import make_optimizer
from gemma_decisions.sweep import atomic_save
from gemma_decisions.tracking import RunLog, atomic_json
from gemma_decisions.validation_cache import length_batches


def training_batches(rows, batch_size, token_budget, seed):
    """Shuffle examples, bucket locally by length, then shuffle the batches."""
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    batches = []
    for start in range(0, len(rows), batch_size * 16):
        batches.extend(length_batches(rows[start:start + batch_size * 16], batch_size, token_budget))
    rng.shuffle(batches)
    return batches


def load_bank(bank_dir, data_dir):
    index = json.loads((bank_dir / "index.json").read_text())
    if not index["complete"]:
        raise ValueError("Feature export is incomplete; finish/resume export first")
    dataset = json.loads((data_dir / "manifest.json").read_text())
    if file_hash(data_dir / "manifest.json") != index["identity"]["dataset_manifest_sha256"]:
        raise ValueError("Feature bank belongs to different development data")
    if file_hash(bank_dir / "records.json") != dataset["files"]["selection.json"]["sha256"]:
        raise ValueError("Feature-bank records changed")
    return index, json.loads((bank_dir / "records.json").read_text())


def verify_features(bank_dir, index, rows):
    needed = {row["id"]: index["files"][row["id"]] for row in rows}
    for identity, item in tqdm(needed.items(), desc="Verify feature bank", unit="file"):
        if file_hash(bank_dir / item["file"]) != item["sha256"]:
            raise ValueError("Feature checksum changed: " + identity)
    return {identity: item["sha256"] for identity, item in needed.items()}


def train(args, verified_features=None):
    index, records = load_bank(args.bank_dir, args.data_dir)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    head = make_head(index["identity"]["hidden_width"], args.width,
                    args.attention_heads, args.low_steps, args.high_cycles, args.head_type)
    initial_path = getattr(args, "continue_from", None)
    initial = torch.load(initial_path, weights_only=True, map_location="cpu") if initial_path else None
    start_epoch = 0
    if initial is not None:
        previous = initial["config"]
        if previous["feature_identity"] != index["identity"]:
            raise ValueError("Continuation checkpoint belongs to a different feature bank")
        for key in ("width", "head_type", "attention_heads", "low_steps", "high_cycles", "optimizer"):
            if previous.get(key, "recurrent" if key == "head_type" else None) != getattr(args, key):
                raise ValueError("Continuation architecture/optimizer differs: " + key)
        head.load_state_dict(initial["head"], strict=True)
        start_epoch = initial["state"].get("completed_epochs", initial["state"]["epoch"] + 1)
    batches = [training_batches(records["splits"]["train"], args.batch_size, args.token_budget,
                                args.seed + start_epoch + epoch) for epoch in range(args.epochs)]
    total_steps = sum(len(rows) for rows in batches)
    configuration = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                     if k not in ("bank_dir", "data_dir", "output_dir", "plan_only", "overfit_url")}
    configuration.update(feature_identity=index["identity"], head_parameters=sum(p.numel() for p in head.parameters()),
                         torch=str(torch.__version__), cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
                         head_source_sha256=file_hash(Path(__file__).parents[1] / "gemma_decisions/model.py"),
                         reasoner_source_sha256=file_hash(Path(__file__).parents[1] / "reasoning/model.py"),
                         trainer_source_sha256=file_hash(Path(__file__).with_name("trainer.py")),
                         reader_source_sha256=file_hash(Path(__file__).with_name("model.py")),
                         metrics_source_sha256=file_hash(Path(__file__).with_name("metrics.py")),
                         optimizer_source_sha256=file_hash(Path(__file__).with_name("optimizers.py")),
                         train_source_sha256=file_hash(__file__),
                         total_steps=total_steps, selection_panel={"final-full": "final fixed 4K full validation",
                             "epoch-full": "fixed 4K full validation after every epoch",
                             "epoch-quick": "fixed 1K quick validation", "none": None}[args.validation_mode],
                         initialization="continued saved head" if initial is not None else "fresh random head",
                         initial_checkpoint_sha256=file_hash(initial_path) if initial_path else None,
                         prior_completed_epochs=start_epoch, added_epochs=args.epochs,
                         optimizer_state="restored momentum" if initial is not None else "fresh",
                         schedule_state="new phase warmup/steady/cooldown", backbone_loaded=False,
                         loss="unweighted native/teacher CE")
    if args.plan_only:
        print(json.dumps(configuration, indent=2)); return
    if (args.output_dir / "experiment.json").exists():
        raise ValueError("Run already exists; choose a new output directory to preserve its checkpoints")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "experiment.json", configuration)
    head = head.to(args.device)
    optimizer = make_optimizer(head, args.optimizer, args.lr, args.weight_decay,
                               args.beta2, args.muon_lr, args.fused_adamw,
                               True if getattr(args, "foreach_adamw", False) else None)
    if initial is not None:
        peaks = [group["lr"] for group in optimizer.param_groups]
        optimizer.load_state_dict(initial["optimizer"])
        for group, peak in zip(optimizer.param_groups, peaks):
            group["lr"] = group["initial_lr"] = peak
    def rate(step):
        if args.warmup_steps and step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        if args.schedule == "constant" or step < total_steps * .8:
            return 1.
        fraction = min(1., (step - total_steps * .8) / max(1., total_steps * .2))
        return .1 + .9 * (1 + math.cos(math.pi * fraction)) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    trainer = Trainer(head, optimizer, scheduler, args.bank_dir, args.device)
    if initial is not None:
        trainer.state = {**initial["state"], "completed_epochs": start_epoch}
        torch.set_rng_state(initial["rng"])
        if args.device == "cuda" and initial.get("cuda_rng"):
            torch.cuda.set_rng_state_all(initial["cuda_rng"])
    logger = RunLog(args.output_dir / "logging", args.output_dir.name, configuration,
                    args.overfit_url, args.overfit_project)
    retained = []
    validation_history = []
    validation_seconds = [0.]
    validation_started = [0.]
    observations = TrainingMetrics()
    def log_batch(trainer, context):
        if trainer.state["step"] % 10 == 0:
            logger.log({"train/" + k: v for k, v in context["metrics"].items()}, trainer.state["step"])
        atomic_json(args.output_dir / "status.json", {"phase": "training", **trainer.state, "updated_at": time.time()})
    def log_validation(trainer, context):
        panel, score = context["panel"], context["metrics"]
        atomic_json(args.output_dir / ("validation-%s-step-%s.json" % (panel, trainer.state["step"])), score)
        values = {"validation/%s/%s" % (panel, k): score[k] for k in
                  ("accuracy", "cross_entropy", "macro_category_cross_entropy")}
        for category, entry in score["by_category"].items():
            for key in ("accuracy", "cross_entropy"):
                values["validation/%s/category/%s/%s" % (panel, category, key)] = entry[key]
        logger.log(values, trainer.state["step"])
        validation_history.append({"epoch": trainer.state["epoch"] + 1,
                                   "step": trainer.state["step"], "panel": panel, **score})
        atomic_json(args.output_dir / "validation-history.json", validation_history)
        print("Epoch %s %s validation: accuracy %.3f%% | CE %.6f | macro-category CE %.6f" %
              (trainer.state["epoch"] + 1, panel, score["accuracy"] * 100,
               score["cross_entropy"], score["macro_category_cross_entropy"]), flush=True)
    def save_checkpoint(trainer, context):
        score = context["metrics"]
        panel = context.get("panel", "quick") if score is not None else None
        path = args.output_dir / ("checkpoint-step-%s.pt" % trainer.state["step"])
        saved = {"head": head.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "config": configuration, "state": trainer.state.copy(),
                 "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if args.device == "cuda" else []}
        if panel is not None:
            saved["validation_" + panel] = score
        atomic_save(saved, path)
        atomic_save(saved, args.output_dir / "last.pt")
        retained[:] = [row for row in retained if row["file"] != path.name]
        retained.append({"file": path.name, "step": trainer.state["step"],
                         "validation_panel": panel,
                         **({key: score[key] for key in ("macro_category_cross_entropy", "accuracy")} if score else {}),
                         "sha256": file_hash(path)})
        retained.sort(key=lambda row: (row.get("macro_category_cross_entropy", float("inf")), -row["step"]))
        keep = {row["file"] for row in retained[:3]}
        scored = [row for row in retained if "accuracy" in row]
        if scored:
            keep.add(max(scored, key=lambda row: row["accuracy"])["file"])
        for removed in [row for row in retained if row["file"] not in keep]:
            (args.output_dir / removed["file"]).unlink()
        retained[:] = [row for row in retained if row["file"] in keep]
        atomic_json(args.output_dir / "checkpoints.json", {"selection_panel": panel,
                    "best": retained[0] if score else None,
                    "best_accuracy": max(scored, key=lambda row: row["accuracy"]) if scored else None,
                    "retained": retained})
    def validation_start(trainer, context):
        if args.device == "cuda":
            torch.cuda.synchronize()
        validation_started[0] = time.perf_counter()
        atomic_json(args.output_dir / "status.json", {"phase": "validating_" + context["panel"],
                                                     **trainer.state, "updated_at": time.time()})
    def validation_end(trainer, context):
        if args.device == "cuda":
            torch.cuda.synchronize()
        validation_seconds[0] += time.perf_counter() - validation_started[0]
    trainer.on("batch_end", log_batch)
    trainer.on("batch_end", observations.batch)
    trainer.on("after_forward", observations.forward)
    trainer.on("validation_start", validation_start)
    trainer.on("validation_end", validation_end)
    trainer.on("validation_end", log_validation)
    trainer.on("epoch_end", save_checkpoint)
    try:
        print("Project: %s | Run: %s" % (args.overfit_project, args.output_dir.name), flush=True)
        print("%s %s head: %s parameters, width %s, %s recurrent updates" %
              ("Continued" if initial is not None else "Fresh", args.head_type,
               f"{configuration['head_parameters']:,}", args.width, args.low_steps * args.high_cycles), flush=True)
        print("8K cached training | %s passes | %s optimizer updates | LR %g" %
              (args.epochs, total_steps, args.lr), flush=True)
        atomic_json(args.output_dir / "status.json", {"phase": "preparing_cache", **trainer.state,
                                                     "updated_at": time.time()})
        needed = list(records["splits"]["train"])
        if args.validation_mode == "epoch-quick":
            needed.extend(records["splits"]["validation-quick"])
        if args.validation_mode in ("final-full", "epoch-full") or args.full_validation:
            needed.extend(records["splits"]["validation-full"])
        verification_started = time.perf_counter()
        scan_features = getattr(args, "verify_features", False)
        if verified_features is None:
            if scan_features:
                verified_features = verify_features(args.bank_dir, index, needed)
            else:
                verified_features = {row["id"]: index["files"][row["id"]]["sha256"] for row in needed}
        elif any(verified_features.get(row["id"]) != index["files"][row["id"]]["sha256"] for row in needed):
            raise ValueError("Shared verification does not cover these features")
        verification_seconds = time.perf_counter() - verification_started
        panel = "full" if args.validation_mode == "epoch-full" else "quick"
        validation_batches = (list(length_batches(records["splits"]["validation-" + panel],
                                                  args.batch_size, args.token_budget))
                              if args.validation_mode in ("epoch-quick", "epoch-full") else None)
        if initial is not None and args.validation_mode == "epoch-full" and "validation_full" in initial:
            initial_score = initial["validation_full"]
            log_validation(trainer, {"panel": "full", "metrics": initial_score})
            save_checkpoint(trainer, {"panel": "full", "metrics": initial_score})
        del initial
        if args.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        training_started = time.perf_counter()
        trainer.train(args.epochs, batches, validation_batches, panel)
        if args.device == "cuda":
            torch.cuda.synchronize()
        training_and_validation_seconds = time.perf_counter() - training_started
        training_seconds = training_and_validation_seconds - validation_seconds[0]
        summary = {**observations.report(), "training_seconds": training_seconds,
                   "training_examples_per_second": observations.rows / training_seconds,
                   "cache_setup_seconds": verification_seconds,
                   "feature_file_hash_scan": scan_features,
                   "feature_hash_manifest_sha256": digest(verified_features),
                   "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else 0.,
                   "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30 if args.device == "cuda" else 0.,
                   "head_parameters": configuration["head_parameters"],
                   "epoch_validation_seconds": validation_seconds[0],
                   "training_and_epoch_validation_seconds": training_and_validation_seconds,
                   "validation_mode": args.validation_mode,
                   "metric_scope": "training forward before each optimizer update; tail is last 1000 examples",
                   "timing_scope": "training plus logging/checkpoints; excludes startup, feature verification and final validation"}
        if args.validation_mode == "final-full":
            atomic_json(args.output_dir / "status.json", {"phase": "validating_full", **trainer.state,
                                                         "updated_at": time.time()})
            score = trainer.evaluate(list(length_batches(records["splits"]["validation-full"],
                                            args.batch_size, args.token_budget)), "full")
            summary["validation_full"] = score
            save_checkpoint(trainer, {"panel": "full", "metrics": score})
            print("Final cached validation (%s rows): accuracy %.3f%% | CE %.6f | macro-category CE %.6f" %
                  (score["rows"], score["accuracy"] * 100, score["cross_entropy"],
                   score["macro_category_cross_entropy"]), flush=True)
            for category, entry in score["by_category"].items():
                print("  %-20s %6.2f%%  CE %.4f  n=%s" %
                      (category, entry["accuracy"] * 100, entry["cross_entropy"], entry["rows"]), flush=True)
        elif args.validation_mode == "epoch-full":
            summary["validation_full"] = validation_history[-1]
            summary["best_validation_full"] = min(validation_history, key=lambda row: row["macro_category_cross_entropy"])
            summary["best_accuracy_full"] = max(validation_history, key=lambda row: row["accuracy"])
        elif args.full_validation:
            best = torch.load(args.output_dir / retained[0]["file"], weights_only=True, map_location=args.device)
            head.load_state_dict(best["head"])
            trainer.state = best["state"].copy()
            trainer.evaluate(list(length_batches(records["splits"]["validation-full"], args.batch_size, args.token_budget)), "full")
        atomic_json(args.output_dir / "status.json", {"phase": "completed", **trainer.state, "updated_at": time.time()})
        atomic_json(args.output_dir / "summary.json", summary)
        logger.log({"summary/" + k: v for k, v in summary.items() if isinstance(v, (int, float))}, trainer.state["step"])
        print("Training: %.2fs | online CE %.6f | last %s examples CE %.6f | peak allocated %.3f GiB" %
              (training_seconds, summary["training_online_ce"], summary["training_tail_rows"],
               summary["training_tail_ce"], summary["peak_allocated_gib"]), flush=True)
        logger.finish()
        return summary
    except BaseException as error:
        atomic_json(args.output_dir / "status.json", {"phase": "failed", **trainer.state, "error": str(error)})
        logger.finish("failed")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank-dir", "data-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--head-type", choices=("recurrent", "norm", "mlp", "bigru"), default="recurrent")
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--low-steps", type=int, default=2)
    parser.add_argument("--high-cycles", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--continue-from", type=Path,
                        help="Restore head/optimizer/RNG into a fresh output directory with a new phase schedule")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--token-budget", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--optimizer", choices=("adamw", "muon"), default="adamw")
    parser.add_argument("--beta2", type=float, default=.999)
    parser.add_argument("--muon-lr", type=float, default=.01)
    parser.add_argument("--fused-adamw", action="store_true")
    parser.add_argument("--foreach-adamw", action="store_true",
                        help="Use foreach AdamW, matching the original baseline's automatic CUDA dispatch")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--schedule", choices=("wsd", "constant"), default="wsd")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument("--validation-mode", choices=("epoch-quick", "epoch-full", "final-full", "none"), default="epoch-quick",
                        help="epoch-full: cached 4K after every epoch; final-full: one at end; none: no validation")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--verify-features", action="store_true",
                        help="Opt in to scanning cached feature-file hashes; validation is controlled separately")
    parser.add_argument("--overfit-url", default="http://127.0.0.1:6006")
    parser.add_argument("--overfit-project", default="Bazza architecture")
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--image-id", required=True)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.token_budget, args.lr) <= 0 or args.warmup_steps < 0:
        parser.error("Training sizes and LR must be positive; warmup cannot be negative")
    if args.validation_mode == "none" and args.full_validation:
        parser.error("--validation-mode none cannot be combined with --full-validation")
    if args.fused_adamw and args.foreach_adamw:
        parser.error("Choose either fused or foreach AdamW")
    train(args)


if __name__ == "__main__":
    main()
