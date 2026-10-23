"""Train Bazza from Python settings; scripts/run.sh manages Docker job lifetime."""

import argparse
import json
import random
import signal
import time
from functools import partial
from pathlib import Path

import torch

from callbacks import Checkpoint, EarlyStopping, ExperimentLogger, Observations
from configuration import (
    load_config,
    make_model,
    prepare,
    record_settings,
    warmup_steady_cosine,
)
from data import BatchLoader, length_batches, training_batches
from logging_backend import RunLog
from losses import DecisionLoss
from optimizers import make_optimizer
from trainer import Trainer
from utils import atomic_json, file_hash, job_lock


def _epoch_batches(epoch, *, rows, config):
    return training_batches(
        rows, config.batch_size, config.token_budget, config.seed + epoch
    )


def main(args):
    config = load_config(args.config)
    selection, backbone, identity = prepare(
        config, args.device, settings_path=args.config
    )
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    rows = selection["splits"]["train"]

    batches = partial(_epoch_batches, rows=rows, config=config)
    if args.plan_only:
        print(
            json.dumps(
                {
                    "train_rows": len(rows),
                    "epochs": config.epochs,
                    "updates": sum(len(batches(e)) for e in range(config.epochs)),
                    "backbone": type(backbone).__name__,
                    "configuration": config.to_dict(),
                },
                indent=2,
            )
        )
        return
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    with job_lock(directory):
        path = directory / "experiment.json"
        if args.initialize_from:
            if args.resume:
                raise ValueError("Choose initialization or resume, not both")
            identity["initial_checkpoint_sha256"] = file_hash(args.initialize_from)
        elif args.resume and path.exists():
            previous = json.loads(path.read_text())
            if "initial_checkpoint_sha256" in previous:
                identity["initial_checkpoint_sha256"] = previous[
                    "initial_checkpoint_sha256"
                ]
        if path.exists():
            if not args.resume or json.loads(path.read_text()) != identity:
                raise ValueError(
                    "Existing/different experiment; exact --resume or a fresh output directory required"
                )
        elif args.resume:
            raise ValueError("No experiment to resume")
        else:
            atomic_json(path, identity)
        record_settings(config, args.config, directory)
        model, readout = make_model(config, selection, backbone=backbone)
        model.to(args.device)
        loader = BatchLoader(
            backbone.prepare_batch, args.device, config.prefetch_batches
        )
        optimizer = make_optimizer(
            model.head,
            name=config.optimizer,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            beta2=config.beta2,
            muon_lr=config.muon_learning_rate,
        )
        trainer = Trainer(
            model,
            optimizer,
            DecisionLoss(config.weighted_loss),
            prepare_batch=loader,
            device=args.device,
            gradient_clip=config.gradient_clip,
            precision=config.precision,
            identity=identity,
            progress=not args.no_progress,
            checkpoint_model=model.head,
            checkpoint_extra={"readout": readout, "config": config.to_dict()},
        )
        trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: (
                warmup_steady_cosine(
                    trainer.state["examples_seen"],
                    total=len(rows) * config.epochs,
                    warmup=config.warmup_examples,
                    decay_start=config.decay_start_examples,
                    minimum=config.minimum_lr_factor,
                    warmup_offset=config.warmup_batch_offset,
                )
                if config.schedule == "wsd"
                else 1.0
            ),
        )
        if args.initialize_from:
            model.head.load_state_dict(
                torch.load(args.initialize_from, map_location="cpu", weights_only=True)[
                    "head"
                ],
                strict=True,
            )
        observations = Observations()
        trainer.on("after_forward", observations.forward, key="observations")
        trainer.on("batch_end", observations.batch, key="observations")
        if config.early_stopping:
            trainer.on(
                "validation_end",
                EarlyStopping(
                    patience=config.early_stop_patience,
                    min_examples=config.early_stop_min_examples,
                    min_delta=config.early_stop_min_delta,
                ),
                key="early_stopping",
            )
        Checkpoint(
            directory,
            keep=config.keep_best_checkpoints,
            every_examples=config.save_every_examples,
        ).attach(trainer)
        log = RunLog(
            directory / "logging",
            directory.name,
            identity,
            config.overfit_url,
            config.overfit_project,
        )
        logger = ExperimentLogger(directory, log, config.log_every_steps)
        for event in (
            "train_start",
            "batch_end",
            "validation_start",
            "validation_end",
            "train_end",
            "exception",
        ):
            trainer.on(event, logger)
        if args.resume and (directory / "last.pt").exists():
            trainer.load_state_dict(
                torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
            )
        previous_handlers = {
            sig: signal.signal(
                sig, lambda signum, frame: trainer.request_stop("signal " + str(signum))
            )
            for sig in (signal.SIGTERM, signal.SIGINT)
        }
        panels = {
            name: list(
                length_batches(
                    selection["splits"][split],
                    config.validation_batch_size,
                    config.validation_token_budget,
                )
            )
            for name, split in config.validation_panels.items()
        }
        started = time.perf_counter()
        try:
            trainer.train(
                batches,
                panels,
                epochs=config.epochs,
                validation_every_examples=config.validation_every_examples,
                initial_validation=config.initial_validation,
            )
            summary = {
                "phase": "stopped" if trainer.state["stop_requested"] else "completed",
                **trainer.state,
                "session_seconds": time.perf_counter() - started,
                "readout": readout,
            }
            if observations.rows:
                summary["training"] = observations.report()
            atomic_json(directory / "summary.json", summary)
            if summary["phase"] == "completed":
                atomic_json(directory / "complete.json", summary)
                if config.test_config_path:
                    from evaluate import main as evaluate

                    retained = json.loads((directory / "checkpoints.json").read_text())[
                        "checkpoints"
                    ]
                    if not retained:
                        raise ValueError(
                            "Automatic test requires a validation-selected checkpoint"
                        )
                    best = retained[0]
                    test_dir = directory / "test"
                    model.head.cpu()
                    if hasattr(backbone, "encoder"):
                        backbone.encoder = None
                    if torch.device(args.device).type == "cuda":
                        torch.cuda.empty_cache()
                    logger.status(trainer, "testing", checkpoint=best)
                    try:
                        report = evaluate(
                            argparse.Namespace(
                                config=config.test_config_path,
                                checkpoint=directory / best["file"],
                                output_dir=test_dir,
                                split=config.test_split,
                                device=args.device,
                                resume=(test_dir / "plan.json").exists(),
                                plan_only=False,
                                no_progress=args.no_progress,
                            )
                        )
                    except BaseException as error:
                        logger.status(trainer, "failed", stage="test", error=str(error))
                        log.finish("failed")
                        raise
                    if report is None:
                        logger.status(trainer, "stopped", stage="test")
                    else:
                        log.log(
                            {
                                "test/macro_track_accuracy": report[
                                    "macro_track_accuracy"
                                ],
                                "test/accuracy": report["accuracy"],
                            },
                            trainer.state["examples_seen"],
                        )
                        atomic_json(
                            directory / "workflow-complete.json",
                            {
                                "selected_checkpoint": best,
                                "test_rows": report["evaluated_rows"],
                                "test_report_sha256": file_hash(
                                    test_dir / "report.json"
                                ),
                            },
                        )
                        logger.status(trainer, "completed", stage="workflow")
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.py")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    main(args)
