"""Plain functions and stateful event handlers; no callback base class."""

import time
from pathlib import Path

from metrics import TrainingMetrics, checkpoint_key
from utils import atomic_json, file_hash


class Checkpoint:
    def __init__(self, directory, *, keep=3, panel="full", every_examples=1000):
        self.directory = Path(directory)
        self.keep, self.panel, self.every = keep, panel, every_examples
        self.retained, self.managed = [], []
        self.last_saved_examples = 0

    def attach(self, trainer):
        for event in ("batch_end", "validation_end", "train_end"):
            trainer.on(event, self, key="checkpoint")
        trainer.on("checkpoint_saved", self.publish, key="checkpoint")

    def __call__(self, trainer, context):
        step, examples = trainer.state["step"], trainer.state["examples_seen"]
        if context["event"] == "batch_end":
            if (
                context["stage"] != "train"
                or not self.every
                or examples // self.every <= self.last_saved_examples // self.every
            ):
                return
        elif context["event"] == "validation_end":
            if context["panel"] != self.panel:
                return
            report = context["metrics"]
            filename = f"checkpoint-step-{step}.pt"
            candidate = {
                "file": filename,
                "step": step,
                "examples_seen": examples,
                **{
                    k: report[k]
                    for k in (
                        "macro_track_accuracy",
                        "macro_track_cross_entropy",
                        "accuracy",
                        "macro_category_cross_entropy",
                    )
                },
            }
            self.retained = sorted(
                [r for r in self.retained if r["step"] != step] + [candidate],
                key=checkpoint_key,
            )[: self.keep]
            if candidate in self.retained:
                if filename not in self.managed:
                    self.managed.append(filename)
                trainer.request_save(self.directory / filename, validation_full=report)
        self.last_saved_examples = examples
        trainer.request_save(self.directory / "last.pt")

    def publish(self, trainer, context):
        if Path(context["path"]).name != "last.pt":
            return
        keep = {row["file"] for row in self.retained}
        # Only paths created by this callback may be pruned.
        for filename in self.managed:
            if filename not in keep:
                (self.directory / filename).unlink(missing_ok=True)
        rows = [
            {**r, "sha256": file_hash(self.directory / r["file"])}
            for r in self.retained
            if (self.directory / r["file"]).exists()
        ]
        atomic_json(
            self.directory / "checkpoints.json",
            {
                "selection_panel": self.panel,
                "selection": "equal-track macro accuracy; macro-track CE tie-breaker; earlier step",
                "checkpoints": rows,
            },
        )

    def state_dict(self):
        return {
            "retained": self.retained,
            "managed": self.managed,
            "last_saved_examples": self.last_saved_examples,
        }

    def load_state_dict(self, saved):
        self.retained, self.managed = saved["retained"], saved["managed"]
        self.last_saved_examples = saved["last_saved_examples"]


class EarlyStopping:
    def __init__(
        self, *, panel="full", patience=8, min_examples=24000, min_delta=0.002
    ):
        self.panel, self.patience, self.min_examples, self.min_delta = (
            panel,
            patience,
            min_examples,
            min_delta,
        )
        self.anchor, self.stale = None, 0

    def __call__(self, trainer, context):
        if context["panel"] != self.panel:
            return
        value = context["metrics"]["macro_track_cross_entropy"]
        if self.anchor is None or value < self.anchor - self.min_delta:
            self.anchor, self.stale = value, 0
        elif trainer.state["examples_seen"] >= self.min_examples:
            self.stale += 1
        if self.stale >= self.patience:
            trainer.request_stop(
                "early stopping: no meaningful validation CE improvement"
            )

    def state_dict(self):
        return {"anchor": self.anchor, "stale": self.stale}

    def load_state_dict(self, state):
        self.anchor, self.stale = state["anchor"], state["stale"]


class Observations(TrainingMetrics):
    def forward(self, trainer, context):
        if context["stage"] == "train":
            super().forward(trainer, context)

    def batch(self, trainer, context):
        if context["stage"] == "train":
            super().batch(trainer, context)

    def state_dict(self):
        return {
            **{
                k: getattr(self, k)
                for k in (
                    "rows",
                    "ce",
                    "hits",
                    "batches",
                    "clipped_batches",
                    "maximum_gradient_norm",
                )
            },
            "tail": list(self.tail),
            "categories": dict(self.categories),
        }

    def load_state_dict(self, saved):
        from collections import deque

        for key, value in saved.items():
            if key == "tail":
                self.tail = deque(value, maxlen=self.tail.maxlen)
            elif key == "categories":
                self.categories.clear()
                self.categories.update(value)
            else:
                setattr(self, key, value)


class ExperimentLogger:
    def __init__(self, directory, log, every_steps=10):
        self.directory, self.log, self.every = Path(directory), log, every_steps

    def __call__(self, trainer, context):
        event = context["event"]
        if event == "batch_end" and context["stage"] == "train":
            if trainer.state["step"] % self.every == 0:
                self.log.log(
                    {"train/" + k: v for k, v in context["metrics"].items()},
                    trainer.state["examples_seen"],
                )
            self.status(trainer, "training")
        elif event == "validation_start":
            self.status(trainer, "validating", panel=context["panel"])
        elif event == "validation_end":
            panel, report = context["panel"], context["metrics"]
            atomic_json(
                self.directory
                / f"validation-{panel}-step-{trainer.state['step']}.json",
                report,
            )
            values = {
                f"validation/{panel}/{key}": report[key]
                for key in (
                    "accuracy",
                    "cross_entropy",
                    "macro_category_cross_entropy",
                    "macro_track_accuracy",
                    "macro_track_cross_entropy",
                )
            }
            for group in ("category", "dataset_source", "task"):
                for name, result in report.get("by_" + group, {}).items():
                    for key in ("accuracy", "cross_entropy"):
                        values[f"validation-full/{panel}/{group}/{name}/{key}"] = (
                            result[key]
                        )
            self.log.log(values, trainer.state["examples_seen"])
            print(
                f"{panel}: {report['accuracy']:.3%} accuracy; macro-track CE {report['macro_track_cross_entropy']:.6f}",
                flush=True,
            )
        elif event == "train_start":
            self.status(trainer, "training")
        elif event == "train_end":
            phase = "stopped" if trainer.state["stop_requested"] else "completed"
            self.status(trainer, phase)
            self.log.finish(phase)
        elif event == "exception":
            self.status(trainer, "failed", error=str(context["error"]))
            self.log.finish("failed")

    def status(self, trainer, phase, **extra):
        atomic_json(
            self.directory / "status.json",
            {"phase": phase, **trainer.state, "updated_at": time.time(), **extra},
        )
