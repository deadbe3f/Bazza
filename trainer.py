"""A small, explicit training loop with ordered, named events."""

import random
from collections import defaultdict
from copy import deepcopy

import torch
from tqdm.auto import tqdm

from losses import DecisionLoss
from metrics import score, selection_metrics
from utils import atomic_save, autocast


class StopRequested(Exception):
    pass


class Trainer:
    EVENTS = frozenset(
        {
            "train_start",
            "train_end",
            "epoch_start",
            "epoch_end",
            "batch_start",
            "batch_end",
            "before_forward",
            "after_forward",
            "before_backward",
            "after_backward",
            "before_optimizer_step",
            "after_optimizer_step",
            "validation_start",
            "validation_end",
            "checkpoint_saved",
            "exception",
        }
    )

    def __init__(
        self,
        model,
        optimizer,
        loss_fn=None,
        scheduler=None,
        *,
        prepare_batch=None,
        device="cuda",
        gradient_clip=1.0,
        identity=None,
        progress=True,
        checkpoint_extra=None,
        checkpoint_model=None,
        precision="bf16",
    ):
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.loss_fn = loss_fn or DecisionLoss()
        self.prepare_batch = prepare_batch or (lambda batch, stage: batch.to(device))
        self.device, self.gradient_clip = device, gradient_clip
        self.identity, self.progress = identity or {}, progress
        self.checkpoint_extra = checkpoint_extra or {}
        self.checkpoint_model = (
            checkpoint_model
            if checkpoint_model is not None
            else getattr(model, "checkpoint_module", model)
        )
        self.precision = precision
        self.callbacks = defaultdict(list)
        self.owners = {}
        self.pending_saves = {}
        self.state = {
            "step": 0,
            "examples_seen": 0,
            "epoch": 0,
            "batch_cursor": 0,
            "completed_epochs": 0,
            "initial_validation_done": False,
            "last_validation_examples": -1,
            "stop_requested": False,
            "stop_reason": None,
        }

    def on(self, event, handler, *, key=None):
        if event not in self.EVENTS:
            raise ValueError("Unknown trainer event: " + event)
        owner = getattr(handler, "__self__", handler)
        if hasattr(owner, "state_dict"):
            if not key:
                raise ValueError("Stateful callbacks require a stable key")
            if key in self.owners and self.owners[key] is not owner:
                raise ValueError("Callback key already owned: " + key)
            self.owners[key] = owner
        self.callbacks[event].append(handler)
        return handler

    def off(self, event, handler):
        self.callbacks[event].remove(handler)

    def emit(self, event, **context):
        if event not in self.EVENTS:
            raise ValueError("Unknown trainer event: " + event)
        for handler in tuple(self.callbacks[event]):
            handler(self, {"event": event, **context})

    def request_stop(self, reason="requested"):
        self.state.update(stop_requested=True, stop_reason=reason)

    def train_batch(self, batch):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        context = {"stage": "train", "batch": batch, "rows": batch.rows}
        self.emit("before_forward", **context)
        with autocast(self.device, self.precision):
            logits = self.model(batch).logits.float()
            loss, losses = self.loss_fn(logits, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        context.update(
            outputs=logits, targets=batch.targets, loss=loss, per_example_loss=losses
        )
        self.emit("after_forward", **context)
        self.emit("before_backward", **context)
        loss.backward()
        self.emit("after_backward", **context)
        self.emit("before_optimizer_step", **context)
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clip, error_if_nonfinite=True
        )
        rates = [group["lr"] for group in self.optimizer.param_groups]
        self.optimizer.step()
        # The sample-based schedule sees the just-committed update, as in v0.5.
        self.state["step"] += 1
        self.state["examples_seen"] += len(batch.rows)
        if self.scheduler is not None:
            self.scheduler.step()
        self.emit("after_optimizer_step", **context)
        result = {
            "loss": float(loss.detach()),
            "cross_entropy": float(loss.detach()),
            "gradient_norm": float(norm),
            "learning_rate": rates[0],
        }
        if len(rates) > 1:
            result["adamw_learning_rate"] = rates[-1]
        if hasattr(self.model, "diagnostics"):
            result.update(self.model.diagnostics())
        return result

    @torch.no_grad()
    def validate(self, batches, panel="full", *, commit=True):
        was_training = self.model.training
        self.emit("validation_start", stage="validation", panel=panel)
        self.model.eval()
        logits, rows = [], []
        try:
            for records in tqdm(
                batches,
                desc="Validation " + panel,
                leave=False,
                disable=not self.progress,
            ):
                if self.state["stop_requested"]:
                    raise StopRequested()
                batch = self.prepare_batch(records, panel)
                context = dict(
                    stage="validation", panel=panel, batch=batch, rows=batch.rows
                )
                self.emit("batch_start", **context)
                self.emit("before_forward", **context)
                with autocast(self.device, self.precision):
                    output = self.model(batch).logits.float()
                    loss, losses = self.loss_fn(output, batch)
                self.emit(
                    "after_forward",
                    **context,
                    outputs=output,
                    targets=batch.targets,
                    loss=loss,
                    per_example_loss=losses,
                )
                logits.extend(
                    value[: len(row["target"])].cpu()
                    for value, row in zip(output, batch.rows)
                )
                rows.extend(batch.rows)
                self.emit("batch_end", **context, metrics={"loss": float(loss)})
            if not rows:
                raise ValueError("Validation panel is empty: " + panel)
            report = {**score(logits, rows), **selection_metrics(logits, rows)}
            self.emit("validation_end", stage="validation", panel=panel, metrics=report)
            if commit:
                self.flush_saves()
            return report
        finally:
            self.model.train(was_training)

    def _validate_panels(self, panels, *, initial=False):
        for panel, batches in panels.items():
            self.validate(batches, panel, commit=False)
        self.state["last_validation_examples"] = self.state["examples_seen"]
        if initial:
            self.state["initial_validation_done"] = True
        self.flush_saves()

    def train(
        self,
        train_batches,
        validation_batches=None,
        *,
        epochs=1,
        validation_every_examples=8000,
        initial_validation=True,
    ):
        """train_batches(epoch) returns a deterministic sequence; cursor is committed per update."""
        panels = validation_batches or {}

        try:
            self.emit("train_start", stage="train")
            if (
                initial_validation
                and panels
                and not self.state["initial_validation_done"]
            ):
                self._validate_panels(panels, initial=True)
            for epoch in range(self.state["completed_epochs"], epochs):
                self._train_epoch(
                    train_batches, panels, epoch, validation_every_examples
                )
        except StopRequested:
            pass
        except BaseException as error:
            try:
                self.emit("exception", stage="train", error=error)
            except BaseException as callback_error:
                error.add_note("Exception callback failed: " + str(callback_error))
            raise
        self.emit("train_end", stage="train")
        self.flush_saves()

    def _train_epoch(self, train_batches, panels, epoch, validation_every_examples):
        """Resume the epoch cursor, replay pending validation, and close prefetch resources."""
        self.state["epoch"] = epoch
        batches = train_batches(epoch)
        self.emit("epoch_start", stage="train", epoch=epoch)

        # Interrupted validation is replayed before another optimizer update.
        cursor = self.state["batch_cursor"]
        if (
            panels
            and cursor
            and self.state["last_validation_examples"] != self.state["examples_seen"]
        ):
            previous = self.state["examples_seen"] - len(batches[cursor - 1])
            if cursor == len(batches) or self._crossed(
                previous, validation_every_examples
            ):
                self._validate_panels(panels)

        bar = tqdm(
            total=sum(len(r) for r in batches),
            initial=sum(len(r) for r in batches[:cursor]),
            desc="Bazza epoch " + str(epoch + 1),
            unit="example",
            disable=not self.progress,
        )

        prepared = None
        try:
            if hasattr(self.prepare_batch, "iter_batches"):
                prepared = self.prepare_batch.iter_batches(batches, cursor, "train")
            else:
                prepared = (
                    (i, self.prepare_batch(batches[i], "train"))
                    for i in range(cursor, len(batches))
                )
            self._train_updates(
                prepared, batches, panels, validation_every_examples, bar
            )
        finally:
            if hasattr(prepared, "close"):
                prepared.close()
            bar.close()

        if self.state["stop_requested"]:
            raise StopRequested()
        self.state.update(completed_epochs=epoch + 1, batch_cursor=0)
        self.emit("epoch_end", stage="train", epoch=epoch)
        self.flush_saves()

    def _train_updates(self, prepared, batches, panels, validation_every_examples, bar):
        """Commit optimizer updates and run validation at the requested boundaries."""
        for index, batch in prepared:
            if self.state["stop_requested"]:
                raise StopRequested()
            records = batches[index]
            before = self.state["examples_seen"]
            self.emit("batch_start", stage="train", batch=batch, rows=records)
            result = self.train_batch(batch)
            self.state["batch_cursor"] = index + 1
            self.emit("batch_end", stage="train", rows=records, metrics=result)
            bar.update(len(records))
            bar.set_postfix(
                loss=f"{result['loss']:.4f}",
                lr=f"{result['learning_rate']:.2g}",
            )
            self.flush_saves()
            if (
                panels
                and not self.state["stop_requested"]
                and (
                    self._crossed(before, validation_every_examples)
                    or index + 1 == len(batches)
                )
            ):
                self._validate_panels(panels)

    def _crossed(self, previous, interval):
        return bool(
            interval and self.state["examples_seen"] // interval > previous // interval
        )

    def request_save(self, path, **extra):
        self.pending_saves[str(path)] = extra

    def flush_saves(self):
        pending, self.pending_saves = self.pending_saves, {}
        for path, extra in pending.items():
            atomic_save(self.state_dict(**extra), path)
            self.emit("checkpoint_saved", path=path)

    def state_dict(self, **extra):
        return {
            "format_version": 1,
            "head": self.checkpoint_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "identity": self.identity,
            "state": deepcopy(self.state),
            "rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.device(self.device).type == "cuda"
            else [],
            "callbacks": {
                key: owner.state_dict() for key, owner in self.owners.items()
            },
            **self.checkpoint_extra,
            **extra,
        }

    def load_state_dict(self, saved):
        if saved["identity"] != self.identity:
            raise ValueError(
                "Resume identity differs; initialize weights for a new experiment instead"
            )
        if set(saved["callbacks"]) != set(self.owners):
            raise ValueError("Resume callback owners differ")
        self.checkpoint_model.load_state_dict(saved["head"], strict=True)
        self.optimizer.load_state_dict(saved["optimizer"])
        self.state.clear()
        self.state.update(saved["state"])
        if self.scheduler:
            self.scheduler.load_state_dict(saved["scheduler"])
        elif saved["scheduler"] is not None:
            raise ValueError("Resume scheduler differs")
        for key, owner in self.owners.items():
            owner.load_state_dict(saved["callbacks"][key])
        torch.set_rng_state(saved["rng"])
        random.setstate(saved["python_rng"])
        if saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        self.state.update(stop_requested=False, stop_reason=None)
