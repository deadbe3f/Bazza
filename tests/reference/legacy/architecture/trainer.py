"""A small head-only Trainer with an explicit event registry."""

from collections import defaultdict
import time

import torch
from tqdm.auto import tqdm

from gemma_decisions.sweep import autocast, collate


def metrics(logits, rows):
    categories, sources = defaultdict(list), defaultdict(list)
    for logit, row in zip(logits, rows):
        target = torch.tensor(row["target"])
        ce = float(-(target * logit.float().log_softmax(-1)).sum())
        hit = float(target[int(logit.argmax())] >= target.max() - 1e-6)
        categories[row["category"]].append((hit, ce))
        sources[row["dataset_source"]].append((hit, ce))
    def aggregate(values):
        return {"rows": len(values), "accuracy": sum(v[0] for v in values) / len(values),
                "cross_entropy": sum(v[1] for v in values) / len(values)}
    result = aggregate([item for values in categories.values() for item in values])
    result["by_category"] = {k: aggregate(v) for k, v in sorted(categories.items())}
    result["by_source"] = {k: aggregate(v) for k, v in sorted(sources.items())}
    result["macro_category_cross_entropy"] = sum(v["cross_entropy"] for v in result["by_category"].values()) / len(categories)
    return result


class Trainer:
    EVENTS = {"train_start", "train_end", "epoch_start", "epoch_end", "batch_start", "batch_end",
              "before_forward", "after_forward", "before_backward", "after_backward",
              "before_optimizer_step", "after_optimizer_step", "validation_start", "validation_end"}

    def __init__(self, model, optimizer, scheduler, bank_dir, device="cuda"):
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.bank_dir, self.device = bank_dir, device
        self.callbacks = defaultdict(list)
        self.state = {"step": 0, "examples_seen": 0, "epoch": 0}

    def on(self, event, callback):
        if event not in self.EVENTS:
            raise ValueError("Unknown trainer event: " + event)
        self.callbacks[event].append(callback)
        return callback

    def emit(self, event, **context):
        for callback in tuple(self.callbacks[event]):
            callback(self, context)

    def train_batch(self, rows):
        self.model.train()
        batch = collate(rows, self.bank_dir, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        self.emit("before_forward", batch=batch, rows=rows)
        with autocast(self.device):
            logits = self.model(*batch[:4]).float()
            per_example_loss = -(batch[4] * logits.log_softmax(-1)).sum(-1)
            loss = per_example_loss.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite head loss")
        self.emit("after_forward", outputs=logits, loss=loss, per_example_loss=per_example_loss,
                  targets=batch[4], rows=rows)
        self.emit("before_backward", loss=loss)
        loss.backward()
        self.emit("after_backward", loss=loss)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
        self.emit("before_optimizer_step")
        learning_rate = self.optimizer.param_groups[0]["lr"]
        auxiliary_rate = self.optimizer.param_groups[-1]["lr"]
        self.optimizer.step()
        self.scheduler.step()
        self.emit("after_optimizer_step")
        self.state["step"] += 1
        self.state["examples_seen"] += len(rows)
        return {"loss": float(loss.detach()), "gradient_norm": float(norm),
                "learning_rate": learning_rate,
                **({"adamw_learning_rate": auxiliary_rate} if len(self.optimizer.param_groups) > 1 else {})}

    @torch.no_grad()
    def evaluate(self, batches, panel="quick"):
        self.emit("validation_start", panel=panel)
        self.model.eval()
        logits, rows = [], []
        for records in tqdm(batches, desc="Cached " + panel + " validation", leave=False):
            batch = collate(records, self.bank_dir, self.device)
            with autocast(self.device):
                output = self.model(*batch[:4]).float().cpu()
            logits.extend(value[:len(row["target"])] for value, row in zip(output, records))
            rows.extend(records)
        score = metrics(logits, rows)
        self.emit("validation_end", panel=panel, metrics=score)
        return score

    def train(self, epochs, train_batches, validation_batches=None, validation_panel="quick"):
        self.emit("train_start")
        start_epoch = self.state.get("completed_epochs", 0)
        for local_epoch in range(epochs):
            epoch = start_epoch + local_epoch
            self.state["epoch"] = epoch
            self.emit("epoch_start", epoch=epoch)
            started, seen, loss_sum = time.perf_counter(), 0, 0.
            progress = tqdm(train_batches[local_epoch], desc="Head epoch %s/%s" %
                            (epoch + 1, start_epoch + epochs), unit="batch")
            for rows in progress:
                self.emit("batch_start", rows=rows)
                result = self.train_batch(rows)
                seen += len(rows)
                loss_sum += result["loss"] * len(rows)
                result["epoch_running_loss"] = loss_sum / seen
                result["examples_per_second"] = seen / max(time.perf_counter() - started, 1e-6)
                self.emit("batch_end", metrics=result)
                progress.set_postfix(loss="%.4f" % result["loss"], lr="%.2g" % result["learning_rate"])
            score = self.evaluate(validation_batches, validation_panel) if validation_batches is not None else None
            self.state["completed_epochs"] = epoch + 1
            self.emit("epoch_end", metrics=score, panel=validation_panel if score is not None else None)
        self.emit("train_end")
