"""Event boundaries and exact committed-progress resumption."""

import tempfile
import unittest
from pathlib import Path

import torch
from fixtures import PrecomputedBackbone

from callbacks import Checkpoint, EarlyStopping, Observations
from data import collate_records
from model import Bazza, BazzaDecisionHead
from trainer import Trainer


def records():
    return [
        dict(
            id=str(i),
            group_id=str(i),
            ids=[1] * 5,
            positions=[1, 4],
            target=[float(i % 2 == 0), float(i % 2 == 1)],
            category="tools",
            dataset_source="fixture",
            source="fixture",
            kind="nool",
            task_key="fixture/choice",
            options=["yes", "no"],
            semantic_class="yes",
            balance_schema="fixture",
        )
        for i in range(8)
    ]


def build(directory):
    torch.manual_seed(11)
    model = BazzaDecisionHead(
        torch.randn(26, 16),
        30,
        encoder_width=16,
        width=8,
        attention_heads=2,
        low_steps=1,
        high_cycles=1,
        gate_width=4,
        rank=2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    tensors = {r["id"]: torch.randn(5, 16) for r in records()}

    def prepare(rows, split):
        return collate_records(rows, [tensors[r["id"]] for r in rows])

    trainer = Trainer(
        Bazza(PrecomputedBackbone(), model),
        optimizer,
        checkpoint_model=model,
        prepare_batch=prepare,
        device="cpu",
        identity={"fixture": 1},
        progress=False,
    )
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda _: 1 / (1 + trainer.state["step"])
    )
    observations = Observations()
    trainer.on("after_forward", observations.forward, key="observations")
    trainer.on("batch_end", observations.batch, key="observations")
    Checkpoint(directory, every_examples=2).attach(trainer)
    trainer.on("validation_end", EarlyStopping(min_examples=100), key="stopping")
    return trainer


class TrainerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_events_stages_order_and_unsubscribe(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = build(Path(directory))
            events = []
            for event in Trainer.EVENTS:
                trainer.on(
                    event, lambda t, c: events.append((c["event"], c.get("stage")))
                )
            handler = trainer.on("batch_start", lambda t, c: None)
            trainer.off("batch_start", handler)
            with self.assertRaises(ValueError):
                trainer.on("batch_strat", handler)
            with self.assertRaises(ValueError):
                trainer.on("batch_end", Observations())
            trainer.train(
                lambda epoch: [records()[:2]],
                {"full": [records()[2:4]]},
                initial_validation=False,
            )
            train = [name for name, stage in events if stage == "train"]
            self.assertEqual(
                train,
                [
                    "train_start",
                    "epoch_start",
                    "batch_start",
                    "before_forward",
                    "after_forward",
                    "before_backward",
                    "after_backward",
                    "before_optimizer_step",
                    "after_optimizer_step",
                    "batch_end",
                    "epoch_end",
                    "train_end",
                ],
            )
            validation = [name for name, stage in events if stage == "validation"]
            self.assertEqual(
                validation,
                [
                    "validation_start",
                    "batch_start",
                    "before_forward",
                    "after_forward",
                    "batch_end",
                    "validation_end",
                ],
            )
            self.assertEqual(trainer.owners["observations"].rows, 2)
            saved = torch.load(Path(directory) / "last.pt", weights_only=True)
            self.assertEqual(saved["state"]["completed_epochs"], 1)
            self.assertEqual(
                saved["callbacks"]["stopping"], trainer.owners["stopping"].state_dict()
            )

    def test_interrupted_resume_matches_uninterrupted_including_callbacks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batches = [records()[i : i + 2] for i in range(0, 8, 2)]
            full = build(root / "full")
            full.train(
                lambda epoch: batches,
                {"full": [records()[:2]]},
                validation_every_examples=4,
            )
            interrupted = build(root / "resume")

            def stop(t, c):
                if c["stage"] == "train" and t.state["step"] == 2:
                    t.request_stop("test")

            interrupted.on("batch_end", stop)
            interrupted.train(
                lambda epoch: batches,
                {"full": [records()[:2]]},
                validation_every_examples=4,
            )
            saved = torch.load(root / "resume/last.pt", weights_only=True)
            self.assertEqual(saved["state"]["batch_cursor"], 2)
            resumed = build(root / "resume")
            resumed.load_state_dict(saved)
            resumed.train(
                lambda epoch: batches,
                {"full": [records()[:2]]},
                validation_every_examples=4,
            )
            for key, value in full.model.state_dict().items():
                self.assertTrue(
                    torch.equal(value, resumed.model.state_dict()[key]), key
                )
            self.assertEqual(full.state, resumed.state)
            self.assertEqual(
                full.scheduler.state_dict(), resumed.scheduler.state_dict()
            )
            self.assertEqual(
                full.owners["observations"].state_dict(),
                resumed.owners["observations"].state_dict(),
            )
            self.assertEqual(
                full.owners["stopping"].state_dict(),
                resumed.owners["stopping"].state_dict(),
            )
            self.assertEqual(
                full.owners["checkpoint"].state_dict(),
                resumed.owners["checkpoint"].state_dict(),
            )
            saved["identity"] = {"changed": True}
            with self.assertRaises(ValueError):
                build(root / "other").load_state_dict(saved)

    def test_callback_order_changes_apply_next_emission_and_exceptions_propagate(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = build(Path(directory))
            events = []

            def first(t, c):
                events.append("first")
                t.off("batch_start", first)
                t.on("batch_start", lambda t, c: events.append("later"))

            trainer.on("batch_start", first)
            trainer.emit("batch_start")
            trainer.emit("batch_start")
            self.assertEqual(events, ["first", "later"])
            trainer.on("exception", lambda t, c: events.append(str(c["error"])))

            def fail(t, c):
                raise RuntimeError("forward failed")

            trainer.on("before_forward", fail)
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                trainer.train(lambda epoch: [records()[:2]], initial_validation=False)
            self.assertIn("forward failed", events)


if __name__ == "__main__":
    unittest.main()
