"""Combine request gating and residual representation correction in one pass."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import math

import torch
from torch.nn import functional as F

from architecture.data import digest, file_hash
from architecture.metrics import TrainingMetrics
from architecture.model import make_head
from architecture.optimizers import make_optimizer
from architecture.readout_ablation import LetterReadout, load_letters
from architecture.readout_refinement import (
    AnswerAdapter, RequestGateReadout, RefinementTrainer, confidence_features, final_state,
)
from architecture.train import load_bank, training_batches
from gemma_decisions.sweep import atomic_save
from gemma_decisions.tracking import RunLog, atomic_json
from gemma_decisions.validation_cache import length_batches


class MixedReadout(RequestGateReadout):
    def __init__(self, decision, letter, encoder_width=3840, reasoner_width=1024,
                 gate_width=32, rank=64):
        component_rng = torch.get_rng_state()
        super().__init__(decision, letter, encoder_width, gate_width)
        # Match each component's separate trial initialization. Adding the gate
        # must not move the adapter's CPU initialization along the RNG sequence.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(component_rng)
            self.answer_adapter = AnswerAdapter(reasoner_width, encoder_width, rank)
        self.last_relative_delta = None

    def forward(self, hidden, mask, positions, option_mask):
        high = []
        handle = self.decision.decision.reasoner.register_forward_hook(
            lambda module, inputs, output: high.append(output.high_state))
        try:
            decision = self.decision(hidden, mask, positions, option_mask)
        finally:
            handle.remove()
        if len(high) != 1:
            raise ValueError("Expected exactly one reasoner forward")
        original = final_state(hidden, mask)
        delta = self.answer_adapter(high.pop())
        prior = F.linear(original + delta, self.letter.weight[:option_mask.shape[1]]).float()
        cap = self.letter.softcap
        if cap:
            prior = cap * (prior / cap).tanh()
        prior = prior.masked_fill(~option_mask, -1e4)
        state = F.silu(self.request_project(self.request_norm(original)))
        stats = torch.cat((confidence_features(decision, option_mask),
                           confidence_features(prior, option_mask),
                           option_mask.sum(-1, keepdim=True).float().log()), -1)
        gate = self.action_score(torch.cat((state, stats), -1)).float().sigmoid()
        self.last_gate = gate.detach()
        self.last_relative_delta = (delta.detach().float().square().mean(-1).sqrt() /
                                    original.detach().float().square().mean(-1).sqrt().clamp_min(1e-8))
        return (decision + gate * prior).masked_fill(~option_mask, -1e4)

    def diagnostics(self):
        return {**super().diagnostics(),
                "answer_delta_relative_rms_mean": float(self.last_relative_delta.mean())}


def run(args):
    if (args.output_dir / "experiment.json").exists():
        raise ValueError("Run exists; choose a fresh directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    index, records = load_bank(args.bank_dir, args.data_dir)
    reference = json.loads((args.reference_dir / "request-gate/experiment.json").read_text())
    if reference["feature_identity"] != index["identity"]:
        raise ValueError("Reference belongs to a different feature bank")
    weight, readout = load_letters(args.base_dir, index["identity"])
    torch.manual_seed(reference["seed"])
    decision = make_head(3840, 1024, 4, 2, 2, "bigru")
    core_hash = digest({k: hashlib.sha256(v.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
                        for k, v in decision.state_dict().items()})
    if core_hash != reference["initial_core_sha256"]:
        raise ValueError("Initial decision core differs from individual trials")
    model = MixedReadout(decision, LetterReadout(weight, readout["softcap"])).cuda()
    batches = training_batches(records["splits"]["train"], 32, 32768, reference["seed"])
    if digest([[r["id"] for r in batch] for batch in batches]) != reference["train_batch_order_sha256"]:
        raise ValueError("Batch order differs")
    config = {**reference, "variant": "mixed-request-residual", "initial_core_sha256": core_hash,
              "reference_dir": str(args.reference_dir), "new_caches": False,
              "gate_confidence_source": "corrected pretrained logits; detached confidence features",
              "component_initialization": "gate and adapter each match their separate trial RNG position",
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "image_tag": args.image_tag, "image_id": args.image_id,
              "source_sha256": {p: file_hash(Path(__file__).parent / p) for p in
                ["readout_mixed.py", "readout_refinement.py", "readout_ablation.py", "model.py", "trainer.py", "optimizers.py", "train.py"]}}
    atomic_json(args.output_dir / "experiment.json", config)
    optimizer = make_optimizer(model, "muon", 3e-5, .01, .999, .003, False)
    def rate(step):
        if step < 20:
            return (step + 1) / 20
        if step < len(batches) * .8:
            return 1.
        x = min(1., (step - len(batches) * .8) / max(1., len(batches) * .2))
        return .1 + .9 * (1 + math.cos(math.pi * x)) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    trainer = RefinementTrainer(model, optimizer, scheduler, args.bank_dir)
    logger = RunLog(args.output_dir / "logging", "mixed-request-residual", config,
                    args.overfit_url, "Bazza readout ablation")
    observations = TrainingMetrics()
    trainer.on("after_forward", observations.forward)
    trainer.on("batch_end", observations.batch)
    def status(phase, **extra):
        atomic_json(args.output_dir / "status.json", {"phase": phase, "updated_at": time.time(), **extra})
    def log(t, ctx):
        if t.state["step"] % 10 == 0:
            logger.log({"train/" + k: v for k, v in ctx["metrics"].items()}, t.state["step"])
            status("training", **t.state)
    trainer.on("batch_end", log)
    try:
        status("training")
        torch.cuda.synchronize()
        begin = time.perf_counter()
        trainer.train(1, [batches])
        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - begin
        status("validating")
        begin = time.perf_counter()
        score = trainer.evaluate(list(length_batches(records["splits"]["validation-full"], 32, 32768)))
        torch.cuda.synchronize()
        summary = {"name": config["variant"], **observations.report(), **score,
                   "training_seconds": training_seconds, "validation_seconds": time.perf_counter() - begin,
                   "trainable_parameters": config["trainable_parameters"]}
        saved = {"head": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "config": config, "readout": readout, "state": trainer.state, "validation_full": score}
        atomic_save(saved, args.output_dir / f'checkpoint-step-{trainer.state["step"]}.pt')
        atomic_json(args.output_dir / "validation-full.json", score)
        atomic_json(args.output_dir / "results.json", summary)
        values = {"validation/full/" + k: score[k] for k in ["accuracy", "cross_entropy", "macro_category_cross_entropy"]}
        for group in ["category", "source", "kind"]:
            for key, entry in score["by_" + group].items():
                for metric in ["accuracy", "cross_entropy"]:
                    values[f"validation/full/{group}/{key}/{metric}"] = entry[metric]
        logger.log(values, trainer.state["step"]); logger.finish()
        status("completed", accuracy=score["accuracy"], macro_category_cross_entropy=score["macro_category_cross_entropy"])
        print(json.dumps({k: summary[k] for k in ["name", "accuracy", "cross_entropy", "macro_category_cross_entropy",
                                               "training_seconds", "validation_seconds"]}, indent=2), flush=True)
    except BaseException as error:
        status("failed", error=str(error), traceback=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["bank-dir", "data-dir", "base-dir", "reference-dir", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--overfit-url", default="http://127.0.0.1:6006")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
