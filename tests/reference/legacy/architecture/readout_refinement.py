"""Request gating and residual answer adapters, with the readout recipe fixed."""

import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import torch
from torch import nn
from torch.nn import functional as F

from architecture.data import digest, file_hash
from architecture.metrics import TrainingMetrics
from architecture.model import make_head
from architecture.optimizers import make_optimizer
from architecture.readout_ablation import AblationTrainer, HybridReadout, LetterReadout, load_letters
from architecture.train import load_bank, training_batches
from gemma_decisions.sweep import atomic_save
from gemma_decisions.tracking import RunLog, atomic_json
from gemma_decisions.validation_cache import length_batches


CASES = ("hybrid-control", "request-gate", "residual-hybrid", "residual-only")


def final_state(hidden, mask):
    last = mask.long().sum(-1) - 1
    return hidden[torch.arange(len(hidden), device=hidden.device), last]


def confidence_features(logits, mask):
    # Confidence is observable at inference. Detach to stop the scorer learning
    # to manipulate the gate indirectly through confidence, rather than answers.
    probs = logits.detach().float().masked_fill(~mask, -1e4).softmax(-1)
    count = mask.sum(-1).float()
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(-1) / count.log()
    top = probs.topk(2, dim=-1).values
    return torch.stack((entropy, top[:, 0] - top[:, 1]), -1)


class RequestGateReadout(nn.Module):
    def __init__(self, decision, letter, encoder_width=3840, gate_width=32):
        super().__init__()
        self.decision, self.letter = decision, letter
        self.request_norm = nn.LayerNorm(encoder_width)
        self.request_project = nn.Linear(encoder_width, gate_width)
        # Name the gate's output projection action_score so the existing optimizer
        # puts it on the unchanged auxiliary AdamW LR, as for the original scorer.
        self.action_score = nn.Linear(gate_width + 5, 1)
        nn.init.zeros_(self.action_score.weight)
        nn.init.constant_(self.action_score.bias, math.log(.1 / .9))
        self.last_gate = None

    def forward(self, hidden, mask, positions, option_mask):
        decision = self.decision(hidden, mask, positions, option_mask)
        prior = self.letter(hidden, mask, positions, option_mask)
        state = F.silu(self.request_project(self.request_norm(final_state(hidden, mask))))
        stats = torch.cat((confidence_features(decision, option_mask),
                           confidence_features(prior, option_mask),
                           option_mask.sum(-1, keepdim=True).float().log()), -1)
        gate = self.action_score(torch.cat((state, stats), -1)).float().sigmoid()
        self.last_gate = gate.detach()
        return (decision + gate * prior).masked_fill(~option_mask, -1e4)

    def diagnostics(self):
        return {"request_gate_mean": float(self.last_gate.mean()),
                "request_gate_min": float(self.last_gate.min()),
                "request_gate_max": float(self.last_gate.max())}


class AnswerAdapter(nn.Module):
    def __init__(self, reasoner_width=1024, encoder_width=3840, rank=64):
        super().__init__()
        self.norm = nn.LayerNorm(reasoner_width)
        self.down = nn.Linear(reasoner_width, rank, bias=False)
        self.action_score = nn.Linear(rank, encoder_width, bias=False)
        nn.init.zeros_(self.action_score.weight)

    def forward(self, high):
        return self.action_score(F.silu(self.down(self.norm(high))))


class ResidualAnswerReadout(nn.Module):
    def __init__(self, decision, letter, hybrid, reasoner_width=1024, encoder_width=3840, rank=64):
        super().__init__()
        self.decision, self.letter, self.hybrid = decision, letter, hybrid
        self.answer_adapter = AnswerAdapter(reasoner_width, encoder_width, rank)
        if hybrid:
            self.prior_gate = nn.Parameter(torch.tensor(math.atanh(.1)))
        self.last_relative_delta = None

    def forward(self, hidden, mask, positions, option_mask):
        # Capture the existing reasoner's output from its one normal forward.
        # A scoped hook avoids duplicating/reimplementing the reader or reasoner.
        high = []
        handle = self.decision.decision.reasoner.register_forward_hook(
            lambda module, inputs, output: high.append(output.high_state))
        try:
            decision = self.decision(hidden, mask, positions, option_mask)
        finally:
            handle.remove()
        if len(high) != 1:
            raise ValueError("Expected exactly one reasoner forward")
        delta = self.answer_adapter(high.pop())
        original = final_state(hidden, mask)
        self.last_relative_delta = (delta.detach().float().square().mean(-1).sqrt() /
                                    original.detach().float().square().mean(-1).sqrt().clamp_min(1e-8))
        logits = F.linear(original + delta, self.letter.weight[:option_mask.shape[1]]).float()
        cap = self.letter.softcap
        if cap:
            logits = cap * (logits / cap).tanh()
        if self.hybrid:
            logits = decision + self.prior_gate.tanh() * logits
        return logits.masked_fill(~option_mask, -1e4)

    def diagnostics(self):
        values = {"answer_delta_relative_rms_mean": float(self.last_relative_delta.mean())}
        if self.hybrid:
            values["pretrained_prior_coefficient"] = float(self.prior_gate.detach().tanh())
        return values


def build_readout(name, decision, weight, softcap):
    letter = LetterReadout(weight, softcap)
    if name == "hybrid-control":
        return HybridReadout(decision, letter)
    if name == "request-gate":
        return RequestGateReadout(decision, letter)
    return ResidualAnswerReadout(decision, letter, hybrid=name == "residual-hybrid")


class RefinementTrainer(AblationTrainer):
    def train_batch(self, rows):
        result = super().train_batch(rows)
        if hasattr(self.model, "diagnostics"):
            result.update(self.model.diagnostics())
        return result


def run(args):
    torch.set_num_threads(8)
    if (args.output_dir / "ablation.json").exists():
        raise ValueError("Run exists; choose a new directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index, records = load_bank(args.bank_dir, args.data_dir)
    weight, readout = load_letters(args.base_dir, index["identity"])
    train_batches = training_batches(records["splits"]["train"], 32, 32768, 20261013)
    val_batches = list(length_batches(records["splits"]["validation-full"], 32, 32768))
    common = {"seed": 20261013, "epochs": 1, "width": 1024, "head_type": "bigru",
              "attention_heads": 4, "low_steps": 2, "high_cycles": 2,
              "batch_size": 32, "token_budget": 32768, "optimizer": "muon",
              "muon_lr": .003, "lr": 3e-5, "weight_decay": .01, "beta2": .999,
              "warmup_steps": 20, "schedule": "wsd; final 20% cosine to 0.1 peak",
              "total_steps": len(train_batches), "gradient_clip": 1., "loss": "unchanged CE",
              "examples": len(records["splits"]["train"]),
              "validation_rows": len(records["splits"]["validation-full"]),
              "feature_identity": index["identity"], "image_tag": args.image_tag, "image_id": args.image_id,
              "torch": str(torch.__version__), "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
              "train_batch_order_sha256": digest([[r["id"] for r in b] for b in train_batches]),
              "input_prompt_changed": False, "choice_order_changed": False, "backbone_loaded": False,
              "feature_file_hash_scan": False, "new_caches": False,
              "gate_width": 32, "adapter_rank": 64, "initial_prior_coefficient": .1,
              "gate_inputs": "final frozen state, detached branch entropies/margins, option count; no source/gold",
              "adapter_initialization": "zero up projection; norm/down use normal initialization",
              "source_sha256": {p: file_hash(Path(__file__).parent / p) for p in
                ["readout_refinement.py", "readout_ablation.py", "model.py", "trainer.py", "optimizers.py", "train.py"]}}
    atomic_json(args.output_dir / "ablation.json", {"common": common, "readout": readout, "cases": list(CASES),
                "temperature_omitted": "post-softcap scalar temperature is redundant with scalar prior gate",
                "reference_run": "readout-20261013-144123/hybrid-ce",
                "interpretation": "same cached prompt, not Mapika native answer-slot protocol"})
    results, initial_hashes = [], set()
    def status(phase, **extra):
        atomic_json(args.output_dir / "status.json", {"phase": phase, "updated_at": time.time(), **extra})
    started = time.perf_counter()
    try:
        for name in CASES:
            status("training", trial=name)
            torch.manual_seed(common["seed"])
            decision = make_head(3840, 1024, 4, 2, 2, "bigru")
            initial_hash = digest({k: hashlib.sha256(v.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
                                   for k, v in decision.state_dict().items()})
            initial_hashes.add(initial_hash)
            model = build_readout(name, decision, weight, readout["softcap"]).cuda()
            optimizer = make_optimizer(model, "muon", 3e-5, .01, .999, .003, False)
            def rate(step):
                if step < 20:
                    return (step + 1) / 20
                if step < len(train_batches) * .8:
                    return 1.
                x = min(1., (step - len(train_batches) * .8) / max(1., len(train_batches) * .2))
                return .1 + .9 * (1 + math.cos(math.pi * x)) / 2
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
            trainer = RefinementTrainer(model, optimizer, scheduler, args.bank_dir)
            config = {**common, "variant": name, "initial_core_sha256": initial_hash,
                      "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                      "residual_only_note": "original scalar scorer computed but unused in loss" if name == "residual-only" else None}
            trial = args.output_dir / name
            logger = RunLog(trial / "logging", "refine-" + name, config, args.overfit_url, "Bazza readout ablation")
            atomic_json(trial / "experiment.json", config)
            observations = TrainingMetrics()
            trainer.on("after_forward", observations.forward)
            trainer.on("batch_end", observations.batch)
            def log(t, ctx):
                if t.state["step"] % 10 == 0:
                    logger.log({"train/" + k: v for k, v in ctx["metrics"].items()}, t.state["step"])
                    status("training", trial=name, **t.state)
            trainer.on("batch_end", log)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            trainer.train(1, [train_batches])
            torch.cuda.synchronize()
            train_seconds = time.perf_counter() - begin
            status("validating", trial=name)
            begin = time.perf_counter()
            score = trainer.evaluate(val_batches)
            torch.cuda.synchronize()
            val_seconds = time.perf_counter() - begin
            summary = {"name": name, **observations.report(), "accuracy": score["accuracy"],
                       "cross_entropy": score["cross_entropy"], "macro_category_cross_entropy": score["macro_category_cross_entropy"],
                       "by_category": score["by_category"], "by_source": score["by_source"], "by_kind": score["by_kind"],
                       "ordinal": score.get("ordinal"), "training_seconds": train_seconds,
                       "validation_seconds": val_seconds, "trainable_parameters": config["trainable_parameters"],
                       "initial_core_sha256": initial_hash}
            saved = {"head": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "config": config, "readout": readout, "state": trainer.state, "validation_full": score}
            atomic_save(saved, trial / f'checkpoint-step-{trainer.state["step"]}.pt')
            atomic_json(trial / "validation-full.json", score)
            values = {"validation/full/" + k: score[k] for k in ["accuracy", "cross_entropy", "macro_category_cross_entropy"]}
            for group in ["category", "source", "kind"]:
                for key, entry in score["by_" + group].items():
                    for metric in ["accuracy", "cross_entropy"]:
                        values[f"validation/full/{group}/{key}/{metric}"] = entry[metric]
            logger.log(values, len(train_batches)); logger.finish()
            results.append(summary)
            atomic_json(args.output_dir / "results.json", results)
            print(f'{name}: {score["accuracy"]:.3%}, CE {score["cross_entropy"]:.6f}, '
                  f'macro CE {score["macro_category_cross_entropy"]:.6f}', flush=True)
            del saved, model, decision, trainer, optimizer, scheduler
            gc.collect(); torch.cuda.empty_cache()
        if len(initial_hashes) != 1:
            raise ValueError("Core initialization differed")
        fields = ["name", "accuracy", "cross_entropy", "macro_category_cross_entropy", "training_seconds",
                  "validation_seconds", "trainable_parameters"]
        with (args.output_dir / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(results)
        status("completed", trials=len(results), wall_seconds=time.perf_counter() - started)
    except BaseException as error:
        status("failed", error=str(error), traceback=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["bank-dir", "data-dir", "base-dir", "output-dir"]:
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--overfit-url", default="http://127.0.0.1:6006")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
