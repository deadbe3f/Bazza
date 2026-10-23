"""Matched one-pass readout/ordinal-loss experiments using existing token features.

The frozen letter readout uses the existing prompt's final token. This is not
a reproduction of Mapika's Answer: ( prompt. No backbone or dataset is changed.
"""

import argparse
import csv
import gc
import json
import math
from pathlib import Path
import time
import traceback

import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from tokenizers import Tokenizer

from architecture.data import digest, file_hash
from architecture.metrics import TrainingMetrics
from architecture.model import make_head
from architecture.optimizers import make_optimizer
from architecture.train import load_bank, training_batches
from architecture.trainer import Trainer
from gemma_decisions.sweep import atomic_save, autocast, collate
from gemma_decisions.tracking import RunLog, atomic_json
from gemma_decisions.validation_cache import length_batches


CASES = (
    ("decision-ce", False, 0.),
    ("decision-ordinal", False, 1.),
    ("hybrid-ce", True, 0.),
    ("hybrid-ordinal", True, 1.),
)


class LetterReadout(nn.Module):
    def __init__(self, weight, softcap):
        super().__init__()
        self.register_buffer("weight", weight.clone())
        self.softcap = softcap

    def forward(self, hidden, mask, positions, option_mask):
        last = mask.long().sum(-1) - 1
        final = hidden[torch.arange(len(hidden), device=hidden.device), last]
        logits = F.linear(final, self.weight[:option_mask.shape[1]]).float()
        if self.softcap:
            logits = self.softcap * (logits / self.softcap).tanh()
        return logits.masked_fill(~option_mask, -1e4)


class HybridReadout(nn.Module):
    def __init__(self, decision, letter, initial_prior=.1):
        super().__init__()
        self.decision, self.letter = decision, letter
        # Deterministic scalar: adding this branch must not change initialization
        # of the shared decision core. AdamW uses the same auxiliary LR as before.
        self.prior_gate = nn.Parameter(torch.tensor(math.atanh(initial_prior)))

    def forward(self, hidden, mask, positions, option_mask):
        decision = self.decision(hidden, mask, positions, option_mask)
        prior = self.letter(hidden, mask, positions, option_mask)
        return (decision + self.prior_gate.tanh() * prior).masked_fill(~option_mask, -1e4)


def ordinal_penalty(logits, targets, rows):
    """Mean squared CDF distance for ordered score rows only; no new labels.

    The loss is bounded and respects adjacent vs distant levels. Choice/noul
    rows remain exactly CE. Soft teacher distributions stay distributions.
    """
    penalty = logits.new_zeros(len(rows))
    for i, row in enumerate(rows):
        if row["kind"] == "score":
            count = len(row["target"])
            p = logits[i, :count].softmax(-1)
            delta = p.cumsum(-1)[:-1] - targets[i, :count].cumsum(-1)[:-1]
            penalty[i] = delta.square().mean()
    return penalty


class AblationTrainer(Trainer):
    def __init__(self, *args, ordinal_weight=0., **kwargs):
        super().__init__(*args, **kwargs)
        self.ordinal_weight = ordinal_weight

    def train_batch(self, rows):
        self.model.train()
        batch = collate(rows, self.bank_dir, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        self.emit("before_forward", batch=batch, rows=rows)
        with autocast(self.device):
            logits = self.model(*batch[:4]).float()
            ce = -(batch[4] * logits.log_softmax(-1)).sum(-1)
            extra = ordinal_penalty(logits, batch[4], rows) if self.ordinal_weight else torch.zeros_like(ce)
            loss = (ce + self.ordinal_weight * extra).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite ablation loss")
        # Observations retain CE for comparability even when the objective changes.
        self.emit("after_forward", outputs=logits, loss=loss, per_example_loss=ce,
                  targets=batch[4], rows=rows)
        self.emit("before_backward", loss=loss)
        loss.backward()
        self.emit("after_backward", loss=loss)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
        self.emit("before_optimizer_step")
        primary, auxiliary = self.optimizer.param_groups[0]["lr"], self.optimizer.param_groups[-1]["lr"]
        self.optimizer.step()
        self.scheduler.step()
        self.emit("after_optimizer_step")
        self.state["step"] += 1
        self.state["examples_seen"] += len(rows)
        result = {"loss": float(loss.detach()), "cross_entropy": float(ce.detach().mean()),
                  "ordinal_penalty": float(extra.detach().mean()), "gradient_norm": float(norm),
                  "learning_rate": primary, "adamw_learning_rate": auxiliary}
        if isinstance(self.model, HybridReadout):
            result["pretrained_prior_coefficient"] = float(self.model.prior_gate.detach().tanh())
        return result

    @torch.no_grad()
    def evaluate(self, batches, panel="full"):
        # Keep the original aggregate metrics, adding ordered-score diagnostics.
        self.model.eval()
        logits, rows = [], []
        from tqdm.auto import tqdm
        from architecture.trainer import metrics
        for records in tqdm(batches, desc="Cached " + panel + " validation", leave=False):
            tensors = collate(records, self.bank_dir, self.device)
            with autocast(self.device):
                values = self.model(*tensors[:4]).float().cpu()
            logits.extend(v[:len(r["target"])] for v, r in zip(values, records))
            rows.extend(records)
        report = metrics(logits, rows)
        report["by_kind"] = {}
        for kind in sorted({r["kind"] for r in rows}):
            chosen = [(v, r) for v, r in zip(logits, rows) if r["kind"] == kind]
            report["by_kind"][kind] = metrics([v for v, _ in chosen], [r for _, r in chosen])
        ordered = [(v, r) for v, r in zip(logits, rows) if r["kind"] == "score"]
        if ordered:
            mae, cdf = [], []
            for value, row in ordered:
                p, target = value.softmax(-1), torch.tensor(row["target"])
                levels = torch.arange(len(p), dtype=torch.float32)
                mae.append(float(((p * levels).sum() - (target * levels).sum()).abs() / (len(p) - 1)))
                cdf.append(float((p.cumsum(-1)[:-1] - target.cumsum(-1)[:-1]).square().mean()))
            report["ordinal"] = {"rows": len(ordered), "normalized_expected_level_mae": sum(mae) / len(mae),
                                 "cdf_mse": sum(cdf) / len(cdf)}
        return report


def load_letters(base, identity):
    config = json.loads((base / "config.json").read_text())
    if file_hash(base / "tokenizer.json") != identity["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs from feature export")
    tok = Tokenizer.from_file(str(base / "tokenizer.json"))
    ids = [tok.encode(c, add_special_tokens=False).ids for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    if any(len(x) != 1 for x in ids):
        raise ValueError("Expected single-token answer letters")
    key = "model.language_model.embed_tokens.weight"
    with safe_open(base / "model.safetensors", framework="pt", device="cpu") as source:
        view = source.get_slice(key)
        weight = torch.cat([view[x[0]:x[0] + 1] for x in ids]).contiguous()
    if weight.shape != (26, identity["hidden_width"]) or weight.dtype != torch.bfloat16:
        raise ValueError("Readout shape/dtype differs")
    text = config["text_config"]
    if not text.get("tie_word_embeddings"):
        raise ValueError("This extraction expects the audited Gemma tied LM head")
    provenance = {"tensor_key": key, "letter_token_ids": [x[0] for x in ids],
                  "selected_rows_sha256": __import__("hashlib").sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest(),
                  "weight_sha256_from_existing_feature_identity": identity["weight_sha256"],
                  "full_weight_hash_rescan": False,
                  "softcap": text["final_logit_softcapping"],
                  "readout_position": "last valid token of existing cached Bazza prompt",
                  "prompt_changed": False}
    return weight, provenance


def run(args):
    torch.set_num_threads(8)
    if (args.output_dir / "ablation.json").exists():
        raise ValueError("Run exists; choose a new output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index, records = load_bank(args.bank_dir, args.data_dir)
    weight, readout = load_letters(args.base_dir, index["identity"])
    atomic_save({"weight": weight, "provenance": readout}, args.output_dir / "letter-readout.pt")
    batches = training_batches(records["splits"]["train"], 32, 32768, 20261013)
    val_batches = list(length_batches(records["splits"]["validation-full"], 32, 32768))
    common = {"seed": 20261013, "epochs": 1, "width": 1024, "head_type": "bigru",
              "attention_heads": 4, "low_steps": 2, "high_cycles": 2, "batch_size": 32,
              "token_budget": 32768, "optimizer": "muon", "muon_lr": .003, "lr": 3e-5,
              "beta2": .999, "weight_decay": .01, "warmup_steps": 20,
              "schedule": "20-step warmup, steady, final 20% cosine to 0.1 peak",
              "gradient_clip": 1., "total_steps": len(batches), "examples": 8000,
              "feature_identity": index["identity"], "image_tag": args.image_tag,
              "image_id": args.image_id, "torch": str(torch.__version__), "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(), "choice_order_changed": False,
              "validation_rows": len(records["splits"]["validation-full"]),
              "train_batch_order_sha256": digest([[r["id"] for r in b] for b in batches]),
              "initial_pretrained_prior_coefficient": .1, "calibration": "none; T=1",
              "feature_file_hash_scan": False, "backbone_loaded": False,
              "input_prompt_changed": False, "ablation_source_sha256": file_hash(__file__),
              "source_sha256": {p: file_hash(Path(__file__).parent / p) for p in
                                ["model.py", "trainer.py", "optimizers.py", "train.py"]}}
    atomic_json(args.output_dir / "ablation.json", {"common": common, "readout": readout,
                "cases": [{"name": n, "hybrid": h, "ordinal_weight": o} for n, h, o in CASES],
                "untrained_control": "frozen-letter-readout",
                "interpretation": "readout/ordinal objective only; cached original prompt, not Mapika's native prompt"})
    results = []
    def status(phase, **extra):
        atomic_json(args.output_dir / "status.json", {"phase": phase, "updated_at": time.time(), **extra})
    def record_result(name, score, summary, logger):
        row = {"name": name, **summary, "accuracy": score["accuracy"], "cross_entropy": score["cross_entropy"],
               "macro_category_cross_entropy": score["macro_category_cross_entropy"],
               "by_category": score["by_category"], "by_source": score["by_source"],
               "by_kind": score["by_kind"], "ordinal": score.get("ordinal")}
        results.append(row)
        atomic_json(args.output_dir / "results.json", results)
        scalars = {"validation/full/" + k: score[k] for k in ["accuracy", "cross_entropy", "macro_category_cross_entropy"]}
        for label in ["category", "source", "kind"]:
            for k, values in score["by_" + label].items():
                for metric in ["accuracy", "cross_entropy"]:
                    scalars[f"validation/full/{label}/{k}/{metric}"] = values[metric]
        if score.get("ordinal"):
            scalars.update({"validation/full/ordinal/" + k: v for k, v in score["ordinal"].items()})
        logger.log(scalars, len(batches) if summary["training_seconds"] else 0)
        logger.finish()
        print(f'{name}: {score["accuracy"]:.3%}, CE {score["cross_entropy"]:.6f}, '
              f'macro CE {score["macro_category_cross_entropy"]:.6f}', flush=True)
    started = time.perf_counter()
    try:
        status("validating", trial="frozen-letter-readout")
        letter = LetterReadout(weight, readout["softcap"]).cuda()
        trainer = AblationTrainer(letter, None, None, args.bank_dir)
        begin = time.perf_counter()
        score = trainer.evaluate(val_batches)
        logger = RunLog(args.output_dir / "frozen-letter-readout/logging", "frozen-letter-readout",
                        common, args.overfit_url, "Bazza readout ablation")
        atomic_json(args.output_dir / "frozen-letter-readout/validation-full.json", score)
        record_result("frozen-letter-readout", score, {"training_seconds": 0., "trainable_parameters": 0,
                      "validation_seconds": time.perf_counter() - begin}, logger)
        del trainer, letter
        for name, hybrid, ordinal_weight in CASES:
            status("training", trial=name)
            torch.manual_seed(common["seed"])
            decision = make_head(3840, 1024, 4, 2, 2, "bigru")
            initial_hash = digest({k: __import__("hashlib").sha256(v.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
                                   for k, v in decision.state_dict().items()})
            model = HybridReadout(decision, LetterReadout(weight, readout["softcap"])) if hybrid else decision
            model = model.cuda()
            optimizer = make_optimizer(model, "muon", 3e-5, .01, .999, .003, False)
            def rate(step):
                if step < 20:
                    return (step + 1) / 20
                if step < len(batches) * .8:
                    return 1.
                fraction = min(1., (step - len(batches) * .8) / max(1., len(batches) * .2))
                return .1 + .9 * (1 + math.cos(math.pi * fraction)) / 2
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
            trainer = AblationTrainer(model, optimizer, scheduler, args.bank_dir, ordinal_weight=ordinal_weight)
            configuration = {**common, "hybrid": hybrid, "ordinal_weight": ordinal_weight,
                             "initial_core_sha256": initial_hash, "initialization": "fresh common core",
                             "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
            trial_dir = args.output_dir / name
            logger = RunLog(trial_dir / "logging", name, configuration, args.overfit_url, "Bazza readout ablation")
            observations = TrainingMetrics()
            trainer.on("after_forward", observations.forward)
            trainer.on("batch_end", observations.batch)
            def log(t, ctx):
                if t.state["step"] % 10 == 0:
                    logger.log({"train/" + k: v for k, v in ctx["metrics"].items()}, t.state["step"])
                    status("training", trial=name, **t.state)
            trainer.on("batch_end", log)
            atomic_json(trial_dir / "experiment.json", configuration)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            trainer.train(1, [batches])
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            summary = {**observations.report(), "training_seconds": seconds,
                       "trainable_parameters": configuration["trainable_parameters"],
                       "initial_core_sha256": initial_hash}
            if hybrid:
                summary["final_pretrained_prior_coefficient"] = float(model.prior_gate.detach().tanh())
            status("validating", trial=name)
            begin = time.perf_counter()
            score = trainer.evaluate(val_batches)
            torch.cuda.synchronize()
            summary["validation_seconds"] = time.perf_counter() - begin
            saved = {"head": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "state": trainer.state,
                     "config": configuration, "readout": readout, "validation_full": score}
            atomic_save(saved, trial_dir / "checkpoint-step-281.pt")
            atomic_json(trial_dir / "validation-full.json", score)
            record_result(name, score, summary, logger)
            del saved, trainer, model, decision, optimizer, scheduler
            gc.collect()
            torch.cuda.empty_cache()
        hashes = {r["initial_core_sha256"] for r in results if "initial_core_sha256" in r}
        if len(hashes) != 1:
            raise ValueError("Ablation core initialization differed")
        fields = ["name", "accuracy", "cross_entropy", "macro_category_cross_entropy",
                  "training_seconds", "trainable_parameters", "validation_seconds"]
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
