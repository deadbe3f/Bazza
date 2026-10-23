"""Measure frozen Gemma plus recurrent-head inference and head backward fit."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .model import GemmaDecisionHead


def load_kiln(directory):
    spec = importlib.util.spec_from_file_location("kiln_gemma4", directory / "model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def memory():
    free, total = torch.cuda.mem_get_info()
    return {"allocated_mib": torch.cuda.memory_allocated() / 2**20,
            "reserved_mib": torch.cuda.memory_reserved() / 2**20,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "device_free_mib": free / 2**20, "device_total_mib": total / 2**20}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=Path, required=True)
    p.add_argument("--kiln-model-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--quantization", choices=("nf4", "bnb-nf4"), default="bnb-nf4")
    p.add_argument("--image-tag", required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 1024])
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261003)
    implementation = load_kiln(args.kiln_model_dir)
    report = {"checkpoint": implementation.CHECKPOINT, "revision": implementation.REVISION,
              "weight_sha256": implementation.WEIGHT_SHA256,
              "quantization": args.quantization, "image": args.image_tag,
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0),
              "implementation_sha256": hashlib.sha256((args.kiln_model_dir / "model.py").read_bytes()).hexdigest(),
              "scope": "Tokenized synthetic GPU inputs; batch one, frozen backbone, random head; no quality claims",
              "profiles": []}
    start = time.perf_counter()
    backbone = implementation.load_model(args.base_dir, quantization=args.quantization)
    torch.cuda.synchronize()
    report["load_seconds"] = time.perf_counter() - start
    report["backbone_resident"] = memory()
    head = GemmaDecisionHead(backbone.config.hidden_size).cuda()
    report["head_parameters"] = sum(p.numel() for p in head.parameters())
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4)
    for length in args.lengths:
        ids = torch.randint(3, 20000, (1, length), device="cuda")
        mask = torch.ones_like(ids)
        positions = torch.linspace(length // 2, length - 3, 8, device="cuda").long()[None]
        option_mask = torch.ones_like(positions, dtype=torch.bool)
        def forward():
            with torch.no_grad():
                hidden = backbone(ids, mask)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = head(hidden, mask, positions, option_mask)
            return hidden, logits
        head.eval()
        with torch.no_grad():
            for _ in range(2):
                hidden, logits = forward()
                del hidden, logits
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(args.repeats):
                begin = time.perf_counter()
                hidden, logits = forward()
                torch.cuda.synchronize()
                times.append((time.perf_counter() - begin) * 1000)
                del hidden, logits
        item = {"input_tokens": length, "choices": 8,
                "forward_median_ms": statistics.median(times), "forward_samples_ms": times,
                "inference_memory": memory()}
        head.train()
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        hidden, logits = forward()
        loss = F.cross_entropy(logits, torch.tensor([3], device="cuda"))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in head.parameters())
        project_gradient = head.reasoner.project.weight.grad.norm().item()
        if not finite or project_gradient == 0:
            raise AssertionError("Head backward failed")
        optimizer.step()
        torch.cuda.synchronize()
        item["head_training_step_ms"] = (time.perf_counter() - begin) * 1000
        item["head_training_memory"] = memory()
        item["head_gradients_finite"] = finite
        item["projection_gradient_norm"] = project_gradient
        if any(p.grad is not None for p in backbone.parameters()):
            raise AssertionError("Frozen backbone acquired gradients")
        report["profiles"].append(item)
        del hidden, logits, loss
        print(json.dumps(item), flush=True)
    out = args.output_dir / (args.quantization + "_memory.json")
    out.write_text(json.dumps(report, indent=2))
    print("Wrote", str(out), flush=True)


if __name__ == "__main__":
    main()
