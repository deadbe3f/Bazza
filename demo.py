"""Load Bazza and answer a new state/questions decision request."""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import torch

from systemone import SystemOne
from utils import atomic_json, digest, file_hash, job_lock


def _infer(args, backbone, request):
    print("Loading Bazza and the live backbone...", file=sys.stderr, flush=True)
    model = SystemOne.from_checkpoint(
        args.checkpoint,
        backbone_path=backbone,
        device=args.device,
        max_length=args.max_length,
    )
    return model(**request)


def _interrupt(signum, frame):
    raise KeyboardInterrupt("signal " + str(signum))


def load_request(path):
    """Read and validate the top-level decision request."""
    request = json.loads(path.read_text())
    if set(request) != {"state", "questions"}:
        raise ValueError("Request requires exactly state and questions")

    return request


def resolve_backbone(args):
    """Resolve weights from the CLI or optional Python settings."""
    backbone = args.backbone_path
    if backbone is None:
        if args.config is None:
            raise ValueError(
                "Supply --backbone-path (or a Python --config with backbone_path)"
            )
        from configuration import load_config

        backbone = load_config(args.config).backbone_path

    return backbone


def request_identity(args, backbone, request):
    """Record the inputs and implementation used for resumable inference."""
    return {
        "checkpoint_sha256": file_hash(args.checkpoint),
        "request_sha256": digest(request),
        "backbone_path": str(Path(backbone).resolve()),
        "device": args.device,
        "max_length": args.max_length,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "image_id": os.environ.get("BAZZA_IMAGE_ID"),
        "backbone_files": {
            name: file_hash(Path(backbone) / name)
            for name in ("config.json", "tokenizer.json")
            if (Path(backbone) / name).exists()
        },
        "weight_file": (
            {
                "size": (Path(backbone) / "model.safetensors").stat().st_size,
                "mtime_ns": (Path(backbone) / "model.safetensors").stat().st_mtime_ns,
            }
            if (Path(backbone) / "model.safetensors").exists()
            else None
        ),
        "sources": {
            str(p.relative_to(Path(__file__).parent)): file_hash(p)
            for p in [
                Path(__file__),
                Path(__file__).with_name("systemone.py"),
                Path(__file__).with_name("data.py"),
                Path(__file__).with_name("utils.py"),
                *sorted(Path(__file__).parent.glob("model/*.py")),
            ]
        },
    }


def prepare_request(args, root, request, identity):
    """Create a request plan or verify an exact resume."""
    plan = root / "request-plan.json"
    if plan.exists():
        if not args.resume or json.loads(plan.read_text()) != identity:
            raise ValueError(
                "Existing/different demo; use exact --resume or a fresh output directory"
            )
    elif args.resume:
        raise ValueError("No request to resume")
    else:
        atomic_json(plan, identity)
        atomic_json(root / "request.json", request)


def load_response(root):
    """Load a completed response after verifying its checksum."""
    marker = json.loads((root / "complete.json").read_text())
    if marker["response_sha256"] != file_hash(root / "response.json"):
        raise ValueError("Saved response checksum differs")
    return json.loads((root / "response.json").read_text())


def infer_and_save(args, backbone, request, root):
    """Run inference with status, timing, and interruption handling."""
    atomic_json(
        root / "status.json",
        {"phase": "loading_model", "updated_at": time.time()},
    )

    handlers = {
        sig: signal.signal(sig, _interrupt) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    started = time.perf_counter()

    try:
        result = _infer(args, backbone, request)
        atomic_json(root / "response.json", result)
        atomic_json(
            root / "complete.json",
            {
                "response_sha256": file_hash(root / "response.json"),
                "questions": len(result["answers"]),
                "seconds": time.perf_counter() - started,
            },
        )
        atomic_json(
            root / "status.json",
            {"phase": "completed", "updated_at": time.time()},
        )
    except BaseException as error:
        atomic_json(
            root / "status.json",
            {
                "phase": "stopped"
                if isinstance(error, KeyboardInterrupt)
                else "failed",
                "error": str(error),
                "updated_at": time.time(),
            },
        )
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)

    return result


def main(args):
    request = load_request(args.request)
    backbone = resolve_backbone(args)

    if args.plan_only:
        print(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "backbone_path": str(backbone),
                    "request": request,
                    "device": args.device,
                },
                indent=2,
            )
        )
        return

    identity = request_identity(args, backbone, request)

    if args.output_dir is None:
        if args.resume:
            raise ValueError("--resume requires --output-dir")
        result = _infer(args, backbone, request)
    else:
        root = args.output_dir
        root.mkdir(parents=True, exist_ok=True)

        with job_lock(root):
            prepare_request(args, root, request, identity)

            if (root / "complete.json").exists():
                result = load_response(root)
            else:
                result = infer_and_save(args, backbone, request, root)

    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-path", type=Path)
    parser.add_argument(
        "--request", type=Path, default=Path(__file__).parent / "examples/decision.json"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    main(args)
