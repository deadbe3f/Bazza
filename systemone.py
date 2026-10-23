"""System 1 inference using the state/questions/criteria request format."""

import json
import math
from collections.abc import Mapping
from pathlib import Path

import torch

from data import encode_input
from model import Bazza, BazzaDecisionHead, Gemma4
from model.gemma_backbone import matches_gemma_source
from utils import autocast, file_hash


class SystemOne:
    """Wrap a loaded Bazza model and tokenizer with a typed decision API."""

    def __init__(self, model, tokenizer, *, max_length=4096):
        if max_length < 1:
            raise ValueError("max_length must be positive")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = self.model.head.letter.weight.device
        if self.device.type == "cpu" and self.model.head_precision == "bf16":
            self.model.head.float()
            self.model.head_precision = "fp32"

    @classmethod
    def from_checkpoint(
        cls, checkpoint, *, backbone_path, device="cuda", max_length=4096
    ):
        """Load a trained decision checkpoint and the bundled live Gemma backbone.

        Training data, feature caches and experiment configuration files are not
        required. Architecture settings are read from the checkpoint itself.
        """
        from tokenizers import Tokenizer

        from model import gemma

        saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        config = saved.get("config", {})
        state = saved.get("head")
        if not isinstance(state, dict) or "letter.weight" not in state:
            raise ValueError("Expected a Bazza decision-head checkpoint")
        for name, expected in [
            ("backbone_checkpoint", gemma.CHECKPOINT),
            ("backbone_revision", gemma.REVISION),
        ]:
            if config.get(name) != expected:
                raise ValueError("Checkpoint backbone identity differs: " + name)
        required = [
            "hidden_width",
            "attention_heads",
            "low_steps",
            "high_cycles",
            "gate_width",
            "adapter_rank",
            "quantization",
        ]
        missing = [name for name in required if name not in config]
        if missing:
            raise ValueError(
                "Checkpoint lacks architecture settings: " + ", ".join(missing)
            )
        if config["quantization"] == "bnb-nf4":
            from importlib.util import find_spec

            if find_spec("bitsandbytes") is None:
                raise RuntimeError(
                    "Live bnb-nf4 inference requires bitsandbytes; install requirements-live.txt or build the Bazza Docker image"
                )
        weights = Path(backbone_path).expanduser().resolve()
        tokenizer_path = weights / "tokenizer.json"
        expected_tokenizer = (
            saved.get("identity", {}).get("readout_files", {}).get("tokenizer.json")
        )
        if (
            expected_tokenizer is not None
            and file_hash(tokenizer_path) != expected_tokenizer
        ):
            raise ValueError("Tokenizer differs from the training checkpoint")
        expected_source = (
            saved.get("identity", {}).get("source_sha256", {}).get("model/gemma.py")
        )
        if expected_source is not None and not matches_gemma_source(expected_source):
            raise ValueError("Bundled backbone differs from the training checkpoint")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        identity = dict(
            checkpoint=gemma.CHECKPOINT,
            revision=gemma.REVISION,
            weight_sha256=gemma.WEIGHT_SHA256,
            quantization=config["quantization"],
            implementation_sha256=file_hash(Path(gemma.__file__)),
        )
        backbone = Gemma4(
            weights,
            quantization=config["quantization"],
            selection={"identity": identity},
        )
        softcap = saved.get("readout", {}).get("softcap", config.get("softcap"))
        if softcap is None:
            raise ValueError("Checkpoint lacks the readout softcap")
        letter = state["letter.weight"]
        if (
            letter.ndim != 2
            or letter.shape[0] != 26
            or config.get("encoder_width") != letter.shape[1]
        ):
            raise ValueError("Checkpoint encoder/readout dimensions differ")
        head = BazzaDecisionHead(
            letter,
            softcap,
            encoder_width=letter.shape[1],
            width=config["hidden_width"],
            attention_heads=config["attention_heads"],
            low_steps=config["low_steps"],
            high_cycles=config["high_cycles"],
            gate_width=config["gate_width"],
            rank=config["adapter_rank"],
        )
        head.load_state_dict(state, strict=True)
        precision = (
            config.get("precision", "bf16")
            if torch.device(device).type == "cuda"
            else "fp32"
        )
        if precision == "fp32":
            head.float()
        model = Bazza(backbone, head, head_precision=precision).to(device).eval()
        backbone._load()
        return cls(model, tokenizer, max_length=max_length)

    def _prepare(self, state, questions):
        # Validate the complete request before any model execution.
        try:
            json.dumps(state, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("state must be finite JSON-compatible data") from error
        if not isinstance(questions, Mapping) or not questions:
            raise ValueError("questions must be a nonempty mapping")
        prepared = []
        for name, question in questions.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(question, Mapping)
            ):
                raise ValueError("Each question requires a nonempty name and object")
            kind = question.get("type")
            if kind not in ("choice", "noul", "score"):
                raise ValueError("Unsupported question type: " + str(kind))
            instructions, criteria = (
                question.get("instructions"),
                question.get("criteria"),
            )
            if not isinstance(instructions, str) or not instructions.strip():
                raise ValueError("Question instructions must be nonempty text")
            if not isinstance(criteria, Mapping) or not 2 <= len(criteria) <= 26:
                raise ValueError("criteria must contain between 2 and 26 candidates")
            if any(
                not isinstance(k, str)
                or not k
                or not isinstance(v, str)
                or not v.strip()
                for k, v in criteria.items()
            ):
                raise ValueError(
                    "Criterion keys and descriptions must be nonempty strings"
                )
            keys = list(criteria)
            if kind == "noul" and set(keys) != {"true", "false"}:
                raise ValueError("noul criteria must contain true and false")
            if kind == "score":
                try:
                    levels = [float(k) for k in keys]
                except ValueError as error:
                    raise ValueError("score criteria require numeric keys") from error
                if not all(math.isfinite(v) for v in levels) or len(set(levels)) != len(
                    levels
                ):
                    raise ValueError(
                        "score criteria require distinct finite numeric keys"
                    )
            row = dict(
                id=name,
                group_id=name,
                source="systemone",
                kind=kind,
                state=state,
                question=instructions,
                options=list(criteria.values()),
            )
            encoded = encode_input(row, self.tokenizer, self.max_length)
            if encoded is None:
                raise ValueError(
                    f"Question {name!r} exceeds max_length={self.max_length}; input was not truncated"
                )
            prepared.append((name, kind, keys, encoded))
        return prepared

    @torch.inference_mode()
    def decide(self, *, state, questions):
        """Return answers keyed by question name; candidate order is preserved.

        Each question is scored as a separate request, matching training. Values
        in probabilities are the model's softmax scores, without calibration.
        """
        prepared = self._prepare(state, questions)
        answers = {}
        for name, kind, keys, row in prepared:
            batch = self.model.backbone.prepare_batch([row], "inference").to(
                self.device
            )
            with autocast(self.device, self.model.head_precision):
                logits = self.model(batch).logits[0, : len(keys)].float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite decision logits")
            probabilities = dict(zip(keys, logits.softmax(-1).cpu().tolist()))
            chosen = keys[int(logits.argmax())]
            answer = dict(
                type=kind, choice=chosen, value=chosen, probabilities=probabilities
            )
            if kind == "noul":
                answer["noul"] = probabilities["true"]
                answer["value"] = probabilities["true"] > 0.5
            elif kind == "score":
                answer["score"] = sum(float(k) * v for k, v in probabilities.items())
                answer["value"] = answer["score"]
            answers[name] = answer
        return {"answers": answers}

    __call__ = decide
