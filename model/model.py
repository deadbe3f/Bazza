"""Bazza composition; state-dict keys of the existing mixed head are unchanged."""

from dataclasses import dataclass

import torch
from torch import nn

from .readers import make_head
from .readout import LetterReadout, MixedReadout


class BazzaDecisionHead(MixedReadout):
    def __init__(
        self,
        letter_weight,
        softcap,
        encoder_width=3840,
        width=1024,
        attention_heads=4,
        low_steps=2,
        high_cycles=2,
        gate_width=32,
        rank=64,
    ):
        decision = make_head(
            encoder_width, width, attention_heads, low_steps, high_cycles, "bigru"
        )
        super().__init__(
            decision,
            LetterReadout(letter_weight, softcap),
            encoder_width,
            width,
            gate_width,
            rank,
        )

    def forward(self, hidden, attention_mask, option_positions, option_mask):
        return super().forward(hidden, attention_mask, option_positions, option_mask)


@dataclass
class DecisionOutput:
    logits: torch.Tensor
    option_mask: torch.Tensor


class Bazza(nn.Module):
    """One frozen backbone plus a trainable head; the original LM head is retained.

    A backbone adapter must return normalized final token states. Generation and
    System 1/System 2 routing are intentionally separate future operations.
    """

    def __init__(self, backbone, head, *, head_precision="bf16"):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.head = head
        self.head_precision = head_precision

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @property
    def checkpoint_module(self):
        """Reference the frozen backbone by identity; save only the trained head."""
        return self.head

    def forward(self, batch):
        batch = batch.to(self.head.letter.weight.device)
        # Match the exporter/live path: backbone and original RMSNorm run
        # outside the head's autocast context.
        with (
            torch.no_grad(),
            torch.autocast(batch.attention_mask.device.type, enabled=False),
        ):
            features = self.backbone(batch)
        if not torch.isfinite(features.hidden).all():
            raise FloatingPointError("Non-finite frozen features")
        hidden = (
            features.hidden.float()
            if self.head_precision == "fp32"
            else features.hidden
        )
        logits = self.head(
            hidden, batch.attention_mask, batch.option_positions, batch.option_mask
        )
        return DecisionOutput(logits, batch.option_mask)

    def diagnostics(self):
        return self.head.diagnostics()
