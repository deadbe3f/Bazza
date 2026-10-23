"""Adapt the existing recurrent head to a causal Gemma token memory."""
from __future__ import annotations

import torch
from torch import nn

from reasoning.model import ReasoningConfig, ReasoningHead


class GemmaDecisionHead(nn.Module):
    def __init__(self, encoder_width: int = 3840, width: int = 256,
                 attention_heads: int = 4, low_steps: int = 2, high_cycles: int = 2):
        super().__init__()
        self.reasoner = ReasoningHead(ReasoningConfig(
            encoder_width=encoder_width, width=width, attention_heads=attention_heads,
            low_steps=low_steps, high_cycles=high_cycles))

    def forward(self, hidden, attention_mask, option_positions, option_mask):
        # A causal BOS state cannot summarize a request. Prepend the final
        # valid token as the existing head's global-state seed. Option seeds
        # come AFTER each option description so they have read its contents.
        last = attention_mask.long().sum(-1) - 1
        summary = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        memory = torch.cat((summary[:, None], hidden), dim=1)
        mask = torch.cat((torch.ones_like(attention_mask[:, :1]), attention_mask), dim=1)
        positions = option_positions + 1
        result = self.reasoner(memory, mask, positions, positions[:, :0],
                               action_mask=option_mask)
        return result.action_logits
