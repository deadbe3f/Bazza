"""Cached-feature readers around the unchanged Bazza decision core."""

from dataclasses import replace

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from gemma_decisions.model import GemmaDecisionHead


class FeatureReaderHead(nn.Module):
    def __init__(self, encoder_width, width, attention_heads, low_steps, high_cycles, reader):
        super().__init__()
        self.decision = GemmaDecisionHead(encoder_width, width, attention_heads, low_steps, high_cycles)
        # Move the existing projection ahead of the optional token reader. The
        # candidate/global seed construction and recurrent core are unchanged.
        self.project = self.decision.reasoner.project
        self.decision.reasoner.project = nn.Identity()
        self.decision.reasoner.config = replace(self.decision.reasoner.config, encoder_width=width)
        self.input_norm = nn.LayerNorm(encoder_width)
        self.reader = reader
        if reader != "norm":
            self.reader_norm = nn.LayerNorm(width)
            self.gate = nn.Parameter(torch.tensor(0.1))
        if reader == "mlp":
            self.update = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        elif reader == "bigru":
            if width % 2:
                raise ValueError("Bidirectional GRU needs an even decision width")
            self.update = nn.GRU(width, width // 2, batch_first=True, bidirectional=True)
        elif reader != "norm":
            raise ValueError("Unknown feature reader: " + reader)

    def forward(self, hidden, attention_mask, option_positions, option_mask):
        memory = self.project(self.input_norm(hidden))
        if self.reader == "mlp":
            memory = memory + self.gate.tanh() * self.update(self.reader_norm(memory))
        elif self.reader == "bigru":
            lengths = attention_mask.long().sum(-1)
            expected = torch.arange(hidden.shape[1], device=hidden.device)[None] < lengths[:, None]
            if torch.any(lengths < 1) or not torch.equal(attention_mask.bool(), expected):
                raise ValueError("Token GRU expects nonempty, right-padded inputs")
            # Explicit FP32 recurrent compute works with this image's cuDNN;
            # projection/attention retain the baseline's BF16 autocast policy.
            with torch.autocast(hidden.device.type, enabled=False):
                packed = pack_padded_sequence(self.reader_norm(memory.float()), lengths.cpu(),
                                              batch_first=True, enforce_sorted=False)
                output, _ = self.update(packed)
                output, _ = pad_packed_sequence(output, batch_first=True, total_length=hidden.shape[1])
            memory = memory + self.gate.tanh() * output.to(memory.dtype)
        return self.decision(memory, attention_mask, option_positions, option_mask)


def make_head(encoder_width, width=256, attention_heads=4, low_steps=2, high_cycles=2, head_type="recurrent"):
    if head_type == "recurrent":
        return GemmaDecisionHead(encoder_width, width, attention_heads, low_steps, high_cycles)
    return FeatureReaderHead(encoder_width, width, attention_heads, low_steps, high_cycles, head_type)
