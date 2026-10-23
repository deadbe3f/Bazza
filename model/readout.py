"""Frozen letter prior, request gate and residual answer projection."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class LetterReadout(nn.Module):
    def __init__(self, weight, softcap):
        super().__init__()
        self.register_buffer("weight", weight.clone())
        self.softcap = softcap

    def forward(self, hidden, mask, positions, option_mask):
        last = mask.long().sum(-1) - 1
        final = hidden[torch.arange(len(hidden), device=hidden.device), last]
        logits = F.linear(final, self.weight[: option_mask.shape[1]]).float()
        if self.softcap:
            logits = self.softcap * (logits / self.softcap).tanh()
        return logits.masked_fill(~option_mask, -1e4)


class HybridReadout(nn.Module):
    def __init__(self, decision, letter, initial_prior=0.1):
        super().__init__()
        self.decision, self.letter = decision, letter
        # Deterministic scalar: adding this branch must not change initialization
        # of the shared decision core. AdamW uses the same auxiliary LR as before.
        self.prior_gate = nn.Parameter(torch.tensor(math.atanh(initial_prior)))

    def forward(self, hidden, mask, positions, option_mask):
        decision = self.decision(hidden, mask, positions, option_mask)
        prior = self.letter(hidden, mask, positions, option_mask)
        return (decision + self.prior_gate.tanh() * prior).masked_fill(
            ~option_mask, -1e4
        )


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
        nn.init.constant_(self.action_score.bias, math.log(0.1 / 0.9))
        self.last_gate = None

    def forward(self, hidden, mask, positions, option_mask):
        decision = self.decision(hidden, mask, positions, option_mask)
        prior = self.letter(hidden, mask, positions, option_mask)
        state = F.silu(
            self.request_project(self.request_norm(final_state(hidden, mask)))
        )
        stats = torch.cat(
            (
                confidence_features(decision, option_mask),
                confidence_features(prior, option_mask),
                option_mask.sum(-1, keepdim=True).float().log(),
            ),
            -1,
        )
        gate = self.action_score(torch.cat((state, stats), -1)).float().sigmoid()
        self.last_gate = gate.detach()
        return (decision + gate * prior).masked_fill(~option_mask, -1e4)

    def diagnostics(self):
        return {
            "request_gate_mean": float(self.last_gate.mean()),
            "request_gate_min": float(self.last_gate.min()),
            "request_gate_max": float(self.last_gate.max()),
        }


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
    def __init__(
        self, decision, letter, hybrid, reasoner_width=1024, encoder_width=3840, rank=64
    ):
        super().__init__()
        self.decision, self.letter, self.hybrid = decision, letter, hybrid
        self.answer_adapter = AnswerAdapter(reasoner_width, encoder_width, rank)
        if hybrid:
            self.prior_gate = nn.Parameter(torch.tensor(math.atanh(0.1)))
        self.last_relative_delta = None

    def forward(self, hidden, mask, positions, option_mask):
        # Capture the existing reasoner's output from its one normal forward.
        # A scoped hook avoids duplicating/reimplementing the reader or reasoner.
        high = []
        handle = self.decision.decision.reasoner.register_forward_hook(
            lambda module, inputs, output: high.append(output.high_state)
        )
        try:
            decision = self.decision(hidden, mask, positions, option_mask)
        finally:
            handle.remove()
        if len(high) != 1:
            raise ValueError("Expected exactly one reasoner forward")
        delta = self.answer_adapter(high.pop())
        original = final_state(hidden, mask)
        self.last_relative_delta = delta.detach().float().square().mean(
            -1
        ).sqrt() / original.detach().float().square().mean(-1).sqrt().clamp_min(1e-8)
        logits = F.linear(
            original + delta, self.letter.weight[: option_mask.shape[1]]
        ).float()
        cap = self.letter.softcap
        if cap:
            logits = cap * (logits / cap).tanh()
        if self.hybrid:
            logits = decision + self.prior_gate.tanh() * logits
        return logits.masked_fill(~option_mask, -1e4)

    def diagnostics(self):
        values = {
            "answer_delta_relative_rms_mean": float(self.last_relative_delta.mean())
        }
        if self.hybrid:
            values["pretrained_prior_coefficient"] = float(
                self.prior_gate.detach().tanh()
            )
        return values


class MixedReadout(RequestGateReadout):
    def __init__(
        self,
        decision,
        letter,
        encoder_width=3840,
        reasoner_width=1024,
        gate_width=32,
        rank=64,
    ):
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
            lambda module, inputs, output: high.append(output.high_state)
        )
        try:
            decision = self.decision(hidden, mask, positions, option_mask)
        finally:
            handle.remove()
        if len(high) != 1:
            raise ValueError("Expected exactly one reasoner forward")
        original = final_state(hidden, mask)
        delta = self.answer_adapter(high.pop())
        prior = F.linear(
            original + delta, self.letter.weight[: option_mask.shape[1]]
        ).float()
        cap = self.letter.softcap
        if cap:
            prior = cap * (prior / cap).tanh()
        prior = prior.masked_fill(~option_mask, -1e4)
        state = F.silu(self.request_project(self.request_norm(original)))
        stats = torch.cat(
            (
                confidence_features(decision, option_mask),
                confidence_features(prior, option_mask),
                option_mask.sum(-1, keepdim=True).float().log(),
            ),
            -1,
        )
        gate = self.action_score(torch.cat((state, stats), -1)).float().sigmoid()
        self.last_gate = gate.detach()
        self.last_relative_delta = delta.detach().float().square().mean(
            -1
        ).sqrt() / original.detach().float().square().mean(-1).sqrt().clamp_min(1e-8)
        return (decision + gate * prior).masked_fill(~option_mask, -1e4)

    def diagnostics(self):
        return {
            **super().diagnostics(),
            "answer_delta_relative_rms_mean": float(self.last_relative_delta.mean()),
        }
