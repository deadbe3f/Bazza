"""Two-timescale action/slot reasoning with a small copy-capable value decoder.

This is an HRM-inspired candidate, not a reproduction of the puzzle model's
training algorithm. The encoder token states remain available to the decoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ReasoningConfig:
    encoder_width: int = 1024
    width: int = 256
    attention_heads: int = 4
    low_steps: int = 2
    high_cycles: int = 2
    slot_gate_classes: int = 4  # omit, select, copy, generate

    def __post_init__(self) -> None:
        if self.width < 1 or self.width % self.attention_heads:
            raise ValueError("reasoning width must be divisible by attention_heads")
        if self.encoder_width < 1 or self.low_steps < 1 or self.high_cycles < 1:
            raise ValueError("encoder width and recurrence counts must be positive")


@dataclass
class ReasoningOutput:
    action_logits: torch.Tensor
    slot_gate_logits: torch.Tensor
    slot_states: torch.Tensor
    memory: torch.Tensor
    high_state: torch.Tensor


class LowLevelUpdate(nn.Module):
    """Shared update applied repeatedly to the small action/slot workspace."""

    def __init__(self, width: int, attention_heads: int) -> None:
        super().__init__()
        self.high_condition = nn.Linear(width, width)
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, attention_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, attention_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(
        self,
        low: torch.Tensor,
        high: torch.Tensor,
        memory: torch.Tensor,
        workspace_mask: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        conditioned = low + self.high_condition(high)[:, None, :]
        query = self.self_norm(conditioned)
        update, _ = self.self_attention(
            query, query, query, key_padding_mask=~workspace_mask, need_weights=False,
        )
        low = low + update
        update, _ = self.cross_attention(
            self.cross_norm(low), memory, memory,
            key_padding_mask=~memory_mask, need_weights=False,
        )
        low = low + update
        low = low + self.ffn(self.ffn_norm(low))
        return low.masked_fill(~workspace_mask[:, :, None], 0)


class ReasoningHead(nn.Module):
    """Slow global state and fast per-action/per-slot states over encoder memory."""

    def __init__(self, config: ReasoningConfig) -> None:
        super().__init__()
        self.config = config
        self.project = nn.Linear(config.encoder_width, config.width)
        self.high_init = nn.Linear(config.width, config.width)
        self.low_update = LowLevelUpdate(config.width, config.attention_heads)
        self.high_update = nn.GRUCell(config.width, config.width)
        self.action_norm = nn.LayerNorm(config.width)
        self.action_score = nn.Linear(config.width, 1)
        self.slot_norm = nn.LayerNorm(config.width)
        self.slot_gate = nn.Linear(config.width, config.slot_gate_classes)

    @staticmethod
    def _gather(memory: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if positions.ndim != 2 or positions.shape[0] != memory.shape[0]:
            raise ValueError("marker positions must have shape [batch, markers]")
        if torch.any(positions < 0) or torch.any(positions >= memory.shape[1]):
            raise ValueError("marker positions are outside encoder memory")
        index = positions[:, :, None].expand(-1, -1, memory.shape[-1])
        return memory.gather(dim=1, index=index)

    def forward(
        self,
        encoder_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        action_positions: torch.Tensor,
        slot_positions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        slot_mask: torch.Tensor | None = None,
    ) -> ReasoningOutput:
        if encoder_hidden.ndim != 3 or encoder_hidden.shape[-1] != self.config.encoder_width:
            raise ValueError("encoder_hidden has the wrong shape")
        if attention_mask.shape != encoder_hidden.shape[:2]:
            raise ValueError("attention_mask has the wrong shape")
        if action_positions.shape[1] < 1:
            raise ValueError("at least one action is required")
        if action_mask is None:
            action_mask = torch.ones_like(action_positions, dtype=torch.bool)
        if slot_mask is None:
            slot_mask = torch.ones_like(slot_positions, dtype=torch.bool)
        if action_mask.shape != action_positions.shape or slot_mask.shape != slot_positions.shape:
            raise ValueError("marker masks must match marker positions")
        memory_mask = attention_mask.bool()
        if not torch.all(memory_mask.any(dim=1)):
            raise ValueError("each input needs at least one unmasked token")
        workspace_mask = torch.cat((action_mask.bool(), slot_mask.bool()), dim=1)
        if not torch.all(workspace_mask.any(dim=1)):
            raise ValueError("each input needs an unmasked marker")

        memory = self.project(encoder_hidden)
        low = torch.cat((self._gather(memory, action_positions),
                         self._gather(memory, slot_positions)), dim=1)
        low = low.masked_fill(~workspace_mask[:, :, None], 0)
        high = self.high_init(memory[:, 0])
        for _ in range(self.config.high_cycles):
            for _ in range(self.config.low_steps):
                low = self.low_update(low, high, memory, workspace_mask, memory_mask)
            pooled = (low * workspace_mask[:, :, None]).sum(dim=1) / workspace_mask.sum(
                dim=1, keepdim=True,
            ).clamp_min(1)
            high = self.high_update(pooled, high)

        n_actions = action_positions.shape[1]
        action_states = low[:, :n_actions] + high[:, None, :]
        slot_states = low[:, n_actions:] + high[:, None, :]
        action_logits = self.action_score(self.action_norm(action_states)).squeeze(-1).float()
        action_logits = action_logits.masked_fill(~action_mask.bool(), -1e4)
        slot_gate_logits = self.slot_gate(self.slot_norm(slot_states)).float()
        slot_gate_logits = slot_gate_logits.masked_fill(~slot_mask.bool()[:, :, None], -1e4)
        return ReasoningOutput(action_logits, slot_gate_logits, slot_states, memory, high)


class SmallValueDecoder(nn.Module):
    """Short token decoder with a learned choice between generating and copying."""

    def __init__(self, vocab_size: int, width: int = 256) -> None:
        super().__init__()
        if vocab_size < 2 or width < 1:
            raise ValueError("vocab_size and width must be positive")
        self.vocab_size = vocab_size
        self.width = width
        self.embedding = nn.Embedding(vocab_size, width)
        self.memory_key = nn.Linear(width, width, bias=False)
        self.query = nn.Linear(width, width, bias=False)
        self.initial = nn.Linear(width, width)
        self.cell = nn.GRUCell(2 * width, width)
        self.output_norm = nn.LayerNorm(width)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.generate_gate = nn.Linear(2 * width, 1)

    def _prepare(
        self,
        slot_states: torch.Tensor,
        memory: torch.Tensor,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if slot_states.ndim != 3 or memory.ndim != 3:
            raise ValueError("slot_states and memory must have [batch, length, width] shape")
        batch, slots, width = slot_states.shape
        if width != self.width or memory.shape[0] != batch or memory.shape[2] != width:
            raise ValueError("decoder hidden widths or batch sizes do not match")
        if source_ids.shape != memory.shape[:2] or source_mask.shape != source_ids.shape:
            raise ValueError("source IDs and mask must match memory")
        if slots < 1 or not torch.all(source_mask.bool().any(dim=1)):
            raise ValueError("at least one slot and one source token are required")
        if torch.any(source_ids < 0) or torch.any(source_ids >= self.vocab_size):
            raise ValueError("source token ID is outside decoder vocabulary")
        expanded_memory = memory[:, None].expand(-1, slots, -1, -1).reshape(
            batch * slots, memory.shape[1], width,
        )
        expanded_ids = source_ids[:, None].expand(-1, slots, -1).reshape(batch * slots, -1)
        expanded_mask = source_mask.bool()[:, None].expand(-1, slots, -1).reshape(batch * slots, -1)
        hidden = torch.tanh(self.initial(slot_states.reshape(batch * slots, width)))
        keys = self.memory_key(expanded_memory)
        return hidden, expanded_memory, keys, expanded_ids, expanded_mask

    def _step(
        self,
        token: torch.Tensor,
        hidden: torch.Tensor,
        memory: torch.Tensor,
        keys: torch.Tensor,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.bmm(keys, self.query(hidden)[:, :, None]).squeeze(-1)
        scores = scores.float().div(self.width ** 0.5).masked_fill(~source_mask, -1e4)
        attention = torch.softmax(scores, dim=-1)
        context = torch.bmm(attention.to(memory.dtype)[:, None, :], memory).squeeze(1)
        token_embedding = self.embedding(token).to(context.dtype)
        hidden = self.cell(torch.cat((token_embedding, context), dim=-1), hidden)
        logits = F.linear(self.output_norm(hidden), self.embedding.weight, self.output_bias)
        generated = torch.softmax(logits.float(), dim=-1)
        copied = torch.zeros_like(generated).scatter_add_(1, source_ids, attention)
        gate = torch.sigmoid(self.generate_gate(torch.cat((hidden, context), dim=-1))).float()
        probabilities = gate * generated + (1 - gate) * copied
        return probabilities, hidden

    def generate(
        self,
        slot_states: torch.Tensor,
        memory: torch.Tensor,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        *,
        bos_token_id: int,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Greedy fixed-length decoding; fixed length makes latency comparable."""
        if not 0 <= bos_token_id < self.vocab_size or max_new_tokens < 1:
            raise ValueError("invalid BOS token or decoding length")
        batch, slots, _ = slot_states.shape
        hidden, expanded_memory, keys, expanded_ids, expanded_mask = self._prepare(
            slot_states, memory, source_ids, source_mask,
        )
        token = torch.full((batch * slots,), bos_token_id, device=source_ids.device, dtype=torch.long)
        result = []
        finished = torch.zeros(batch * slots, device=source_ids.device, dtype=torch.bool)
        for _ in range(max_new_tokens):
            probabilities, hidden = self._step(
                token, hidden, expanded_memory, keys, expanded_ids, expanded_mask,
            )
            token = probabilities.argmax(dim=-1)
            result.append(token)
            if eos_token_id is not None:
                finished |= token == eos_token_id
                if torch.all(finished):
                    break
        return torch.stack(result, dim=-1).reshape(batch, slots, len(result))

    def teacher_forced_nll(
        self,
        slot_states: torch.Tensor,
        memory: torch.Tensor,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        bos_token_id: int,
        pad_token_id: int | None = None,
    ) -> torch.Tensor:
        """Mean token NLL for later supervised training; no training is run here."""
        batch, slots, steps = target_ids.shape
        if slot_states.shape[:2] != (batch, slots) or steps < 1:
            raise ValueError("target IDs do not match slot states")
        hidden, expanded_memory, keys, expanded_ids, expanded_mask = self._prepare(
            slot_states, memory, source_ids, source_mask,
        )
        target = target_ids.reshape(batch * slots, steps)
        token = torch.full((batch * slots,), bos_token_id, device=target.device, dtype=torch.long)
        losses = []
        for step in range(steps):
            probabilities, hidden = self._step(
                token, hidden, expanded_memory, keys, expanded_ids, expanded_mask,
            )
            gold_probability = probabilities.gather(1, target[:, step:step + 1]).squeeze(1)
            losses.append(-gold_probability.clamp_min(1e-12).log())
            token = target[:, step]
        losses = torch.stack(losses, dim=-1)
        if pad_token_id is None:
            return losses.mean()
        valid = target != pad_token_id
        return (losses * valid).sum() / valid.sum().clamp_min(1)


class StructuredAgent(nn.Module):
    """Encoder, recurrent workspace, typed heads, and optional value decoder."""

    def __init__(self, encoder: nn.Module, config: ReasoningConfig, vocab_size: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.reasoner = ReasoningHead(config)
        self.decoder = SmallValueDecoder(vocab_size, config.width)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        action_positions: torch.Tensor,
        slot_positions: torch.Tensor,
        *,
        generated_slot_indices: torch.Tensor | None = None,
        bos_token_id: int = 0,
        max_new_tokens: int = 16,
    ) -> tuple[ReasoningOutput, torch.Tensor | None]:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        result = self.reasoner(hidden, attention_mask, action_positions, slot_positions)
        if generated_slot_indices is None:
            return result, None
        if generated_slot_indices.ndim != 1 or generated_slot_indices.numel() < 1:
            raise ValueError("generated_slot_indices must be a nonempty vector")
        slots = result.slot_states.index_select(1, generated_slot_indices)
        values = self.decoder.generate(
            slots, result.memory, input_ids, attention_mask,
            bos_token_id=bos_token_id, max_new_tokens=max_new_tokens,
        )
        return result, values
