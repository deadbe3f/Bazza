"""Frozen backbone forwards during head training; no hidden states on disk."""
from __future__ import annotations

import torch


class OnlineFeatures:
    def __init__(self, backbone, device, backbone_batch_size=1, letter_ids=None):
        self.backbone = backbone.requires_grad_(False).eval()
        self.device = device
        self.backbone_batch_size = backbone_batch_size
        self.letter_ids = letter_ids
        self.capture_baseline = False
        # Evaluation predictions are small CPU vectors, never token features.
        self.baselines = {}

    def collate(self, records):
        length = max(len(r["ids"]) for r in records)
        choices = max(len(r["target"]) for r in records)
        ids = torch.zeros(len(records), length, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        positions = torch.zeros(len(records), choices, dtype=torch.long, device=self.device)
        option_mask = torch.zeros_like(positions, dtype=torch.bool)
        targets = torch.zeros(len(records), choices, device=self.device)
        for i, record in enumerate(records):
            tokens, count = len(record["ids"]), len(record["target"])
            ids[i, :tokens] = torch.tensor(record["ids"], device=self.device)
            mask[i, :tokens] = 1
            positions[i, :count] = torch.tensor(record["positions"], device=self.device)
            option_mask[i, :count] = True
            targets[i, :count] = torch.tensor(record["target"], device=self.device)
        hidden_chunks = []
        # no_grad tensors can be saved by the head's backward pass. Do not use
        # inference_mode here: the head must learn from these constant inputs.
        with torch.no_grad():
            for start in range(0, len(records), self.backbone_batch_size):
                end = min(len(records), start + self.backbone_batch_size)
                chunk_length = max(len(r["ids"]) for r in records[start:end])
                hidden = self.backbone(ids[start:end, :chunk_length], mask[start:end, :chunk_length])
                if self.capture_baseline:
                    if self.letter_ids is None:
                        raise ValueError("Answer-letter IDs are required for stock evaluation")
                    last = mask[start:end].sum(-1) - 1
                    summary = hidden[torch.arange(end - start, device=self.device), last]
                    logits = self.backbone.logits(summary, self.letter_ids[:choices]).cpu()
                    for row, record in zip(logits, records[start:end]):
                        self.baselines[record["id"]] = row[:len(record["target"])].clone()
                # Batch one uses the exact unpadded forward from the audited
                # exporter. Pad only afterwards for the head's larger batch.
                hidden_chunks.append(torch.nn.functional.pad(hidden, (0, 0, 0, length - chunk_length)))
        hidden = torch.cat(hidden_chunks) if len(hidden_chunks) > 1 else hidden_chunks[0]
        return hidden, mask, positions, option_mask, targets

    def stock_predictions(self, records):
        return [self.baselines[r["id"]] for r in records]
