"""Backbone output contract and model-independent feature-cache machinery."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class BackboneOutput:
    hidden: torch.Tensor
    layer_states: dict[int, torch.Tensor]


class FeatureCacheMixin:
    """Prepare cached token states and bind them to their source bank."""

    def prepare_batch(self, rows, split):
        batch = self.store.prepare(rows, split)
        batch.split = split
        batch.feature_space = (
            "raw-final" if self.store.normalization is not None else "normalized-final"
        )
        batch.feature_source = self.cache_identity
        return batch

    def identity(self):
        return self.store.identity()


class FeatureCache(FeatureCacheMixin, nn.Module):
    """Frozen token-state source with optional stored final RMS normalization."""

    def __init__(
        self,
        directory=None,
        *,
        selection,
        format="pt",
        banks=None,
        split_map=None,
        layer=None,
        max_open_shards=128,
    ):
        super().__init__()
        if format in ("layer", "routed") and layer is None:
            raise ValueError("A layer-shard cache requires an explicit final layer")
        self.layer = layer
        from features.store import FeatureStore

        settings = {
            "kind": format,
            "root": directory,
            "layer": f"layer-{layer}" if layer is not None else None,
            "max_open_shards": max_open_shards,
            "split_map": split_map or {"full": "validation", "retention": "validation"},
        }
        if banks:
            settings["banks"] = banks
        self.store = FeatureStore(settings, selection, "cpu")
        from utils import digest

        self.cache_identity = digest(
            {
                "files": self.store.identity(),
                "root": str(directory),
                "banks": {name: str(path) for name, path in (banks or {}).items()},
            }
        )
        self.register_buffer("_device", torch.empty(0), persistent=False)

    def forward(self, batch):
        from features.bank import normalize_final

        if batch.hidden is None:
            batch = self.prepare_batch(batch.rows, batch.split)
        expected = (
            "raw-final" if self.store.normalization is not None else "normalized-final"
        )
        if (
            batch.feature_space != expected
            or batch.feature_source != self.cache_identity
        ):
            raise ValueError(
                "Cached batch belongs to a different feature space or cache"
            )
        raw = batch.hidden.to(self._device.device)
        if self.store.normalization is None:
            return BackboneOutput(raw, {})
        return BackboneOutput(
            normalize_final(raw, self.store.normalization), {self.layer: raw}
        )
