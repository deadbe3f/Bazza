"""Load Python settings and assemble explicit model components."""

import os
import runpy
from pathlib import Path

import torch

from data import load_selection
from model import Bazza, BazzaDecisionHead, Gemma4, Gemma4FeatureCache
from settings import Config
from utils import ROOT, atomic_json, file_hash


def load_config(path):
    path = Path(path).resolve()
    if path.suffix != ".py":
        raise ValueError(
            "Training settings must be a Python file exporting config = Config(...)"
        )
    config = runpy.run_path(str(path)).get("config")
    if not isinstance(config, Config):
        raise TypeError("Settings file must export config = Config(...)")
    return config.resolved(path.parent)


def make_backbone(config, selection):
    if config.backbone_mode == "live":
        return Gemma4(
            config.backbone_path, quantization=config.quantization, selection=selection
        )
    return Gemma4FeatureCache(
        config.feature_cache_path,
        selection=selection,
        format=config.feature_format,
        banks=config.feature_banks,
        split_map=config.feature_split_map,
        layers=config.feature_layers,
        max_open_shards=config.max_open_shards,
    )


def prepare(config, device, *, settings_path=None):
    selection = load_selection(config.data_selection)
    expected = selection.get("identity", {})
    for key, value in (
        ("checkpoint", config.backbone_checkpoint),
        ("revision", config.backbone_revision),
        ("quantization", config.quantization),
        ("hidden_width", config.encoder_width),
    ):
        if key in expected and value is not None and expected[key] != value:
            raise ValueError("Settings and selected backbone differ: " + key)
    if config.target_key != "target":
        for rows in selection["splits"].values():
            for row in rows:
                if config.target_key not in row:
                    raise ValueError(
                        "Requested target unavailable: "
                        + row["id"]
                        + "/"
                        + config.target_key
                    )
                row["_cache_target"] = row["target"]
                row["target"] = row[config.target_key]
    backbone = make_backbone(config, selection)
    files = [*ROOT.glob("*.py"), *ROOT.glob("model/*.py"), *ROOT.glob("features/*.py")]
    identity = {
        "configuration": config.to_dict(),
        "selection_sha256": file_hash(config.data_selection),
        "feature_files": backbone.identity(),
        "source_sha256": {
            str(p.relative_to(ROOT)): file_hash(p) for p in sorted(files)
        },
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device": device,
        "image_id": os.environ.get("BAZZA_IMAGE_ID"),
        "hardware": torch.cuda.get_device_name()
        if torch.device(device).type == "cuda" and torch.cuda.is_available()
        else str(device),
    }
    if settings_path is not None:
        identity["settings_sha256"] = file_hash(settings_path)
    if config.letter_readout_path is not None:
        identity["letter_readout_sha256"] = file_hash(config.letter_readout_path)
    elif config.backbone_path is not None:
        identity["readout_files"] = {
            name: file_hash(config.backbone_path / name)
            for name in ("config.json", "tokenizer.json")
        }
    return selection, backbone, identity


def record_settings(config, source, directory):
    source = Path(source)
    destination = Path(directory) / "config-source.py"
    if not destination.exists():
        destination.write_bytes(source.read_bytes())
    atomic_json(Path(directory) / "resolved-config.json", config.to_dict())


def make_model(config, selection, saved=None, *, backbone=None):
    if saved is not None:
        weight = saved["head"]["letter.weight"]
        softcap = saved.get("readout", {}).get("softcap", config.softcap)
    elif config.letter_readout_path is not None:
        artifact = torch.load(
            config.letter_readout_path, map_location="cpu", weights_only=True
        )
        weight, softcap = artifact["weight"], artifact["softcap"]
    else:
        from model.gemma_backbone import load_letters

        weight, readout = load_letters(config.backbone_path, selection["identity"])
        softcap = readout["softcap"]
    if config.encoder_width != weight.shape[1] or config.softcap != softcap:
        raise ValueError("Configured encoder width/softcap differs from the readout")
    head = BazzaDecisionHead(
        weight,
        softcap,
        encoder_width=config.encoder_width,
        width=config.hidden_width,
        attention_heads=config.attention_heads,
        low_steps=config.low_steps,
        high_cycles=config.high_cycles,
        gate_width=config.gate_width,
        rank=config.adapter_rank,
    )
    if saved is not None:
        head.load_state_dict(saved["head"], strict=True)
    if config.precision == "fp32":
        head.float()
    return Bazza(
        backbone if backbone is not None else make_backbone(config, selection),
        head,
        head_precision=config.precision,
    ), {"softcap": softcap}


def warmup_steady_cosine(
    examples, *, total, warmup=640, decay_start=25000, minimum=0.1, warmup_offset=32
):
    import math

    if examples < warmup:
        return min(1.0, (examples + warmup_offset) / warmup)
    if examples < decay_start:
        return 1.0
    x = min(1.0, max(0.0, (examples - decay_start) / max(1, total - decay_start)))
    return minimum + (1 - minimum) * (1 + math.cos(math.pi * x)) / 2
