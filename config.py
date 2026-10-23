"""Editable training settings. Model/optimizer construction stays in train.py."""

import os
from pathlib import Path

from settings import Config


def env_path(name, default):
    return Path(os.environ.get(name) or default).expanduser().resolve()


repository_root = Path(__file__).resolve().parent
root = env_path("BAZZA_ROOT", repository_root / "build")
project_root = env_path("BAZZA_PROJECT_ROOT", repository_root.parent)
data_root = env_path("BAZZA_DATA_ROOT", root / "data")
cache_root = env_path("BAZZA_CACHE_ROOT", root / "features/bazza")
model_root = env_path("BAZZA_MODEL_ROOT", root / "weights")
run_root = env_path("BAZZA_RUN_ROOT", root / "training/bazza")

config = Config(
    root=root,
    project_root=project_root,
    data_root=data_root,
    cache_root=cache_root,
    model_root=model_root,
    run_root=run_root,
    data_selection=env_path(
        "BAZZA_SELECTION_PATH", data_root / "groundbeef/v0.5/selection.json"
    ),
    backbone_mode="cache",  # 'live' uses Gemma4; 'cache' uses Gemma4FeatureCache
    backbone_path=model_root / "Gemma4-12B-it",
    backbone_revision="707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
    quantization="bnb-nf4",
    feature_format="routed",  # 'pt', 'layer', or 'routed'
    feature_banks={
        "old": cache_root / "groundbeef-v0.4/gemma4-12b-it-nf4-layer-taps",
        "addon": cache_root / "groundbeef-v0.5/addon-gemma4-12b-it-nf4-final",
        "remaining": cache_root / "groundbeef-v0.5/remaining-gemma4-12b-it-nf4-final",
    },
    feature_split_map={"full": "validation", "retention": "validation"},
    feature_layers=(48,),
    prefetch_batches=8,
    encoder_width=3840,
    hidden_width=1024,
    attention_heads=4,
    low_steps=2,
    high_cycles=2,
    gate_width=32,
    adapter_rank=64,
    softcap=30.0,
    target_key="target",
    weighted_loss=False,
    seed=20261013,
    epochs=1,
    batch_size=32,
    token_budget=32768,
    optimizer="muon",
    learning_rate=3e-5,
    muon_learning_rate=0.003,
    weight_decay=0.01,
    beta2=0.999,
    gradient_clip=1.0,
    precision="bf16",
    schedule="wsd",
    warmup_examples=640,
    warmup_batch_offset=32,
    decay_start_examples=25000,
    minimum_lr_factor=0.1,
    save_every_examples=1000,
    log_every_steps=10,
    validation_panels={"full": "validation", "retention": "retention"},
    validation_batch_size=32,
    validation_token_budget=32768,
    validation_every_examples=8000,
    initial_validation=True,
    keep_best_checkpoints=3,
    early_stopping=False,
    overfit_url=os.environ.get("OVERFIT_URL", ""),
    overfit_project="Bazza",
)
